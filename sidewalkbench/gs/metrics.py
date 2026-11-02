"""Route metrics + CSV report (coco-gs schema).

Per route we record the 8 metric keys::

    success_rate, collision_rate, freeze_rate, timeout_rate, off_sidewalk_rate,
    route_completion, spl, average_speed_mps

The first five are 0/1 (which terminating outcome fired); ``route_completion``
is PG's (EvalTracker): progress at the end of the route from the START projection over
the route length between the start and goal projections,
``clip((final_progress - start_progress) / (goal_progress - start_progress), 0, 1)``, 1.0 on success;
``spl`` is SPL as PG's EvalTracker ``overall`` per episode: success x min(that final centerline progress /
the distance travelled (summed every 20 Hz env step), 1);
``average_speed_mps`` is the mean of per-step |v_xy| over the route (reference
``xy_speeds_mps`` mean).

The CSV (``evaluation_metrics.csv``) has one ``overall`` row -- the route-weighted
mean (every route counts once) -- followed by one ``route`` row per route.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

from . import checks

METRIC_KEYS = (
    "success_rate",
    "collision_rate",
    "freeze_rate",
    "timeout_rate",
    "off_sidewalk_rate",
    "route_completion",
    "spl",
    "average_speed_mps",
)

FIELDNAMES = (
    "row_type",
    "scene",
    "num_routes",
    "route_index",
    "route_id",
    "route_source",
    "outcome",
    "reason",
    *METRIC_KEYS,
)

_OUTCOME_TO_RATE = {
    checks.SUCCESS: "success_rate",
    checks.COLLISION: "collision_rate",
    checks.FREEZE: "freeze_rate",
    checks.TIMEOUT: "timeout_rate",
    checks.OFF_SIDEWALK: "off_sidewalk_rate",
}


class RouteMetricAccumulator:
    """Tracks the latest progress + speed over one route; produces its metric row."""

    def __init__(self):
        self._progress_m = 0.0
        self._speed_sum = 0.0
        self._speed_n = 0

    def update(self, progress_m: float, speed_mps: float) -> None:
        self._progress_m = float(progress_m)
        self._speed_sum += float(speed_mps)
        self._speed_n += 1

    def finalize(self, *, outcome: str, start_progress_m: float, goal_progress_m: float,
                 path_length_m: float = 0.0) -> dict[str, float]:
        """``path_length_m``: the distance the robot travelled on the route (for SPL)."""
        rates = {k: 0.0 for k in ("success_rate", "collision_rate", "freeze_rate", "timeout_rate", "off_sidewalk_rate")}
        if outcome in _OUTCOME_TO_RATE:
            rates[_OUTCOME_TO_RATE[outcome]] = 1.0
        # start->goal route length (NOT absolute arc length).
        route_length_m = abs(float(goal_progress_m) - float(start_progress_m))
        if outcome == checks.SUCCESS:
            completion = 1.0
        elif route_length_m <= 1e-9:
            completion = 0.0
        else:
            directed = max(0.0, self._progress_m - float(start_progress_m))
            completion = max(0.0, min(1.0, directed / route_length_m))
        # SPL as PG's EvalTracker "overall" per episode: success x min(progress along the centerline / distance
        # travelled, 1) -- progress = final centerline progress from the start projection (PG optimal_path)
        progress = self._progress_m - float(start_progress_m)
        spl = (min(progress / path_length_m, 1.0) if path_length_m > 1e-6 else 0.0) if outcome == checks.SUCCESS else 0.0
        avg_speed = (self._speed_sum / self._speed_n) if self._speed_n else 0.0
        return {**rates, "route_completion": completion, "spl": spl, "average_speed_mps": avg_speed}


@dataclass
class RouteResult:
    scene: str
    route_index: int
    route_id: str
    route_source: str
    outcome: str
    reason: str
    metrics: dict[str, float]


@dataclass
class MetricsReport:
    scene: str
    routes: list[RouteResult] = field(default_factory=list)

    def add(self, result: RouteResult) -> None:
        self.routes.append(result)

    def _mean_metrics(self, results: list[RouteResult]) -> dict[str, float]:
        n = len(results)
        if n == 0:
            return {k: 0.0 for k in METRIC_KEYS}
        return {k: sum(r.metrics[k] for r in results) / n for k in METRIC_KEYS}

    def rows(self) -> list[dict]:
        """One ``overall`` row (route-weighted mean over every route) + the route rows."""
        rows = [self._row("overall", scene=self.scene, num_routes=len(self.routes),
                          metrics=self._mean_metrics(self.routes))]
        for r in self.routes:
            rows.append(self._row("route", scene=r.scene, num_routes="", route_index=r.route_index,
                                  route_id=r.route_id, route_source=r.route_source,
                                  outcome=r.outcome, reason=r.reason, metrics=r.metrics))
        return rows

    @staticmethod
    def _row(row_type, *, scene, num_routes, route_index="", route_id="", route_source="",
             outcome="", reason="", metrics) -> dict:
        return {
            "row_type": row_type, "scene": scene, "num_routes": num_routes, "route_index": route_index,
            "route_id": route_id, "route_source": route_source, "outcome": outcome, "reason": reason,
            **{k: metrics[k] for k in METRIC_KEYS},
        }

    def write_csv(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(FIELDNAMES))
            w.writeheader()
            for row in self.rows():
                w.writerow(row)

    def summary_line(self, *, requested: int | None = None) -> str:
        m = self._mean_metrics(self.routes)
        n = len(self.routes)
        return (
            f"Summary: routes={n}/{requested if requested is not None else n} success={m['success_rate']*100:.1f}% "
            f"collision={m['collision_rate']*100:.1f}% freeze={m['freeze_rate']*100:.1f}% "
            f"timeout={m['timeout_rate']*100:.1f}% off_sidewalk={m['off_sidewalk_rate']*100:.1f}% "
            f"completion={m['route_completion']*100:.1f}% spl={m['spl']:.3f} speed={m['average_speed_mps']:.2f}m/s"
        )

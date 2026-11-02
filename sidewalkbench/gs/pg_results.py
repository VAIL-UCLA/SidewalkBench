"""PG-format results for the GS eval (``sidewalkbench.pg.utils.eval.EvalTracker`` schema).

A GS run must write the same ``results.pkl`` a PG run does, so
``sidewalkbench/gs/tools/collect_results.py`` and any
downstream table code consume both without special cases. This module
reimplements EvalTracker's per-episode bookkeeping and its ``summarize()`` /
``save()`` (joblib) with the identical 27 keys, formulas, console lines and NaN /
inf conventions. Human-related quantities take their "no pedestrians" values
(in-personal-space fractions 0 -> outside fractions 1.0, min human distance inf,
human collisions 0), exactly as a PG run with ``num_peds_per_terrain: 0``.

Per-episode values (EvalTracker.step):
  * path_length      : integrated XY distance travelled
  * optimal_path     : directed centerline progress at termination (s_end - s_start,
                       NOT clamped, as in PG)
  * route_completion : clip(optimal_path / route_length, 0, 1), forced to 1.0 on success
  * episode_time     : steps * dt (only successes are kept in the time list)
  * time_efficiency  : optimal_path / episode_time
  * energy_efficiency: optimal_path / (sum |tau * qdot| dt) * 1000  (0 if no energy)
  * path_efficiency  : min(optimal_path / path_length, 1)
"""

from __future__ import annotations

import numpy as np


class PGEpisodeAccumulator:
    """Per-env running sums for one episode (path length, energy)."""

    def __init__(self):
        self.reset(None)

    def reset(self, xy) -> None:
        self.path_length = 0.0
        self.energy_j = 0.0
        self.last_xy = None if xy is None else np.asarray(xy, dtype=float).copy()

    def step(self, xy, power_w: float, dt: float) -> None:
        xy = np.asarray(xy, dtype=float)
        if self.last_xy is not None:
            self.path_length += float(np.linalg.norm(xy - self.last_xy))
        self.last_xy = xy.copy()
        self.energy_j += float(power_w) * float(dt)

    def moved_to(self, xy) -> None:
        """The robot was teleported to ``xy``: measure from there on, keeping the sums so far."""
        self.last_xy = np.asarray(xy, dtype=float).copy()


class PGResults:
    """Episode lists + EvalTracker-identical ``summarize()`` / ``save()``."""

    def __init__(self, num_envs: int, planned_episodes: int):
        self.num_envs = int(num_envs)
        self.planned_episodes = int(planned_episodes)
        self.episode_num = [0] * self.num_envs
        self.success: list[bool] = []
        self.collision: list[bool] = []
        self.oob: list[bool] = []
        self.timeout: list[bool] = []
        self.freeze: list[bool] = []
        self.col_static: list[bool] = []
        self.col_dynamic: list[bool] = []
        self.route_completion: list[float] = []
        self.time_efficiency: list[float] = []
        self.energy_efficiency: list[float] = []
        self.in_ps_frac: list[float] = []
        self.in_proj_ps_frac: list[float] = []
        self.min_human_distance: list[float] = []
        self.path_length: list[float] = []
        self.optimal_path: list[float] = []
        self.episode_time: list[float] = []

    def add_episode(
        self,
        env_idx: int,
        *,
        success: bool,
        collision_static: bool,
        oob: bool,
        timeout: bool,
        freeze: bool,
        path_length: float,
        directed_progress_m: float,
        route_length_m: float,
        episode_time_s: float,
        energy_j: float,
    ) -> None:
        is_collision_dynamic = False
        is_any_collision = bool(collision_static) or is_collision_dynamic
        optimal_path = float(directed_progress_m)  # PG: s_current - s_start, unclamped
        if route_length_m > 0:
            route_completion = min(optimal_path / route_length_m, 1.0)
        else:
            route_completion = 0.0
        route_completion = max(route_completion, 0.0)
        if success:
            route_completion = 1.0
        ep_time = float(episode_time_s)
        time_efficiency = (optimal_path / ep_time) if ep_time > 0 else 0.0
        energy_efficiency = (optimal_path / energy_j) * 1000 if energy_j > 0 else 0.0
        in_ps_frac = 0.0
        in_proj_ps_frac = 0.0
        min_human_distance = float("inf")

        self.success.append(bool(success))
        self.collision.append(is_any_collision)
        self.oob.append(bool(oob))
        self.timeout.append(bool(timeout))
        self.freeze.append(bool(freeze))
        self.col_static.append(bool(collision_static))
        self.col_dynamic.append(is_collision_dynamic)
        self.path_length.append(float(path_length))
        self.route_completion.append(route_completion)
        if success:
            self.episode_time.append(ep_time)
        self.time_efficiency.append(time_efficiency)
        self.energy_efficiency.append(energy_efficiency)
        self.in_ps_frac.append(in_ps_frac)
        self.in_proj_ps_frac.append(in_proj_ps_frac)
        self.min_human_distance.append(min_human_distance)
        self.optimal_path.append(optimal_path)
        self.episode_num[env_idx] += 1

        path_eff_ep = min((optimal_path / path_length) if path_length > 1e-6 else 0.0, 1.0)
        overall_ep = float(success) * path_eff_ep
        print(
            f"[Eval] Env {env_idx} episode {sum(self.episode_num)}/{self.planned_episodes}  "
            f"success={bool(success)}  col_static={bool(collision_static)}  col_dyn={is_collision_dynamic}  "
            f"oob={bool(oob)}  timeout={bool(timeout)}  freeze={bool(freeze)}  "
            f"route_comp={route_completion:.3f}  path_len={path_length:.3f}m  opt_path={optimal_path:.3f}m  "
            f"path_eff={path_eff_ep:.4f}  time={ep_time:.2f}s  "
            f"time_eff={time_efficiency:.3f}  energy_eff={energy_efficiency:.4f}  "
            f"out_ps={1.0 - in_ps_frac:.4f}  out_proj_ps={1.0 - in_proj_ps_frac:.4f}  "
            f"min_hum_dist={min_human_distance:.3f}  overall={overall_ep:.4f}",
            flush=True,
        )

    def summarize(self, env_config: dict | None = None, pg_config: dict | None = None) -> dict | None:
        total = len(self.success)
        if total == 0:
            print("[Eval] No episodes completed.")
            return None

        succ_mask = np.array(self.success, dtype=bool)

        def _succ(values):
            return np.array(values)[succ_mask] if succ_mask.any() else np.array([])

        succ_time_efficiency = _succ(self.time_efficiency)
        succ_energy_efficiency = _succ(self.energy_efficiency)
        succ_in_ps = _succ(self.in_ps_frac)
        succ_in_proj_ps = _succ(self.in_proj_ps_frac)
        succ_min_human_distance = _succ(self.min_human_distance)

        all_path_eff = np.array(
            [min(o / p, 1.0) if p > 1e-6 else 0.0 for o, p in zip(self.optimal_path, self.path_length)]
        )
        succ_path_eff = all_path_eff[succ_mask] if succ_mask.any() else np.array([])

        all_timeout_combined = np.array([t or f for t, f in zip(self.timeout, self.freeze)], dtype=float)
        all_env_collision = np.array([o or s for o, s in zip(self.oob, self.col_static)], dtype=float)
        all_outside_ps = 1.0 - np.array(self.in_ps_frac)
        succ_outside_ps = 1.0 - succ_in_ps if len(succ_in_ps) > 0 else np.array([])
        all_outside_proj_ps = 1.0 - np.array(self.in_proj_ps_frac)
        succ_outside_proj_ps = 1.0 - succ_in_proj_ps if len(succ_in_proj_ps) > 0 else np.array([])

        success_rate = np.mean(self.success)
        overall = np.mean(np.array(self.success, dtype=float) * all_path_eff)

        results = {
            "success_rate": success_rate,
            "human_collision_rate": np.mean(self.col_dynamic),
            "timeout_rate": np.mean(all_timeout_combined),
            "env_collision_rate": np.mean(all_env_collision),
            "avg_route_completion": np.mean(self.route_completion),
            "time_efficiency": np.mean(self.time_efficiency),
            "path_efficiency": np.mean(all_path_eff),
            "energy_efficiency": np.mean(self.energy_efficiency) if self.energy_efficiency else 0.0,
            "time_efficiency_success": np.mean(succ_time_efficiency) if len(succ_time_efficiency) > 0 else float("nan"),
            "path_efficiency_success": np.mean(succ_path_eff) if len(succ_path_eff) > 0 else float("nan"),
            "energy_efficiency_success": (
                np.mean(succ_energy_efficiency) if len(succ_energy_efficiency) > 0 else float("nan")
            ),
            "avg_frac_outside_ps": np.mean(all_outside_ps),
            "avg_frac_outside_proj_ps": np.mean(all_outside_proj_ps),
            "avg_min_human_distance": np.mean(self.min_human_distance) if self.min_human_distance else float("inf"),
            "avg_frac_outside_ps_success": np.mean(succ_outside_ps) if len(succ_outside_ps) > 0 else float("nan"),
            "avg_frac_outside_proj_ps_success": (
                np.mean(succ_outside_proj_ps) if len(succ_outside_proj_ps) > 0 else float("nan")
            ),
            "avg_min_human_distance_success": (
                np.mean(succ_min_human_distance) if len(succ_min_human_distance) > 0 else float("inf")
            ),
            "overall": overall,
            "total_episodes": total,
            "num_envs": self.num_envs,
        }
        if env_config is not None:
            results.update(
                {
                    "scenario_type": env_config.get("scenario_type", "default"),
                    "model_type": env_config.get("model_type", ""),
                    "terrain_type": env_config.get("terrain_type", ""),
                    "ped_lateral": env_config.get("ped_lateral", "center"),
                    "ped_crossing_direction": env_config.get("ped_crossing_direction", "left_to_right"),
                }
            )
        if pg_config is not None:
            results.update(
                {
                    "num_peds_per_terrain": pg_config.get("num_peds_per_terrain", 0),
                    "num_static_obstacles": pg_config.get("num_walkable_objects_per_terrain", 0),
                }
            )

        print("\n" + "=" * 70)
        print(f"  EVALUATION RESULTS  ({total} episodes across {self.num_envs} envs)")
        print("=" * 70)
        print("  --- TASK-RELATED ---")
        print(f"  Success Rate              : {results['success_rate']:.4f}")
        print(f"  Human Collision Rate      : {results['human_collision_rate']:.4f}")
        print(f"  Timeout Rate (freeze+to)  : {results['timeout_rate']:.4f}")
        print(f"  Env Collision Rate (oob+s): {results['env_collision_rate']:.4f}")
        print(f"  Avg Route Completion      : {results['avg_route_completion']:.4f}")
        print("  --- EFFICIENCY-RELATED ---")
        print(
            f"  Time Efficiency           : {results['time_efficiency']:.3f} m/s  (success: {results['time_efficiency_success']:.3f})"
        )
        print(
            f"  Path Efficiency           : {results['path_efficiency']:.4f}  (success: {results['path_efficiency_success']:.4f})"
        )
        print(
            f"  Energy Efficiency         : {results['energy_efficiency']:.4f} m/J  (success: {results['energy_efficiency_success']:.4f})"
        )
        print("  --- HUMAN-RELATED ---")
        print(
            f"  Avg Frac Outside PS       : {results['avg_frac_outside_ps']:.4f}  (success: {results['avg_frac_outside_ps_success']:.4f})"
        )
        print(
            f"  Avg Frac Outside Proj PS  : {results['avg_frac_outside_proj_ps']:.4f}  (success: {results['avg_frac_outside_proj_ps_success']:.4f})"
        )
        print(
            f"  Avg Min Human Distance    : {results['avg_min_human_distance']:.3f} m   (success: {results['avg_min_human_distance_success']:.3f} m)"
        )
        print("  --- OVERALL ---")
        print(f"  Overall (mean(succ * PE * OS)): {results['overall']:.4f}")
        print("=" * 70, flush=True)
        return results

    def save(self, path: str, env_config: dict | None = None, pg_config: dict | None = None) -> dict | None:
        """Summarize and save results to a pickle file (joblib, like EvalTracker.save)."""
        import joblib

        results = self.summarize(env_config, pg_config)
        if results is not None:
            joblib.dump(results, path)
            print(f"Saved results to {path}", flush=True)
        return results

"""
Scene graph that extracts POIs (points of interest) from the terrain and spawned
static objects so that the LLM program generator can reference concrete locations.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Affordance / action vocabulary
# ---------------------------------------------------------------------------

# Atomic actions a pedestrian can perform.  Each maps to a Motion-LCM prompt
# (Step 4 of the plan) and optionally requires a POI with a matching affordance.
ACTIONS = {
    "walk_to": {"needs_poi": True, "affordance": None, "motion_prompt": "a person is walking forward"},
    "sit": {"needs_poi": True, "affordance": "sittable", "motion_prompt": "a person sits down on a bench"},
    "stand_up": {"needs_poi": False, "affordance": None, "motion_prompt": "a person stands up"},
    "idle": {"needs_poi": False, "affordance": None, "motion_prompt": "a person is standing still"},
    "wait": {"needs_poi": True, "affordance": None, "motion_prompt": "a person is standing and waiting"},
    "use_phone": {
        "needs_poi": False,
        "affordance": None,
        "motion_prompt": "a person is standing and looking at their phone",
    },
    "look_around": {"needs_poi": False, "affordance": None, "motion_prompt": "a person is standing and looking around"},
    "cross_street": {"needs_poi": True, "affordance": "crosswalk", "motion_prompt": "a person is walking forward"},
    "enter_building": {"needs_poi": True, "affordance": "enterable", "motion_prompt": "a person is walking forward"},
    "exit_building": {"needs_poi": True, "affordance": "enterable", "motion_prompt": "a person is walking forward"},
    "eat": {"needs_poi": True, "affordance": "sittable", "motion_prompt": "a person is sitting and eating"},
    "drink": {"needs_poi": True, "affordance": "sittable", "motion_prompt": "a person is sitting and drinking"},
    "use_vending": {
        "needs_poi": True,
        "affordance": "vending",
        "motion_prompt": "a person is standing in front of a vending machine",
    },
    "board_bus": {"needs_poi": True, "affordance": "bus_stop", "motion_prompt": "a person is walking forward"},
    "alight_bus": {"needs_poi": True, "affordance": "bus_stop", "motion_prompt": "a person is walking forward"},
}

# Map scene category names → affordance tags
_ALL_AFFORDANCES = ["sittable", "vending", "bus_stop"]  # debug: every object gets all

_CATEGORY_AFFORDANCES: dict[str, list[str]] = {
    k: list(_ALL_AFFORDANCES)
    for k in [
        "Bench",
        "Chair",
        "BusStop",
        "Vending_machine",
        "FoodTruck",
        "Building",
        "Crosswalk",
        "Telephone_booth",
        "TrashCan",
        "Mailbox",
        "Advertising_board",
        "Lamp_post",
        "FireHydrant",
        "Bollard",
        "Tree",
        "Vegetation",
        "Bonsai",
        "Cone",
    ]
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class POI:
    """A point of interest in the scene."""

    name: str  # e.g. "Bench_0"
    category: str  # e.g. "Bench"
    affordances: list[str]  # e.g. ["sittable"]
    position: np.ndarray  # world (x, y, z)
    yaw: float = 0.0  # orientation
    bbox: np.ndarray | None = None  # (length, width, height)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "affordances": self.affordances,
            "position": self.position.tolist(),
        }


# ---------------------------------------------------------------------------
# Scene graph
# ---------------------------------------------------------------------------


class SceneGraph:
    """
    Builds a list of POIs from:
      1. Static objects spawned in each terrain cell (clone_static_states)
      2. Terrain geometry (walkable region endpoints as generic waypoints)

    The graph is per-terrain-cell so each pedestrian group gets its own
    local set of reachable POIs.
    """

    def __init__(self):
        # Build a flat mapping: category_name -> affordances
        self._affordance_map = dict(_CATEGORY_AFFORDANCES)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_scene(
        self,
        clone_static_states: dict[int, torch.Tensor | list],
        clone_static_categories: dict[int, list[str]],
        terrain_origins: np.ndarray,  # [rows, cols, 3]
        terrain_walkable_samplers,  # [rows][cols] callables
        terrain_size: tuple[float, float],
        num_cols: int,
        terrain_centerline_splines: list[list] | None = None,
        terrain_indices: list[int] | None = None,
        rngs: dict[int, np.random.RandomState] | None = None,
    ) -> dict[int, list[POI]]:
        """
        Returns {terrain_idx: [POI, ...]} for every terrain cell that has
        static objects or walkable area.

        If terrain_centerline_splines is provided, waypoints use arc-length
        along the centerline (for curvy terrains); otherwise they fall back
        to terrain_width * frac for straight sidewalks.
        """
        pois_per_terrain: dict[int, list[POI]] = {}
        num_rows = terrain_origins.shape[0]

        for tidx in (range(num_rows * num_cols) if terrain_indices is None else terrain_indices):
            i, j = tidx // num_cols, tidx % num_cols
            origin = terrain_origins[i, j]
            pois: list[POI] = []

            # --- POIs from static objects ---
            if clone_static_states and tidx in clone_static_states:
                states = clone_static_states[tidx]
                categories = clone_static_categories.get(tidx, [])
                if states is not None and (isinstance(states, torch.Tensor) and states.numel() > 0):
                    for obj_idx in range(states.shape[0]):
                        s = states[obj_idx]
                        pos = s[:3].cpu().numpy()
                        bbox = s[3:6].cpu().numpy()
                        yaw = float(s[6].cpu())
                        cat = categories[obj_idx] if obj_idx < len(categories) else "object"
                        affs = self._affordance_map.get(cat, list(_ALL_AFFORDANCES))
                        pois.append(
                            POI(
                                name=f"{cat}_{tidx}_{obj_idx}",
                                category=cat,
                                affordances=list(affs),
                                position=pos,
                                yaw=yaw,
                                bbox=bbox,
                            )
                        )

            # --- Walkable-region endpoint POIs (start / end of sidewalk) ---
            spline_data = None
            if terrain_centerline_splines is not None:
                spline_data = terrain_centerline_splines[i][j]
            if spline_data is not None:
                arc_to_xy, total_len = spline_data
                for label, frac in [("sidewalk_start", 0.1), ("sidewalk_end", 0.9)]:
                    s = frac * total_len
                    xy = arc_to_xy(np.array([s]))
                    pos = np.array([float(xy[0, 0]), float(xy[0, 1]), 0.0]) + origin
                    pois.append(
                        POI(
                            name=f"{label}_{tidx}",
                            category="waypoint",
                            affordances=[],
                            position=pos,
                        )
                    )
            else:
                for label, frac in [("sidewalk_start", 0.1), ("sidewalk_end", 0.9)]:
                    sampler = terrain_walkable_samplers[i][j]
                    xx, yy = sampler(walkable=True, num_samples=1, rng=None if rngs is None else rngs[tidx])
                    yy = np.array([frac * terrain_size[1]])
                    pos = np.array([float(xx[0]), float(yy[0]), 0.0]) + origin
                    pois.append(
                        POI(
                            name=f"{label}_{tidx}",
                            category="waypoint",
                            affordances=[],
                            position=pos,
                        )
                    )

            pois_per_terrain[tidx] = pois

        return pois_per_terrain

    def get_manifest(self, pois: list[POI]) -> str:
        """Return a compact text description of available POIs for LLM prompting."""
        if not pois:
            return "No points of interest in this area."
        lines = []
        for p in pois:
            aff_str = ", ".join(p.affordances) if p.affordances else "none"
            lines.append(f"- {p.name} (category={p.category}, affordances={aff_str})")
        return "\n".join(lines)

    def find_by_affordance(self, pois: list[POI], affordance: str) -> list[POI]:
        return [p for p in pois if affordance in p.affordances]

    def find_by_name(self, pois: list[POI], name: str) -> POI | None:
        for p in pois:
            if p.name == name:
                return p
        return None

    def nearest_poi(self, pois: list[POI], position: np.ndarray, affordance: str | None = None) -> POI | None:
        candidates = self.find_by_affordance(pois, affordance) if affordance else pois
        if not candidates:
            return None
        dists = [np.linalg.norm(p.position[:2] - position[:2]) for p in candidates]
        return candidates[int(np.argmin(dists))]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

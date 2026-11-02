# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Runtime terrain texture and sky randomization.

At startup, up to ten MDL materials per surface category (walkable, terrain,
road) are sampled from all available files and spawned. Synchronized resets
reuse these pools. The ``other`` group keeps its vertex-color material and is
re-tinted, while the dome light samples from a pool of up to ten HDR maps.
"""

from __future__ import annotations

from collections.abc import Sequence
from glob import glob

import numpy as np

# Discover the full asset pools instead of maintaining a curated subset.
# Paths are relative to the repository root. Unknown non-walkable groups may
# use either terrain or road materials.
CATEGORY_PATTERNS = {
    "walkable": ("./assets/materials/material_sidewalk_mdl/*.mdl",),
    "terrain": ("./assets/materials/material_terrain_mdl/*.mdl",),
    "road": ("./assets/materials/material_road_mdl/*.mdl",),
    "non_walkable": (
        "./assets/materials/material_terrain_mdl/*.mdl",
        "./assets/materials/material_road_mdl/*.mdl",
    ),
}
BACKGROUND_HDR_PATTERN = "./assets/materials/material_background_hdr/*.hdr"
TEXTURE_POOL_SIZE = 10

# Per-episode multiplicative tint applied to the vertex-colored ``other`` group.
OTHER_TINT_SCALE_RANGE = (0.6, 1.4)
OTHER_TINT_OFFSET_RANGE = (-0.08, 0.08)

# Per-episode dome-light intensity range [unitless], matching the startup range.
HDRI_INTENSITY_RANGE = (500.0, 1000.0)


def _collect_paths(patterns: Sequence[str]) -> list[str]:
    """Return the sorted union of all files matching ``patterns``."""
    return sorted({path for pattern in patterns for path in glob(pattern)})


def _sample_paths(paths: list[str], rng: np.random.Generator) -> list[str]:
    """Choose a startup pool without replacement, retaining small pools in full."""
    if len(paths) <= TEXTURE_POOL_SIZE:
        return paths
    indices = rng.choice(len(paths), size=TEXTURE_POOL_SIZE, replace=False)
    return [paths[int(index)] for index in indices]


def _material_category(mdl_path: str | None, list_kind: str) -> str:
    """Classify a group's original material into a surface category.

    Args:
        mdl_path: The group's original ``.mdl`` path, if any.
        list_kind: ``"walkable"`` or ``"non_walkable"`` for the fallback.
    """
    if mdl_path:
        path = mdl_path.lower()
        if "material_road_mdl" in path:
            return "road"
        if "material_terrain_mdl" in path:
            return "terrain"
        if "material_sidewalk_mdl" in path or "/ground/" in path:
            return "walkable"
    return "walkable" if list_kind == "walkable" else "non_walkable"


def _get_stage():
    import omni.usd

    return omni.usd.get_context().get_stage()


def _spawn_material(prim_path: str, mdl_path: str) -> None:
    """Spawn an MDL material prim at ``prim_path`` using the Isaac Lab spawner."""
    import isaaclab.sim as sim_utils
    from pxr import UsdGeom

    UsdGeom.Xform.Define(_get_stage(), prim_path.rsplit("/", 1)[0])
    cfg = sim_utils.MdlFileCfg(mdl_path=mdl_path, project_uvw=True, texture_scale=(1.0, 1.0))
    cfg.func(prim_path, cfg)


def _bind_material(mesh_prim_path: str, material_path: str) -> None:
    from pxr import UsdShade

    stage = _get_stage()
    mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
    material_prim = stage.GetPrimAtPath(material_path)
    if not mesh_prim.IsValid() or not material_prim.IsValid():
        return
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(
        UsdShade.Material(material_prim), bindingStrength=UsdShade.Tokens.strongerThanDescendants
    )


def _set_dome(prim_path: str, hdr_path: str, intensity: float) -> None:
    """Set the dome light's HDR environment map and intensity."""
    from pxr import UsdLux

    stage = _get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return
    UsdLux.DomeLight(prim).CreateTextureFileAttr(hdr_path)
    intensity_attr = prim.GetAttribute("inputs:intensity")
    if intensity_attr and intensity_attr.IsValid():
        intensity_attr.Set(float(intensity))


def _cache_vertex_colors(mesh_prim_path: str) -> np.ndarray | None:
    """Read the original ``displayColor`` primvar of a mesh, if present [0, 1]."""
    stage = _get_stage()
    prim = stage.GetPrimAtPath(mesh_prim_path)
    if not prim.IsValid():
        return None
    attr = prim.GetAttribute("primvars:displayColor")
    if not attr or not attr.IsValid():
        return None
    values = attr.Get()
    return None if values is None else np.asarray(values, dtype=np.float32)


def _tint_vertex_colors(mesh_prim_path: str, base_colors: np.ndarray | None, rng: np.random.Generator) -> None:
    """Apply a random per-episode tint to a mesh's cached vertex colors."""
    if base_colors is None:
        return
    from pxr import UsdGeom

    stage = _get_stage()
    prim = stage.GetPrimAtPath(mesh_prim_path)
    attr = prim.GetAttribute("primvars:displayColor") if prim.IsValid() else None
    if not attr or not attr.IsValid():
        return
    scale = rng.uniform(*OTHER_TINT_SCALE_RANGE, size=3).astype(np.float32)
    offset = rng.uniform(*OTHER_TINT_OFFSET_RANGE, size=3).astype(np.float32)
    tinted = np.clip(base_colors * scale + offset, 0.0, 1.0)
    UsdGeom.Primvar(attr).SetInterpolation(UsdGeom.Tokens.vertex)
    attr.Set(tinted)


class TerrainMaterialRandomizer:
    """Rebinds terrain visual materials and the dome light globally, seeded per episode.

    The terrain is one shared prim (``/World/ground``), so a swap affects every
    environment. Evaluators invoke :meth:`apply` at the initial reset and each
    synchronized round or row boundary. Individual retries preserve textures.

    Pools are built per surface category (walkable / terrain / road), inferred
    from each group's original material path, so a road mesh only ever receives
    road textures and so on.
    """

    def __init__(
        self,
        ground_prim_path: str,
        walkable_group_mdl_paths: Sequence[str | None],
        non_walkable_group_mdl_paths: Sequence[str | None],
        seed: int = 0,
        sky_light_prim_path: str | None = None,
    ):
        self._seed = int(seed)
        self._episode = 0
        self._groups: list[tuple[str, str]] = [
            (f"{ground_prim_path}/walkable_{i}/mesh", _material_category(mdl, "walkable"))
            for i, mdl in enumerate(walkable_group_mdl_paths)
        ] + [
            (f"{ground_prim_path}/non_walkable_{i}/mesh", _material_category(mdl, "non_walkable"))
            for i, mdl in enumerate(non_walkable_group_mdl_paths)
        ]
        self._other_mesh_prim = f"{ground_prim_path}/other/mesh"
        self._sky_light_prim_path = sky_light_prim_path

        setup_rng = np.random.default_rng(self._seed)
        self._pools: dict[str, list[str]] = {}
        for category in sorted({category for _, category in self._groups}):
            candidates = _sample_paths(_collect_paths(CATEGORY_PATTERNS[category]), setup_rng)
            self._pools[category] = self._build_pool(
                f"{ground_prim_path}/materials/{category}", candidates
            )

        self._hdr_pool = (
            _sample_paths(_collect_paths((BACKGROUND_HDR_PATTERN,)), setup_rng)
            if self._sky_light_prim_path else []
        )

        self._other_base_colors = _cache_vertex_colors(self._other_mesh_prim)

    @staticmethod
    def _build_pool(prim_root: str, mdl_paths: list[str]) -> list[str]:
        pool = []
        for k, mdl_path in enumerate(mdl_paths):
            # USD prim names cannot start with a digit, so prefix the pool slot.
            path = f"{prim_root}/mat_{k}"
            _spawn_material(path, mdl_path)
            pool.append(path)
        return pool

    def apply(self) -> None:
        """Re-sample terrain materials, the dome light, and colors for this episode."""
        rng = np.random.default_rng(self._seed + self._episode)
        self._episode += 1
        for mesh_prim, category in self._groups:
            pool = self._pools.get(category, [])
            if pool:
                _bind_material(mesh_prim, pool[int(rng.integers(len(pool)))])
        if self._hdr_pool:
            hdr = self._hdr_pool[int(rng.integers(len(self._hdr_pool)))]
            _set_dome(self._sky_light_prim_path, hdr, float(rng.uniform(*HDRI_INTENSITY_RANGE)))
        _tint_vertex_colors(self._other_mesh_prim, self._other_base_colors, rng)

# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Shared loaders for terrain building/car asset metadata.

These read the USD filenames and their adjacent ``adj_parameter_folder`` JSON
descriptors and are shared by the procedural terrains (see
:mod:`sidewalkbench.pg.terrains.block_sidewalk` and
:mod:`sidewalkbench.pg.terrains.intersection`).
"""

from __future__ import annotations

import json
import os

# Cars present in the shipped asset bundle.
_ALLOWED_CAR_UUIDS = {
    "e4087350c4d84d8391de09b405c96488",
    "d999506c52cf4313aad5220f4af2da6c",
    "03a6d639648540e3ad45ed38c9b19d1f",
    "389f9ebcaeb84cbfb020a88a25751edc",
    "5416965277554e7682fcc2dc4961a99d",
    "af108d4773df4722a62420f65dd5b8fe",
    "f3c9f419c0924e968deed17196248c98",
    "d893ecd4f4d94f509aa7312306d74c61",
    "c6dc0d2275524bb1a43ddc398d2be3bc",
    "1533810cfbf146ac813da2b03f12befa",
}


def _asset_dirs(asset_root: str = "assets") -> tuple[str, str]:
    return os.path.join(asset_root, "usds"), os.path.join(asset_root, "adj_parameter_folder")


def load_building_infos(asset_root: str = "assets") -> list[dict]:
    """Load metadata for every building USD in the asset bundle."""
    usd_dir, adj_dir = _asset_dirs(asset_root)
    building_infos = []
    for fn in sorted(os.listdir(usd_dir)):
        if "building" not in fn.lower() or "non_metric" in fn:
            continue
        uuid_part = fn[:-4].rsplit("_", 1)[-1]
        json_path = os.path.join(adj_dir, f"structures_Building-{uuid_part}.json")
        if not os.path.exists(json_path):
            continue
        obj_info = json.load(open(json_path))
        building_infos.append(
            {
                "length": obj_info["general"]["length"],
                "width": obj_info["general"]["width"],
                "height": obj_info["general"]["height"],
                "usd_path": os.path.join(usd_dir, fn),
                "obj_info": obj_info,
                "usd_name": fn,
                "uuid": uuid_part,
            }
        )
    return building_infos


def load_car_infos(asset_root: str = "assets") -> list[dict]:
    """Load metadata for the allowed car USDs in the asset bundle."""
    usd_dir, adj_dir = _asset_dirs(asset_root)
    # JSON filenames may carry a model-name prefix before the UUID.
    json_by_uuid = {}
    for jf in os.listdir(adj_dir):
        if not jf.endswith(".json"):
            continue
        stem = jf[4:-5]  # strip 'car_' and '.json'
        uuid_part = stem.split("-", 1)[-1] if "-" in stem else stem
        json_by_uuid[uuid_part] = os.path.join(adj_dir, jf)

    car_infos = []
    for fn in sorted(os.listdir(usd_dir)):
        if "non_metric" in fn:
            continue
        uuid_part = fn[4:-4]
        if uuid_part not in _ALLOWED_CAR_UUIDS:
            continue
        json_path = json_by_uuid.get(uuid_part)
        if not json_path:
            continue
        obj_info = json.load(open(json_path))
        car_infos.append(
            {
                "length": obj_info["general"]["length"],
                "width": obj_info["general"]["width"],
                "height": obj_info["general"]["height"],
                "usd_path": os.path.join(usd_dir, fn),
                "obj_info": obj_info,
                "usd_name": fn,
                "uuid": uuid_part,
            }
        )
    return car_infos

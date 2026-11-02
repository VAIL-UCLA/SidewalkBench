"""Robot geometry and action configurations shared by PG and GS evaluation.

Profiles can be read without starting Isaac Sim. Call ``get_robot_configs`` only
after AppLauncher has initialized the simulator.
"""

from dataclasses import dataclass
import math
import re

import numpy as np


@dataclass(frozen=True)
class RobotProfile:
    name: str
    bbox: tuple[float, float, float]
    height_offset: float
    camera_height: float
    camera_link: str
    camera_offset: tuple[float, float, float]
    camera_intrinsics: tuple[float, ...]
    ground_contact_pattern: str
    wheel_joint_pattern: str | None = None
    wheel_radius: float | None = None
    wheel_direction: float = 1.0

    def as_dict(self) -> dict:
        return dict(bbox=list(self.bbox), height_offset=self.height_offset,
                    camera_height=self.camera_height,
                    camera_intrinsics=np.array(self.camera_intrinsics).reshape(3, 3))

    @property
    def pedestrian_radius(self) -> float:
        # Preserve the existing COCO crowd radius; cover the other footprints.
        return 0.5 if self.name == "coco" else math.hypot(*self.bbox[:2]) / 2

    def collision_body_mask(self, body_names) -> list[bool]:
        """Ignore normal wheel/foot support contacts with the scanned ground."""
        return [re.fullmatch(self.ground_contact_pattern, name, re.IGNORECASE) is None for name in body_names]


_COCO_K = (790.0 * 512 / 1920, 0.0, 256.0, 0.0, 790.0 * 512 / 1920, 144.0, 0.0, 0.0, 1.0)
ROBOT_PROFILES = {
    "coco": RobotProfile(
        "coco", (0.75953799486, 0.60900770638, 0.70764710009), 0.27668, 0.27668 + 0.184,
        "base_link", (0.38243, 0.0, 0.184), _COCO_K, ".*wheel.*", ".*wheel_joint", 0.11),
    "go2": RobotProfile(
        "go2", (0.5738245, 0.324026, 0.537749), 0.44891, 0.44891 + 0.04297,
        "base", (0.32715, -0.00003, 0.0429),
        (609.65997906439202 * 512 / 1280, 0.0, 635.92105141186175 * 512 / 1280,
         0.0, 773.27749708858391 * 512 / 1280, 347.8374060632205 * 512 / 1280, 0.0, 0.0, 1.0),
        ".*(foot|calf)"),
    "g1": RobotProfile(
        "g1", (0.40, 0.45, 1.27), 0.74, 0.74 + 0.42,
        "torso_link", (0.07, 0.0, 0.42),
        (210.67, 0.0, 256.0, 0.0, 210.67, 144.0, 0.0, 0.0, 1.0), ".*ankle_roll_link"),
    "m20": RobotProfile(
        "m20", (0.81733334, 0.50425637, 0.6753079), 0.5900668874382973, 0.5900668874382973 + 0.03738,
        "base_link", (0.37646, 0.0, 0.03738),
        (210.67, 0.0, 256.0, 0.0, 210.67, 144.0, 0.0, 0.0, 1.0),
        ".*wheel.*", ".*_wheel_joint", 0.09, -1.0),
}


def get_robot_profile(name: str) -> RobotProfile:
    try:
        return ROBOT_PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown robot {name!r}; choose from {', '.join(ROBOT_PROFILES)}") from None


def get_robot_configs(name: str):
    """Return a fresh articulation config and (v, w) action config."""
    get_robot_profile(name)
    if name == "coco":
        from .coco import COCO_CFG, COCOWaypointActionsCfg

        return COCO_CFG.copy(), COCOWaypointActionsCfg()
    from .legged import G1ActionsCfg, GO2ActionsCfg, M20ActionsCfg, M20_CFG
    if name == "m20":
        return M20_CFG.copy(), M20ActionsCfg()
    from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG, UNITREE_GO2_CFG

    return (UNITREE_GO2_CFG.copy(), GO2ActionsCfg()) if name == "go2" else (G1_MINIMAL_CFG.copy(), G1ActionsCfg())

"""Compatibility patches for IsaacLab/IsaacSim version mismatches."""

import sys


def patch_physx_manager_for_cloner():
    """Add missing fabric notice handler methods to PhysxManager.

    IsaacLab's isaaclab_physx replaces SimulationManager with PhysxManager,
    but IsaacSim 6.0's Cloner.disable_change_listener() expects methods that
    PhysxManager doesn't have. This adds no-op stubs so the Cloner works.
    """
    module = sys.modules.get("isaacsim.core.simulation_manager")
    if module is None:
        return

    cls = getattr(module, "SimulationManager", None)
    if cls is None:
        return

    if not hasattr(cls, "is_fabric_usd_notice_handler_enabled"):
        cls.is_fabric_usd_notice_handler_enabled = classmethod(lambda cls, stage_id: False)

    if not hasattr(cls, "enable_fabric_usd_notice_handler"):
        cls.enable_fabric_usd_notice_handler = classmethod(lambda cls, stage_id, enabled: None)

    if not hasattr(cls, "enable_usd_notice_handler"):
        cls.enable_usd_notice_handler = classmethod(lambda cls, enabled: None)

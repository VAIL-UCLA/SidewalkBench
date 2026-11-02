# Copyright (c) 2022-2025, The SidewalkBench-PG Project Developers.
# Author: Honglin He
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import warnings

import gymnasium as gym
import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import EventManager
from isaaclab.sim.utils.stage import use_stage
from isaaclab.utils.timer import Timer
from sidewalkbench.pg.scene.sidewalk_scene_pg import SidewalkScenePG

logger = logging.getLogger(__name__)


class AbstractEnv(ManagerBasedRLEnv, gym.Env):
    def _init_sim(self):
        """Complete environment initialization after the SimulationContext is created.

        Separated from :meth:`__init__` so that the caller can tear down the
        :class:`SimulationContext` singleton if this method raises.
        """
        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # -- controls camera/Kit rendering in step().
        # When False, the Kit app loop (app.update()) and camera/RTX sensor updates are
        # skipped, but standalone visualizers (Newton, Rerun, Viser) continue to update.
        # This is because Kit bundles camera rendering with its app loop and the two
        # cannot be separated.  Non-Kit visualizers have independent step() methods
        # that do not trigger camera or GUI updates, so they remain active.
        self.render_enabled: bool = True

        # allocate dictionary to store metrics
        self.extras = {}

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            # set the stage context for scene creation steps which use the stage
            with use_stage(self.sim.stage):
                self.scene = SidewalkScenePG(self.cfg.scene)
                self.scene.generate_scene()
                self.scene.dt = self.physics_dt
                self.scene.decimation = self.cfg.decimation
            self.sim.register_interactive_scene(self.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # Base ManagerBasedEnv creates self.video_recorders in its own _init_sim, which this
        # override replaces. Keep the attribute defined because ManagerBasedEnv.step iterates it.
        self.video_recorders = []

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
            # since the reset can trigger callbacks which use the stage,
            # we need to set the stage context here
            with use_stage(self.sim.stage):
                self.sim.reset()
            # update scene to pre populate data buffers for assets and sensors.
            # this is needed for the observation manager to get valid tensors for initialization.
            # this shouldn't cause an issue since later on, users do a reset over all the environments
            # so the lazy buffers would be reset.
            self.scene.update(dt=self.physics_dt)
        # let the physics backend know about the env decimation so it can
        # fold the full loop into a single step() when possible
        self.sim.physics_manager.set_decimation(self.cfg.decimation)
        self._physics_handles_decimation = self.sim.physics_manager.handles_decimation()
        # add timeline event to load managers
        self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None
        self.has_rtx_sensors = self.sim.get_setting("/isaaclab/render/rtx_sensors")
        # initialize observation buffers
        self.obs_buf = {}

        # Synchronized-episode support: when ``defer_resets`` is set, terminated
        # environments are not reset immediately; instead they are recorded in
        # ``sync_done`` until the caller resets all environments together via
        # :meth:`force_reset_all`. Disabled by default.
        self.defer_resets = False
        self.sync_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Let the scene freeze pedestrians of environments that finished the round.
        self.scene.frozen_env_mask = self.sync_done

        # export IO descriptors if requested
        if self.cfg.export_io_descriptors:
            self.export_IO_descriptors()

        # show deprecation message for rerender_on_reset
        if self.cfg.rerender_on_reset:
            msg = (
                "\033[93m\033[1m[DEPRECATION WARNING] ManagerBasedEnvCfg.rerender_on_reset is deprecated. Use"
                " ManagerBasedEnvCfg.num_rerenders_on_reset instead.\033[0m"
            )
            warnings.warn(
                msg,
                FutureWarning,
                stacklevel=2,
            )
            if self.cfg.num_rerenders_on_reset == 0:
                self.cfg.num_rerenders_on_reset = 1

    def reset(self, seed=None, env_ids=None, options=None):
        """Reseed scene sampling along with the simulator on an explicit seed."""
        if seed is not None:
            self.scene.randomization.reseed(seed)
            self.scene._animation_cache.clear()
            self.scene.pg_config["seed"] = int(seed)
        return super().reset(seed=seed, env_ids=env_ids, options=options)

    def _reset_idx(self, env_ids):
        """Defer resets to the round boundary when ``defer_resets`` is set."""
        if self.defer_resets:
            ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids)
            self.sync_done[ids.to(device=self.device, dtype=torch.long)] = True
            return
        super()._reset_idx(env_ids)

    def step(self, action: torch.Tensor):
        """Step the environment, suppressing deferred post-reset re-renders.

        While ``defer_resets`` is active, terminated environments are not reset by
        the managers, but :meth:`ManagerBasedRLEnv.step` would still re-render the
        scene ``num_rerenders_on_reset`` times on every step as long as any
        environment remains terminated. Skip those renders here; the round
        boundary re-renders once via :meth:`force_reset_all`.
        """
        if not self.defer_resets:
            return super().step(action)
        saved_rerenders = self.cfg.num_rerenders_on_reset
        self.cfg.num_rerenders_on_reset = 0
        try:
            return super().step(action)
        finally:
            self.cfg.num_rerenders_on_reset = saved_rerenders

    def force_reset_envs(self, env_ids):
        """Retry selected environments without changing the other trials.

        Scene assets and start/goal are resampled only for the selected terrain
        cells. Global textures, pedestrian timing, and other environments'
        observation histories are preserved. No physics step is added.
        """
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).flatten().unique()
        if ids.numel() == 0:
            return self.obs_buf, self.extras
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise ValueError("Retry environment indices are out of range")
        keep = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        keep[ids] = False
        terrain = self.scene.terrain
        cells = terrain.terrain_levels * terrain.terrain_origins.shape[1] + terrain.terrain_types
        if bool(torch.isin(cells[keep], cells[ids]).any()):
            raise ValueError("Retry requires separate terrain cells for the other environments")

        # ManagerBasedEnv.reset appends observations for the whole batch, even
        # when env_ids selects one robot. Restore the histories of other robots.
        histories = []
        for group in self.observation_manager._group_obs_term_history_buffer.values():
            for buffer in group.values():
                if buffer._buffer is not None:
                    histories.append((buffer, buffer.buffer[keep].clone(), buffer._num_pushes[keep].clone()))
        was_deferring = self.defer_resets
        was_paused = self.scene.pause_pedestrians
        clock = self.scene.count
        material_randomizer = self.scene._terrain_material_randomizer
        was_retrying = self.scene.retrying_reset
        self.defer_resets = False
        self.scene.retrying_reset = True
        self.scene.pause_pedestrians = True
        self.scene._terrain_material_randomizer = None
        try:
            _, extras = self.reset(env_ids=ids)
        finally:
            for buffer, history, pushes in histories:
                buffer.buffer[keep] = history
                buffer._num_pushes[keep] = pushes
            self.scene._terrain_material_randomizer = material_randomizer
            self.scene.retrying_reset = was_retrying
            self.scene.pause_pedestrians = was_paused
            self.scene.count = clock
            self.sync_done[ids] = False
            self.defer_resets = was_deferring
        self.obs_buf = self.observation_manager.compute(update_history=False)
        return self.obs_buf, extras

    def force_reset_all(self):
        """Reset every environment at once, ignoring and restoring ``defer_resets``.

        Used at synchronized round boundaries so the global scene randomization
        (``reset_obj``/``reset_base`` events, hence terrain texture and start/goal)
        runs once for all environments, then a new round begins together.
        """
        was_deferring = self.defer_resets
        self.defer_resets = False
        try:
            env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            obs, extras = self.reset(env_ids=env_ids)
        finally:
            self.sync_done[:] = False
            self.defer_resets = was_deferring
        return obs, extras

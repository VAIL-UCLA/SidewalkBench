"""GS variant regression checks (python -m unittest sidewalkbench.gs.test_variants)."""

import argparse
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from sidewalkbench.gs.annotation import Route
from sidewalkbench.gs.camera import camera_pose
from sidewalkbench.gs.config import add_variant_arguments, output_directory, resolve_arguments
from sidewalkbench.gs.goals import GSGoalManager
from sidewalkbench.gs.sampler import RouteRuntime
from sidewalkbench.gs.tools.make_tables import run_dirs, display_models
from sidewalkbench.models.base_model import BaseModel
from sidewalkbench.models.controller_config import controller_from_config
from sidewalkbench.pg.primitives.robot.registry import get_robot_profile


def settings(*argv):
    parser = argparse.ArgumentParser()
    add_variant_arguments(parser)
    return resolve_arguments(parser.parse_args(argv))


def runtime(points, goal=None):
    points = np.asarray(points, dtype=float)
    route = Route("test", "short_horizon", (), points[0], 0.0,
                  points[-1] if goal is None else np.array(goal), points)
    return RouteRuntime.from_route(route, None)


class ConfigurationTests(unittest.TestCase):
    def test_default_keeps_original_output_path_and_model_defaults(self):
        cfg = settings()
        self.assertEqual(str(output_directory(cfg, "/repo")),
                         "/repo/experiments/gs/FlowPilot_beverly_hills_crosswalk_gs_0_42")
        self.assertIsNone(controller_from_config(cfg, "FlowPilot"))

    def test_yaml_and_cli_have_the_same_variant_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "variant.yaml"
            path.write_text("model_type: MBRA\nrobot_type: go2\nuse_goal: true\ncontroller: pd\n"
                            "pd:\n  waypoint_mode: lookahead\n  lookahead_s: 1.0\n  max_v: 0.8\n")
            cfg = settings("--config", str(path))
            cli = settings("--model_type", "MBRA", "--robot_type", "go2", "--use_goal", "--controller", "pd",
                           "--pd_waypoint_mode", "lookahead", "--pd_lookahead_s", "1.0", "--max_v", "0.8")
            self.assertEqual(output_directory(cfg, tmp), output_directory(cli, tmp))
            override = settings("--config", str(path), "--no-use_goal", "--robot_type", "m20")
            self.assertFalse(override["use_goal"])
            self.assertEqual(override["robot_type"], "m20")
            self.assertNotEqual(output_directory(cfg, tmp), output_directory(override, tmp))

    def test_each_controller_or_goal_variant_has_a_distinct_output(self):
        variants = [(), ("--use_goal",), ("--controller", "pp"), ("--pd_waypoint_mode", "lookahead"),
                    ("--pd_waypoint_mode", "lookahead", "--pd_lookahead_s", "0.5"),
                    ("--robot_type", "go2"), ("--robot_type", "g1"), ("--model_hz", "2")]
        self.assertEqual(len({output_directory(settings(*v), "/repo") for v in variants}), len(variants))

    def test_invalid_options_fail_early(self):
        for args in (("--goal_segment_length", "0"), ("--model_hz", "nan"),
                     ("--goal_advance_margin", "30")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                settings(*args)
        with self.assertRaisesRegex(ValueError, "Unknown pd"):
            controller_from_config({"pd": {"lookahed_s": 1}}, "MBRA")
        with self.assertRaisesRegex(ValueError, "controller"):
            controller_from_config({"controller": "other"}, "MBRA")

    def test_reports_keep_controller_variants_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            for suffix in ("", "_go2_goal_pd_test"):
                path = Path(tmp) / ("ViNT_scan_gs_0_42" + suffix)
                path.mkdir()
                (path / "run_config.json").write_text(json.dumps({"model_type": "ViNT", "variant": suffix}))
            rows = run_dirs([tmp], "_gs_0_")
            self.assertEqual({row[0] for row in rows}, {"ViNT", "ViNT_go2_goal_pd_test"})
            labels = {row[1]: row[0] for row in display_models({r[0] for r in rows}) if row is not None}
            self.assertIn("ViNT_go2_goal_pd_test", labels)

    def test_shared_pd_modes_change_actual_command(self):
        half = BaseModel(device="cpu", controller=controller_from_config({"controller": "pd"}, "test"))
        one = BaseModel(device="cpu", controller=controller_from_config(
            {"controller": "pd", "pd": {"waypoint_mode": "lookahead", "lookahead_s": 1}}, "test"))
        for model in (half, one):
            model._info = {"output_fps": 2}
        path = torch.zeros(1, 8, 2)
        path[0, :, 0] = torch.arange(1, 9) / 2
        path[0, -1, 1] = 2
        half_v, half_w = half._pd_control(path)
        one_v, one_w = one._pd_control(path)
        self.assertEqual(half._controller.heading_waypoint_index, 7)
        self.assertEqual(one._controller.heading_waypoint_index, 1)
        torch.testing.assert_close(half_v, one_v)
        self.assertGreater(half_w.item(), 0)
        self.assertEqual(one_w.item(), 0)
        pp = controller_from_config({"controller": "pp", "pp": {"lookahead_base": 0.7}}, "test")
        self.assertEqual(pp.type, "pp")
        self.assertEqual(pp.pp.lookahead_base, 0.7)


class GoalTests(unittest.TestCase):
    def manager(self, n=2, **kw):
        return GSGoalManager(n, ground_z=lambda x, y: .1 * x, camera_height=.5,
                             intrinsics=np.eye(3), width=8, height=6, **kw)

    def test_point_goal_uses_robot_frame_and_exact_endpoint(self):
        goals = self.manager(mode="short_horizon")
        goals.reset(0, runtime([[0, 0], [10, 0]], goal=[10, 2]))
        goals.reset(1, runtime([[0, 0], [0, 5]]))
        q = torch.tensor([[0, 0, 0, 1], [0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)]])
        local = goals.goal_xy_local(torch.tensor([[8., 1.], [0., 1.]]), q)
        np.testing.assert_allclose(local, [[2, 1], [4, 0]], atol=1e-6)

    def test_long_goals_follow_route_and_advance_after_recovery(self):
        goals = self.manager(n=1, mode="long_horizon")
        goals.reset(0, runtime([[0, 0], [40, 0], [40, 40]]))
        np.testing.assert_allclose(goals.xy[0], [30, 0])
        goals.update([[28, 0]])
        np.testing.assert_allclose(goals.xy[0], [30, 0])
        goals.update([[29, 0]])
        np.testing.assert_allclose(goals.xy[0], [40, 20])
        self.assertAlmostEqual(goals.yaw[0], math.pi / 2, places=6)
        goals.update([[40, 39]])
        np.testing.assert_allclose(goals.xy[0], [40, 40])
        goals.update([[40, 40]])
        self.assertEqual(goals.target_s[0], 80)

    def test_goal_images_are_cached_and_reset_per_environment(self):
        class Renderer:
            def __init__(self):
                self.calls = []

            def render(self, pos, rot, ks, **kw):
                self.calls.append((pos.clone(), rot.clone(), kw))
                return torch.stack([torch.full((3, 6, 8), x / 100) for x in pos[:, 0]])

        goals = self.manager(mode="short_horizon", capture_images=True)
        goals.reset(0, runtime([[0, 0], [10, 0]]))
        goals.reset(1, runtime([[0, 0], [20, 0]]))
        render = Renderer()
        goals.render_pending(render, extras=lambda i: f"cones{i}")
        np.testing.assert_allclose(render.calls[0][0], [[10, 0, 1.5], [20, 0, 2.5]])
        self.assertEqual(render.calls[0][2]["humans"], ["cones0", "cones1"])
        old = goals.images[1].clone()
        goals.render_pending(render)
        self.assertEqual(len(render.calls), 1)
        goals.reset(0, runtime([[0, 0], [30, 0]]))
        goals.set_camera_height(0, .8)
        goals.render_pending(render)
        self.assertEqual(len(render.calls[1][0]), 1)
        self.assertAlmostEqual(render.calls[1][0][0, 2].item(), 3.8, places=6)
        torch.testing.assert_close(goals.images[1], old)
        self.assertAlmostEqual(goals.images[0].mean().item(), .3, places=6)


class EmbodimentTests(unittest.TestCase):
    def test_camera_mount_rotates_with_its_body(self):
        q = torch.tensor([[0., 0., math.sin(math.pi / 4), math.cos(math.pi / 4)]])
        pos, rot = camera_pose(torch.tensor([[10., 20., 1.]]), q, (1., 0., .2))
        torch.testing.assert_close(pos, torch.tensor([[10., 21., 1.2]]))
        torch.testing.assert_close(rot[0, :, 2], torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)

    def test_support_contacts_match_each_embodiment(self):
        self.assertEqual(get_robot_profile("coco").collision_body_mask(["base_link", "front_left_wheel"]), [True, False])
        self.assertEqual(get_robot_profile("go2").collision_body_mask(["base", "FL_calf", "FL_foot"]), [True, False, False])
        self.assertEqual(get_robot_profile("g1").collision_body_mask(["pelvis", "left_knee_link", "left_ankle_roll_link"]),
                         [True, True, False])
        self.assertEqual(get_robot_profile("m20").collision_body_mask(["base_link", "fl_wheel_link"]), [True, False])
        for name in ("coco", "go2", "g1", "m20"):
            profile = get_robot_profile(name)
            self.assertGreaterEqual(profile.pedestrian_radius, math.hypot(*profile.bbox[:2]) / 2)


if __name__ == "__main__":
    unittest.main()

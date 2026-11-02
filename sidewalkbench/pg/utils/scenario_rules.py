"""Short-horizon outcome rules shared by evaluation and visualization."""

import torch


GESTURING_MIN_PROGRESS = 0.35
GROUP_SCENARIOS = ("conversing", "queueing")


def pedestrian_gap_segments(scene, terrain_idx):
    """Return finite segments joining members of the scripted pedestrian group.

    Queue members retain the order assigned by the terrain sampler, so only
    consecutive members are connected. Conversing groups connect every pair.
    Coordinates are world XY, like ``clone_dynamic_states`` and robot poses.
    """
    scenario = getattr(scene, "scenario_type", None)
    states = getattr(scene, "clone_dynamic_states", None)
    if scenario not in GROUP_SCENARIOS or states is None:
        return None
    people = states.get(int(terrain_idx))
    if people is None or len(people) < 2:
        return None
    xy = people[:, :2]
    if scenario == "queueing":
        return torch.stack((xy[:-1], xy[1:]), dim=1)
    pairs = torch.combinations(torch.arange(len(xy), device=xy.device), r=2)
    return xy[pairs]


def crosses_pedestrian_gaps(previous_xy, current_xy, segments, epsilon=1e-6):
    """Check robot-center movement against finite segments, including contact.

    Returns one boolean per robot. Swept segments catch crossings even when a
    policy step starts and ends on opposite sides of a gap. A stationary robot,
    degenerate pedestrian pair, or non-finite coordinate is not a crossing.
    """
    a, b = previous_xy[:, None, :], current_xy[:, None, :]
    c, d = segments[None, :, 0, :], segments[None, :, 1, :]

    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]

    def straddles(u, v):
        return ((u >= -epsilon) & (v <= epsilon)) | ((v >= -epsilon) & (u <= epsilon))

    ab, cd = b - a, d - c
    sides = straddles(cross(ab, c - a), cross(ab, d - a))
    sides &= straddles(cross(cd, a - c), cross(cd, b - c))
    overlap = (torch.maximum(torch.minimum(a, b), torch.minimum(c, d))
               <= torch.minimum(torch.maximum(a, b), torch.maximum(c, d)) + epsilon).all(dim=-1)
    valid = (ab.square().sum(dim=-1) > epsilon ** 2) & (cd.square().sum(dim=-1) > epsilon ** 2)
    valid &= torch.isfinite(a).all(dim=-1) & torch.isfinite(b).all(dim=-1)
    valid &= torch.isfinite(c).all(dim=-1) & torch.isfinite(d).all(dim=-1)
    return (sides & overlap & valid).any(dim=-1)


def pedestrian_group_intrusion(env):
    """Termination term for driving between conversing or queued pedestrians."""
    result = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if getattr(env.scene, "scenario_type", None) not in GROUP_SCENARIOS:
        return result
    current = env.scene["robot"].data.root_pos_w[:, :2]
    if not hasattr(env, "_pedestrian_gap_previous_xy"):
        env._pedestrian_gap_previous_xy = current.clone()
        env._pedestrian_gap_crossed = result.clone()
    previous = env._pedestrian_gap_previous_xy
    crossed = env._pedestrian_gap_crossed
    # Reset only the environments whose episode restarted, including selective
    # retries. A teleport must not be interpreted as driving through the group.
    just_reset = env.episode_length_buf <= 1
    previous[just_reset] = current[just_reset]
    crossed[just_reset] = False
    terrain = env.scene.terrain
    cells = terrain.terrain_levels * terrain.terrain_origins.shape[1] + terrain.terrain_types
    for cell in cells.unique().tolist():
        segments = pedestrian_gap_segments(env.scene, cell)
        if segments is None:
            continue
        mask = cells == cell
        crossed[mask] |= crosses_pedestrian_gaps(previous[mask], current[mask], segments.to(current))
    previous.copy_(current)
    return crossed.clone()


def classify_outcome(scenario, *, goal_reached, freeze, progress, collision_static=False,
                     collision_dynamic=False, oob=False, group_intrusion=False):
    """Return (success, failure_reason) using the benchmark's scenario rules."""
    if group_intrusion:
        return False, "between_pedestrians"
    if scenario == "gesturing":
        if goal_reached:
            return False, "ignore_gesture"
        if freeze and progress >= GESTURING_MIN_PROGRESS:
            return True, None
    elif goal_reached:
        return True, None
    for occurred, reason in ((collision_static, "collision_static"), (collision_dynamic, "collision_dynamic"),
                             (oob, "out_of_bound"), (freeze, "freeze")):
        if occurred:
            return False, reason
    return False, "timeout"


def outcome_label(failure_reason):
    """Human-readable label for the same failure saved in evaluation results."""
    return {
        "ignore_gesture": "Ignore Gesture",
        "between_pedestrians": "Between Pedestrians",
        "collision_static": "Obstacle Collision",
        "collision_dynamic": "Pedestrian Collision",
        "out_of_bound": "Out of Bounds",
        "freeze": "Freeze",
        "timeout": "Timeout",
        None: None,
    }[failure_reason]

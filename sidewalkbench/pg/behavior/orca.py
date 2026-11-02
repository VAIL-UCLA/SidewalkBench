"""
Lightweight 2D ORCA (Optimal Reciprocal Collision Avoidance) for pedestrian
collision avoidance.  Pure numpy, no external dependencies.

Based on: J. van der Berg et al., "Reciprocal n-body Collision Avoidance",
Robotics Research, 2011.

Reference implementation inspired by Muon/pyorca (MIT license).
Content was rephrased for compliance with licensing restrictions.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import det as det2

# ---------------------------------------------------------------------------
# Geometric primitives
# ---------------------------------------------------------------------------


def _perp(v: np.ndarray) -> np.ndarray:
    """90-degree clockwise rotation of a 2-D vector."""
    return np.array([v[1], -v[0]])


def _norm_sq(v: np.ndarray) -> float:
    return float(np.dot(v, v))


def _normalized(v: np.ndarray) -> np.ndarray:
    l = _norm_sq(v)
    if l < 1e-12:
        return np.array([1.0, 0.0])
    return v / np.sqrt(l)


# ---------------------------------------------------------------------------
# Half-plane intersection (2-D linear program)
# ---------------------------------------------------------------------------


class _Line:
    __slots__ = ("point", "direction")

    def __init__(self, point: np.ndarray, direction: np.ndarray):
        self.point = np.asarray(point, dtype=float)
        self.direction = _normalized(np.asarray(direction, dtype=float))


def _halfplane_optimize(lines: list[_Line], optimal: np.ndarray, max_speed: float) -> np.ndarray:
    """Find the point closest to *optimal* inside the intersection of
    half-planes, clamped to a disk of radius *max_speed*.
    Returns zero velocity if constraints are infeasible."""
    result = optimal.copy()
    for i, line in enumerate(lines):
        if np.dot(result - line.point, line.direction) >= 0:
            continue
        # Project onto the new constraint line, bounded by previous lines
        tangent = _perp(line.direction)
        proj = np.dot(optimal - line.point, tangent)
        # Clamp to feasible interval from previous constraints
        left, right = -1e6, 1e6
        for prev in lines[:i]:
            num = np.dot(prev.direction, line.point - prev.point)
            den = det2(np.array([tangent, prev.direction]))
            if abs(den) < 1e-12:
                if num < 0:
                    return np.zeros(2)
                continue
            offset = num / den
            if den > 0:
                right = min(right, offset)
            else:
                left = max(left, offset)
            if left > right:
                return np.zeros(2)
        proj = np.clip(proj, left, right)
        result = line.point + tangent * proj
    # Clamp to max speed disk
    if _norm_sq(result) > max_speed * max_speed:
        result = _normalized(result) * max_speed
    return result


# ---------------------------------------------------------------------------
# ORCA velocity computation
# ---------------------------------------------------------------------------


def _orca_halfplane(
    agent_pos: np.ndarray,
    agent_vel: np.ndarray,
    agent_radius: float,
    other_pos: np.ndarray,
    other_vel: np.ndarray,
    other_radius: float,
    tau: float,
    dt: float,
    responsibility: float = 0.5,
) -> _Line:
    """Compute the ORCA half-plane constraint for one agent-agent pair.

    Args:
        responsibility: fraction of avoidance effort taken by this agent.
            0.5 = standard reciprocal (both share equally).
            Higher values (e.g. 0.8) make this agent dodge more.
    """
    x = other_pos - agent_pos
    v = agent_vel - other_vel
    r = agent_radius + other_radius
    x_sq = _norm_sq(x)

    if x_sq > r * r:
        # Not currently overlapping
        adjusted = x / tau * (1.0 - (r * r) / x_sq)
        if np.dot(v - adjusted, adjusted) < 0:
            # Project onto truncation disk
            w = v - x / tau
            u = _normalized(w) * (r / tau) - w
            n = _normalized(w)
        else:
            # Project onto cone side
            leg = np.sqrt(max(x_sq - r * r, 0.0))
            sine = np.copysign(r, det2(np.array([v, x])))
            rot = np.array([[leg, sine], [-sine, leg]])
            rotated = rot.dot(x) / x_sq
            n = _perp(rotated)
            if sine < 0:
                n = -n
            u = rotated * np.dot(v, rotated) - v
    else:
        # Already overlapping — push apart within dt
        w = v - x / dt
        u = _normalized(w) * (r / dt) - w
        n = _normalized(w)

    return _Line(agent_vel + u * responsibility, n)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_orca_velocity(
    agent_pos: np.ndarray,  # (2,)
    agent_vel: np.ndarray,  # (2,)
    preferred_vel: np.ndarray,  # (2,)
    agent_radius: float,
    max_speed: float,
    other_agents: list[tuple[np.ndarray, np.ndarray, float]],  # [(pos, vel, radius), ...]
    static_obstacles: list[tuple[np.ndarray, float]],  # [(pos, radius), ...]
    tau: float = 3.0,  # time horizon for agents
    dt: float = 0.2,  # simulation timestep
    agent_responsibility: float = 0.5,  # avoidance responsibility for agent-agent pairs (1.0 = full, 0.5 = shared)
) -> np.ndarray:
    """Compute a collision-free velocity for a single agent using ORCA.

    This is the scalar reference implementation; production uses
    :func:`compute_orca_velocities`. Pairs are treated symmetrically with
    ``agent_responsibility``; static obstacles are handled as zero-velocity peers
    that this agent avoids fully (responsibility 1.0).

    Args:
        agent_pos: Agent position [m], shape ``(2,)``.
        agent_vel: Agent velocity [m/s], shape ``(2,)``.
        preferred_vel: Desired velocity [m/s], shape ``(2,)``.
        agent_radius: Agent collision radius [m].
        max_speed: Maximum speed [m/s].
        other_agents: ``[(pos2d, vel2d, radius), ...]`` peer agents.
        static_obstacles: ``[(pos2d, radius), ...]`` circular obstacles.
        tau: Time horizon [s].
        dt: Simulation timestep [s].
        agent_responsibility: Fraction of avoidance this agent takes (0.5 =
            reciprocal sharing).

    Returns:
        New velocity [m/s], shape ``(2,)``, closest to ``preferred_vel`` while
        satisfying the ORCA constraints.
    """
    lines: list[_Line] = []

    # Static obstacle constraints (full responsibility)
    for obs_pos, obs_r in static_obstacles:
        dist = np.linalg.norm(obs_pos - agent_pos)
        if dist < (agent_radius + obs_r) * 3:  # only nearby obstacles
            lines.append(
                _orca_halfplane(
                    agent_pos,
                    agent_vel,
                    agent_radius,
                    obs_pos,
                    np.zeros(2),
                    obs_r,
                    tau,
                    dt,
                    responsibility=1.0,
                )
            )

    # Agent-agent constraints
    for other_pos, other_vel, other_r in other_agents:
        dist = np.linalg.norm(other_pos - agent_pos)
        if dist < (agent_radius + other_r) * 5:  # only nearby agents
            lines.append(
                _orca_halfplane(
                    agent_pos,
                    agent_vel,
                    agent_radius,
                    other_pos,
                    other_vel,
                    other_r,
                    tau,
                    dt,
                    responsibility=agent_responsibility,
                )
            )

    return _halfplane_optimize(lines, preferred_vel, max_speed)


# ---------------------------------------------------------------------------
# Vectorized ORCA (batch over a whole pedestrian population)
# ---------------------------------------------------------------------------
#
# The scalar API above is kept as the reference implementation (and is exercised
# by tests/test_orca_batch.py).  The functions below evaluate every agent in a
# terrain cell at once, which removes the O(P^2) Python per-pair overhead in
# PedestrianBehaviorManager.tick().


def _normalized_batch(v: np.ndarray) -> np.ndarray:
    """Row-wise normalization for 2-D vectors, defaulting to ``+x``."""
    norm_sq = np.sum(v * v, axis=-1, keepdims=True)
    safe = np.sqrt(np.maximum(norm_sq, 1e-12))
    return np.where(norm_sq < 1e-12, np.array([1.0, 0.0], dtype=v.dtype), v / safe)


def _perp_batch(v: np.ndarray) -> np.ndarray:
    """90-degree clockwise rotation, applied to the last axis."""
    return np.stack([v[..., 1], -v[..., 0]], axis=-1)


def _orca_halfplanes_batch(
    agent_pos: np.ndarray,  # (P, 2)
    agent_vel: np.ndarray,  # (P, 2)
    agent_radius: np.ndarray,  # (P,)
    other_pos: np.ndarray,  # (P, K, 2)
    other_vel: np.ndarray,  # (P, K, 2)
    other_radius: np.ndarray,  # (P, K)
    tau: float,
    dt: float,
    responsibility: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized ORCA half-plane for many ``(agent, other)`` pairs.

    Returns ``(point, direction)``, each shaped ``(P, K, 2)``.  Callers mask out
    invalid pairs; this function never raises on degenerate inputs.
    """
    x = other_pos - agent_pos[:, None, :]
    v = agent_vel[:, None, :] - other_vel
    r = agent_radius[:, None] + other_radius
    x_sq = np.sum(x * x, axis=-1)
    x_sq_safe = np.maximum(x_sq, 1e-12)

    overlapping = x_sq <= r * r

    # Cone-side branch
    leg = np.sqrt(np.maximum(x_sq - r * r, 0.0))
    det_vx = v[..., 0] * x[..., 1] - v[..., 1] * x[..., 0]
    sine = np.copysign(r, det_vx)
    rot = np.empty(x.shape[:-1] + (2, 2), dtype=x.dtype)
    rot[..., 0, 0] = leg
    rot[..., 0, 1] = sine
    rot[..., 1, 0] = -sine
    rot[..., 1, 1] = leg
    rotated = np.einsum("...ij,...j->...i", rot, x) / x_sq_safe[..., None]
    n_cone = _perp_batch(rotated)
    n_cone = np.where((sine < 0)[..., None], -n_cone, n_cone)
    u_cone = rotated * np.sum(v * rotated, axis=-1, keepdims=True) - v

    # Non-overlapping: choose between truncation disk and cone side
    adjusted = x / tau * (1.0 - (r * r) / x_sq_safe)[..., None]
    project_disk = np.sum((v - adjusted) * adjusted, axis=-1) < 0.0

    w_disk = v - x / tau
    n_disk = _normalized_batch(w_disk)
    u_disk = n_disk * (r / tau)[..., None] - w_disk

    n_free = np.where(project_disk[..., None], n_disk, n_cone)
    u_free = np.where(project_disk[..., None], u_disk, u_cone)

    # Overlapping: push apart within dt
    w_push = v - x / dt
    n_push = _normalized_batch(w_push)
    u_push = n_push * (r / dt)[..., None] - w_push

    u = np.where(overlapping[..., None], u_push, u_free)
    n = np.where(overlapping[..., None], n_push, n_free)

    point = agent_vel[:, None, :] + u * responsibility
    direction = _normalized_batch(n)
    return point, direction


def _halfplane_optimize_batch(
    points: np.ndarray,  # (P, L, 2)
    directions: np.ndarray,  # (P, L, 2)
    valid: np.ndarray,  # (P, L) bool
    optimal: np.ndarray,  # (P, 2)
    max_speed: float,
) -> np.ndarray:
    """Batched 2-D linear program mirroring :func:`_halfplane_optimize`.

    Each agent is solved independently; agents that hit an infeasible constraint
    return zero velocity and are frozen for the remaining lines, exactly like the
    scalar implementation.
    """
    num_agents, num_lines, _ = points.shape
    # ``_Line`` normalizes directions on construction; mirror that here.
    directions = _normalized_batch(directions)
    # Drop constraints that are invalid for every agent (keeps the pair
    # precomputation small in sparse scenes).
    column_active = valid.any(axis=0)
    if not column_active.all():
        points = points[:, column_active]
        directions = directions[:, column_active]
        valid = valid[:, column_active]
        num_lines = points.shape[1]
    result = np.array(optimal, dtype=float, copy=True)
    done = np.zeros(num_agents, dtype=bool)
    eps = 1e-12

    # The per-pair coefficients below do not depend on the intermediate result,
    # so precompute them once and then vectorize the inner accumulation loop.
    dirs_x = directions[..., 0]
    dirs_y = directions[..., 1]
    tangent = _perp_batch(directions)
    tangent_x = tangent[..., 0]
    tangent_y = tangent[..., 1]
    self_term = np.sum(directions * points, axis=-1)  # (P, L)
    pair_num = np.einsum("pjk,pik->pij", directions, points) - self_term[:, None, :]
    pair_den = tangent_x[:, :, None] * dirs_y[:, None, :] - tangent_y[:, :, None] * dirs_x[:, None, :]

    for i in range(num_lines):
        active = valid[:, i] & (~done)
        if not active.any():
            continue
        p_i = points[:, i]
        d_i = directions[:, i]
        inside = np.sum((result - p_i) * d_i, axis=-1) >= 0.0
        need = active & (~inside)
        if not need.any():
            continue

        tangent_i = _perp_batch(d_i)
        proj = np.sum((optimal - p_i) * tangent_i, axis=-1)
        left = np.full(num_agents, -1e6)
        right = np.full(num_agents, 1e6)
        infeasible = np.zeros(num_agents, dtype=bool)

        if i > 0:
            from_prev = valid[:, :i] & need[:, None]  # (P, i)
            den = pair_den[:, i, :i]
            num = pair_num[:, i, :i]
            degenerate = np.abs(den) < eps
            infeasible |= (from_prev & degenerate & (num < 0.0)).any(axis=1)
            offset = num / np.where(degenerate, 1.0, den)
            right_vals = np.where(from_prev & (den > 0.0), offset, np.inf).min(axis=1)
            left_vals = np.where(from_prev & (den < 0.0), offset, -np.inf).max(axis=1)
            right = np.minimum(right_vals, 1e6)
            left = np.maximum(left_vals, -1e6)

        infeasible |= need & (left > right)
        proj = np.clip(proj, left, right)
        new_result = p_i + tangent_i * proj[:, None]
        result = np.where((need & (~infeasible))[:, None], new_result, result)
        result = np.where(infeasible[:, None], 0.0, result)
        done = done | infeasible

    speed = np.linalg.norm(result, axis=-1, keepdims=True)
    over = speed > max_speed
    return np.where(over, result / np.maximum(speed, eps) * max_speed, result)


def compute_orca_velocities(
    positions: np.ndarray,  # (P, 2)
    velocities: np.ndarray,  # (P, 2)
    preferred: np.ndarray,  # (P, 2)
    radii: np.ndarray,  # (P,) or scalar
    max_speed: float,
    static_obstacles: list[tuple[np.ndarray, float]] | None = None,
    robot: tuple[np.ndarray, np.ndarray, float] | None = None,
    tau: float = 3.0,
    dt: float = 0.2,
    neighbor_factor: float = 5.0,
    obstacle_factor: float = 3.0,
    agent_responsibility: float = 0.5,
) -> np.ndarray:
    """Vectorized ORCA velocity for a whole pedestrian population.

    All pairs (including the optional ``robot``) share avoidance symmetrically
    with ``agent_responsibility``; static obstacles are zero-velocity peers the
    pedestrians avoid fully (responsibility 1.0).

    Args:
        positions: Agent positions [m], shape ``(P, 2)``.
        velocities: Agent velocities [m/s], shape ``(P, 2)``.
        preferred: Preferred velocities [m/s], shape ``(P, 2)``.
        radii: Agent collision radii [m], shape ``(P,)`` or scalar.
        max_speed: Maximum speed [m/s].
        static_obstacles: ``[(pos2d, radius), ...]`` circular obstacles.
        robot: Optional ``(pos2d, vel2d, radius)`` treated as an ordinary peer.
        tau: Time horizon [s].
        dt: Simulation timestep [s].
        neighbor_factor: Neighbor cutoff as a multiple of the summed radii.
        obstacle_factor: Static-obstacle cutoff as a multiple of the summed radii.
        agent_responsibility: Fraction of avoidance each agent takes (0.5 =
            reciprocal sharing).

    Returns:
        New velocities [m/s], shape ``(P, 2)``.
    """
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    preferred = np.asarray(preferred, dtype=float)
    num_agents = positions.shape[0]
    if num_agents == 0:
        return np.zeros((0, 2))

    radii = np.asarray(radii, dtype=float).reshape(-1)
    if radii.shape[0] == 1:
        radii = np.repeat(radii, num_agents)
    max_speed = float(max_speed)

    line_points: list[np.ndarray] = []
    line_dirs: list[np.ndarray] = []
    line_valid: list[np.ndarray] = []

    # Static obstacle constraints first (full responsibility), matching the
    # ordering of the scalar reference implementation.
    if static_obstacles:
        obs_pos = np.asarray([obs[0] for obs in static_obstacles], dtype=float).reshape(-1, 2)
        obs_rad = np.asarray([obs[1] for obs in static_obstacles], dtype=float).reshape(-1)
        num_obs = obs_pos.shape[0]
        points, dirs = _orca_halfplanes_batch(
            positions,
            velocities,
            radii,
            np.broadcast_to(obs_pos[None, :, :], (num_agents, num_obs, 2)),
            np.zeros((num_agents, num_obs, 2)),
            np.broadcast_to(obs_rad[None, :], (num_agents, num_obs)),
            tau,
            dt,
            1.0,
        )
        dist = np.linalg.norm(obs_pos[None, :, :] - positions[:, None, :], axis=-1)
        valid = dist < (radii[:, None] + obs_rad[None, :]) * obstacle_factor
        line_points.append(points)
        line_dirs.append(dirs)
        line_valid.append(valid)

    # Agent-agent constraints (a peer list ordered by index, self excluded)
    other_pos = np.broadcast_to(positions[None, :, :], (num_agents, num_agents, 2))
    other_vel = np.broadcast_to(velocities[None, :, :], (num_agents, num_agents, 2))
    other_rad = np.broadcast_to(radii[None, :], (num_agents, num_agents))
    points, dirs = _orca_halfplanes_batch(
        positions, velocities, radii, other_pos, other_vel, other_rad, tau, dt, agent_responsibility
    )
    pair_dist = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    pair_valid = (~np.eye(num_agents, dtype=bool)) & (pair_dist < (radii[:, None] + radii[None, :]) * neighbor_factor)
    line_points.append(points)
    line_dirs.append(dirs)
    line_valid.append(pair_valid)

    # Robot as an extra peer agent, appended last
    if robot is not None:
        robot_pos = np.asarray(robot[0], dtype=float).reshape(1, 2)
        robot_vel = np.asarray(robot[1], dtype=float).reshape(1, 2)
        robot_rad = np.asarray(robot[2], dtype=float).reshape(1)
        points, dirs = _orca_halfplanes_batch(
            positions,
            velocities,
            radii,
            np.broadcast_to(robot_pos[None, :, :], (num_agents, 1, 2)),
            np.broadcast_to(robot_vel[None, :, :], (num_agents, 1, 2)),
            np.broadcast_to(robot_rad[None, :], (num_agents, 1)),
            tau,
            dt,
            agent_responsibility,
        )
        dist = np.linalg.norm(robot_pos[None, :, :] - positions[:, None, :], axis=-1)
        valid = dist < (radii[:, None] + robot_rad[None, :]) * neighbor_factor
        line_points.append(points)
        line_dirs.append(dirs)
        line_valid.append(valid)

    all_points = np.concatenate(line_points, axis=1)
    all_dirs = np.concatenate(line_dirs, axis=1)
    all_valid = np.concatenate(line_valid, axis=1)
    return _halfplane_optimize_batch(all_points, all_dirs, all_valid, preferred, max_speed)

import numpy as np
import torch
import trimesh
from pxr import Usd
from scipy.spatial.transform import Rotation as R

from isaaclab.sim.spawners.meshes import meshes_cfg
from isaaclab.sim.spawners.meshes.meshes import _spawn_mesh_geom_from_mesh
from isaaclab.sim.utils import clone, get_current_stage


def _quat_to_yaw(quat: torch.tensor) -> torch.tensor:
    x, y, z, w = quat.unbind(-1)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: torch.tensor) -> torch.tensor:
    half_yaw = yaw / 2
    return torch.stack(
        [torch.zeros_like(half_yaw), torch.zeros_like(half_yaw), torch.sin(half_yaw), torch.cos(half_yaw)], dim=-1
    )


def lookat_from_pose(camera_pos, camera_quat):
    """
    camera_pos:  (N, 3)
    camera_quat: (N, 4) in (x, y, z, w)
    returns:
      eye: (N, 3)
      at:  (N, 3)
      up:  (N, 3)
    """
    eye = np.asarray(camera_pos.cpu(), dtype=float)
    quat = np.asarray(camera_quat.cpu(), dtype=float)

    if eye.ndim != 2 or eye.shape[1] != 3:
        raise ValueError(f"camera_pos must be (N,3), got {eye.shape}")
    if quat.ndim != 2 or quat.shape[1] != 4 or quat.shape[0] != eye.shape[0]:
        raise ValueError(f"camera_quat must be (N,4) with same N, got {quat.shape}")

    r = R.from_quat(quat)  # batched rotations

    f_local = np.tile([1.0, 0.0, 0.0], (eye.shape[0], 1))  # local +X forward
    u_local = np.tile([0.0, 0.0, 1.0], (eye.shape[0], 1))  # local +Z up

    forward_world = r.apply(f_local)  # (N,3)
    up_world = r.apply(u_local)  # (N,3)

    at = eye + forward_world
    up = up_world
    return eye, at, up


def perspective(intrinsics, near=0.1, far=100.0, device="cuda", dtype=torch.float32):
    """
    Build OpenGL-style batched projection matrices from intrinsics.

    Args:
        intrinsics: (N, 3, 3), pixel-space K
        near, far: clip planes

    Assumes:
        image width  W = 2 * cx
        image height H = 2 * cy
        pixel origin is top-left (so y is flipped into NDC).
    """
    K = torch.as_tensor(intrinsics, device=device, dtype=dtype)
    fx = K[:, 0, 0]
    fy = K[:, 1, 1]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    N = K.shape[0]
    P = torch.zeros((N, 4, 4), device=device, dtype=dtype)

    P[:, 0, 0] = fx / cx
    P[:, 1, 1] = fy / cy
    P[:, 0, 2] = 0
    P[:, 1, 2] = 0
    P[:, 2, 2] = (far + near) / (near - far)
    P[:, 2, 3] = (2.0 * far * near) / (near - far)
    P[:, 3, 2] = -1.0

    return P


def look_at(eye, at, up, device="cuda", dtype=torch.float32):
    eye = torch.as_tensor(eye, device=device, dtype=dtype)
    at = torch.as_tensor(at, device=device, dtype=dtype)
    up = torch.as_tensor(up, device=device, dtype=dtype)
    N = eye.shape[0]
    eps = 1e-8
    z = eye - at
    z = z / (torch.linalg.norm(z, dim=-1, keepdim=True) + eps)

    x = torch.cross(up, z, dim=-1)
    x = x / (torch.linalg.norm(x, dim=-1, keepdim=True) + eps)

    y = torch.cross(z, x, dim=-1)

    V = torch.zeros((N, 4, 4), device=device, dtype=dtype)
    V[:, 0, :3] = x
    V[:, 1, :3] = y
    V[:, 2, :3] = z
    V[:, 3, 3] = 1.0
    V[:, :3, 3] = -torch.bmm(V[:, :3, :3], eye.unsqueeze(-1)).squeeze(-1)

    return V


# todo speed up collision checking
def check_collision_bbox(robot_states: torch.Tensor, obstacle_states: torch.Tensor) -> torch.Tensor:
    """Check 2D oriented bounding box collision between robots and obstacles on the x,y plane.

    Uses the Separating Axis Theorem (SAT) for OBB collision detection with yaw rotation.

    Args:
        robot_states: [num_envs, 7] (x, y, z, bbox_x, bbox_y, bbox_z, yaw)
        obstacle_states: [num_obstacles, 7] (x, y, z, bbox_x, bbox_y, bbox_z, yaw)

    Returns:
        torch.Tensor: [num_envs] bool tensor representing each robot's collision state
    """
    num_envs = robot_states.shape[0]
    num_obstacles = obstacle_states.shape[0]
    device = robot_states.device

    # Extract robot data
    robot_pos = robot_states[:, :2]  # [num_envs, 2] (x, y)
    robot_half_extents = robot_states[:, 3:5] / 2.0  # [num_envs, 2] (bbox_x/2, bbox_y/2)
    robot_yaw = robot_states[:, 6]  # [num_envs]

    # Extract obstacle data (shared across all envs)
    obs_pos = obstacle_states[:, :2]  # [num_obstacles, 2]
    obs_half_extents = obstacle_states[:, 3:5] / 2.0  # [num_obstacles, 2] (bbox_x/2, bbox_y/2)
    obs_yaw = obstacle_states[:, 6]  # [num_obstacles]

    # Compute rotation matrices for robot: [num_envs, 2]
    cos_r = torch.cos(robot_yaw)
    sin_r = torch.sin(robot_yaw)
    # Robot local axes (columns are x-axis and y-axis in world frame)
    robot_axis_x = torch.stack([cos_r, sin_r], dim=-1)  # [num_envs, 2]
    robot_axis_y = torch.stack([-sin_r, cos_r], dim=-1)  # [num_envs, 2]

    # Compute rotation matrices for obstacles: [num_obstacles, 2]
    cos_o = torch.cos(obs_yaw)
    sin_o = torch.sin(obs_yaw)
    obs_axis_x = torch.stack([cos_o, sin_o], dim=-1)  # [num_obstacles, 2]
    obs_axis_y = torch.stack([-sin_o, cos_o], dim=-1)  # [num_obstacles, 2]

    # Vector from robot center to obstacle center: [num_envs, num_obstacles, 2]
    # robot_pos: [num_envs, 2] -> [num_envs, 1, 2]
    # obs_pos: [num_obstacles, 2] -> [1, num_obstacles, 2]
    delta = obs_pos.unsqueeze(0) - robot_pos.unsqueeze(1)  # [num_envs, num_obstacles, 2]

    # Expand robot axes for broadcasting: [num_envs, 1, 2]
    robot_axis_x_exp = robot_axis_x.unsqueeze(1)
    robot_axis_y_exp = robot_axis_y.unsqueeze(1)
    robot_half_x = robot_half_extents[:, 0].unsqueeze(1)  # [num_envs, 1]
    robot_half_y = robot_half_extents[:, 1].unsqueeze(1)  # [num_envs, 1]

    # Expand obstacle axes for broadcasting: [1, num_obstacles, 2]
    obs_axis_x_exp = obs_axis_x.unsqueeze(0)  # [1, num_obstacles, 2]
    obs_axis_y_exp = obs_axis_y.unsqueeze(0)  # [1, num_obstacles, 2]
    obs_half_x = obs_half_extents[:, 0].unsqueeze(0)  # [1, num_obstacles]
    obs_half_y = obs_half_extents[:, 1].unsqueeze(0)  # [1, num_obstacles]

    # SAT: Check separation on 4 axes (2 from robot, 2 from obstacle)
    # For each axis, compute projection of delta and sum of projected half-extents
    # If |delta · axis| > sum of projected half-extents, there's separation

    def check_axis_separation(axis, delta, r_hx, r_hy, r_ax, r_ay, o_hx, o_hy, o_ax, o_ay):
        """Check if there's separation along the given axis.

        Args:
            axis: [num_envs, 1, 2] or [1, num_obstacles, 2] - the separation axis
            delta: [num_envs, num_obstacles, 2] - vector between centers
            r_hx, r_hy: robot half extents [num_envs, 1]
            r_ax, r_ay: robot local axes [num_envs, 1, 2]
            o_hx, o_hy: obstacle half extents [1, num_obstacles]
            o_ax, o_ay: obstacle local axes [1, num_obstacles, 2]
        """
        # Project delta onto axis
        delta_proj = torch.abs((delta * axis).sum(dim=-1))  # [num_envs, num_obstacles]

        # Project robot half-extents onto axis
        robot_proj = r_hx * torch.abs((r_ax * axis).sum(dim=-1)) + r_hy * torch.abs((r_ay * axis).sum(dim=-1))

        # Project obstacle half-extents onto axis
        obs_proj = o_hx * torch.abs((o_ax * axis).sum(dim=-1)) + o_hy * torch.abs((o_ay * axis).sum(dim=-1))

        # Separation exists if delta projection > sum of half-extent projections
        return delta_proj > (robot_proj + obs_proj)

    # Check all 4 axes
    sep_robot_x = check_axis_separation(
        robot_axis_x_exp,
        delta,
        robot_half_x,
        robot_half_y,
        robot_axis_x_exp,
        robot_axis_y_exp,
        obs_half_x,
        obs_half_y,
        obs_axis_x_exp,
        obs_axis_y_exp,
    )
    sep_robot_y = check_axis_separation(
        robot_axis_y_exp,
        delta,
        robot_half_x,
        robot_half_y,
        robot_axis_x_exp,
        robot_axis_y_exp,
        obs_half_x,
        obs_half_y,
        obs_axis_x_exp,
        obs_axis_y_exp,
    )
    sep_obs_x = check_axis_separation(
        obs_axis_x_exp,
        delta,
        robot_half_x,
        robot_half_y,
        robot_axis_x_exp,
        robot_axis_y_exp,
        obs_half_x,
        obs_half_y,
        obs_axis_x_exp,
        obs_axis_y_exp,
    )
    sep_obs_y = check_axis_separation(
        obs_axis_y_exp,
        delta,
        robot_half_x,
        robot_half_y,
        robot_axis_x_exp,
        robot_axis_y_exp,
        obs_half_x,
        obs_half_y,
        obs_axis_x_exp,
        obs_axis_y_exp,
    )

    # No collision if separated on any axis
    separated = sep_robot_x | sep_robot_y | sep_obs_x | sep_obs_y  # [num_envs, num_obstacles]

    # Collision occurs if NOT separated
    collision_per_obstacle = ~separated  # [num_envs, num_obstacles]

    # Robot is in collision if it collides with any obstacle
    collision = collision_per_obstacle.any(dim=-1)  # [num_envs]

    return collision


@clone
def spawn_mesh_cuboid(
    prim_path: str,
    cfg: meshes_cfg.MeshCuboidCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> Usd.Prim:
    """Create a USD-Mesh cuboid prim with the given attributes.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration instance.
        translation: The translation to apply to the prim w.r.t. its parent prim. Defaults to None, in which case
            this is set to the origin.
        orientation: The orientation in (x, y, z, w) to apply to the prim w.r.t. its parent prim. Defaults to None,
            in which case this is set to identity.
        **kwargs: Additional keyword arguments, like ``clone_in_fabric``.

    Returns:
        The created prim.

    Raises:
        ValueError: If a prim already exists at the given path.
    """
    # create a trimesh box
    transform = np.eye(4)
    transform[2, 3] = cfg.size[2] / 2.0  # shift up by half height to align with the ground
    box = trimesh.creation.box(cfg.size, transform=transform)
    # obtain stage handle
    stage = get_current_stage()
    # spawn the cuboid as a mesh
    _spawn_mesh_geom_from_mesh(prim_path, cfg, box, translation, orientation, None, stage=stage)
    # return the prim
    return stage.GetPrimAtPath(prim_path)


def create_prim_from_mesh(prim_path: str, mesh: trimesh.Trimesh, **kwargs):
    """Create a USD prim with mesh defined from vertices and triangles.

    The function creates a USD prim with a mesh defined from vertices and triangles. It performs the
    following steps:

    - Create a USD Xform prim at the path :obj:`prim_path`.
    - Create a USD prim with a mesh defined from the input vertices and triangles at the path :obj:`{prim_path}/mesh`.
    - Assign a physics material to the mesh at the path :obj:`{prim_path}/physicsMaterial`.
    - Assign a visual material to the mesh at the path :obj:`{prim_path}/visualMaterial`.

    Args:
        prim_path: The path to the primitive to be created.
        mesh: The mesh to be used for the primitive.

    Keyword Args:
        translation: The translation of the terrain. Defaults to None.
        orientation: The orientation of the terrain. Defaults to None.
        visual_material: The visual material to apply. Defaults to None.
        physics_material: The physics material to apply. Defaults to None.
    """
    # need to import these here to prevent isaacsim launching when importing this module
    # from pxr import UsdGeom
    from pxr import Sdf, UsdGeom

    import isaaclab.sim as sim_utils
    import isaaclab.sim.utils.prims as prim_utils

    # create parent prim
    prim_utils.create_prim(prim_path, "Xform")
    # create mesh prim
    prim = prim_utils.create_prim(
        f"{prim_path}/mesh",
        "Mesh",
        translation=kwargs.get("translation"),
        orientation=kwargs.get("orientation"),
        attributes={
            "points": mesh.vertices,
            "faceVertexIndices": mesh.faces.flatten(),
            "faceVertexCounts": np.asarray([3] * len(mesh.faces)),
            "subdivisionScheme": "bilinear",
        },
    )

    # apply collider properties
    collider_cfg = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
    sim_utils.define_collision_properties(prim.GetPrimPath(), collider_cfg)
    # add rgba color to the mesh primvars
    if mesh.visual.vertex_colors is not None:
        # obtain color from the mesh
        rgba_colors = np.asarray(mesh.visual.vertex_colors).astype(np.float32) / 255.0
        # displayColor is a primvar attribute that is used to color the mesh
        color_prim_attr = prim.GetAttribute("primvars:displayColor")
        color_prim_var = UsdGeom.Primvar(color_prim_attr)
        color_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        color_prim_attr.Set(rgba_colors[:, :3])
        # displayOpacity is a primvar attribute that is used to set the opacity of the mesh
        display_prim_attr = prim.GetAttribute("primvars:displayOpacity")
        display_prim_var = UsdGeom.Primvar(display_prim_attr)
        display_prim_var.SetInterpolation(UsdGeom.Tokens.vertex)
        display_prim_var.Set(rgba_colors[:, 3])

        try:
            usd_mesh = UsdGeom.Mesh(prim)
            primvars_api = UsdGeom.PrimvarsAPI(usd_mesh)
            st_primvar = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
            st_primvar.Set(mesh.visual.uvs)
        except Exception:
            pass

    # create visual material
    if kwargs.get("visual_material") is not None:
        visual_material_cfg: sim_utils.VisualMaterialCfg = kwargs.get("visual_material")
        # spawn the material
        visual_material_cfg.func(f"{prim_path}/visualMaterial", visual_material_cfg)
        sim_utils.bind_visual_material(prim.GetPrimPath(), f"{prim_path}/visualMaterial")
    # create physics material
    if kwargs.get("physics_material") is not None:
        physics_material_cfg: sim_utils.RigidBodyMaterialCfg = kwargs.get("physics_material")
        # spawn the material
        physics_material_cfg.func(f"{prim_path}/physicsMaterial", physics_material_cfg)
        sim_utils.bind_physics_material(prim.GetPrimPath(), f"{prim_path}/physicsMaterial")

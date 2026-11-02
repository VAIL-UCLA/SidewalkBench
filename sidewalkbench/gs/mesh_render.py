"""Triangle meshes rasterized with nvdiffrast and depth-composited over a gsplat render.

A :class:`MeshDepth` rasterizes a static mesh's z-depth (the scene's collision mesh, for
``gsplat_render``). A :class:`MeshCompositor` holds one vertex-colored mesh and, for a batch of
cameras, keeps whichever of mesh / background is closer per pixel, lit like PG's Isaac scenes
(``shade``, below): the articulated robot (video chase view, a :class:`LinkedMesh` posed from the
sim). The cones are Gaussians lit with ``shade`` (``cones.py``).
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import trimesh

from .obs import rotmat_from_quat_xyzw

# PG's Coco lights (sidewalkbench/pg/primitives/navigation/random_env_cfg.py): a dome light with this HDR sky and a
# distant sun pointing straight down, both intensity ~890. Diffuse: radiance = albedo / pi * (sky + sun
# irradiance), then RTX's camera response (exposure, ACES tone curve, sRGB). The HDR's azimuth convention
# and the exposure were set against an RTX render of matte spheres under these lights (~9/255 mean error).
SKY_HDR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "materials",
                       "kloofendal_48d_partly_cloudy_puresky_4k.hdr")
LIGHT_INTENSITY = 890.0
SKY_AZIMUTH_DEG = 330.0  # RTX maps the latlong image mirrored, rotated by this about +z
EXPOSURE = 2.07e-3


def _sh9(n):
    """Real spherical harmonics up to order 2 (unnormalized) of unit vectors (..., 3), numpy or torch."""
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    xp = torch if isinstance(n, torch.Tensor) else np
    return xp.stack([xp.ones_like(x), y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y], -1)


def sky_irradiance_sh() -> np.ndarray:
    """(9, 3) so that the dome light's irradiance on a surface with normal n is ``_sh9(n) @ M``."""
    sky = cv2.resize(cv2.imread(SKY_HDR, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1], (256, 128),
                     interpolation=cv2.INTER_AREA)
    h, w = sky.shape[:2]
    th, ph = np.meshgrid((np.arange(h) + 0.5) / h * np.pi,
                         np.radians(SKY_AZIMUTH_DEG) - (np.arange(w) + 0.5) / w * 2 * np.pi, indexing="ij")
    d = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    c = np.array([0.282095] + [0.488603] * 3 + [1.092548, 1.092548, 0.315392, 1.092548, 0.546274])
    y = _sh9(d) * c * (np.sin(th) * np.pi / h * 2 * np.pi / w)[..., None]  # basis x solid angle
    band = np.array([np.pi] + [2 * np.pi / 3] * 3 + [np.pi / 4] * 5)  # cosine-lobe convolution per order
    return LIGHT_INTENSITY * (band * c)[:, None] * np.einsum("hwk,hwc->kc", y, sky)


def irradiance(n, sky):
    """PG's light on surfaces with unit normals ``n`` (..., 3): dome (``sky_irradiance_sh``) + sun straight down."""
    return _sh9(n) @ sky + LIGHT_INTENSITY * n[..., 2:].clamp(min=0.0)


def shade(albedo, irr):
    """Linear albedo under irradiance ``irr`` -> sRGB: albedo / pi * irr, then RTX's camera response."""
    x = EXPOSURE * albedo / np.pi * irr
    x = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)  # ACES tone curve
    return x.clamp(1e-6, 1.0) ** (1 / 2.2)  # sRGB


class MeshDepth:
    """A static triangle mesh rasterized for its z-depth (the scene's collision mesh)."""

    def __init__(self, mesh: trimesh.Trimesh, device):
        import nvdiffrast.torch as dr  # deferred (CUDA import)

        self.dr, self.ctx = dr, dr.RasterizeCudaContext(device=device)
        self.verts = torch.as_tensor(np.array(mesh.vertices, dtype=np.float32), device=device)
        self.tri = torch.as_tensor(mesh.faces, dtype=torch.int32, device=device).contiguous()

    def _rasterize(self, v, cam_pos_w, rot_c2w, ks, width, height, near=0.01, far=1000.0):
        """Vertices (B,V,3) -> (rast, clip, camera-frame z (B,V)) for B pinhole cameras."""
        x, y, z = torch.einsum("bji,bvj->bvi", rot_c2w, v - cam_pos_w[:, None]).unbind(-1)  # camera frame
        fx, fy, cx, cy = (ks[:, i, j, None] for i, j in ((0, 0), (1, 1), (0, 2), (1, 2)))
        clip = torch.stack([2 * fx / width * x + (2 * cx / width - 1) * z,
                            2 * fy / height * y + (2 * cy / height - 1) * z,
                            (far + near) / (far - near) * z - 2 * far * near / (far - near), z], -1).contiguous()
        return self.dr.rasterize(self.ctx, clip, self.tri, (height, width))[0], clip, z

    def depth(self, cam_pos_w, rot_c2w, ks, width, height):
        """The mesh's z-depth (B,H,W,1; inf where it is not seen)."""
        rast, _, z = self._rasterize(self.verts.expand(len(cam_pos_w), -1, -1), cam_pos_w, rot_c2w, ks, width, height)
        a, _ = self.dr.interpolate(z[..., None].contiguous(), rast, self.tri)
        return torch.where(rast[..., 3:] > 0, a, torch.inf)


class MeshCompositor(MeshDepth):
    """A vertex-colored mesh, shaded under PG's lights and depth-composited over a render."""

    def __init__(self, mesh: trimesh.Trimesh, device):
        super().__init__(mesh, device)
        t = lambda a: torch.as_tensor(np.array(a, dtype=np.float32), device=device)  # noqa: E731
        self.normals, self.base = t(mesh.vertex_normals), t(mesh.visual.vertex_colors[:, :3] / 255.0)
        self.sky = t(sky_irradiance_sh())

    def composite(self, rgb, depth, cam_pos_w, rot_c2w, ks, width, height, *, verts_w, normals_w):
        """rgb (B,H,W,3) + z-depth (B,H,W,1; inf = empty) -> (rgb, depth) with the mesh, posed by
        ``verts_w`` / ``normals_w`` (B,V,3), in front where closer."""
        rast, clip, z = self._rasterize(verts_w, cam_pos_w, rot_c2w, ks, width, height)
        attr = torch.cat([self.base.expand(len(cam_pos_w), -1, -1), irradiance(normals_w, self.sky), z[..., None]], -1)
        a, _ = self.dr.interpolate(attr.contiguous(), rast, self.tri)  # (B, H, W, 7): base, irradiance, z
        color = shade(a[..., :3], a[..., 3:6])
        hit = (rast[..., 3:] > 0) & (a[..., -1:] < depth)
        rgb = self.dr.antialias(torch.where(hit, color, rgb).contiguous(), rast, clip, self.tri)
        return rgb, torch.where(hit, a[..., -1:], depth)


def load_usd_links(usd_path: str, link_names: list[str]) -> tuple[trimesh.Trimesh, np.ndarray]:
    """The visible meshes of an articulation USD, each in its link's frame, merged into one
    vertex-colored mesh + a per-vertex index into ``link_names``. Faces take their GeomSubset's
    (else the mesh's) material diffuse color."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usd_path)
    xc = UsdGeom.XformCache()

    def color(prim) -> np.ndarray:
        mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        for sh in mat.GetPrim().GetChildren() if mat else ():
            for name in ("diffuseColor", "diffuse_color_constant"):
                inp = UsdShade.Shader(sh).GetInput(name)
                if inp and inp.Get() is not None:
                    return np.asarray(inp.Get(), dtype=float)
        return np.full(3, 0.5)

    verts, faces, colors, links = [], [], [], []
    for p in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not p.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(p).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        link = p.GetParent()
        while link and link.GetName() not in link_names:
            link = link.GetParent()
        if not link:
            continue
        mesh = UsdGeom.Mesh(p)
        rel = np.array(xc.GetLocalToWorldTransform(p) * xc.GetLocalToWorldTransform(link).GetInverse())  # row-vector
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=float) @ rel[:3, :3] + rel[3, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        # fan-triangulate the polygons: triangle (s, s+k, s+k+1) for k = 1 .. count-2
        face = np.repeat(np.arange(len(counts)), counts - 2)
        start = (np.cumsum(counts) - counts)[face]
        k = np.arange(len(face)) - np.repeat(np.cumsum(counts - 2) - (counts - 2), counts - 2) + 1
        tris = np.stack([idx[start], idx[start + k], idx[start + k + 1]], 1)
        fcol = np.tile(color(p), (len(counts), 1))
        for s in UsdGeom.Subset.GetAllGeomSubsets(mesh):
            fcol[np.asarray(s.GetIndicesAttr().Get())] = color(s.GetPrim())
        vcol = np.full((len(pts), 3), 0.5)
        vcol[tris.ravel()] = np.repeat(fcol[face], 3, axis=0)
        faces.append(tris + sum(len(v) for v in verts))
        verts.append(pts)
        colors.append(vcol)
        links.append(np.full(len(pts), link_names.index(link.GetName())))
    mesh = trimesh.Trimesh(np.concatenate(verts), np.concatenate(faces),
                           vertex_colors=(np.concatenate(colors) * 255).astype(np.uint8), process=False)
    return mesh, np.concatenate(links)


class LinkedMesh:
    """A mesh whose every vertex rides on one of L rigid bodies (the robot's links), posed from the sim."""

    def __init__(self, mesh: trimesh.Trimesh, link: np.ndarray, device):
        self.comp = MeshCompositor(mesh, device)
        self.link = torch.as_tensor(link, device=device)

    def posed(self, body_pos: torch.Tensor, body_quat_xyzw: torch.Tensor):
        """A ``GaussianRenderer.render`` mesh layer: body poses (B, L, 3) / (B, L, 4), or (L, ...) for one
        env, in the scene frame."""
        pos, quat = body_pos.float().reshape(-1, *body_pos.shape[-2:]), body_quat_xyzw.float().reshape(-1, *body_quat_xyzw.shape[-2:])
        r = rotmat_from_quat_xyzw(quat.reshape(-1, 4)).reshape(*quat.shape[:2], 3, 3)[:, self.link]  # (B, V, 3, 3)
        v = torch.einsum("bvij,vj->bvi", r, self.comp.verts) + pos[:, self.link]
        n = torch.einsum("bvij,vj->bvi", r, self.comp.normals)
        return self.comp, v, n


def robot_mesh(usd_path: str, body_names: list[str], device) -> LinkedMesh:
    """The articulated robot's visual meshes, one link per sim body."""
    return LinkedMesh(*load_usd_links(usd_path, list(body_names)), device)

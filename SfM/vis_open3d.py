"""
Open3D-based visualization for Structure-from-Motion results (GUI mode).
- Point cloud plotting with subsampling & quantile clipping
- Optional camera frustums (from (R, t), K, image size)
- Equal aspect / nice default view
- Interactive window (Visualizer) and screen capture PNG
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.colors as mcolors


@dataclass
class ViewSpec:
    elev_deg: float = 20.0   # pitch (+ up)
    azim_deg: float = -60.0  # yaw (+ left)
    dist_scale: float = 1.6  # distance vs bbox size


class SfMVisualizerO3D:
    def __init__(self):
        self._geoms = []
        self._bbox = None
        self._bbox_points = None  # bbox from point clouds only
        self._vis = None
        self._win_created = False
        # store as RGBA internally; GUI will use only RGB
        self._bg_color = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # ---------- helpers ----------
    @staticmethod
    def _to_o3d_color(color_hex_or_rgb, n: int):
        # hex "#RRGGBB" or tuple/list in [0,1]
        if isinstance(color_hex_or_rgb, str) and color_hex_or_rgb.startswith("#"):
            s = color_hex_or_rgb.lstrip('#')
            # support 3-digit shorthand like #555 → #555555
            if len(s) == 3:
                s = ''.join([c*2 for c in s])
            if len(s) != 6:
                raise ValueError(f"Invalid hex color: {color_hex_or_rgb}")
            rgb = tuple(int(s[i:i+2], 16)/255.0 for i in (0, 2, 4))
        else:
            rgb = tuple(color_hex_or_rgb)
        return np.tile(np.array(rgb, dtype=np.float32)[None, :], (n, 1))

    @staticmethod
    def _spherical_front(elev_deg, azim_deg):
        # Open3D expects "front" (camera looking direction).
        el = np.deg2rad(elev_deg)
        az = np.deg2rad(azim_deg)
        # yaw=az (around Z), pitch=el (from xy-plane)
        x = np.cos(el) * np.cos(az)
        y = np.cos(el) * np.sin(az)
        z = np.sin(el)
        front = np.array([x, y, z], dtype=np.float64)
        # choose a reasonable up vector (avoid collinearity near poles)
        up = np.array([0, 0, 1], dtype=np.float64)
        if abs(np.dot(front, up)) > 0.95:
            up = np.array([0, 1, 0], dtype=np.float64)
        return front / np.linalg.norm(front), up / np.linalg.norm(up)

    @staticmethod
    def _camera_frustum_lineset(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                                img_w: int, img_h: int, scale: float = 0.1,
                                color=(0.2, 0.4, 1.0)):
        """
        Build a wireframe camera frustum Lineset in world coordinates.
        - K: 3x3
        - R: 3x3, t: (3,) world_T_cam: X_w = R * X_c + t   (R, t are camera-to-world)
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        # depth 'z' controls frustum length in camera coords
        z = scale
        corners_px = np.array([[0,       0],
                               [img_w,   0],
                               [img_w, img_h],
                               [0,     img_h]], dtype=np.float64)

        # back-project image corners at depth z in camera coordinates
        corners_cam = []
        for u, v in corners_px:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            corners_cam.append([x, y, z])
        corners_cam = np.asarray(corners_cam)  # (4, 3)

        # camera center in camera coords
        C_cam = np.zeros((1, 3), dtype=np.float64)  # [0,0,0]

        # transform to world: X_w = R * X_c + t
        def cam2world(Xc):
            return (R @ Xc.T).T + t[None, :]

        corners_w = cam2world(corners_cam)
        C_w = cam2world(C_cam)[0]

        # Lines: 5 nodes: C + 4 corners
        points = np.vstack([C_w, corners_w])  # idx: 0..4
        lines = [
            [0, 1], [0, 2], [0, 3], [0, 4],  # from C to corners
            [1, 2], [2, 3], [3, 4], [4, 1]   # rectangle rim
        ]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.array(color, dtype=np.float64), (len(lines), 1))
        )
        return ls

    @staticmethod
    def _camera_frustum_mesh(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                             img_w: int, img_h: int, scale: float = 0.1,
                             color=(1.0, 0.0, 0.0)):
        """
        Build a filled camera frustum as TriangleMesh (COLMAP-like filled pyramid).
        Assumes (R, t) are cam2world: X_w = R X_c + t.
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

        z = scale
        corners_px = np.array([[0,       0],
                               [img_w,   0],
                               [img_w, img_h],
                               [0,     img_h]], dtype=np.float64)
        corners_cam = []
        for u, v in corners_px:
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            corners_cam.append([x, y, z])
        corners_cam = np.asarray(corners_cam)
        C_cam = np.zeros((1, 3), dtype=np.float64)

        def cam2world(Xc):
            return (R @ Xc.T).T + t[None, :]

        corners_w = cam2world(corners_cam)
        C_w = cam2world(C_cam)[0]

        # vertices: C + 4 corners
        verts = np.vstack([C_w, corners_w])
        # faces: sides (4 triangles) + image plane (2 triangles)
        faces = np.array([
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 4],
            [0, 4, 1],
            [1, 2, 3],
            [1, 3, 4],
        ], dtype=np.int32)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        col = np.asarray(color, dtype=np.float64)
        mesh.vertex_colors = o3d.utility.Vector3dVector(np.tile(col, (verts.shape[0], 1)))
        return mesh

    def _update_bbox(self):
        if not self._geoms:
            self._bbox = None
            self._bbox_points = None
            return

        mins_all, maxs_all = [], []
        mins_pts, maxs_pts = [], []
        for g in self._geoms:
            aabb = g.get_axis_aligned_bounding_box()
            if aabb is None:
                continue
            lo = np.asarray(aabb.get_min_bound())
            hi = np.asarray(aabb.get_max_bound())
            mins_all.append(lo); maxs_all.append(hi)
            # prefer bbox from point clouds only for view fitting
            if isinstance(g, o3d.geometry.PointCloud):
                mins_pts.append(lo); maxs_pts.append(hi)

        if mins_all:
            lo = np.min(np.vstack(mins_all), axis=0)
            hi = np.max(np.vstack(maxs_all), axis=0)
            self._bbox = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        else:
            self._bbox = None

        if mins_pts:
            lo = np.min(np.vstack(mins_pts), axis=0)
            hi = np.max(np.vstack(maxs_pts), axis=0)
            self._bbox_points = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        else:
            self._bbox_points = None

    # ---------- public API ----------
    def set_background(self, rgb=(1, 1, 1), alpha: float = 1.0):
        """Store background as RGBA float32 [0,1]."""
        r, g, b = rgb
        self._bg_color = np.array([r, g, b, alpha], dtype=np.float32)

    def add_point_cloud(self, X: np.ndarray, *,
                        every: int = 1,
                        sample_max: Optional[int] = 100_000,
                        clip_quantile: Optional[float] = 0.995,
                        color="#555555",
                        scalars: Optional[np.ndarray] = None,
                        cmap: str = "viridis",
                        estimate_normals: bool = False):
        """
        X: (N,3) float
        """
        if X is None or len(X) == 0:
            return

        # Normalize point cloud to reasonable scale
        P = np.asarray(X[::max(1, every)], dtype=np.float64)
        center = P.mean(axis=0)
        P = P - center  # Center the points
        scale = np.abs(P).max()
        if scale > 0:
            P = P / scale * 2.0  # Scale points to a more reasonable size range
            
        print(f"After every sampling: {len(P)} points")
        
        if sample_max is not None and len(P) > sample_max:
            idx = np.random.choice(len(P), sample_max, replace=False)
            P = P[idx]
            print(f"After max sampling: {len(P)} points")

        if clip_quantile is not None:
            q = float(clip_quantile)
            lo = np.quantile(P, 1.0 - q, axis=0)
            hi = np.quantile(P, q, axis=0)
            # pad a bit
            pad = 0.02 * np.linalg.norm(hi - lo)
            lo -= pad
            hi += pad
            # crop with AABB
            aabb = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
            # temporarily create pcd to crop
            tmp = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
            tmp = tmp.crop(aabb)
            P = np.asarray(tmp.points)
            print(f"After quantile clipping: {len(P)} points")

        if len(P) == 0:
            print("Warning: No points remaining after filtering!")
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(P)
        if scalars is not None:
            s = np.asarray(scalars).reshape(-1)
            if len(s) != len(P):
                # fallback if size mismatch
                pcd.colors = o3d.utility.Vector3dVector(self._to_o3d_color(color, len(P)))
            else:
                # robust normalize to 2-98 percentile to avoid outliers dominating
                vmin = float(np.nanpercentile(s, 2))
                vmax = float(np.nanpercentile(s, 98))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
                rgba = cm.get_cmap(cmap)(norm(s))  # (N,4)
                rgb = rgba[:, :3].astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(rgb)
        else:
            pcd.colors = o3d.utility.Vector3dVector(self._to_o3d_color(color, len(P)))
        if estimate_normals:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        print(f"Created point cloud with {len(P)} points")
        self._geoms.append(pcd)
        print(f"Total geometries after adding point cloud: {len(self._geoms)}")
        self._update_bbox()
        print(f"[viz] add_point_cloud -> points={len(P)}; geoms={len(self._geoms)}")

    def add_cameras(self,
                    cams: List[Tuple[np.ndarray, np.ndarray]],
                    K: Optional[np.ndarray] = None,
                    img_size: Tuple[int, int] = (1920, 1080),
                    scale: float = 0.1,
                    color=(1.0, 0.0, 0.0),
                    show_axes: bool = True,
                    axes_scale: float = 0.05,
                    filled: bool = True):
        """
        cams: list of (R, t) with world_T_cam: X_w = R X_c + t
        """
        w, h = img_size
        for (R, t) in cams:
            if K is not None:
                if filled:
                    g = self._camera_frustum_mesh(K, R, t.reshape(-1), w, h, scale=scale, color=color)
                else:
                    g = self._camera_frustum_lineset(K, R, t.reshape(-1), w, h, scale=scale, color=color)
                self._geoms.append(g)
            if show_axes:
                # place a small coordinate frame at camera center
                C_w = (R @ np.zeros(3)).reshape(3,) + t.reshape(3,)
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axes_scale)
                # frame in cam coords; bring to world: rotation R and translation t
                frame.rotate(R, center=np.zeros(3))
                frame.translate(C_w)
                self._geoms.append(frame)
        self._update_bbox()

    def add_camera_path(self,
                    cams: List[Tuple[np.ndarray, np.ndarray]],
                    pose_type: str = "auto",   # "auto"|"cam2world"|"world2cam"
                    color: Tuple[float,float,float] = (1.0, 0.0, 0.0),
                    normalize: bool = True,
                    max_len_ratio: float = 2.0):
        """
        Draw a polyline through camera centers.
        pose_type:
        - "cam2world":  X_w = R X_c + t   → center = t
        - "world2cam":  X_c = R X_w + t   → center = -R^T t
        - "auto": pick whichever aligns better with the point-cloud bbox.
        normalize:
        - shift path to start at origin and (if needed) scale it so its extent
            is not wildly larger than the point-cloud extent.
        max_len_ratio:
        - if path extent > max_len_ratio * point_extent → scale down.
        """
        if not cams:
            return

        # build candidates
        c_c2w, c_w2c = [], []
        for R, t in cams:
            R = np.asarray(R).reshape(3,3)
            t = np.asarray(t).reshape(3,)
            c_c2w.append(t)
            c_w2c.append((-R.T @ t))
        C_c2w = np.stack(c_c2w, axis=0)
        C_w2c = np.stack(c_w2c, axis=0)

        # get points bbox (for alignment)
        if self._bbox_points is None:
            self._update_bbox()
        pts_bbox = self._bbox_points if self._bbox_points is not None else self._bbox

        def _score(C):
            if pts_bbox is None or len(C) == 0:
                return 0.0
            pc = pts_bbox.get_center()
            pe = np.linalg.norm(pts_bbox.get_extent()) + 1e-9
            Cc = C.mean(axis=0)
            Ce = (C.max(axis=0) - C.min(axis=0))
            Ce = np.linalg.norm(Ce)
            # distance of centers + extent mismatch (smaller is better)
            return np.linalg.norm(Cc - pc) / pe + abs(Ce/pe - 1.0)

        # choose pose_type if auto
        if pose_type == "auto":
            s_c2w = _score(C_c2w)
            s_w2c = _score(C_w2c)
            use = "cam2world" if s_c2w <= s_w2c else "world2cam"
        else:
            use = pose_type

        centers = C_c2w if use == "cam2world" else C_w2c

        # normalize (shift + optional scale)
        if normalize and (pts_bbox is not None):
            centers = centers.copy()
            centers -= centers.mean(axis=0)  # Center the camera path
            path_ext = np.linalg.norm(centers.max(0) - centers.min(0)) + 1e-9
            pts_ext  = np.linalg.norm(pts_bbox.get_extent()) + 1e-9
            # Scale camera path to match point cloud scale
            scale = (2.0 * pts_ext) / path_ext
            centers *= scale

        # too short to draw
        if len(centers) < 2:
            return

        lines = [[i, i+1] for i in range(len(centers)-1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(centers),
            lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        )
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
        )
        self._geoms.append(ls)
        self._update_bbox()
        print(f"[viz] add_camera_path: use={use}, normalize={normalize}")

    # ---------- GUI window ----------
    def _ensure_window(self, window_name="SfM (Open3D)", width=1280, height=900, visible=True):
        if self._win_created:
            return
        self._vis = o3d.visualization.Visualizer()
        ok = self._vis.create_window(window_name=window_name, width=width, height=height, visible=visible)
        if not ok:
            raise RuntimeError("Open3D: create_window failed.")
        for g in self._geoms:
            self._vis.add_geometry(g)
        opt = self._vis.get_render_option()
        # use only RGB for GUI background
        bg_rgb = self._bg_color[:3] if isinstance(self._bg_color, np.ndarray) and self._bg_color.shape[0] >= 3 \
                 else np.array([1,1,1], dtype=np.float32)
        opt.background_color = bg_rgb
        opt.point_size = 3.0  # 점 크기 증가
        opt.line_width = 2.0  # 선 굵기 유지
        self._win_created = True

    def _fit_view(self, view: ViewSpec):
        if self._bbox is None or self._bbox_points is None:
            self._update_bbox()
        # Prefer fitting to points only; fall back to all geoms
        bbox = self._bbox_points if self._bbox_points is not None else self._bbox
        if bbox is None:
            print("[viz] WARNING: no bbox to fit.")
            return
        ctr = self._vis.get_view_control()
        center = bbox.get_center()
        extent = float(np.linalg.norm(bbox.get_extent()))
        front, up = self._spherical_front(view.elev_deg, view.azim_deg)
        ctr.set_lookat(center)
        ctr.set_front(front)
        ctr.set_up(up)
        # 장면 크기에 따라 동적으로 zoom 조정
        zoom = 0.7
        if extent > 0:
            zoom = np.clip(0.35 + 0.15 * np.log10(extent + 1e-6), 0.2, 0.8)
        ctr.set_zoom(zoom)
        print(f"[viz] fit_view: center={center}, extent={extent:.3f}, zoom={zoom:.3f}")

    def show(self, view: ViewSpec = ViewSpec(), window_name="SfM (Open3D)",
             width=1280, height=900):
        """Open interactive window; you can move/zoom and take screenshots manually."""
        self._ensure_window(window_name=window_name, width=width, height=height, visible=True)
        self._fit_view(view)
        self._vis.run()
        self._vis.destroy_window()
        self._win_created = False

    def save_png(self, png_path: Path, view: ViewSpec = ViewSpec(),
                 width=1600, height=1200, show_window: bool = False):
        png_path = Path(png_path)
        png_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_window(window_name="SfM (Open3D)", width=width, height=height, visible=True)
        self._fit_view(view)

        self._vis.poll_events()
        self._vis.update_renderer()
        self._vis.capture_screen_image(str(png_path), do_render=True)

        if show_window:
            self._vis.run()

        self._vis.destroy_window()
        self._win_created = False
        print(f"[Open3D] Saved PNG to {png_path}")

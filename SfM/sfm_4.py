"""
Structure from Motion pipeline (custom PnP + RANSAC + Triangulation).
OpenCV is used for Fundamental matrix only.
"""
import numpy as np
import cv2 as cv
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from triangulation_custom import linear_triangulate_points
from pnp_custom import pnp_ransac_solve


class SfMPipeline:
    def __init__(self, confidence: float = 0.99, min_matches: int = 50):
        self.confidence = confidence
        self.min_matches = min_matches
        self.K = None
        self.K_inv = None
        self.cameras: List[Tuple[np.ndarray, np.ndarray]] = []
        self.points_3d: np.ndarray = np.zeros((0, 3), dtype=np.float32)
        self.point_tracks: Dict[Tuple[int, int], int] = {}
        self.observations: List[Tuple[int, int, float, float]] = []
        self.view_id_map: Dict[int, int] = {}
        self.inv_view_id_map: Dict[int, int] = {}

    def set_intrinsics(self, K: np.ndarray):
        self.K = K.astype(np.float64).copy()
        self.K_inv = np.linalg.inv(self.K)
        self.fx, self.fy = float(K[0, 0]), float(K[1, 1])
        self.cx, self.cy = float(K[0, 2]), float(K[1, 2])
        if not hasattr(self, 'pending_pnp'):
            self.pending_pnp = {}
        self.pending_pnp.clear()

    def _ensure_cam_slot_for_ext(self, vid_ext: int) -> int:
        if vid_ext in self.view_id_map:
            return self.view_id_map[vid_ext]
        vid_int = len(self.cameras)
        self.view_id_map[vid_ext] = vid_int
        self.inv_view_id_map[vid_int] = vid_ext
        return vid_int

    def build_index(self, matches_data: list):
        self.matches_by_view = defaultdict(list)
        for m in matches_data:
            v1, v2 = int(m['view1_id']), int(m['view2_id'])
            self.matches_by_view[v1].append(m)
            self.matches_by_view[v2].append(m)

    # ---------- initialization (two-view) ----------
    def initialize_from_two_view(self, matches: Dict) -> bool:
        v0_ext = int(matches['view1_id'])
        v1_ext = int(matches['view2_id'])
        pts1 = matches['pts1'].astype(np.float32)
        pts2 = matches['pts2'].astype(np.float32)
        if pts1.shape[0] < 50:
            return False
        # Fundamental via OpenCV (allowed)
        F, Fmask = cv.findFundamentalMat(pts1, pts2, method=cv.FM_RANSAC, ransacReprojThreshold=2.0, confidence=0.995)
        if F is None or Fmask is None:
            return False
        Fmask = Fmask.ravel().astype(bool)
        if Fmask.sum() < 30:
            return False
        E = self.K.T @ F @ self.K
        # Enforce rank-2
        U, S, Vt = np.linalg.svd(E)
        S[2] = 0.0
        E = U @ np.diag(S) @ Vt
        # Decompose E (four solutions) and pick with cheirality
        R, t, inl_idx = self._recover_pose_from_E(E, pts1[Fmask], pts2[Fmask])
        if R is None:
            return False
        # map back to original keypoint indices using 'pairs'
        pairs = matches.get('pairs', None)
        if pairs is None:
            return False
        idx_all = np.flatnonzero(Fmask)
        inl_global = idx_all[inl_idx]
        kp1_idx = pairs[inl_global, 0].astype(np.int32)
        kp2_idx = pairs[inl_global, 1].astype(np.int32)

        i0 = self._ensure_cam_slot_for_ext(v0_ext)
        i1 = self._ensure_cam_slot_for_ext(v1_ext)
        while len(self.cameras) <= max(i0, i1):
            self.cameras.append((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
        self.cameras[i0] = (np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32))
        self.cameras[i1] = (R.astype(np.float32), t.astype(np.float32))

        # seed triangulation (custom) with correct kp indices
        kps1 = matches.get('kps1_all', matches['pts1'])
        kps2 = matches.get('kps2_all', matches['pts2'])
        pts1_px = kps1[kp1_idx][:, :2].astype(np.float64)
        pts2_px = kps2[kp2_idx][:, :2].astype(np.float64)

        P0 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.K @ np.hstack([R, t])
        X = linear_triangulate_points(P0, P1, pts1_px, pts2_px)
        # cheirality & reprojection filter
        keep = self._triage_triangulated(R, t, X, pts1_px, pts2_px, reproj_thresh=4.0)
        if keep.sum() == 0:
            return False
        X_keep = X[keep].astype(np.float32)
        kp1_keep = kp1_idx[keep]
        kp2_keep = kp2_idx[keep]
        base = len(self.points_3d)
        self.points_3d = np.vstack([self.points_3d, X_keep])
        for j, Xj in enumerate(X_keep):
            pid = base + j
            x1, y1 = pts1_px[keep][j]
            x2, y2 = pts2_px[keep][j]
            self.point_tracks[(i0, int(kp1_keep[j]))] = pid
            self.point_tracks[(i1, int(kp2_keep[j]))] = pid
            self.observations.append((i0, pid, float(x1), float(y1)))
            self.observations.append((i1, pid, float(x2), float(y2)))
        return True

    def _recover_pose_from_E(self, E: np.ndarray, pts1: np.ndarray, pts2: np.ndarray):
        # Normalize by K
        K = self.K
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p1 = np.column_stack([(pts1[:, 0] - cx) / fx, (pts1[:, 1] - cy) / fy])
        p2 = np.column_stack([(pts2[:, 0] - cx) / fx, (pts2[:, 1] - cy) / fy])
        U, S, Vt = np.linalg.svd(E)
        if np.linalg.det(U @ Vt) < 0:
            Vt = -Vt
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R_candidates = [U @ W @ Vt, U @ W.T @ Vt]
        t_candidates = [U[:, 2], -U[:, 2]]
        best = (None, None, None)
        best_count = -1
        for R in R_candidates:
            if np.linalg.det(R) < 0:
                R = -R
            for t in t_candidates:
                count, mask = self._cheirality_count(R, t.reshape(3, 1), p1, p2)
                if count > best_count:
                    best_count = count
                    best = (R, t.reshape(3, 1), mask)
        R, t, inl_mask = best
        if R is None:
            return None, None, None
        return R, t, np.flatnonzero(inl_mask)

    def _cheirality_count(self, R, t, p1, p2):
        P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = np.hstack([R, t])
        # back to pixels using K for triangulation consistency
        pts = linear_triangulate_points(self.K @ P0, self.K @ P1,
                                        self._to_px(p1), self._to_px(p2))
        Z0 = pts[:, 2]
        Z1 = (R @ pts.T + t).T[:, 2]
        mask = (Z0 > 0) & (Z1 > 0)
        return int(mask.sum()), mask

    def _to_px(self, pn):
        fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
        uv = np.empty_like(pn)
        uv[:, 0] = pn[:, 0] * fx + cx
        uv[:, 1] = pn[:, 1] * fy + cy
        return uv

    def _triage_triangulated(self, R, t, X, pts1_px, pts2_px, reproj_thresh: float):
        K = self.K
        def project(R, t, X3):
            Xc = (R @ X3.T + t).T
            uv = (Xc[:, :2] / (Xc[:, 2:3] + 1e-12))
            return (K[:2, :2] @ uv.T + K[:2, 2:3]).T
        P0_px = project(np.eye(3), np.zeros((3, 1)), X)
        P1_px = project(R, t, X)
        err0 = np.linalg.norm(P0_px - pts1_px, axis=1)
        err1 = np.linalg.norm(P1_px - pts2_px, axis=1)
        Z0 = X[:, 2]
        Z1 = (R @ X.T + t).T[:, 2]
        mask = (err0 < reproj_thresh) & (err1 < reproj_thresh) & (Z0 > 0) & (Z1 > 0)
        return mask

    # ---------- Next image selection (reuse from sfm_2 logic) ----------
    def _collect_pnp_candidates_from_pairse(self, m:dict, cand_ext: int, reg_ext: int):
        v1, v2 = int(m['view1_id']), int(m['view2_id'])
        if v1 == cand_ext and v2 == reg_ext:
            kps_cand = m.get('kps1_all', m['pts1']); kps_reg = m.get('kps2_all', m['pts2']); cand_side = 1
        elif v2 == cand_ext and v1 == reg_ext:
            kps_cand = m.get('kps2_all', m['pts2']); kps_reg = m.get('kps1_all', m['pts1']); cand_side = 2
        else:
            return []
        pairs = m.get('pairs', None)
        if pairs is None:
            idx1 = m.get('idx1'); idx2 = m.get('idx2')
            if idx1 is None or idx2 is None: return []
            pairs = np.stack([idx1, idx2], axis=1)
        reg_int = self.view_id_map[reg_ext]
        out = []
        for (i1, i2) in pairs:
            kp_reg = int(i2) if cand_side == 1 else int(i1)
            kp_cand = int(i1) if cand_side == 1 else int(i2)
            pid = self.point_tracks.get((reg_int, kp_reg), None)
            if pid is None: continue
            out.append((pid, kp_cand, kps_cand[kp_cand][:2].astype(float)))
        return out

    def select_next_best_image(self, matches_data:list, min_common_points:int=40, min_spread_px: float=50.0, exclude_ids: set | None = None):
        if exclude_ids is None: exclude_ids = set()
        registered_ext_ids = set(self.view_id_map.keys())
        per_cand = defaultdict(lambda: {"pid_set": set(), "kp2d_map": {}, "pid_map": {}})
        for m in matches_data:
            v1 = int(m['view1_id']); v2 = int(m['view2_id'])
            for cand_ext, reg_ext in ((v1, v2), (v2, v1)):
                if cand_ext in exclude_ids or cand_ext in registered_ext_ids or reg_ext not in registered_ext_ids:
                    continue
                triples = self._collect_pnp_candidates_from_pairse(m, cand_ext, reg_ext)
                if not triples: continue
                D = per_cand[cand_ext]
                for pid, kp_idx_cand, xy_cand in triples:
                    if pid in D["pid_map"]: continue
                    D["pid_map"][pid] = kp_idx_cand
                    D["pid_set"].add(pid)
                    D["kp2d_map"][kp_idx_cand] = xy_cand
        if not per_cand: return None, {}
        best_ext, best_score = None, -1
        score_table = {}
        for cand_ext, D in per_cand.items():
            pid_list = list(D["pid_set"])
            score = len(pid_list)
            if score < min_common_points: continue
            pts2d = np.array(list(D["kp2d_map"].values()), dtype=np.float32)
            if pts2d.shape[0] == 0: continue
            xs, ys = pts2d[:, 0], pts2d[:, 1]
            spread = max(xs.max() - xs.min(), ys.max() - ys.min())
            if spread < min_spread_px: continue
            score_table[cand_ext] = {"n_2d3d": score, "spread_px": float(spread)}
            if score > best_score: best_score, best_ext = score, cand_ext
        if best_ext is None: return None, score_table
        D = per_cand[best_ext]
        pid_list = sorted(list(D["pid_set"]))
        kp_list = [D["pid_map"][pid] for pid in pid_list]
        pts3d = self.points_3d[np.array(pid_list, dtype=int)]
        pts2d = np.array([D["kp2d_map"][kp] for kp in kp_list], dtype=np.float32)
        if not hasattr(self, 'pending_pnp'): self.pending_pnp = {}
        self.pending_pnp[best_ext] = {"pid_list": pid_list, "kp_list": kp_list, "pts3d": pts3d.astype(np.float32), "pts2d": pts2d.astype(np.float32)}
        return best_ext, score_table

    # ---------- PnP registration (custom RANSAC) ----------
    def build_pnp_correspondences(self, sfm, cand_ext, matches_data):
        registered_ext = set(sfm.view_id_map.keys())
        if not registered_ext: return None
        pts3d_list, pts2d_list, pid_list, kp_list = [], [], [], []
        for m in matches_data:
            v1, v2 = int(m['view1_id']), int(m['view2_id'])
            if v1 == cand_ext and v2 in registered_ext:
                int_id = sfm.view_id_map[v2]; kps_cand = m['kps1_all']; pairs = m['pairs']
                for cand_kp, reg_kp in pairs:
                    pid = sfm.point_tracks.get((int_id, int(reg_kp)), None)
                    if pid is None: continue
                    pts3d_list.append(sfm.points_3d[int(pid)])
                    pts2d_list.append(kps_cand[int(cand_kp)])
                    pid_list.append(int(pid)); kp_list.append(int(cand_kp))
            elif v2 == cand_ext and v1 in registered_ext:
                int_id = sfm.view_id_map[v1]; kps_cand = m['kps2_all']; pairs = m['pairs']
                for reg_kp, cand_kp in pairs:
                    pid = sfm.point_tracks.get((int_id, int(reg_kp)), None)
                    if pid is None: continue
                    pts3d_list.append(sfm.points_3d[int(pid)])
                    pts2d_list.append(kps_cand[int(cand_kp)])
                    pid_list.append(int(pid)); kp_list.append(int(cand_kp))
        if not pts3d_list: return None
        first_by_pid = {}
        for i, pid in enumerate(pid_list):
            if pid not in first_by_pid: first_by_pid[pid] = i
        keep_idx = list(first_by_pid.values())
        pts3d = np.asarray([pts3d_list[i] for i in keep_idx], dtype=np.float32)
        pts2d = np.asarray([pts2d_list[i] for i in keep_idx], dtype=np.float32)
        pid_list = [pid_list[i] for i in keep_idx]; kp_list = [kp_list[i] for i in keep_idx]
        return pts3d, pts2d, pid_list, kp_list

    def register_image_via_pnp(self, cand_ext: int, use_upnp: bool = False,
                               ransac_reproj_err: float = 3.0,
                               ransac_conf: float = 0.999,
                               ransac_iters: int = 3000,
                               min_inliers: int = 30) -> bool:
        if not hasattr(self, 'pending_pnp') or cand_ext not in self.pending_pnp:
            return False
        if cand_ext in getattr(self, 'view_id_map', {}):
            return False
        pkg = self.pending_pnp[cand_ext]
        pts3d = np.asarray(pkg["pts3d"], dtype=np.float64)
        pts2d = np.asarray(pkg["pts2d"], dtype=np.float64)
        pid_list = list(pkg["pid_list"]); kp_list = list(pkg["kp_list"])
        if pts3d.shape[0] < min_inliers:
            return False
        # custom PnP with RANSAC
        ok, R, t, inlier_mask = pnp_ransac_solve(pts3d, pts2d, self.K, reproj_thresh=ransac_reproj_err,
                                                 max_iters=ransac_iters, confidence=ransac_conf)
        if not ok or inlier_mask.sum() < min_inliers:
            return False
        i_new = self._ensure_cam_slot_for_ext(cand_ext)
        while len(self.cameras) <= i_new:
            self.cameras.append((np.eye(3, dtype=np.float32), np.zeros((3, 1), dtype=np.float32)))
        self.cameras[i_new] = (R.astype(np.float32), t.astype(np.float32))
        sel = np.flatnonzero(inlier_mask)
        for idx in sel:
            pid = int(pid_list[idx]); kpi = int(kp_list[idx]); xy = pts2d[idx]
            self.point_tracks[(i_new, kpi)] = pid
            self.observations.append((i_new, pid, float(xy[0]), float(xy[1])))
        del self.pending_pnp[cand_ext]
        return True

    # ---------- triangulation for new matches ----------
    def triangulate_new_matches_from_view(self, new_ext: int, matches_data: list,
                                          reproj_thresh: float = 5.0) -> tuple[int, int]:
        if new_ext not in self.view_id_map:
            return 0, 0
        K = self.K
        new_int = self.view_id_map[new_ext]
        (R_new, t_new) = self.cameras[new_int]
        n_added_pts = 0
        n_attached_obs = 0

        def project(R, t, X3):
            Xc = (R @ X3.T + t).T
            uv = (Xc[:, :2] / (Xc[:, 2:3] + 1e-12))
            return (K[:2, :2] @ uv.T + K[:2, 2:3]).T

        registered_ext_ids = set(self.view_id_map.keys()) - {new_ext}
        for m in self.matches_by_view.get(new_ext, []):
            v1 = int(m['view1_id']); v2 = int(m['view2_id'])
            if not ((v1 in registered_ext_ids and v2 == new_ext) or (v2 in registered_ext_ids and v1 == new_ext)):
                continue
            if v1 == new_ext:
                reg_ext = v2; kps_new = m.get('kps1_all', m['pts1']); kps_reg = m.get('kps2_all', m['pts2']); new_is_left = True
            else:
                reg_ext = v1; kps_new = m.get('kps2_all', m['pts2']); kps_reg = m.get('kps1_all', m['pts1']); new_is_left = False
            reg_int = self.view_id_map.get(reg_ext, None)
            if reg_int is None: continue
            (R_reg, t_reg) = self.cameras[reg_int]

            pairs = m.get('pairs', None)
            if pairs is None:
                idx1 = m.get('idx1'); idx2 = m.get('idx2')
                if idx1 is None or idx2 is None: continue
                pairs = np.stack([idx1, idx2], axis=1)

            tri_pts_new_2d, tri_pts_reg_2d, tri_kp_new, tri_kp_reg = [], [], [], []
            attach_pid, attach_kp_new, attach_xy_new = [], [], []
            for (i1, i2) in pairs:
                i1 = int(i1); i2 = int(i2)
                kp_new = i1 if new_is_left else i2
                kp_reg = i2 if new_is_left else i1
                pid_reg = self.point_tracks.get((reg_int, kp_reg), None)
                pid_new = self.point_tracks.get((new_int, kp_new), None)
                xy_new = kps_new[kp_new][:2].astype(np.float64)
                xy_reg = kps_reg[kp_reg][:2].astype(np.float64)
                if (pid_reg is not None) and (pid_new is None):
                    attach_pid.append(int(pid_reg)); attach_kp_new.append(int(kp_new)); attach_xy_new.append(xy_new); continue
                if (pid_reg is None) and (pid_new is not None):
                    # back attach to reg if consistent
                    if 'attach_back_pid' not in locals():
                        attach_back_pid, attach_back_kp_reg, attach_back_xy_reg = [], [], []
                    attach_back_pid.append(int(pid_new)); attach_back_kp_reg.append(int(kp_reg)); attach_back_xy_reg.append(xy_reg); continue
                if (pid_reg is not None) and (pid_new is not None):
                    continue
                tri_pts_new_2d.append(xy_new); tri_pts_reg_2d.append(xy_reg)
                tri_kp_new.append(int(kp_new)); tri_kp_reg.append(int(kp_reg))

            if attach_pid:
                pts3d_attach = self.points_3d[np.array(attach_pid, dtype=int)]
                pts2d_attach = np.array(attach_xy_new, dtype=np.float64)
                x_hat = project(R_new, t_new, pts3d_attach)
                err = np.linalg.norm(x_hat - pts2d_attach, axis=1)
                keep = (err < reproj_thresh)
                for pid, kp, xy, k in zip(attach_pid, attach_kp_new, attach_xy_new, keep):
                    if not k: continue
                    self.point_tracks[(new_int, int(kp))] = int(pid)
                    self.observations.append((new_int, int(pid), float(xy[0]), float(xy[1])))
                    n_attached_obs += 1

            if 'attach_back_pid' in locals() and len(attach_back_pid) > 0:
                pts3d_back  = self.points_3d[np.array(attach_back_pid, dtype=int)]
                pts2d_back  = np.array(attach_back_xy_reg, dtype=np.float64)
                x_hat_reg   = project(R_reg, t_reg, pts3d_back)
                err_back    = np.linalg.norm(x_hat_reg - pts2d_back, axis=1)
                keep_b      = (err_back < reproj_thresh)
                for pid, kr, xy, k in zip(attach_back_pid, attach_back_kp_reg, attach_back_xy_reg, keep_b):
                    if not k: continue
                    self.point_tracks[(reg_int, int(kr))] = int(pid)
                    self.observations.append((reg_int, int(pid), float(xy[0]), float(xy[1])))
                    n_attached_obs += 1
                del attach_back_pid, attach_back_kp_reg, attach_back_xy_reg

            if len(tri_pts_new_2d) >= 2:
                pts1 = np.array(tri_pts_reg_2d, dtype=np.float64)
                pts2 = np.array(tri_pts_new_2d, dtype=np.float64)
                P_reg = self.K @ np.hstack([R_reg, t_reg])
                P_new = self.K @ np.hstack([R_new, t_new])
                X = linear_triangulate_points(P_reg, P_new, pts1, pts2)
                Z_reg = (R_reg @ X.T + t_reg).T[:, 2]
                Z_new = (R_new @ X.T + t_new).T[:, 2]
                x_reg_hat = project(R_reg, t_reg, X)
                x_new_hat = project(R_new, t_new, X)
                err_reg = np.linalg.norm(x_reg_hat - pts1, axis=1)
                err_new = np.linalg.norm(x_new_hat - pts2, axis=1)
                keep = (Z_reg > 0) & (Z_new > 0) & (err_reg < reproj_thresh) & (err_new < reproj_thresh)
                if keep.any():
                    X_keep = X[keep]
                    kp_new_keep = np.array(tri_kp_new, dtype=int)[keep]
                    kp_reg_keep = np.array(tri_kp_reg, dtype=int)[keep]
                    base_pid = len(self.points_3d)
                    self.points_3d = np.vstack([self.points_3d, X_keep.astype(np.float32)])
                    for j, Xj in enumerate(X_keep):
                        pid = base_pid + j
                        xn = kps_new[kp_new_keep[j]][:2]
                        xr = kps_reg[kp_reg_keep[j]][:2]
                        self.point_tracks[(new_int, int(kp_new_keep[j]))] = pid
                        self.observations.append((new_int, pid, float(xn[0]), float(xn[1])))
                        self.point_tracks[(reg_int, int(kp_reg_keep[j]))] = pid
                        self.observations.append((reg_int, pid, float(xr[0]), float(xr[1])))
                        n_added_pts += 1
        return n_added_pts, n_attached_obs



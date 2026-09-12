import numpy as np

# --- Camera Center :  C = -R^T t ---
def camera_center(R: np.ndarray, t: np.ndarray):
    t = t.reshape(3,1)
    return (-R.T @ t).reshape(-1)

# --- print (3,3) matrix better ---
def format_mat3(M: np.ndarray, w=11, p=6):
    rows = []
    for r in range(3):
        rows.append("  " + " ".join(f"{M[r,c]:{w}.{p}f}" for c in range(3)))
    return "\n".join(rows)

# print (3,4) [R|t] matrix
def format_Rt_block(R: np.ndarray, t: np.ndarray, w=11, p=6):
    R = np.asarray(R).reshape(3,3)
    t = np.asarray(t).reshape(3,1)
    Rt = np.hstack([R, t])  # (3,4)
    lines = []
    for r in range(3):
        left  = " ".join(f"{Rt[r,c]:{w}.{p}f}" for c in range(3))
        right = f"{Rt[r,3]:{w}.{p}f}"
        lines.append(f"  {left}  | {right}")
    return "\n".join(lines)


# --- main : print results pretty.... ---
def print_calibration_summary(K, R_list, t_list, title="Zhang's Method",
                              show_camera_center=True, max_views=None):
    fx, fy, cx, cy, skew = float(K[0,0]), float(K[1,1]), float(K[0,2]), float(K[1,2]), float(K[0,1])

    line = "=" * 72
    sub  = "-" * 72
    print(line)
    print(f"{title}".center(72))
    print(line)

    # Intrinsics
    print("INTRINSICS (K)")
    print(sub)
    print(f"fx = {fx:.6f}   fy = {fy:.6f}   cx = {cx:.6f}   cy = {cy:.6f}   skew = {skew:.6f}\n")
    print("K =")
    print(format_mat3(K, w=12, p=6))
    print()

    # Extrinsics header
    print("EXTRINSICS  [R | t]  (per view, world -> camera)")
    print(sub)
    for i, (R, t) in enumerate(zip(R_list, t_list)):
        print(f"[View {i:02d}]   X_c = R_i * X_w + t_i")
        print(format_Rt_block(R, t, w=12, p=6))
        print("-" * 72)
    print(line)

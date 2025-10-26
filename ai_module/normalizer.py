import numpy as np
from scipy.signal import savgol_filter

L_HIP, R_HIP = 23, 24
L_SHOULDER, R_SHOULDER = 11, 12

def _center_scale_frame(frame):
    pts = frame[:, :3].copy()
    vis = frame[:, 3]
    if vis[L_HIP] > 0.5 and vis[R_HIP] > 0.5:
        pelvis = (pts[L_HIP] + pts[R_HIP]) / 2.0
        pts -= pelvis
    ref = []
    if vis[L_SHOULDER] > 0.5 and vis[R_SHOULDER] > 0.5:
        ref.append(np.linalg.norm(pts[L_SHOULDER] - pts[R_SHOULDER]))
    if vis[L_HIP] > 0.5 and vis[R_HIP] > 0.5:
        ref.append(np.linalg.norm(pts[L_HIP] - pts[R_HIP]))
    scale = np.median(ref) if ref else 1.0
    pts /= max(scale, 1e-6)
    out = np.zeros_like(frame, dtype=np.float32)
    out[:, :3] = pts
    out[:, 3] = vis
    return out

def ema_smooth(seq, alpha=0.3):
    seq = np.array(seq, dtype=np.float32)
    sm = seq.copy()
    for t in range(1, len(seq)):
        sm[t, :, :3] = alpha*seq[t, :, :3] + (1-alpha)*sm[t-1, :, :3]
    return sm

def normalize_sequence(frames, smoothing="ema"):
    if not frames:
        return []
    normed = [_center_scale_frame(f) for f in frames]
    return ema_smooth(normed) if smoothing == "ema" else normed

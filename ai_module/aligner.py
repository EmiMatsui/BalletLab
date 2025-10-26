# ai_module/aligner.py
import numpy as np

def _frame_feat(frame):
    """(33,4)->重み付き(x,y,z)フラット"""
    xyz = frame[:, :3]
    vis = frame[:, 3:4]
    return (xyz * np.clip(vis, 0.0, 1.0)).reshape(-1)

def _l2n_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    return X / n

def ratio_align(ideal_seq, user_seq):
    """長さ比で対応付け（超高速）"""
    I, U = len(ideal_seq), len(user_seq)
    if U <= 1 or I <= 1:
        return np.zeros(U, dtype=np.int32)
    return np.round(np.linspace(0, I-1, U)).astype(np.int32)

def simple_dtw(ideal_seq, user_seq):
    """無制約DTW（コサイン距離）"""
    I, U = len(ideal_seq), len(user_seq)
    if I == 0 or U == 0:
        return np.zeros(U, dtype=np.int32)

    Ifeat = _l2n_rows(np.stack([_frame_feat(f) for f in ideal_seq]))  # (I,D)
    Ufeat = _l2n_rows(np.stack([_frame_feat(f) for f in user_seq]))   # (U,D)
    C = 1.0 - Ifeat @ Ufeat.T  # (I,U)

    inf = np.float32(1e9)
    dp  = np.full((I, U), inf, dtype=np.float32)
    ptr = np.full((I, U, 2), -1, dtype=np.int16)

    dp[0, 0] = C[0, 0]
    for i in range(1, I):
        dp[i, 0] = C[i, 0] + dp[i-1, 0]
        ptr[i, 0] = [i-1, 0]
    for j in range(1, U):
        dp[0, j] = C[0, j] + dp[0, j-1]
        ptr[0, j] = [0, j-1]

    for i in range(1, I):
        for j in range(1, U):
            # 3方向から最小
            choices = [(dp[i-1,j-1], (i-1,j-1)),
                       (dp[i-1,j],   (i-1,j)),
                       (dp[i,  j-1], (i,  j-1))]
            best = min(choices, key=lambda x: x[0])
            dp[i, j] = C[i, j] + best[0]
            ptr[i, j] = best[1]

    # 復元して user_idx -> ideal_idx
    i, j = I-1, U-1
    path = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        pi, pj = ptr[i, j]
        if pi < 0 or pj < 0:
            break
        i, j = pi, pj
    path.reverse()

    mapping = np.zeros(U, dtype=np.int32)
    for ii, jj in path:
        mapping[jj] = ii
    for jj in range(1, U):
        if mapping[jj] == 0:
            mapping[jj] = mapping[jj-1]
    return mapping

def simple_dtw_band(ideal_seq, user_seq, band_ratio=0.08):
    """Sakoe–Chibaバンド付きDTW（コサイン距離）"""
    I, U = len(ideal_seq), len(user_seq)
    if I == 0 or U == 0:
        return np.zeros(U, dtype=np.int32)

    Ifeat = _l2n_rows(np.stack([_frame_feat(f) for f in ideal_seq]))
    Ufeat = _l2n_rows(np.stack([_frame_feat(f) for f in user_seq]))
    W = max(1, int(max(I, U) * band_ratio))

    inf = np.float32(1e9)
    dp  = np.full((I, U), inf, dtype=np.float32)
    ptr = np.full((I, U, 2), -1, dtype=np.int16)

    dp[0, 0] = 1.0 - float(Ifeat[0] @ Ufeat[0])
    for j in range(1, min(U, W+1)):
        c = 1.0 - float(Ifeat[0] @ Ufeat[j])
        dp[0, j] = dp[0, j-1] + c
        ptr[0, j] = [0, j-1]

    for i in range(1, I):
        j_start = max(0, i - W)
        j_end   = min(U - 1, i + W)
        # 左端
        if j_start <= j_end:
            c = 1.0 - float(Ifeat[i] @ Ufeat[j_start])
            if dp[i-1, j_start] < inf:
                dp[i, j_start] = dp[i-1, j_start] + c
                ptr[i, j_start] = [i-1, j_start]
        # 本体
        for j in range(j_start+1, j_end+1):
            c = 1.0 - float(Ifeat[i] @ Ufeat[j])
            candidates = [
                (dp[i-1, j-1], (i-1, j-1)),
                (dp[i-1, j],   (i-1, j)),
                (dp[i,   j-1], (i,   j-1)),
            ]
            best = min(candidates, key=lambda x: x[0])
            dp[i, j] = best[0] + c
            ptr[i, j] = best[1]

    # 末端をバンド内から選ぶ
    j_candidates = [j for j in range(max(0, I-1-W), min(U-1, I-1+W)+1) if dp[I-1, j] < inf]
    if not j_candidates:
        return ratio_align(ideal_seq, user_seq)

    j = min(j_candidates, key=lambda jj: dp[I-1, jj])
    i = I - 1
    path = []
    while i >= 0 and j >= 0 and ptr[i, j, 0] != -1:
        path.append((i, j))
        i, j = int(ptr[i, j, 0]), int(ptr[i, j, 1])
    path.append((0, 0))
    path.reverse()

    mapping = np.zeros(U, dtype=np.int32)
    for ii, jj in path:
        mapping[jj] = ii
    for jj in range(1, U):
        if mapping[jj] == 0:
            mapping[jj] = mapping[jj-1]
    return mapping

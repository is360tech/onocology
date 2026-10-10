"""Handcrafted image features (first-order, GLCM texture, LBP, gradient, intensity histogram). Pure functions, no file I/O.
All texture features are computed inside a mask; quantisation uses robust percentiles of the masked pixels."""
import numpy as np
from scipy import stats as st, ndimage as ndi
from skimage.feature import graycomatrix, local_binary_pattern

LEVELS = 32
GLCM_NAMES = ["contrast", "dissimilarity", "homogeneity", "asm", "correlation", "entropy"]

def quantise(x, mask, levels=LEVELS):
    """uint8 image with 1..levels-1 inside the mask and 0 outside (level 0 is dropped from the GLCM)."""
    v = x[mask].astype(np.float64)
    lo, hi = np.percentile(v, (1, 99)); q = np.clip((x.astype(np.float64) - lo) / max(hi - lo, 1e-6), 0, 1)
    q = 1 + np.floor(q * (levels - 2)).astype(np.int32)
    return np.where(mask, q, 0).astype(np.uint8)

def first_order(x, mask, prefix="fo"):
    v = x[mask].astype(np.float64)
    if v.size < 20: return {}
    h, _ = np.histogram(v, bins=32, range=(0, 255)); p = h[h > 0] / h.sum()
    q = np.percentile(v, [10, 25, 50, 75, 90])
    f = dict(mean=v.mean(), std=v.std(), skew=float(st.skew(v)), kurt=float(st.kurtosis(v)), entropy=float(-(p * np.log2(p)).sum()),
             p10=q[0], p25=q[1], p50=q[2], p75=q[3], p90=q[4], iqr=q[3] - q[1], mad=float(np.median(np.abs(v - q[2]))))
    return {f"{prefix}_{k}": float(val) for k, val in f.items()}

def glcm_features(q, dists=(1, 2, 4), angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4), levels=LEVELS, prefix="glcm"):
    G = graycomatrix(q, list(dists), list(angles), levels=levels, symmetric=True, normed=False)[1:, 1:]   # drop background level 0
    i, j = np.meshgrid(np.arange(1, levels), np.arange(1, levels), indexing="ij"); out = {}
    for di, d in enumerate(dists):
        rows = []
        for ai in range(len(angles)):
            P = G[:, :, di, ai].astype(np.float64); s = P.sum()
            if s == 0: continue
            P = P / s; mi, mj = (i * P).sum(), (j * P).sum(); si, sj = np.sqrt(((i - mi) ** 2 * P).sum()), np.sqrt(((j - mj) ** 2 * P).sum())
            nz = P[P > 0]
            rows.append([((i - j) ** 2 * P).sum(), (np.abs(i - j) * P).sum(), (P / (1 + (i - j) ** 2)).sum(), (P ** 2).sum(),
                         ((i - mi) * (j - mj) * P).sum() / (si * sj + 1e-9), -(nz * np.log2(nz)).sum()])
        if rows:
            for n, val in zip(GLCM_NAMES, np.mean(rows, 0)): out[f"{prefix}_d{d}_{n}"] = float(val)
    return out

def lbp_features(x, mask, prefix="lbp"):
    out = {}; er = ndi.binary_erosion(mask, iterations=3)
    if er.sum() < 20: er = mask
    for P, R in ((8, 1), (16, 2)):
        code = local_binary_pattern(x, P, R, method="uniform"); h = np.bincount(code[er].astype(int), minlength=P + 2)[: P + 2].astype(float)
        h = h / max(h.sum(), 1)
        for k, val in enumerate(h): out[f"{prefix}{P}_{k}"] = float(val)
    return out

def gradient_features(x, mask, prefix="grad"):
    xf = x.astype(np.float32); gx, gy = ndi.sobel(xf, 1), ndi.sobel(xf, 0); g = np.hypot(gx, gy)[mask]; lap = ndi.laplace(xf)[mask]
    return {f"{prefix}_mean": float(g.mean()), f"{prefix}_std": float(g.std()), f"{prefix}_lapvar": float(lap.var())} if g.size else {}

def hist_features(x, mask, bins=16, prefix="hist"):
    h, _ = np.histogram(x[mask], bins=bins, range=(0, 255)); h = h / max(h.sum(), 1)
    return {f"{prefix}_{k}": float(val) for k, val in enumerate(h)}

def extract(x, mask, with_hist=False):
    """x: uint8 2-D array, mask: bool 2-D array (same shape). Returns a flat feature dict."""
    if mask.sum() < 100: return {}
    f = {}; f.update(first_order(x, mask)); f.update(gradient_features(x, mask)); f.update(glcm_features(quantise(x, mask))); f.update(lbp_features(x, mask))
    if with_hist: f.update(hist_features(x, mask))
    return f

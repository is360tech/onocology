"""Mammogram preprocessing algorithms (pure functions, no file I/O). Canonical frame: chest wall on the LEFT.
Every geometric step is recorded in a Transform so lesion coordinates and predictions can be mapped between frames."""
from dataclasses import dataclass, field, asdict
import numpy as np
from scipy import ndimage as ndi, stats
from skimage import filters, measure, morphology, exposure, transform as sktf

# ------------------------------------------------------------------ geometry record
@dataclass
class Transform:
    """original std frame (row 0 = top) -> model-input frame. Order: flip, crop, pad, scale."""
    width: int = 1024
    height: int = 1024
    flipped: bool = False
    crop: tuple = (0, 0, 1024, 1024)      # x0, y0, x1, y1 (in the flipped frame)
    pad: tuple = (0, 0)                   # pad_x, pad_y added left/top after cropping
    scale: float = 1.0                    # resize factor after padding
    out_size: int = 512

    def map_point(self, x, y):
        if self.flipped: x = self.width - 1 - x
        x, y = x - self.crop[0] + self.pad[0], y - self.crop[1] + self.pad[1]
        return x * self.scale, y * self.scale

    def map_radius(self, r): return r * self.scale

    def inverse_point(self, xm, ym):
        x, y = xm / self.scale - self.pad[0] + self.crop[0], ym / self.scale - self.pad[1] + self.crop[1]
        if self.flipped: x = self.width - 1 - x
        return x, y

    def to_dict(self): return {**asdict(self), "crop": list(self.crop), "pad": list(self.pad)}
    @staticmethod
    def from_dict(d): d = dict(d); d["crop"] = tuple(d["crop"]); d["pad"] = tuple(d["pad"]); return Transform(**d)

# ------------------------------------------------------------------ breast segmentation
def breast_mask(img, sigma=3.0, min_area_frac=0.02):
    """Multi-Otsu (3 classes) lowest threshold separates background from tissue; keep the largest connected component,
    which also removes burned-in labels, tape and markers that are not attached to the breast."""
    f = filters.gaussian(img.astype(np.float32), sigma=sigma, preserve_range=True)
    try: thr = filters.threshold_multiotsu(f, classes=3)[0]
    except ValueError: thr = filters.threshold_otsu(f)
    m = f > thr
    lab = measure.label(m, connectivity=2)
    if lab.max() == 0: return np.zeros_like(m), float(thr)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    if sizes.max() < min_area_frac * m.size: return np.zeros_like(m), float(thr)
    m = lab == sizes.argmax()
    m = ndi.binary_fill_holes(m)
    m = ndi.binary_opening(m, structure=morphology.disk(5))
    lab = measure.label(m, connectivity=2)
    if lab.max() > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0; m = lab == sizes.argmax()
    return m, float(thr)

def chest_wall_side(mask):
    """The chest-wall side of a mammogram is cut straight (vertical edge); the nipple side is curved.
    Returns ('left'|'right', confidence in [0,1])."""
    rows = np.where(mask.any(1))[0]
    if len(rows) < 10: return "left", 0.0
    first = np.array([np.argmax(mask[r]) for r in rows]); last = np.array([mask.shape[1] - 1 - np.argmax(mask[r][::-1]) for r in rows])
    sl = float(np.mean(first <= first.min() + 6)); sr = float(np.mean(last >= last.max() - 6))
    return ("left", sl - sr) if sl >= sr else ("right", sr - sl)

def bbox(mask, margin=8):
    ys, xs = np.where(mask)
    if len(xs) == 0: return (0, 0, mask.shape[1], mask.shape[0])
    return (int(max(xs.min() - margin, 0)), int(max(ys.min() - margin, 0)), int(min(xs.max() + 1 + margin, mask.shape[1])), int(min(ys.max() + 1 + margin, mask.shape[0])))

# ------------------------------------------------------------------ pectoral muscle (MLO, chest wall on the left)
def pectoral_mask(img, mask, fx=0.65, fy=0.75, min_frac=0.01, max_frac=0.40):
    """Baseline (non-learned) pectoral-muscle removal for MLO views with the chest wall on the left.
    1) top-left region of the breast box; 2) upper multi-Otsu class = muscle / dense tissue candidates; 3) component reaching
    the top-left corner; 4) boundary points (right edge per row, excluding rows cut by the region limit); 5) Theil-Sen line;
    6) remove the triangle left of the line. Returns (pectoral_mask, ok, info); ok=False leaves the image unchanged
    (the failure is recorded and reported, never silently ignored)."""
    x0, y0, x1, y1 = bbox(mask, 0); H, W = y1 - y0, x1 - x0
    info = dict(ok=False, reason="")
    if H < 100 or W < 100: info["reason"] = "small"; return np.zeros_like(mask), False, info
    sm = filters.gaussian(img.astype(np.float32), sigma=6, preserve_range=True)[y0:y1, x0:x1]; mc = mask[y0:y1, x0:x1]
    yy, xx = np.mgrid[0:H, 0:W]; R = mc & (xx < fx * W) & (yy < fy * H)
    if R.sum() < 2000: info["reason"] = "region"; return np.zeros_like(mask), False, info
    try: thr = filters.threshold_multiotsu(sm[R], classes=3)[1]
    except ValueError: info["reason"] = "otsu"; return np.zeros_like(mask), False, info
    bright = ndi.binary_opening(R & (sm > thr), structure=morphology.disk(4)); lab = measure.label(bright, connectivity=2)
    best, bs = 0, 1e18
    for r in measure.regionprops(lab):
        if r.area < 1500: continue
        s_ = (r.coords[:, 0] + r.coords[:, 1]).min()
        if s_ < bs: bs, best = s_, r.label
    if best == 0 or bs > 0.15 * (W + H): info["reason"] = "not_at_corner"; return np.zeros_like(mask), False, info
    comp = lab == best; rows = np.where(comp.any(1))[0]
    xmax = np.array([np.where(comp[r])[0].max() for r in rows], float)
    keep = xmax < fx * W - 4; rows2, x2 = rows[keep].astype(float), xmax[keep]
    if len(rows2) < 40: info["reason"] = "short"; return np.zeros_like(mask), False, info
    slope, x_top, _, _ = stats.theilslopes(x2, rows2)                  # line x = slope*y + x_top
    if slope >= -0.1: info["reason"] = "slope"; info["slope"] = float(slope); return np.zeros_like(mask), False, info
    y_left = -x_top / slope; area_frac = 0.5 * x_top * y_left / max(mc.sum(), 1)
    info.update(slope=float(slope), x_top=float(x_top), y_left=float(y_left), area_frac=float(area_frac))
    if not (0.05 * W < x_top < 1.0 * W and 0.05 * H < y_left < 1.0 * H and min_frac < area_frac < max_frac):
        info["reason"] = "implausible"; return np.zeros_like(mask), False, info
    pm = np.zeros_like(mask); pm[y0:y1, x0:x1] = (xx < (x_top + slope * yy)) & mc
    info["ok"] = True; return pm, True, info

# ------------------------------------------------------------------ enhancement and resizing
def enhance(img, mask, clip=(0.5, 99.5), clahe_clip=0.02, kernel=64):
    """Percentile clipping inside the breast, rescale to [0,1], CLAHE, output uint8 (outside mask = 0)."""
    vals = img[mask].astype(np.float32)
    if vals.size < 100: return img.astype(np.uint8)
    lo, hi = np.percentile(vals, clip); x = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    x = exposure.equalize_adapthist(x, kernel_size=kernel, clip_limit=clahe_clip)
    return (np.where(mask, x, 0) * 255).astype(np.uint8)

def to_square(img, out_size):
    h, w = img.shape[:2]; side = max(h, w); py, px = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side), img.dtype); canvas[py:py + h, px:px + w] = img
    s = out_size / side
    out = sktf.resize(canvas.astype(np.float32), (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True)
    out = np.rint(out).astype(img.dtype)
    return out, (int(px), int(py)), float(s)

def to_square_mask(mask, out_size):
    """Resize a boolean mask with the same geometry as to_square (threshold at 0.5; never floor interpolated values)."""
    h, w = mask.shape; side = max(h, w); py, px = (side - h) // 2, (side - w) // 2
    canvas = np.zeros((side, side), np.float32); canvas[py:py + h, px:px + w] = mask
    return sktf.resize(canvas, (out_size, out_size), order=1, preserve_range=True, anti_aliasing=True) > 0.5

# ------------------------------------------------------------------ variants (cumulative ablation ladder)
VARIANT_DOC = {"P0": "raw (resize only)", "P1": "+ orientation (chest wall left)", "P2": "+ artefact removal + breast crop",
               "P3": "+ pectoral muscle removal", "P4": "+ percentile clip + CLAHE (full pipeline)"}

def prepare_geometry(img, chest_side="left"):
    """Compute the breast mask / orientation / pectoral mask once per image and share it between variants."""
    m, thr = breast_mask(img); flip = chest_side == "right"
    x = img[:, ::-1].copy() if flip else img.copy(); mf = m[:, ::-1].copy() if flip else m.copy()
    pm, ok, info = pectoral_mask(x, mf)
    return dict(img=img, m=m, x=x, mf=mf, pm=pm, ok=ok, info=info, flip=flip, area=float(m.mean()))

def run_variant(img, variant, chest_side="left", out_size=512, geo=None):
    """img: 2-D uint8 standardised image (row 0 = top). chest_side: side of the chest wall in the ORIGINAL image."""
    g = geo or prepare_geometry(img, chest_side); H, W = img.shape; tf = Transform(width=W, height=H, out_size=out_size)
    flags = {"breast_area_frac": g["area"]}
    if variant == "P0":
        out, pad, s = to_square(img, out_size); tf.pad, tf.scale = pad, s
        mm = to_square_mask(g["m"], out_size)
        return dict(image=out, breast=mm, pectoral=np.zeros(mm.shape, bool), transform=tf, flags=flags)
    x, m, pm = g["x"].copy(), g["mf"].copy(), np.zeros_like(g["mf"]); tf.flipped = g["flip"]
    if variant in ("P2", "P3", "P4"):
        x = np.where(m, x, 0).astype(x.dtype); tf.crop = bbox(m)
    if variant in ("P3", "P4"):
        flags["pectoral_ok"] = bool(g["ok"]); flags["pectoral"] = g["info"]
        if g["ok"]: pm = g["pm"]; x = np.where(pm, 0, x).astype(x.dtype); m = m & ~pm
    if variant == "P4": x = enhance(x, m)
    c = tf.crop; xc, mc, pc = x[c[1]:c[3], c[0]:c[2]], m[c[1]:c[3], c[0]:c[2]], pm[c[1]:c[3], c[0]:c[2]]
    out, pad, s = to_square(xc, out_size); tf.pad, tf.scale = pad, s
    mo, po = to_square_mask(mc, out_size), to_square_mask(pc, out_size)
    return dict(image=out, breast=mo, pectoral=po, transform=tf, flags=flags)

# ------------------------------------------------------------------ helpers used by QC and patch sampling
def decide_side(parity_side, image_side, image_conf, min_conf=0.3):
    """Chest-wall side: file-number parity (MIAS convention) unless the image evidence is confident and disagrees."""
    if image_side == parity_side: return parity_side, "agree"
    return (image_side, "image_override") if image_conf >= min_conf else (parity_side, "parity_low_conf")

def oriented_masks(img, chest_side):
    """Original-frame image -> (oriented image, breast mask, pectoral mask, flipped) with chest wall on the left."""
    g = prepare_geometry(img, chest_side); return g["x"], g["mf"], g["pm"], g["flip"]

def thumb(img, size=32):
    """Small normalised thumbnail for near-duplicate detection (Pearson correlation between thumbnails)."""
    h = img.shape[0] // size; return img[:h * size, :h * size].astype(np.float32).reshape(size, h, size, h).mean((1, 3)).ravel()

def bright_text_components(img, mask, thr=150, min_area=60, max_area=30000):
    """Bright compact objects outside the breast (film labels, markers): count of connected components."""
    im, mk = img[::2, ::2], mask[::2, ::2]                      # half resolution is enough and 4x faster
    outside = (im > thr) & (ndi.distance_transform_edt(~mk) > 5)
    lab = measure.label(outside, connectivity=2)
    if lab.max() == 0: return 0
    areas = np.bincount(lab.ravel())[1:]; return int(((areas >= min_area / 4) & (areas <= max_area / 4)).sum())

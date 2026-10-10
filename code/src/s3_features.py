"""Step 3: analysis engine, feature extraction (outcome-blind: labels are NOT read here except to define which ROIs exist).
  python bc.py patches                 build lesion + density-matched normal ROI table (needs the frozen partition)
  python bc.py features [--variants P0,...,P4]
Writes  processed/.../annotations/ann_v1/patches.csv
        processed/.../features/pp_v1.0.0/variant=Pk/{features_images.csv, features_patches.csv}
ROI definition: lesion circle (MIAS x, y from bottom-left, radius) -> square window of side clip(2.4 * radius) in the model frame, resized to 96x96.
Normal ROIs: sampled from normal images of the SAME partition, matched on background density, radius drawn from the lesion radius distribution."""
import os, json, argparse
import numpy as np, pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import transform as sktf
import config as C, preprocess_lib as L, feature_lib as F, partition as P, status

def _read_ann():
    return pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "annotations.csv"))

def build_patches(seed=C.SEED):
    man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); ann = _read_ann(); rng = np.random.default_rng(seed)
    part = {p: "DEV" for p in P.load_partition("DEV")}; part.update({p: "MS" for p in P.load_partition("MS")})
    les = ann[(ann.cls != "NORM") & ann.has_roi & ann.severity.notna()].copy()
    les = les.drop(columns=["bg"]).merge(man[["image_id", "bg", "laterality"]], on="image_id")
    rows = [dict(patch_id=r.lesion_id, group="lesion", image_id=r.image_id, patient_id=r.patient_id, partition=part[r.patient_id], severity=r.severity,
                 lesion_class=r.cls, bg=r.bg, roi_x=float(r.x), roi_y=float(1024 - r.y), roi_r=float(r.radius)) for r in les.itertuples()]
    normals = man[man.image_class == "normal"].copy(); normals["partition"] = normals.patient_id.map(part)
    used = set()
    for pname in ["DEV", "MS"]:
        radii = les[les.patient_id.map(part) == pname].radius.values
        for bg in ["F", "G", "D"]:
            want = [r for r in rows if r["group"] == "lesion" and r["partition"] == pname and r["bg"] == bg]
            pool = normals[(normals.partition == pname) & (normals.bg == bg)].image_id.tolist(); rng.shuffle(pool)
            for k, w in enumerate(sorted(want, key=lambda z: z["patch_id"])):
                iid = pool[k % len(pool)] if pool else None
                if iid is None: continue
                mrow = man[man.image_id == iid].iloc[0]
                sd = C.image_dir("std", mrow.patient_id, iid); img = np.array(Image.open(os.path.join(sd, "image.png"))).astype(np.uint8)
                side = json.load(open(os.path.join(sd, "qc.json")))["chest_side_used"]; x, m, pm, flip = L.oriented_masks(img, side)
                allowed = m & ~ndi.binary_dilation(pm, iterations=15) if pm.any() else m; dist = ndi.distance_transform_edt(allowed)
                r = float(rng.choice(radii)); ok = False
                for _ in range(8):
                    ys, xs = np.where(dist >= max(r * 0.9, 15))
                    if len(xs): j = rng.integers(len(xs)); cx, cy = float(xs[j]), float(ys[j]); ok = True; break
                    r *= 0.8
                if not ok: continue
                if flip: cx = 1024 - 1 - cx
                rows.append(dict(patch_id=f"N_{iid}_{k}", group="normal", image_id=iid, patient_id=mrow.patient_id, partition=pname, severity=None, lesion_class="NORM",
                                 bg=bg, roi_x=cx, roi_y=cy, roi_r=r))
    df = pd.DataFrame(rows); out = os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "patches.csv"); df.to_csv(out, index=False); return df

def _load(row, variant):
    d = C.image_dir("pre", row["patient_id"], row["image_id"], variant)
    img = np.array(Image.open(os.path.join(d, "model_input.png"))); m = np.array(Image.open(os.path.join(d, "breast_mask.png"))) > 0
    pm = np.array(Image.open(os.path.join(d, "pectoral_mask.png"))) > 0; tf = L.Transform.from_dict(json.load(open(os.path.join(d, "transform.json"))))
    return img, m & ~pm, tf

def patch_features(img, tf, cx, cy, r):
    xm, ym = tf.map_point(cx, cy); rm = tf.map_radius(r); side = int(np.clip(C.PATCH_SCALE * rm, C.PATCH_MIN, C.PATCH_MAX)); h = side // 2
    pad = np.pad(img, h + 2); x0, y0 = int(round(xm)) - h + h + 2, int(round(ym)) - h + h + 2
    crop = pad[y0:y0 + side, x0:x0 + side]
    if crop.shape != (side, side): return None
    patch = (sktf.resize(crop.astype(np.float32), (C.PATCH_OUT, C.PATCH_OUT), order=1, preserve_range=True, anti_aliasing=True)).astype(np.uint8)
    mask = patch > 0
    if mask.mean() < 0.5: mask = np.ones_like(mask)
    f = F.extract(patch, mask); f["valid_frac"] = float(mask.mean()); f["side_px"] = side; return f

def run_variant(variant, partition="ALL"):
    man = P.select(pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")), partition); patches = pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "patches.csv"))
    fd = C.feature_dir(variant); os.makedirs(fd, exist_ok=True); ri, rp = [], []
    for r in man.to_dict("records"):
        img, mask, tf = _load(r, variant); sub = (slice(0, None, 2), slice(0, None, 2)); x, m = img[sub], mask[sub]
        ys, xs = np.where(m)
        if len(xs) > 200:
            x, m = x[ys.min():ys.max() + 1, xs.min():xs.max() + 1], m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            f = F.extract(x, m, with_hist=True)
        else: f = {}
        ri.append(dict(image_id=r["image_id"], patient_id=r["patient_id"], **f))
        for pr in patches[patches.image_id == r["image_id"]].to_dict("records"):
            g = patch_features(img, tf, pr["roi_x"], pr["roi_y"], pr["roi_r"])
            if g: rp.append(dict(patch_id=pr["patch_id"], image_id=r["image_id"], patient_id=r["patient_id"], **g))
    pd.DataFrame(ri).to_csv(os.path.join(fd, "features_images.csv"), index=False); pd.DataFrame(rp).to_csv(os.path.join(fd, "features_patches.csv"), index=False)
    return len(ri), len(rp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["patches", "features"]); ap.add_argument("--variants", default=",".join(C.VARIANTS)); ap.add_argument("--partition", default="ALL", choices=["ALL", "DEV", "MS"]); a = ap.parse_args()
    if a.what == "patches":
        df = build_patches(); print(df.groupby(["partition", "group"]).size().to_string()); print("severity (lesion):", df[df.group == "lesion"].groupby(["partition", "severity"]).size().to_dict())
    else:
        for v in a.variants.split(","): ni, npch = run_variant(v, a.partition); print(f"{v}: {ni} image rows, {npch} patch rows")
        man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); status.mark("FEATURES_DONE", man.image_id)

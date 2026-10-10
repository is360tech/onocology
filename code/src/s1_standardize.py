"""Step 1: standardise (canonical image + sidecar) and quality control.
  python bc.py standardize
Reads   raw/.../original.pgm       Writes  processed/.../standardized/<patient>/<image_id>/image.png (uint16 container, 8-bit values),
        meta.json, qc.json;  manifests/qc_summary.csv
QC (label-blind): breast segmentation area, chest-wall side (image evidence vs file-number parity), full-frame mask, burned-in
label/marker components, exposure, duplicates (difference hash)."""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
import config as C, preprocess_lib as L

def _one(r, force=False):
    d = C.image_dir("std", r["patient_id"], r["image_id"]); os.makedirs(d, exist_ok=True)
    if not force and os.path.exists(os.path.join(d, "qc.json")): return json.load(open(os.path.join(d, "qc.json")))   # resumable
    raw = os.path.join(C.WS, r["raw_path"]); img = np.array(Image.open(raw))
    m, thr = L.breast_mask(img); side_img, conf = L.chest_wall_side(m); parity = "left" if r["laterality"] == "L" else "right"
    side, why = L.decide_side(parity, side_img, conf)
    Image.fromarray(img.astype(np.uint16)).save(os.path.join(d, "image.png"))
    area = float(m.mean()); med = float(np.median(img[m])) if m.any() else 0.0; ntxt = L.bright_text_components(img, m)
    qc = dict(image_id=r["image_id"], breast_area_frac=area, mask_threshold=thr, chest_side_parity=parity, chest_side_image=side_img,
              chest_side_conf=float(conf), chest_side_used=side, chest_side_status=why, full_frame_mask=bool(area > 0.75),
              n_label_components=ntxt, median_intensity=med, thumb=L.thumb(img).tolist())
    qc["qc_pass"] = bool(m.any() and 0.08 <= area <= 0.75 and 20 <= med <= 235)
    json.dump(qc, open(os.path.join(d, "qc.json"), "w"), indent=1)
    json.dump(dict(image_id=r["image_id"], patient_id=r["patient_id"], refnum=r["refnum"], laterality=r["laterality"], view=r["view"],
                   bit_depth=8, shape=list(img.shape), pixel_spacing_mm=r["pixel_spacing_mm"], source=C.SOURCE, licence="research_only"),
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    return qc

def run(n_jobs=-1, force=False, limit=None):
    """Resumable: images with an existing qc.json are skipped. limit = max NEW images to process in this call."""
    C.ensure_dirs(); man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); recs = man.to_dict("records")
    todo = [r for r in recs if force or not os.path.exists(os.path.join(C.image_dir("std", r["patient_id"], r["image_id"]), "qc.json"))]
    if limit: todo = todo[:limit]
    Parallel(n_jobs=n_jobs)(delayed(_one)(r, force) for r in todo)
    if any(not os.path.exists(os.path.join(C.image_dir("std", r["patient_id"], r["image_id"]), "qc.json")) for r in recs): return None, []
    qcs = [_one(r) for r in recs]; q = pd.DataFrame(qcs)
    T = np.array(q.thumb.tolist()); T = (T - T.mean(1, keepdims=True)) / (T.std(1, keepdims=True) + 1e-6); Cm = T @ T.T / T.shape[1]; np.fill_diagonal(Cm, 0)
    dup = [(q.image_id[i], q.image_id[j]) for i, j in zip(*np.where(np.triu(Cm > 0.995)))]   # near-identical thumbnails
    q["dup_of"] = ""
    for a, b in dup: q.loc[q.image_id == b, "dup_of"] = a
    q.drop(columns=["thumb"]).merge(man[["image_id", "refnum", "patient_id"]], on="image_id").to_csv(os.path.join(C.MAN, "qc_summary.csv"), index=False)
    return q, dup

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--force", action="store_true"); ap.add_argument("--limit", type=int)
    a = ap.parse_args(); q, dup = run(a.jobs, a.force, a.limit)
    if q is None: print("partial run done; run again to continue"); sys.exit(0)
    print(f"standardised {len(q)} images | QC pass {q.qc_pass.mean():.1%} | full-frame masks {int(q.full_frame_mask.sum())} | "
          f"chest-side status {q.chest_side_status.value_counts().to_dict()} | near-duplicates {len(dup)}")

"""Step 2: preprocessing variants (cumulative ablation ladder P0..P4).
  python bc.py preprocess [--variants P0,P1,P2,P3,P4]
Writes  processed/.../preprocessed/pp_v1.0.0/variant=Pk/<patient>/<image_id>/{model_input.png, breast_mask.png, pectoral_mask.png,
        transform.json, flags.json};  manifests/preprocess_summary_Pk.csv
transform.json lets lesion coordinates / predictions map between the original and model-input frames."""
import os, sys, json, argparse
import numpy as np, pandas as pd
from PIL import Image
from joblib import Parallel, delayed
import config as C, preprocess_lib as L

def _done(r, v): return os.path.exists(os.path.join(C.image_dir("pre", r["patient_id"], r["image_id"], v), "flags.json"))

def _one(r, variants, force=False):
    todo = [v for v in variants if force or not _done(r, v)]
    if todo:
        sd = C.image_dir("std", r["patient_id"], r["image_id"])
        img = np.array(Image.open(os.path.join(sd, "image.png"))).astype(np.uint8); qc = json.load(open(os.path.join(sd, "qc.json")))
        geo = L.prepare_geometry(img, qc["chest_side_used"])
        for v in todo:
            res = L.run_variant(img, v, qc["chest_side_used"], C.MODEL_SIZE, geo=geo); d = C.image_dir("pre", r["patient_id"], r["image_id"], v); os.makedirs(d, exist_ok=True)
            Image.fromarray(res["image"].astype(np.uint8)).save(os.path.join(d, "model_input.png"))
            Image.fromarray((res["breast"] * 255).astype(np.uint8)).save(os.path.join(d, "breast_mask.png"))
            Image.fromarray((res["pectoral"] * 255).astype(np.uint8)).save(os.path.join(d, "pectoral_mask.png"))
            json.dump(res["transform"].to_dict(), open(os.path.join(d, "transform.json"), "w"))
            json.dump(res["flags"], open(os.path.join(d, "flags.json"), "w"), default=float)      # written last = completion marker
    out = []
    for v in variants:
        fl = json.load(open(os.path.join(C.image_dir("pre", r["patient_id"], r["image_id"], v), "flags.json")))
        out.append(dict(variant=v, image_id=r["image_id"], breast_area_frac=fl["breast_area_frac"], pectoral_ok=fl.get("pectoral_ok"),
                        pectoral_reason=(fl.get("pectoral") or {}).get("reason"), pectoral_area_frac=(fl.get("pectoral") or {}).get("area_frac")))
    return out

def run(variants, n_jobs=-1, force=False, limit=None, partition="ALL"):
    """Resumable: variants with an existing flags.json are skipped. limit = max images with missing variants to process per call."""
    C.ensure_dirs(); import partition as PT
    man = PT.select(pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")), partition); recs = man.to_dict("records")
    todo = [r for r in recs if force or any(not _done(r, v) for v in variants)]
    if limit: todo = todo[:limit]
    Parallel(n_jobs=n_jobs)(delayed(_one)(r, variants, force) for r in todo)
    if any(not _done(r, v) for r in recs for v in variants): return None
    flat = pd.DataFrame([x for r in recs for x in _one(r, variants)])
    for v, g in flat.groupby("variant"): g.to_csv(os.path.join(C.MAN, f"preprocess_summary_{v}.csv"), index=False)
    return flat

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", default=",".join(C.VARIANTS)); ap.add_argument("--jobs", type=int, default=-1); ap.add_argument("--force", action="store_true"); ap.add_argument("--limit", type=int); ap.add_argument("--partition", default="ALL", choices=["ALL", "DEV", "MS"])
    a = ap.parse_args(); f = run(a.variants.split(","), a.jobs, a.force, a.limit, a.partition)
    if f is None: print("partial run done; run again to continue"); sys.exit(0)
    for v, g in f.groupby("variant"):
        extra = f" | pectoral removed in {g.pectoral_ok.sum()}/{len(g)} ({g.pectoral_ok.mean():.1%})" if g.pectoral_ok.notna().any() else ""
        print(f"{v} ({L.VARIANT_DOC[v]}): {len(g)} images{extra}")

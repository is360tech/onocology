"""Objective check that lesion coordinates are mapped correctly (origin bottom-left). For mass-type lesions (CIRC/SPIC/MISC) the mean
intensity inside the annotated circle should exceed the surrounding ring more often with the CORRECT mapping than with wrong ones.
  python bc.py align --partition DEV"""
import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
from PIL import Image
import config as C, partition as PT

ap = argparse.ArgumentParser(); ap.add_argument("--partition", default="DEV", choices=["DEV", "MS", "ALL"]); a = ap.parse_args()
ann = pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "annotations.csv")); man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv"))
ann = PT.select(ann, a.partition); ann = ann[ann.cls.isin(["CIRC", "SPIC", "MISC"]) & ann.has_roi]
maps = {"correct (x, 1024-y)": lambda x, y: (x, 1024 - y), "y not flipped (x, y)": lambda x, y: (x, y), "x mirrored": lambda x, y: (1023 - x, 1024 - y), "x mirrored, y not flipped": lambda x, y: (1023 - x, y)}
res = {k: [] for k in maps}; yy, xx = np.mgrid[0:1024, 0:1024]
for r in ann.itertuples():
    img = np.array(Image.open(os.path.join(C.image_dir("std", r.patient_id, r.image_id), "image.png"))).astype(float)
    for k, f in maps.items():
        cx, cy = f(r.x, r.y); d = np.hypot(xx - cx, yy - cy); inside = img[d <= r.radius]; ring = img[(d > 1.3 * r.radius) & (d <= 2.0 * r.radius)]
        if inside.size > 20 and ring.size > 20: res[k].append((inside.mean() - ring.mean()) / (ring.std() + 1e-6))
print(f"mass lesions in {a.partition}: {len(ann)}")
tab = pd.DataFrame([dict(mapping=k, n=len(v), median_contrast_z=np.median(v), frac_inside_brighter=np.mean(np.array(v) > 0)) for k, v in res.items()]).round(3); print(tab.to_string(index=False))
ok = tab.iloc[0].frac_inside_brighter > tab.iloc[1:].frac_inside_brighter.max() + 0.10; print("PASS: correct mapping is clearly best" if ok else "FAIL: check coordinate mapping"); sys.exit(0 if ok else 1)

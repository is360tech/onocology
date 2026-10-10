"""Label-permutation control (negative control): shuffle labels, repeat the grouped CV, compare with the observed AUC.
  python bc.py permute --partition DEV --task severity_roi --variant P4 --model slda --n-perm 30
A leak-free pipeline gives a null AUC centred near 0.5."""
import os, sys, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd
import config as C, s4_experiments as S
from evalkit import run_cv, metrics

ap = argparse.ArgumentParser(); ap.add_argument("--partition", default="DEV", choices=["DEV", "MS"]); ap.add_argument("--task", default=C.PRIMARY_TASK); ap.add_argument("--variant", default=C.PRIMARY_VARIANT)
ap.add_argument("--model", default=C.PRIMARY_MODEL); ap.add_argument("--n-perm", type=int, default=30); a = ap.parse_args()
kind = S.run_kind(a.partition)
if a.partition == "MS": S._ledger("MS", "permutation_control", "permutation")
df, cols, key = S.load_task(a.task, a.variant, a.partition); df = df.sort_values(key).reset_index(drop=True)
X, y, g = df[cols].values.astype(float), df.y.values, df.patient_id.values; t = C.TASKS[a.task]
P, classes, _ = run_cv(X, y, g, a.model, C.SEED, 3); obs = metrics(y, P.mean(0), classes, t["positive"])["auc"]
rng = np.random.default_rng(C.SEED); null = []
for i in range(a.n_perm):
    yp = rng.permutation(y); Pp, cl, _ = run_cv(X, yp, g, a.model, C.SEED + 100 + i, 1); null.append(metrics(yp, Pp.mean(0), cl, t["positive"])["auc"])
null = np.array([v for v in null if np.isfinite(v)]); p = (1 + (null >= obs).sum()) / (1 + len(null))
print(f"{a.task}/{a.variant}/{a.model} on {a.partition}: observed AUC {obs:.3f} | permuted AUC mean {null.mean():.3f} (sd {null.std():.3f}, 95th pct {np.percentile(null, 95):.3f}) | permutation p = {p:.3f}")
ok = abs(null.mean() - 0.5) < 0.08; print("PASS: null centred near 0.5 (no leakage detected)" if ok else "FAIL: permuted AUC far from 0.5, investigate leakage"); sys.exit(0 if ok else 1)

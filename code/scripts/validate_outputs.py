"""Acceptance report for the processing stages (T2-T4). python bc.py validate [--partition ALL|DEV|MS]"""
import os, sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
import config as C, partition as PT
ap = argparse.ArgumentParser(); ap.add_argument("--partition", default="ALL", choices=["ALL", "DEV", "MS"]); a = ap.parse_args()
fails, L = [], ["# Validation report", ""]
def chk(name, ok, detail): L.append(f"- {'PASS' if ok else 'FAIL'}  {name}: {detail}"); (None if ok else fails.append(name))
man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); chk("ingest counts", len(man) == 322 and man.patient_id.nunique() == 161, f"{len(man)} images, {man.patient_id.nunique()} patients (expected 322 / 161)")
try: PT.load(); fz = json.load(open(os.path.join(C.DATASET_ROOT, "partition_freeze.json"))); chk("partition frozen + hashes verify", True, json.dumps(fz["counts"]))
except SystemExit as e: chk("partition frozen", False, str(e))
m = PT.select(man, a.partition) if os.path.exists(os.path.join(C.DATASET_ROOT, "partition_freeze.json")) else man; L.append(f"\nScope: {a.partition} ({len(m)} images)\n")
qc = pd.read_csv(os.path.join(C.MAN, "qc_summary.csv")); qc = qc[qc.image_id.isin(m.image_id)]; chk("QC pass rate >= 98%", qc.qc_pass.mean() >= 0.98, f"{qc.qc_pass.mean():.1%}; full-frame masks {int(qc.full_frame_mask.sum())}; chest-side {qc.chest_side_status.value_counts().to_dict()}")
for v in C.VARIANTS:
    done = sum(os.path.exists(os.path.join(C.image_dir("pre", r.patient_id, r.image_id, v), "flags.json")) for r in m.itertuples()); chk(f"preprocessing {v} complete", done == len(m), f"{done}/{len(m)}")
    if v in ("P3", "P4") and done:
        ok = [json.load(open(os.path.join(C.image_dir("pre", r.patient_id, r.image_id, v), "flags.json"))).get("pectoral_ok", False) for r in m.itertuples() if os.path.exists(os.path.join(C.image_dir("pre", r.patient_id, r.image_id, v), "flags.json"))]
        chk(f"pectoral removal success {v} >= 75%", sum(ok) / len(ok) >= 0.75, f"{sum(ok)}/{len(ok)} = {sum(ok) / len(ok):.1%} (failures are recorded, image left unchanged)"); 
for v in C.VARIANTS:
    f = os.path.join(C.feature_dir(v), "features_images.csv"); chk(f"features {v}", os.path.exists(f), f"{len(pd.read_csv(f)) if os.path.exists(f) else 0} image rows")
L.append("- INFO  frozen protocol (needed before the MS run): " + ((__import__("provenance").sha(C.PROTOCOL_FILE) or "")[:16] or "not frozen yet"))
if a.partition == "MS" and not os.path.exists(C.PROTOCOL_FILE): fails.append("protocol not frozen")
L += ["", "## Result", "PASS" if not fails else "FAIL: " + "; ".join(fails)]; os.makedirs(os.path.join(C.REP), exist_ok=True); p = os.path.join(C.REP, "validation_report.md"); open(p, "w").write("\n".join(L)); print("\n".join(L)); print("\nreport:", p); sys.exit(1 if fails else 0)

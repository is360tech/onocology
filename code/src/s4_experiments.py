"""Step 4: pre-specified experiments (tasks x preprocessing variants x models) with governance for the manuscript partition.
  python bc.py protocol                       copy docs/protocol.yaml into the workspace and print its SHA-256 (freeze)
  python bc.py experiments --partition DEV    development runs (free to repeat)
  python bc.py experiments --partition MS     manuscript run: needs the frozen protocol; the FIRST run per protocol hash is 'confirmatory',
                                              later runs are recorded as 'exploratory'. Every MS access is written to manifests/ms_access_log.csv
Outputs: research/.../experiments/<exp_id>/{results.csv, comparisons.csv, oof/*.csv, config.json}"""
import os, sys, json, glob, shutil, argparse, datetime, getpass
import numpy as np, pandas as pd
import config as C, partition as PT, provenance
from evalkit import run_cv, metrics, boot_ci, paired_boot
from models import paz_summary, paz_gate

FEAT_PREFIX = ("fo_", "grad_", "glcm_", "lbp8_", "lbp16_", "hist_")
EXP_ROOT = os.path.join(C.RESEARCH, "experiments")

def freeze_protocol(force=False):
    src = os.path.join(C.HOME, "docs", "protocol.yaml"); os.makedirs(EXP_ROOT, exist_ok=True)
    if os.path.exists(C.PROTOCOL_FILE) and not force: raise SystemExit("protocol already frozen; a changed protocol needs --force and a written justification")
    shutil.copy2(src, C.PROTOCOL_FILE); return provenance.sha(C.PROTOCOL_FILE)

def _ledger(partition, kind, exp_id):
    os.makedirs(C.MAN, exist_ok=True); new = not os.path.exists(C.ACCESS_LEDGER)
    row = pd.DataFrame([dict(time=datetime.datetime.now().isoformat(timespec="seconds"), user=getpass.getuser(), partition=partition, kind=kind, exp_id=exp_id,
                             protocol_sha256=provenance.sha(C.PROTOCOL_FILE), code_sha256=provenance.record()["code_sha256"])])
    row.to_csv(C.ACCESS_LEDGER, mode="a", header=new, index=False)

def run_kind(partition):
    if partition == "DEV": return "development"
    if not os.path.exists(C.PROTOCOL_FILE): raise SystemExit("MS run needs a frozen protocol: python bc.py protocol")
    ph = provenance.sha(C.PROTOCOL_FILE); led = pd.read_csv(C.ACCESS_LEDGER) if os.path.exists(C.ACCESS_LEDGER) else pd.DataFrame(columns=["partition", "kind", "protocol_sha256"])
    prior = led[(led.partition == "MS") & (led.kind == "confirmatory") & (led.protocol_sha256 == ph)]
    return "exploratory" if len(prior) else "confirmatory"

def load_task(task, variant, partition):
    t = C.TASKS[task]; fd = C.feature_dir(variant); man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); qc = pd.read_csv(os.path.join(C.MAN, "qc_summary.csv"))[["image_id", "qc_pass"]]
    ids = set(PT.load_partition(partition)) if partition != "ALL" else set(man.patient_id)
    if t["level"] == "patch":
        f = pd.read_csv(os.path.join(fd, "features_patches.csv")); pt = pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "patches.csv"))
        df = f.drop(columns=["patient_id", "image_id"]).merge(pt[["patch_id", "image_id", "patient_id", "group", "severity", "bg"]], on="patch_id"); key = "patch_id"
        if task == "severity_roi": df = df[(df.group == "lesion") & df.severity.isin(t["classes"])]; df["y"] = df.severity
        else: df["y"] = df.group
    else:
        f = pd.read_csv(os.path.join(fd, "features_images.csv")).drop(columns=["patient_id"]); df = f.merge(man[["image_id", "patient_id", "bg", "abnormal"]], on="image_id"); key = "image_id"
        df["y"] = df.bg if task == "density_img" else np.where(df.abnormal == 1, "abnormal", "normal")
    df = df.merge(qc, on="image_id" if key == "patch_id" else "image_id"); df = df[df.qc_pass & df.patient_id.isin(ids)]
    cols = [c for c in df.columns if c.startswith(FEAT_PREFIX)]; df = df.dropna(subset=cols, how="all")
    return df.reset_index(drop=True), cols, key

def main(a):
    C.ensure_dirs(); kind = run_kind(a.partition); os.makedirs(EXP_ROOT, exist_ok=True)
    n = len(glob.glob(os.path.join(EXP_ROOT, f"exp_{datetime.date.today():%Y%m%d}_*"))) + 1; exp_id = f"exp_{datetime.date.today():%Y%m%d}_{n:03d}_{a.partition}"
    out = os.path.join(EXP_ROOT, exp_id); os.makedirs(os.path.join(out, "oof"), exist_ok=True)
    if a.partition == "MS": _ledger("MS", kind, exp_id)
    print(f"{exp_id}  kind={kind}  partition={a.partition}  repeats={a.repeats}  boot={a.boot}")
    rows, store = [], {}
    for task in a.tasks:
        t = C.TASKS[task]; data = {v: load_task(task, v, a.partition) for v in a.variants}
        key = data[a.variants[0]][2]; common = set.intersection(*[set(d[0][key]) for d in data.values()])       # identical rows for every variant
        for v in a.variants:
            df, cols, _ = data[v]; df = df[df[key].isin(common)].sort_values(key).reset_index(drop=True)
            X, y, g = df[cols].values.astype(float), df.y.values, df.patient_id.values
            if len(np.unique(y)) < 2 or min(pd.Series(y).value_counts()) < 6: print(f"  skip {task}/{v}: too few samples {pd.Series(y).value_counts().to_dict()}"); continue
            for mname in a.models:
                P, classes, picks = run_cv(X, y, g, mname, C.SEED, a.repeats, a.jobs); Pbar = P.mean(0)
                per = pd.DataFrame([metrics(y, P[r], classes, t["positive"]) for r in range(len(P))]); pos = t["positive"]
                ci_auc = boot_ci(y, Pbar, g, lambda yy, pp: metrics(yy, pp, classes, pos)["auc"], a.boot); ci_ba = boot_ci(y, Pbar, g, lambda yy, pp: metrics(yy, pp, classes, pos)["bal_acc"], a.boot)
                pz = paz_summary(y, Pbar, classes); m0 = metrics(y, Pbar, classes, pos)
                rows.append(dict(task=task, variant=v, model=mname, n=len(y), n_patients=len(set(g)), classes="/".join(classes), class_counts=json.dumps(pd.Series(y).value_counts().to_dict()),
                                 auc=m0["auc"], auc_lo=ci_auc[0], auc_hi=ci_auc[1], auc_repeat_sd=per.auc.std(), bal_acc=m0["bal_acc"], bal_lo=ci_ba[0], bal_hi=ci_ba[1],
                                 acc=m0["acc"], f1_macro=m0["f1_macro"], sens=m0.get("sens"), spec=m0.get("spec"), brier=m0.get("brier"), ece=m0.get("ece"),
                                 paz_ratio=pz["paz_ratio"], paz_state=pz["batch_state"], acc_confident=pz["acc_confident"], coverage=pz["coverage"], kind=kind))
                mg, amb, paz_method_used = paz_gate(Pbar); o = df[[key, "patient_id", "y"]].copy()
                for j, c in enumerate(classes): o[f"p_{c}"] = Pbar[:, j]
                # column renamed from the old "paz_margin" to "paz_score" + "paz_method": paz_gate() now
                # auto-resolves margin (2-class) vs entropy (3+-class, e.g. density_img) per task, so a
                # fixed "margin" label would silently mislabel entropy scores for density_img.
                o["paz_score"] = mg; o["paz_method"] = paz_method_used; o["state"] = np.where(amb, "AMBIGUOUS", "STABLE"); o.to_csv(os.path.join(out, "oof", f"{task}__{v}__{mname}.csv"), index=False)
                store[(task, v, mname)] = (y, Pbar, g, classes, pos); r = rows[-1]
                print(f"  {task:<22} {v} {mname:<7} n={r['n']:<4} AUC {r['auc']:.3f} [{r['auc_lo']:.3f},{r['auc_hi']:.3f}]  bal.acc {r['bal_acc']:.3f}  PAZ {r['paz_ratio']:.2f}", flush=True)
    res = pd.DataFrame(rows); res.to_csv(os.path.join(out, "results.csv"), index=False)
    comps = []                                                              # pre-specified paired comparisons
    for task in a.tasks:
        for (t_, v, m), (y, P, g, classes, pos) in store.items():
            if t_ != task: continue
            fn = lambda yy, pp: metrics(yy, pp, classes, pos)["auc"]
            if m == C.PRIMARY_MODEL and v != a.variants[0] and (task, a.variants[0], m) in store:
                d = paired_boot(y, P, store[(task, a.variants[0], m)][1], g, fn, a.boot); comps.append(dict(task=task, comparison=f"{v} vs {a.variants[0]} ({m})", delta_auc=d[0], lo=d[1], hi=d[2], p=d[3]))
            if v == C.PRIMARY_VARIANT and m != C.PRIMARY_MODEL and (task, v, C.PRIMARY_MODEL) in store:
                d = paired_boot(y, P, store[(task, v, C.PRIMARY_MODEL)][1], g, fn, a.boot); comps.append(dict(task=task, comparison=f"{m} vs {C.PRIMARY_MODEL} ({v})", delta_auc=d[0], lo=d[1], hi=d[2], p=d[3]))
    pd.DataFrame(comps).to_csv(os.path.join(out, "comparisons.csv"), index=False)
    json.dump(dict(exp_id=exp_id, kind=kind, partition=a.partition, tasks=a.tasks, variants=a.variants, models=a.models, repeats=a.repeats, boot=a.boot,
                   provenance=provenance.record(), partition_freeze=json.load(open(os.path.join(C.DATASET_ROOT, "partition_freeze.json")))), open(os.path.join(out, "config.json"), "w"), indent=1, default=str)
    print("saved", out); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--partition", choices=["DEV", "MS"], required=True); ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--variants", default=",".join(C.VARIANTS)); ap.add_argument("--models", default=",".join(C.MODELS)); ap.add_argument("--repeats", type=int, default=C.N_REPEATS)
    ap.add_argument("--boot", type=int, default=C.N_BOOT); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--quick", action="store_true"); ap.add_argument("--freeze-protocol", action="store_true")
    a = ap.parse_args()
    if a.freeze_protocol: print("protocol sha256:", freeze_protocol()); sys.exit(0)
    a.tasks, a.variants, a.models = a.tasks.split(","), a.variants.split(","), a.models.split(",")
    if a.quick: a.repeats, a.boot = 2, 200
    main(a)

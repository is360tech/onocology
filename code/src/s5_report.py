"""Step 5: manuscript tables and figures from ONE experiment directory (nothing is re-computed here).
  python bc.py report [--exp exp_...]      (default: latest experiment; MS reports are labelled with the run kind)
Writes research/.../manuscript/<exp_id>/{tables/*.md|csv, figures/*.png, RESULTS_SUMMARY.md}"""
import os, sys, glob, json, argparse
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve
import config as C, partition as PT, preprocess_lib as L
from models import paz_summary

def md(df):
    cols = list(df.columns); lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False): lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    return "\n".join(lines)
ci = lambda v, lo, hi: f"{v:.2f} [{lo:.2f}, {hi:.2f}]"

def latest_exp(partition=None):
    d = sorted(glob.glob(os.path.join(C.RESEARCH, "experiments", "exp_*" + (f"_{partition}" if partition else ""))))
    if not d: raise SystemExit("no experiment found: run `python bc.py experiments ...` first")
    return d[-1]

def fig_preprocessing(out, partition):
    man = PT.select(pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")), partition).sort_values("refnum")
    pick = [man[man.bg == b].iloc[0] for b in ["F", "G", "D"]] + [man[man.image_class == "malignant"].iloc[0]]
    fig, ax = plt.subplots(len(pick), 5, figsize=(11, 2.3 * len(pick)))
    for i, r in enumerate(pick):
        for j, v in enumerate(C.VARIANTS):
            f = os.path.join(C.image_dir("pre", r.patient_id, r.image_id, v), "model_input.png")
            ax[i, j].imshow(np.array(Image.open(f)), cmap="gray", vmin=0, vmax=255); ax[i, j].axis("off")
            if i == 0: ax[i, j].set_title(f"{v}\n{L.VARIANT_DOC[v]}", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_preprocessing_examples.png"), dpi=200); plt.close()

def fig_alignment(out, partition):
    ann = PT.select(pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "annotations.csv")), partition)
    ann = ann[ann.has_roi & ann.cls.isin(["CIRC", "SPIC", "MISC"])].sort_values("lesion_id").head(8)
    fig, ax = plt.subplots(2, 4, figsize=(11, 5.6))
    for a_, r in zip(ax.ravel(), ann.itertuples()):
        d = C.image_dir("pre", r.patient_id, r.image_id, C.PRIMARY_VARIANT); tf = L.Transform.from_dict(json.load(open(os.path.join(d, "transform.json"))))
        cx, cy = tf.map_point(r.x, 1024 - r.y); a_.imshow(np.array(Image.open(os.path.join(d, "model_input.png"))), cmap="gray"); a_.add_patch(plt.Circle((cx, cy), tf.map_radius(r.radius), fill=False, color="yellow", lw=1.2))
        a_.set_title(f"{r.cls} {r.severity}", fontsize=8); a_.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(out, "fig_annotation_alignment.png"), dpi=200); plt.close()

def main(exp):
    cfg = json.load(open(os.path.join(exp, "config.json"))); res = pd.read_csv(os.path.join(exp, "results.csv")); comp = pd.read_csv(os.path.join(exp, "comparisons.csv")) if os.path.getsize(os.path.join(exp, "comparisons.csv")) > 5 else pd.DataFrame()
    eid = os.path.basename(exp); out = os.path.join(C.RESEARCH, "manuscript", eid); os.makedirs(os.path.join(out, "tables"), exist_ok=True); os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    kind = cfg["kind"]; banner = {"confirmatory": "CONFIRMATORY run on the manuscript partition (frozen protocol).", "exploratory": "EXPLORATORY run: NOT for the manuscript's primary results.",
                                  "development": "DEVELOPMENT run on the DEV partition: for pipeline design only, NOT manuscript results."}[kind]
    T = os.path.join(out, "tables"); F = os.path.join(out, "figures")
    fz = cfg["partition_freeze"]["counts"]; pt = pd.read_csv(os.path.join(C.PROC, "annotations", C.ANNOT_VERSION, "patches.csv"))
    t1 = pd.DataFrame([dict(partition=k, patients=v["patients"], images=v["images"], normal=v["normal"], benign=v["benign"], malignant=v["malignant"],
                            roi_benign=int(((pt.partition == k) & (pt.severity == "B")).sum()), roi_malignant=int(((pt.partition == k) & (pt.severity == "M")).sum())) for k, v in fz.items()])
    t2 = res[res.variant == C.PRIMARY_VARIANT].assign(AUC=lambda d: [ci(*x) for x in zip(d.auc, d.auc_lo, d.auc_hi)], BalAcc=lambda d: [ci(*x) for x in zip(d.bal_acc, d.bal_lo, d.bal_hi)])[["task", "model", "n", "n_patients", "AUC", "BalAcc", "paz_ratio", "acc_confident"]]
    t3 = res[res.model == C.PRIMARY_MODEL].assign(AUC=lambda d: [ci(*x) for x in zip(d.auc, d.auc_lo, d.auc_hi)])[["task", "variant", "n", "AUC", "bal_acc", "ece"]]
    t5 = res[(res.variant == C.PRIMARY_VARIANT)][["task", "model", "paz_ratio", "paz_state", "coverage", "acc_confident", "acc"]]
    for n, t in [("table1_dataset", t1), ("table2_main_results", t2), ("table3_preprocessing_ablation", t3), ("table4_paired_comparisons", comp), ("table5_paz_gate", t5)]:
        t.to_csv(os.path.join(T, n + ".csv"), index=False); open(os.path.join(T, n + ".md"), "w").write(f"<!-- {banner} -->\n" + (md(t) if len(t) else "(none)") + "\n")
    # figures
    pv, pm = C.PRIMARY_VARIANT, C.PRIMARY_MODEL; bin_tasks = [t for t, d in C.TASKS.items() if len(d["classes"]) == 2 and t in res.task.unique()]
    if bin_tasks:
        fig, ax = plt.subplots(1, len(bin_tasks), figsize=(4.2 * len(bin_tasks), 4))
        for a_, t in zip(np.atleast_1d(ax), bin_tasks):
            for m in [x for x in C.MODELS if x in res.model.unique()]:
                f = os.path.join(exp, "oof", f"{t}__{pv}__{m}.csv")
                if os.path.exists(f):
                    o = pd.read_csv(f); pos = C.TASKS[t]["positive"]; fpr, tpr, _ = roc_curve((o.y == pos).astype(int), o[f"p_{pos}"]); r = res[(res.task == t) & (res.variant == pv) & (res.model == m)].iloc[0]
                    a_.plot(fpr, tpr, label=f"{m} AUC {r.auc:.2f}", lw=2 if m == pm else 1)
            a_.plot([0, 1], [0, 1], "k:", lw=.8); a_.set_title(t, fontsize=9); a_.set_xlabel("1 - specificity"); a_.set_ylabel("sensitivity"); a_.legend(fontsize=7)
        plt.tight_layout(); plt.savefig(os.path.join(F, "fig_roc.png"), dpi=200); plt.close()
        fig, ax = plt.subplots(1, len(bin_tasks), figsize=(4.2 * len(bin_tasks), 4))
        for a_, t in zip(np.atleast_1d(ax), bin_tasks):
            f = os.path.join(exp, "oof", f"{t}__{pv}__{pm}.csv")
            if os.path.exists(f):
                o = pd.read_csv(f); pos = C.TASKS[t]["positive"]; yb = (o.y == pos).astype(int).values; p = o[f"p_{pos}"].values; e = np.linspace(0, 1, 6); xs, ys = [], []
                for lo, hi in zip(e[:-1], e[1:]):
                    s = (p >= lo) & (p <= hi if hi == 1 else p < hi)
                    if s.sum() >= 3: xs.append(p[s].mean()); ys.append(yb[s].mean())
                a_.plot([0, 1], [0, 1], "k:"); a_.plot(xs, ys, "o-"); a_.set_title(f"{t} calibration ({pm}, {pv})", fontsize=8); a_.set_xlabel("predicted"); a_.set_ylabel("observed")
        plt.tight_layout(); plt.savefig(os.path.join(F, "fig_calibration.png"), dpi=200); plt.close()
    tasks = list(res.task.unique()); fig, ax = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.6))
    for a_, t in zip(np.atleast_1d(ax), tasks):
        d = res[(res.task == t) & (res.model == pm)].sort_values("variant")
        a_.errorbar(range(len(d)), d.auc, yerr=[d.auc - d.auc_lo, d.auc_hi - d.auc], fmt="o-", capsize=3); a_.set_xticks(range(len(d))); a_.set_xticklabels(d.variant); a_.axhline(0.5, color="k", ls=":", lw=.8); a_.set_title(f"{t} ({pm})", fontsize=8); a_.set_ylabel("AUC")
    plt.tight_layout(); plt.savefig(os.path.join(F, "fig_preprocessing_ablation.png"), dpi=200); plt.close()
    pf = os.path.join(exp, "oof", f"{C.PRIMARY_TASK}__{pv}__{pm}.csv")
    if os.path.exists(pf):
        o = pd.read_csv(pf); classes = sorted(o.y.unique()); P = o[[f"p_{c}" for c in classes]].values; ep = np.linspace(0, 0.6, 13); rows = [paz_summary(o.y.values, P, classes, e) for e in ep]
        fig, ax = plt.subplots(figsize=(4.5, 3.6)); ax.plot([r["coverage"] for r in rows], [r["acc_confident"] for r in rows], "o-"); ax.set_xlabel("coverage (cases outside the PAZ)"); ax.set_ylabel("accuracy on covered cases"); ax.set_title(f"PAZ trade-off: {C.PRIMARY_TASK}", fontsize=8)
        plt.tight_layout(); plt.savefig(os.path.join(F, "fig_paz_coverage_accuracy.png"), dpi=200); plt.close()
    part = cfg["partition"]; fig_preprocessing(F, part); fig_alignment(F, part)
    open(os.path.join(out, "RESULTS_SUMMARY.md"), "w").write(f"# {eid}\n\n**{banner}**\n\nPartition: {part} | repeats: {cfg['repeats']} | bootstrap: {cfg['boot']} | code sha256: {cfg['provenance']['code_sha256'][:12]} | "
        f"protocol sha256: {str(cfg['provenance']['protocol_sha256'])[:12]}\n\n## Table 2 (primary variant {pv})\n\n{md(t2)}\n\n## Table 3 (preprocessing ablation, {pm})\n\n{md(t3)}\n\n## Table 4 (paired comparisons, delta AUC)\n\n{md(comp) if len(comp) else '(none)'}\n")
    print("report written to", out); return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--exp"); ap.add_argument("--partition"); a = ap.parse_args(); main(a.exp or latest_exp(a.partition))

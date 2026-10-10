"""OncoSync staging API — read-only, serves the Case Review UI and QA Dashboard from REAL pipeline
outputs (never mock/invented data). Scope: internal research/QA + OncoSync staging, per the
deployment-scope decision recorded in deploy/README_DEPLOY.md. This is explicitly NOT a regulatory or
production-clinical deployment — every response carries that disclaimer, and severity_roi is
unconditionally redacted server-side (safety.py), not left to the caller's trust.

Reads directly from the real workspace (config.py's own C.WS, C.RESEARCH, C.PROTOCOL_FILE,
C.ACCESS_LEDGER) so there is exactly one source of truth with the actual bc.py pipeline — no
duplicated constants, no synced-by-hand copies of TASKS or paths.
"""
import os
import sys
import glob
import hashlib
from functools import lru_cache
from typing import Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, os.path.join(os.environ.get("BC_HOME", "/app"), "code", "src"))
import config as C  # noqa: E402  (path must be set before this import)

import safety  # noqa: E402

app = FastAPI(title="OncoSync Staging API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET"], allow_headers=["*"])

SCOPE_DISCLAIMER = (
    "Internal research/QA and OncoSync staging only. Not a regulatory submission artifact, not "
    "validated for and not authorized for production clinical use."
)


def _sha256(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _exp_root() -> str:
    return os.path.join(C.RESEARCH, "experiments")


def _latest_exp(partition_suffix: str = "_MS", kind_filter: Optional[str] = "confirmatory") -> Optional[str]:
    """Most recent experiment directory for the given partition, optionally filtered to a ledger kind
    (confirmatory/exploratory/development) via manifests/ms_access_log.csv. Falls back to the most
    recent directory by name if the ledger doesn't resolve a kind for it (e.g. DEV runs, which are
    never ledgered)."""
    root = _exp_root()
    if not os.path.isdir(root):
        return None
    cands = sorted(d for d in os.listdir(root) if d.endswith(partition_suffix))
    if not cands:
        return None
    if kind_filter is None:
        return cands[-1]
    if os.path.exists(C.ACCESS_LEDGER):
        led = pd.read_csv(C.ACCESS_LEDGER)
        ok = set(led[led.kind == kind_filter].exp_id)
        cands = [c for c in cands if c in ok] or cands
    return cands[-1]


@app.get("/health")
def health():
    return {"status": "ok", "scope": SCOPE_DISCLAIMER}


@app.get("/protocol/active")
def protocol_active():
    """The currently frozen protocol (docs/protocol.yaml as copied to research/experiments/protocol.yaml
    by `bc.py protocol`), its hash, and the version history visible in the governance ledger."""
    if not os.path.exists(C.PROTOCOL_FILE):
        raise HTTPException(404, "no protocol frozen yet (run: bc.py protocol)")
    proto = yaml.safe_load(open(C.PROTOCOL_FILE))
    sha = _sha256(C.PROTOCOL_FILE)
    history = []
    if os.path.exists(C.ACCESS_LEDGER):
        led = pd.read_csv(C.ACCESS_LEDGER)
        for h, grp in led.groupby("protocol_sha256"):
            history.append(dict(
                protocol_sha256=h,
                first_seen=grp.time.min(),
                last_seen=grp.time.max(),
                kinds=sorted(grp.kind.unique().tolist()),
                n_runs=int(len(grp)),
                active=(h == sha),
            ))
    return dict(active_sha256=sha, protocol=proto, version_history=history, scope=SCOPE_DISCLAIMER)


@app.get("/governance/ledger")
def governance_ledger(limit: int = Query(200, le=2000)):
    """The append-only MS-access ledger (manifests/ms_access_log.csv) verbatim. This endpoint is
    read-only by construction — there is no write path for the ledger anywhere in this API."""
    if not os.path.exists(C.ACCESS_LEDGER):
        return {"rows": [], "scope": SCOPE_DISCLAIMER}
    led = pd.read_csv(C.ACCESS_LEDGER).tail(limit)
    return {"rows": led.to_dict(orient="records"), "scope": SCOPE_DISCLAIMER}


@app.get("/qa/summary")
def qa_summary(exp_id: Optional[str] = None, variant: str = None):
    """PAZ ratio + AUC-by-task table for one experiment's primary variant (defaults to the most recent
    confirmatory MS run and C.PRIMARY_VARIANT), i.e. the same numbers the QA_Dashboard canvas shows."""
    exp_id = exp_id or _latest_exp()
    if exp_id is None:
        raise HTTPException(404, "no MS experiment found")
    variant = variant or C.PRIMARY_VARIANT
    path = os.path.join(_exp_root(), exp_id, "results.csv")
    if not os.path.exists(path):
        raise HTTPException(404, f"results.csv not found for {exp_id}")
    res = pd.read_csv(path)
    sub = res[res.variant == variant]
    rows = sub[["task", "model", "auc", "auc_lo", "auc_hi", "bal_acc", "paz_ratio", "paz_state",
                "coverage", "acc_confident", "kind"]].to_dict(orient="records")
    for r in rows:
        if r["task"] == safety.SEVERITY_TASK:
            r["clinical_warning"] = safety.SEVERITY_WARNING
    return dict(exp_id=exp_id, variant=variant, rows=rows, scope=SCOPE_DISCLAIMER)


@app.get("/cases")
def cases(exp_id: Optional[str] = None, limit: int = Query(25, le=200)):
    """Patient-level case list built by joining each task's real out-of-fold posteriors (oof/*.csv)
    on patient_id, primary variant/model only. severity_roi is passed through safety.redact_severity()
    unconditionally -- see safety.py for why there is no parameter to turn that off.

    This is a staging/QA view over real MIAS predictions, not a patient worklist tied to any real
    clinical system -- MIAS is a public research dataset; there is no live EHR or PACS integration
    here, and this endpoint should not be pointed at real-patient data without a fresh, separate
    safety and access review."""
    exp_id = exp_id or _latest_exp()
    if exp_id is None:
        raise HTTPException(404, "no MS experiment found")
    oof_dir = os.path.join(_exp_root(), exp_id, "oof")
    if not os.path.isdir(oof_dir):
        raise HTTPException(404, f"oof/ not found for {exp_id}")

    per_task = {}
    for task in C.TASKS:
        f = os.path.join(oof_dir, f"{task}__{C.PRIMARY_VARIANT}__{C.PRIMARY_MODEL}.csv")
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)
        pcols = [c for c in df.columns if c.startswith("p_")]
        keep = ["patient_id"] + pcols + (["paz_score", "paz_method"] if "paz_score" in df.columns else ["paz_margin"]) + ["state"]
        keep = [c for c in keep if c in df.columns]
        # one row per patient for this task (a patient can have >1 patch; keep the first, deterministic,
        # and labeled as such -- this is a staging simplification, not a clinical adjudication rule)
        per_task[task] = df[keep].drop_duplicates("patient_id", keep="first")

    if "abnormal_img" not in per_task:
        raise HTTPException(404, "abnormal_img oof missing — cannot build a patient-level case list")

    base = per_task["abnormal_img"].rename(columns=lambda c: c if c == "patient_id" else f"abnormal_img__{c}")
    for task, df in per_task.items():
        if task == "abnormal_img":
            continue
        renamed = df.rename(columns=lambda c: c if c == "patient_id" else f"{task}__{c}")
        base = base.merge(renamed, on="patient_id", how="left")

    base = base.head(limit)
    out = []
    for _, row in base.iterrows():
        case = dict(patient_id=row["patient_id"], tasks={})
        for task in C.TASKS:
            prefix = f"{task}__"
            rec = {k[len(prefix):]: v for k, v in row.items() if k.startswith(prefix) and pd.notna(v)}
            if not rec:
                continue
            case["tasks"][task] = safety.redact_severity(task, rec)
            case["tasks"][task]["disclaimer"] = safety.task_disclaimer(task)
        out.append(case)
    return dict(exp_id=exp_id, variant=C.PRIMARY_VARIANT, model=C.PRIMARY_MODEL, n=len(out),
                cases=out, scope=SCOPE_DISCLAIMER)

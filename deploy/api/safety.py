"""Server-side safety gating for the OncoSync staging API.

This exists because the OncoSync case-review UI (Main.dc.html) already enforces two things in the
client — bold protocol badges, and severity_roi always rendered with a hard-coded chance-performance
warning — but a client-side-only control is not a control: anyone who can call this API directly
(a script, a future integration, a different UI) bypasses it. The same two rules are enforced here,
server-side, so no caller can get an unqualified severity output no matter how they ask.

Ground truth for *why* severity_roi is gated, not just *that* it is: under confirmatory testing
(exp_20261009_003_MS, Protocol v1.1, hash d87e25ff84b15cec1cd45ef7b128f2d97eab6e23fe9eca61bb4ac1bd7a9727f0),
the malignancy task scores AUC 0.502, statistically indistinguishable from chance (permutation null).
That is a model-readiness fact, not a configuration flag, so it is not something a caller can turn off
by passing a parameter -- there is deliberately no override argument on redact_severity().
"""

SEVERITY_TASK = "severity_roi"
SEVERITY_WARNING = (
    "INDETERMINATE — malignancy (severity_roi) model performance is not distinguishable from chance "
    "under confirmatory testing (AUC 0.502, 95% CI spans 0.5). This output must not be used to support "
    "a diagnosis; mandatory specialist review applies to every case regardless of the value below."
)


def redact_severity(task: str, record: dict) -> dict:
    """Mutates nothing in place; returns a copy with severity_roi fields overridden for display.
    Raw probabilities are kept under raw_probabilities for audit/QA use, never surfaced as a display
    value — this mirrors the manuscript's own convention of reporting both the as-run and the
    governance-corrected numbers rather than silently picking one."""
    out = dict(record)
    if task == SEVERITY_TASK:
        out["display_label"] = "INDETERMINATE"
        out["requires_specialist_review"] = True
        out["clinical_warning"] = SEVERITY_WARNING
        out["raw_probabilities"] = {k: v for k, v in record.items() if k.startswith("p_")}
        for k in list(out.keys()):
            if k.startswith("p_"):
                del out[k]
    else:
        out["requires_specialist_review"] = bool(record.get("state") == "AMBIGUOUS")
    return out


def task_disclaimer(task: str) -> str:
    if task == SEVERITY_TASK:
        return SEVERITY_WARNING
    return (
        "Internal proof-of-concept model (MIAS cohort only, not externally validated). "
        "Decision support only — independent clinical judgment and local escalation pathways apply."
    )

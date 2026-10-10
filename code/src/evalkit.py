"""Repeated, nested, patient-grouped cross-validation; metrics; patient-level bootstrap; paired comparisons; calibration;
risk-coverage AURC (added 2026-10-10 -- see NOTES_paz_merge.md, section 2, for why this is a reconstruction, not a
recovered original)."""
import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, brier_score_loss
import config as C
from models import make_model, margin_score, entropy_score

def _align(P, mclasses, classes):
    full = np.zeros((P.shape[0], len(classes))); cl = list(classes)
    for j, c in enumerate(mclasses): full[:, cl.index(c)] = P[:, j]
    return full

def cv_repeat(X, y, groups, model_name, seed, n_splits=C.N_SPLITS, inner=C.INNER_SPLITS):
    classes = np.unique(y); P = np.zeros((len(y), len(classes))); picks = []
    per_class = min(len(set(groups[y == c])) for c in classes); k = max(2, min(n_splits, per_class)); ki = max(2, min(inner, per_class - 1))
    outer = StratifiedGroupKFold(k, shuffle=True, random_state=seed)
    for tr, te in outer.split(X, y, groups):
        assert set(groups[tr]).isdisjoint(groups[te]), "patient leakage between train and test"
        est, grid = make_model(model_name, seed)
        if grid:
            cvs = list(StratifiedGroupKFold(ki, shuffle=True, random_state=seed).split(X[tr], y[tr], groups[tr]))
            gs = GridSearchCV(est, grid, cv=cvs, scoring="balanced_accuracy", refit=True, error_score=0.0); gs.fit(X[tr], y[tr]); m = gs.best_estimator_; picks.append(gs.best_params_)
        else: m = est.fit(X[tr], y[tr])
        P[te] = _align(m.predict_proba(X[te]), m.classes_, classes)
    return P, picks

def run_cv(X, y, groups, model_name, seed=C.SEED, n_repeats=C.N_REPEATS, n_jobs=1):
    res = Parallel(n_jobs=n_jobs)(delayed(cv_repeat)(X, y, groups, model_name, seed + r) for r in range(n_repeats))
    return np.stack([r[0] for r in res]), np.unique(y), [r[1] for r in res]

def ece(yb, p, bins=10):
    edges = np.linspace(0, 1, bins + 1); tot = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        s = (p >= a) & (p < b if b < 1 else p <= b)
        if s.any(): tot += s.mean() * abs(yb[s].mean() - p[s].mean())
    return float(tot)

def metrics(y, P, classes, positive=None):
    y = np.asarray(y); classes = np.asarray(classes); pred = classes[P.argmax(1)]
    out = dict(acc=float((pred == y).mean()), bal_acc=float(balanced_accuracy_score(y, pred)), f1_macro=float(f1_score(y, pred, average="macro")))
    if len(np.unique(y)) < 2: return {**out, "auc": float("nan")}
    if len(classes) == 2:
        j = list(classes).index(positive); yb = (y == positive).astype(int); p = P[:, j]
        out.update(auc=float(roc_auc_score(yb, p)), sens=float(((pred == positive) & (yb == 1)).sum() / max(yb.sum(), 1)),
                   spec=float(((pred != positive) & (yb == 0)).sum() / max((1 - yb).sum(), 1)), brier=float(brier_score_loss(yb, p)), ece=ece(yb, p))
    else:
        present = [c for c in classes if (y == c).any()]
        out["auc"] = float(roc_auc_score(y, P, multi_class="ovr", labels=classes, average="macro")) if len(present) == len(classes) else float("nan")
    return out

def boot_ci(y, P, groups, fn, n_boot=C.N_BOOT, seed=C.SEED, alpha=0.05):
    """Patient-level bootstrap percentile CI of fn(y, P)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; vals = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); 
        if len(np.unique(y[ii])) < 2: continue
        v = fn(y[ii], P[ii])
        if np.isfinite(v): vals.append(v)
    return (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))) if len(vals) > 20 else (float("nan"), float("nan"))

def paired_boot(y, P1, P2, groups, fn, n_boot=C.N_BOOT, seed=C.SEED):
    """Paired patient-level bootstrap of fn(P1) - fn(P2) on identical samples. Returns (delta, lo, hi, two-sided p)."""
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {g: np.where(groups == g)[0] for g in ug}; d = []
    for _ in range(n_boot):
        ii = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        if len(np.unique(y[ii])) < 2: continue
        a, b = fn(y[ii], P1[ii]), fn(y[ii], P2[ii])
        if np.isfinite(a) and np.isfinite(b): d.append(a - b)
    d = np.array(d);
    if len(d) < 20: return (float("nan"),) * 4
    p = 2 * min((d <= 0).mean(), (d >= 0).mean()); return float(fn(y, P1) - fn(y, P2)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0))

# ---------------------------------------------------------------- Risk-coverage AURC
# Added 2026-10-10. "AURC"/"AURC gain" appear in the manuscript (Sections 3.2-3.6) and in a legacy CSV
# (results_MS_003.csv, tied to the retired/unreproducible exp_20261008_003_MS run under the RETIRED
# protocol hash) but were found, by direct grep, to have NO generating code anywhere in this pipeline.
# This is a NEW implementation of the standard selective-classification "risk-coverage AURC" (El-Yaniv &
# Wiener 2010; Geifman & El-Yaniv 2017), NOT a recovery of whatever produced the legacy numbers -- the
# legacy run's per-case artifacts no longer exist to check against. One internal consistency check was
# possible and is documented in NOTES_paz_merge.md: the legacy "aurc_baseline" column is, for every row,
# an exact integer/81 for severity_roi (n=81 in that MS cohort), consistent with "baseline = this model's
# own overall error rate" -- which is exactly what flat_baseline_risk() returns below. That is corroborating
# evidence for the general shape of the original definition, not proof the two implementations agree in
# every detail (confidence-ranking convention, interpolation, tie-breaking are all still unverifiable).

def confidence_score(proba, method="auto"):
    """Higher = more confident, on a common direction regardless of margin vs entropy. method='auto'
    resolves margin (2-class) vs entropy (3+-class) from proba.shape[1], same rule as models.paz_gate."""
    K = proba.shape[1]
    resolved = ("margin" if K == 2 else "entropy") if method == "auto" else method
    if resolved == "margin":
        return margin_score(proba), resolved
    if resolved == "entropy":
        return 1.0 - entropy_score(proba), resolved
    raise ValueError(f"unknown method: {method}")

def risk_coverage_curve(y, proba, classes, method="auto"):
    """Sort cases by descending confidence; at each coverage level (fraction kept, most-confident-first),
    risk = error rate among the kept cases. Returns (coverage, risk, method_used), each length n."""
    classes = np.asarray(classes); y = np.asarray(y); pred = classes[proba.argmax(1)]
    correct = (pred == y).astype(float)
    conf, used = confidence_score(proba, method)
    order = np.argsort(-conf)
    correct_sorted = correct[order]
    n = len(correct_sorted)
    cum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n
    risk = 1.0 - cum_correct / np.arange(1, n + 1)
    return coverage, risk, used

def flat_baseline_risk(y, proba, classes):
    """Risk of rejecting independently of the score (flat/random rejection): constant at the model's own
    overall error rate at every coverage level, so its AURC is just that error rate."""
    classes = np.asarray(classes); y = np.asarray(y); pred = classes[proba.argmax(1)]
    return float(1.0 - (pred == y).mean())

def aurc(y, proba, classes, method="auto"):
    """Area under the risk-coverage curve (trapezoidal over coverage in (0,1]). Units: probability
    (error rate), lower = better (a more effective confidence ranking concentrates errors at low coverage)."""
    coverage, risk, used = risk_coverage_curve(y, proba, classes, method)
    trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 renamed trapz to trapezoid
    return float(trapz(risk, coverage)), used

def aurc_summary(y, proba, classes, method="auto"):
    """aurc: this model's risk-coverage AURC (lower = better).
    aurc_random: the flat/random-rejection baseline AURC (= overall error rate).
    aurc_gain: aurc_random - aurc (positive = the PAZ score ranks errors into low-confidence cases
    better than chance; this is the opposite sign convention from the legacy aurc_gain column, which
    computed aurc - aurc_baseline -- see NOTES_paz_merge.md for why the legacy sign could not be verified
    and this skill instead follows the sign convention standard in the selective-classification literature,
    where a positive gain means an improvement)."""
    area, used = aurc(y, proba, classes, method)
    base = flat_baseline_risk(y, proba, classes)
    return dict(aurc=area, aurc_random=base, aurc_gain=base - area, method=used)

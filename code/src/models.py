"""Models: NeuroSync SLDA (Tikhonov-regularised LDA), baselines (logistic regression, RBF-SVM, random forest), PAZ ambiguity gate."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
import config as C

class NeuroSyncSLDA(BaseEstimator, ClassifierMixin):
    """Pooled within-class covariance S_W + lam*I (NeuroSync SLDA, Tikhonov regularisation). Inputs must be standardised.
    uniform_priors=True uses equal class priors (balanced decision rule for imbalanced tasks)."""
    def __init__(self, lam=1.0, uniform_priors=True): self.lam = lam; self.uniform_priors = uniform_priors

    def fit(self, X, y):
        X, y = check_X_y(X, y); self.classes_ = unique_labels(y); p = X.shape[1]
        self.means_ = np.vstack([X[y == c].mean(0) for c in self.classes_])
        Sw = sum(((X[y == c] - m).T @ (X[y == c] - m)) for c, m in zip(self.classes_, self.means_)) / max(len(y) - len(self.classes_), 1)
        pri = np.full(len(self.classes_), 1.0 / len(self.classes_)) if self.uniform_priors else np.array([(y == c).mean() for c in self.classes_])
        self.coef_ = np.linalg.solve(Sw + self.lam * np.eye(p), self.means_.T).T
        self.intercept_ = -0.5 * np.einsum("kp,kp->k", self.means_, self.coef_) + np.log(pri); return self

    def decision_function(self, X): check_is_fitted(self); return check_array(X) @ self.coef_.T + self.intercept_
    def predict_proba(self, X):
        s = self.decision_function(X); s = s - s.max(1, keepdims=True); e = np.exp(s); return e / e.sum(1, keepdims=True)
    def predict(self, X): return self.classes_[self.decision_function(X).argmax(1)]

def make_model(name, seed=C.SEED):
    """-> (sklearn Pipeline, param grid for the inner CV or None)."""
    pre = [("imp", SimpleImputer(strategy="median")), ("var", VarianceThreshold(0.0)), ("sc", StandardScaler())]
    if name == "slda": return Pipeline(pre + [("clf", NeuroSyncSLDA())]), {"clf__lam": C.LAMBDA_GRID}
    if name == "logreg": return Pipeline(pre + [("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]), {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    if name == "svm": return Pipeline(pre + [("clf", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed))]), {"clf__C": [0.1, 1.0, 10.0]}
    if name == "rf": return Pipeline(pre + [("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=seed, n_jobs=1))]), None
    raise ValueError(name)

# ---------------------------------------------------------------- PAZ scoring functions
# Merged 2026-10-10 from PAZ_Entropy_Margin_Patch.zip (previously unmerged; validated against
# the standalone epsilon_sweep.py script used for the Protocol v1.1 manuscript analysis --
# see NOTES_paz_merge.md in this directory for the validation record). make_model() and
# NeuroSyncSLDA above are UNCHANGED from the original file; only the PAZ gate section below
# is replaced/extended. The patch bundle's own ComBatTransformer is intentionally NOT merged:
# it depends on neuroHarmonize, is unused anywhere in this breast-imaging pipeline, and is not
# part of this change.

def margin_score(proba):
    """Top-1 minus top-2 posterior probability. Confidence score: HIGH = confident, LOW = ambiguous.
    Binary-oriented: for 3+ classes it ignores all probability mass below the top two, which is the
    specific weakness that showed up as a non-monotonic PAZ coverage-accuracy curve on density_img."""
    s = np.sort(proba, axis=1)
    return s[:, -1] - s[:, -2]


def entropy_score(proba, eps=1e-12):
    """Normalized Shannon entropy of the FULL posterior, scaled to [0, 1] by log(K) (natural log).
    Uncertainty score: LOW = confident (near one-hot), HIGH = ambiguous (near uniform).
    Uses every class's probability, so it is the natural choice once K > 2 (e.g. density_img, 3 classes) --
    note the comparison direction is the OPPOSITE of margin_score (high entropy = ambiguous, not low)."""
    p = np.clip(proba, eps, 1.0)
    H = -(p * np.log(p)).sum(axis=1)
    K = proba.shape[1]
    return H / np.log(K)


def paz_gate(proba, epsilon=None, method="auto"):
    """PAZ ambiguity gate.
    method: 'margin' (default epsilon C.PAZ_EPSILON, flag when score < epsilon)
            'entropy' (default epsilon C.PAZ_EPSILON_ENTROPY, flag when score > epsilon)
            'auto'    (margin for 2-class problems, entropy for 3+ class problems -- resolved from proba.shape[1])
    Returns (score, is_ambiguous, method_used). The method actually used is always returned explicitly,
    because 'auto' resolves silently otherwise and that resolution needs to be recorded with the result."""
    K = proba.shape[1]
    resolved = ("margin" if K == 2 else "entropy") if method == "auto" else method
    if resolved == "margin":
        eps = C.PAZ_EPSILON if epsilon is None else epsilon
        score = margin_score(proba)
        return score, score < eps, resolved
    elif resolved == "entropy":
        eps = C.PAZ_EPSILON_ENTROPY if epsilon is None else epsilon
        score = entropy_score(proba)
        return score, score > eps, resolved
    raise ValueError(f"unknown PAZ method: {method}")


def select_epsilon(proba, y, classes, method="margin", target_coverage=0.5, grid=None, seed=C.SEED):
    """Pick epsilon EMPIRICALLY from held-out (e.g. DEV out-of-fold) probabilities: the threshold that
    maximises accuracy among confident (non-ambiguous) predictions, subject to covering at least
    target_coverage of cases. Do not hand-pick an entropy threshold by guesswork -- the entropy scale
    does not transfer intuition from the margin scale (a fairly confident 3-class prediction like
    [0.7, 0.2, 0.1] already has normalized entropy ~0.73, which would look 'low confidence' on a margin-
    trained intuition). Always call this per task, on DEV data only, before freezing a value into protocol.yaml."""
    classes = np.asarray(classes)
    if grid is None:
        grid = np.linspace(0.05, 0.45, 9) if method == "margin" else np.linspace(0.30, 0.95, 14)
    pred = classes[proba.argmax(1)]
    best = None
    for eps in grid:
        _, amb, _ = paz_gate(proba, eps, method)
        conf = ~amb
        cov = float(conf.mean())
        if cov < target_coverage or conf.sum() == 0:
            continue
        acc = float((pred[conf] == y[conf]).mean())
        if best is None or acc > best[1]:
            best = (float(eps), acc, cov)
    if best is None:
        # nothing met target_coverage; fall back to the midpoint and flag it clearly in the return
        mid = float(grid[len(grid) // 2])
        return mid, float("nan"), float("nan")
    return best  # (epsilon, accuracy_confident_at_that_epsilon, coverage_at_that_epsilon)


def paz_task_validity(observed_auc, permuted_auc_ci):
    """Null guard: is this task's observed AUC actually distinguishable from the permutation-control null?
    Returns False when it is NOT distinguishable -- in that case, the PAZ coverage-accuracy curve for this
    task should be reported with an explicit 'not meaningful for this task' annotation, because posterior
    margin or entropy cannot be expected to track correctness on a task where the underlying classifier
    has no real decision boundary."""
    lo, hi = permuted_auc_ci
    return not (lo <= observed_auc <= hi)


def paz_summary(y, proba, classes, epsilon=None, limit=C.PAZ_RATIO_LIMIT, method="auto"):
    score, amb, used = paz_gate(proba, epsilon, method)
    pred = np.asarray(classes)[proba.argmax(1)]; y = np.asarray(y); conf = ~amb
    return dict(
        method=used,
        epsilon=(epsilon if epsilon is not None else (C.PAZ_EPSILON if used == "margin" else C.PAZ_EPSILON_ENTROPY)),
        paz_ratio=float(amb.mean()),
        batch_state="AMBIGUOUS" if amb.mean() > limit else "STABLE",
        coverage=float(conf.mean()),
        acc_all=float((pred == y).mean()),
        acc_confident=float((pred[conf] == y[conf]).mean()) if conf.any() else float("nan"),
    )

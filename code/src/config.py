"""Central configuration. HOME is derived from where the project was unzipped, so no setup is needed on a Desktop install.
The workspace mirrors the S3 bucket layout of the IS360 Breast Imaging Platform architecture (raw / processed / ml / reports)."""
import os
from pathlib import Path

HOME = os.getenv("BC_HOME") or str(Path(__file__).resolve().parents[2])      # .../IS360_Breast_PoC
WS = os.path.join(HOME, "workspace")
TENANT, SOURCE = "is360", "mias"

def _p(*a): return os.path.join(WS, *a)
RAW  = _p("raw", f"tenant={TENANT}", f"source={SOURCE}")
PROC = _p("processed", f"tenant={TENANT}", f"source={SOURCE}")
RESEARCH = _p("research", f"tenant={TENANT}")      # datasets, partitions, experiments, manuscript assets
MODELS   = _p("models", f"tenant={TENANT}")
RESULTS  = _p("results", f"tenant={TENANT}")
REP  = _p("reports", f"tenant={TENANT}")
AUDIT = _p("audit")
MAN, LOGS, TMP = _p("manifests"), _p("logs"), _p(".tmp")
DIRS = [RAW, PROC, RESEARCH, MODELS, RESULTS, REP, AUDIT, MAN, LOGS, TMP, _p("inbox_extracted")]
DEV_FRAC = 0.30                                    # Option A: two-way patient-level split (development / manuscript)

PIPELINE_VERSION = "pp_v1.0.0"
ANNOT_VERSION = "ann_v1"
DATASET_ID = "ds_brca_mias_001"

# --- image / preprocessing ---
PIXEL_SPACING_MM = 0.2                  # MIAS mini database (200 micron)
MODEL_SIZE = 512
VARIANTS = ["P0", "P1", "P2", "P3", "P4"]
PRIMARY_VARIANT = "P4"
# --- patches (lesion ROI level) ---
PATCH_SCALE, PATCH_MIN, PATCH_MAX, PATCH_OUT = 2.4, 48, 320, 96
NORMAL_PER_LESION = 1
# --- modelling ---
SEED = 42
N_SPLITS, N_REPEATS = 5, 10             # patient-grouped repeated CV (use --repeats 3 for a quick run)
INNER_SPLITS = 3
LAMBDA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
MODELS = ["slda", "logreg", "svm", "rf"]
PRIMARY_MODEL = "slda"
PAZ_EPSILON, PAZ_RATIO_LIMIT = 0.15, 0.15
# Added 2026-10-10 with the PAZ patch merge: fallback default for paz_gate(method='entropy'|'auto')
# on 3+ class tasks, used only when no explicit epsilon is passed. Set to the Protocol v1.1 value
# selected by select_epsilon() on DEV data for density_img (see NOTES_paz_merge.md).
PAZ_EPSILON_ENTROPY = 0.90
N_BOOT = 2000

TASKS = {   # pre-specified in docs/03_STUDY_PROTOCOL.md
    "severity_roi":         dict(level="patch", label="severity", classes=["B", "M"], positive="M", desc="Benign vs malignant (annotated lesion ROI)"),
    "lesion_vs_normal_roi": dict(level="patch", label="lesion", classes=["normal", "lesion"], positive="lesion", desc="Lesion ROI vs density-matched normal ROI"),
    "abnormal_img":         dict(level="image", label="abn", classes=["normal", "abnormal"], positive="abnormal", desc="Normal vs abnormal (whole image)"),
    "density_img":          dict(level="image", label="bg", classes=["F", "G", "D"], positive=None, desc="Background tissue F / G / D (whole image)"),
}
PRIMARY_TASK = "severity_roi"

def ensure_dirs():
    for d in DIRS: os.makedirs(d, exist_ok=True)

def image_dir(kind, patient_id, image_id, variant=None):
    """kind: raw | std | pre"""
    if kind == "raw": return os.path.join(RAW, "images", patient_id, image_id)
    if kind == "std": return os.path.join(PROC, "standardized", patient_id, image_id)
    if kind == "pre": return os.path.join(PROC, "preprocessed", PIPELINE_VERSION, f"variant={variant}", patient_id, image_id)
    raise ValueError(kind)

def feature_dir(variant): return os.path.join(PROC, "features", PIPELINE_VERSION, f"variant={variant}")

PARTITION_SEED, DEV_FRACTION = 20260921, 0.30      # Option A: two-way, patient-level, stratified by worst class
DATASET_ROOT = os.path.join(RESEARCH, "datasets", DATASET_ID)
PROTOCOL_FILE = os.path.join(RESEARCH, "experiments", "protocol.yaml")
ACCESS_LEDGER = os.path.join(MAN, "ms_access_log.csv")

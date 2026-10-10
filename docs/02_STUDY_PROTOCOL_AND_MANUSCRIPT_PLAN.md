# Study protocol and manuscript plan

## 1. Question
Can an interpretable, calibrated pipeline (standardised preprocessing + handcrafted texture features + NeuroSync SLDA with a PAZ ambiguity gate)
characterise breast tissue and annotated lesions in the MIAS database, and which preprocessing steps contribute?

## 2. Data and partitions (Architecture v2, Option A)
MIAS: 322 mediolateral-oblique films, 161 patients (56 benign, 53 normal, 52 malignant by worst breast). Split at patient level, stratified by worst class,
hash-ranked with seed 20260921: DEV 49 patients / 98 images (ROIs 21 benign, 16 malignant); MS 112 patients / 224 images (ROIs 47 benign, 35 malignant).
DEV is used for design and debugging; MS is analysed once under the frozen protocol. Expected precision at AUC 0.80 with 35 malignant / 47 benign ROIs: about +/-0.10 (Hanley-McNeil, ignoring clustering, so realistically wider).

## 3. Tasks and endpoints
| Task | Unit | Classes | Role |
|---|---|---|---|
| severity_roi | annotated lesion ROI | benign vs malignant | **primary** (AUC, malignant positive) |
| lesion_vs_normal_roi | ROI | lesion vs density-matched normal ROI | secondary |
| abnormal_img | image | normal vs abnormal | secondary |
| density_img | image | F / G / D (MIAS background class) | secondary |
ROI location is given by the annotation, so the primary task is classification of a marked region, not detection.

## 4. Methods (as implemented)
* Standardise: 8-bit values kept in a 16-bit PNG container; laterality from file parity; view assumed MLO; outcome-blind QC.
* Preprocessing ladder: P0 resize; P1 + chest wall on the left; P2 + breast mask, label/tape removal, crop; P3 + pectoral muscle removal (edge/threshold + robust line fit; failures recorded, image unchanged); P4 + percentile clipping and CLAHE. Coordinates follow every step through transform.json.
* Features: first-order (12), gradient (3), GLCM at distances 1, 2, 4 (18), uniform LBP P8R1 and P16R2 (30), 16-bin histogram (images only). ROI patch: square window of 2.4 x radius (48-320 px in model space) resized to 96 px. Lesion radius is not a feature.
* Models: NeuroSync SLDA (pooled covariance + lambda I, uniform priors), logistic regression, RBF-SVM, random forest; imputation, variance filter and standardisation inside every training fold.
* Validation: 5-fold patient-grouped stratified CV repeated 10 times; lambda / C chosen in a 3-fold patient-grouped inner loop; an assertion stops the run if a patient appears in train and test.
* Uncertainty: patient-level bootstrap 95% CI (2000 draws) on the repeat-averaged out-of-fold probabilities; paired patient-level bootstrap for delta-AUC (each variant vs P0 for SLDA; each baseline vs SLDA at P4); PAZ: ambiguous when the top-two probability margin is below 0.15; ECE and Brier for binary tasks.
* Controls: label permutation, coordinate alignment, leakage assertion.

## 5. Freeze procedure
1. Ingest, QC, `partition` (hashes recorded). 2. Work on DEV only; log every change. 3. Edit `docs/protocol.yaml` if needed, then `python bc.py protocol` (SHA-256).
4. `experiments --partition MS` once (confirmatory). 5. `report`. Later MS runs are exploratory and must be labelled as such in the paper.

## 6. Expectation management (from development runs on DEV only, wide intervals, not results)
DEV pilot runs gave AUC about 0.78 for lesion vs normal ROI and about 0.83-0.89 for density, benign vs malignant near chance (37 ROIs), and normal vs abnormal image between 0.4 and 0.73 depending on variant.
The primary endpoint may therefore be null on MS. That is a legitimate, publishable finding if reported with intervals; do not switch the primary endpoint after seeing MS results.

## 7. Manuscript outline
1. Introduction: MIAS as a reference database; need for transparent preprocessing and uncertainty reporting.
2. Methods: dataset and partition (Fig. cohort flow from `partition_freeze.json` and QC counts), preprocessing (Fig. `fig_preprocessing_examples`), features, models, validation, statistics, code/data availability (MIAS licence).
3. Results: Table 1 dataset; Table 2 main results; Table 3 ablation (`fig_preprocessing_ablation`); Table 4 paired comparisons; PAZ (`fig_paz_coverage_accuracy`); calibration and ROC (`fig_calibration`, `fig_roc`); annotation alignment (`fig_annotation_alignment`); preprocessing failures (pectoral success rate).
4. Discussion: what preprocessing contributes; limits (1990s film, 161 patients, MLO only, circle annotations, single centre, handcrafted features, no external validation); next steps (external dataset, learned features).
5. Checklists: CLAIM, TRIPOD+AI, STARD items; ethics statement for public de-identified data (confirm with your institution).

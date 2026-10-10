# User guide

## 1. Install
Python 3.9+ (3.11 recommended). Windows: use WSL2 Ubuntu and keep the project inside the Linux home folder (fast disk). 8 GB RAM, 10 GB disk.
```
cd IS360_Breast_PoC/code && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python bc.py setup && python bc.py check && python bc.py test
```
## 2. Data
Copy `archive.zip` (or `all-mias_tar.gz`) to `inbox/`. `python bc.py ingest --src ../inbox/archive.zip` expects 322 images, 161 patients, 330 annotation rows;
it stops if any number differs, and (when h5py is installed) confirms the HDF5 copy equals the PGM files.

## 3. Processing (outcome-blind, all images)
| Command | What it does | Approx. time on 1 core |
|---|---|---|
| `standardize` | canonical image, chest-wall side (file rule vs image evidence), breast area, film labels, near-duplicates, QC flag | 3 min |
| `partition` | seeded patient-level split into DEV (~30%) and MS (~70%), stratified by worst class; writes hashes; refuses to overwrite | seconds |
| `preprocess [--partition DEV]` | five cumulative variants P0..P4 (orientation, artefact removal + crop, pectoral removal, CLAHE) with transform.json | ~1.5 s per image for all five |
| `patches` | lesion ROIs (119 with coordinates) + density-matched normal ROIs, sampled inside the same partition | 1 min |
| `features [--variants P4]` | first-order, GLCM, LBP, gradient, histogram features (whole image and ROI) | ~20 s per variant |
All steps are resumable: re-run the same command to continue.

## 4. Checks
`python bc.py validate` (completion, QC, pectoral success) and `python bc.py align` (lesion coordinates: with the correct mapping 98% of mass lesions are brighter than their surrounding ring, against 57-71% for wrong mappings).

## 5. Experiments
* Development: `python bc.py experiments --partition DEV --quick` (2 repeats) or full settings. Use it to debug, choose thresholds, look at figures. Not reported as results.
* Manuscript: edit `docs/protocol.yaml` if (and only if) needed **before** `python bc.py protocol`. Then `python bc.py experiments --partition MS --jobs 4`.
  The first MS run for a given protocol hash is `confirmatory`; every later run is `exploratory` and labelled so in all tables. Full settings: 10 x 5-fold repeats, 2000 bootstrap draws.
  Time estimate (1 core): about 1 hour; with `--jobs 8` about 10-15 minutes.
* Controls: `python bc.py permute --partition DEV --task severity_roi` (null AUC must be near 0.5).
## 6. Report
`python bc.py report --partition MS` writes tables (Markdown + CSV), figures (PNG, 200 dpi) and `RESULTS_SUMMARY.md` to `workspace/research/tenant=is360/manuscript/<exp_id>/`.
Tables/figures are generated only from the experiment folder, never recomputed.

## 7. Troubleshooting
| Symptom | Fix |
|---|---|
| ingest count mismatch | wrong archive/partial download; re-download MIAS |
| `partition already frozen` | intended. A re-split needs `--force` and a written justification in CHANGELOG.txt |
| `MS run needs a frozen protocol` | run `python bc.py protocol` first |
| `partition file MS.csv changed after freeze` | someone edited a frozen file; restore from backup / re-extract |
| pectoral removal < 75% | report it; do not tune on MS. Tune only on DEV and re-freeze the pipeline version |
| slow SVM | `--models slda,logreg,rf` for development runs |

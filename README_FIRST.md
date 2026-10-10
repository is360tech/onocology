# IS360 Breast Imaging PoC - desktop research package (Phase 1)

Research use only. Not a medical device. Data: MIAS mini database (research-only licence; cite Suckling et al., 1994).
Architecture: IS360 Breast Architecture v2 (Option A: patient-level DEV 30% / MANUSCRIPT 70% split). The `workspace/` folder mirrors the
future S3 buckets (raw, processed, research, models, results, reports, audit), so nothing is reorganised when the portal arrives.

## Quick start (Windows WSL2 / Linux / macOS terminal)
```
cd IS360_Breast_PoC/code
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python bc.py setup && python bc.py check          # T0
python bc.py test                                 # T1  synthetic self-test (~1 min)
# put archive.zip or all-mias_tar.gz into ../inbox/
python bc.py ingest --src ../inbox/archive.zip    # 322 images, 161 patients
python bc.py standardize                          # canonical images + outcome-blind QC
python bc.py partition                            # FREEZE the DEV / MS split (once)
python bc.py preprocess                           # variants P0..P4
python bc.py patches && python bc.py features     # ROI table + features
python bc.py validate && python bc.py align       # T2-T4 acceptance checks
# --- development (free to repeat) ---
python bc.py experiments --partition DEV --quick
python bc.py permute --partition DEV
# --- manuscript (once, after the protocol is frozen) ---
python bc.py protocol                             # freezes docs/protocol.yaml (prints its SHA-256)
python bc.py experiments --partition MS --jobs 4  # first run = CONFIRMATORY; later runs are logged as EXPLORATORY
python bc.py report --partition MS
```
Read: `docs/01_USER_GUIDE.md` (steps), `docs/02_STUDY_PROTOCOL_AND_MANUSCRIPT_PLAN.md`, `docs/03_TEST_PLAN.md` (T0-T7 with sign-off),
`docs/04_GOVERNANCE_AND_LICENCE.md`, `docs/05_FOLDER_STRUCTURE.md`.

## Rules
* Algorithm design and threshold tuning: DEV partition only. The MS partition is analysed once under the frozen protocol.
* Do not look at MS results (or run experiments on MS) before the protocol is frozen. Every MS run is written to `workspace/manifests/ms_access_log.csv`.
* Report all pre-specified results, including negative ones, with confidence intervals.
* Do not redistribute MIAS images. Cite the database in every paper.

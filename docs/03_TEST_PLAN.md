# Test plan (T0-T7)

| ID | Purpose | Command | Pass criterion |
|---|---|---|---|
| T0 | Environment | `python bc.py check` | 0 FAIL |
| T1 | Code self-test on synthetic data | `python bc.py test` | prints ALL TESTS PASSED (parsing incl. *NOTE 3* lines, coordinate transform through P0-P4, deterministic and tamper-evident partition, features, SLDA vs LDA, PAZ, grouped CV + permutation, governance ledger) |
| T2 | Ingest | `python bc.py ingest ...` | 322 images, 161 patients, 330 annotation rows; HDF5 copy identical (322/322) |
| T3 | Standardise + QC | `python bc.py standardize` | QC pass >= 98% (observed 320/322 = 99.4%; 2 full-frame images flagged); chest-side rule agrees with image evidence in >= 99% (observed 321/322) |
| T4 | Geometry | `python bc.py align --partition ALL` | correct mapping clearly best (observed: 98% of 58 mass lesions brighter than the surrounding ring vs 57-71% for wrong mappings) |
| T5 | Preprocessing + features | `python bc.py validate` | all variants complete; pectoral removal success >= 75% (observed on DEV: 83/98 = 84.7%) |
| T6 | Leakage control | `python bc.py permute --partition DEV` | permuted AUC mean within 0.5 +/- 0.08 (observed 0.50, sd 0.06, 12 permutations) |
| T7 | Governance | try `experiments --partition MS` before `protocol` | refused; after freezing: first run confirmatory, later exploratory, each in ms_access_log.csv |

Numbers marked "observed" were produced on the development machine on the real MIAS files. Re-check them on your machine; report deviations, do not adjust thresholds.

## Sign-off
| Test | Date | Run by | PASS/FAIL | Evidence (file) | Lead initials |
|---|---|---|---|---|---|
| T0 | | | | | |
| T1 | | | | | |
| T2 | | | | | |
| T3 | | | | | |
| T4 | | | | | |
| T5 | | | | | |
| T6 | | | | | |
| T7 | | | | | |

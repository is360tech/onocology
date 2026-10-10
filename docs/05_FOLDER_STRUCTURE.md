# Desktop folder structure

```
IS360_Breast_PoC/
├── README_FIRST.md
├── code/            bc.py (entry point) | src/ | scripts/ | tests/ | requirements.txt
├── docs/            guides, protocol.yaml (edit BEFORE freezing), test plan, governance
├── inbox/           MIAS archive (input only)
└── workspace/       mirrors the S3 buckets of Architecture v2
    ├── raw/tenant=is360/source=mias/          images/<patient>/<image_id>/original.pgm | licence/ | source_info.json      (bucket: raw)
    ├── processed/tenant=is360/source=mias/    standardized/  preprocessed/pp_v1.0.0/variant=P0..P4/  annotations/ann_v1/  features/  (bucket: processed)
    ├── research/tenant=is360/                 datasets/ds_brca_mias_001/{partitions/DEV.csv, MS.csv, partition_freeze.json}
    │                                          experiments/{protocol.yaml, exp_.../{results.csv, comparisons.csv, oof/, config.json}}
    │                                          manuscript/exp_.../{tables/, figures/, RESULTS_SUMMARY.md}                       (bucket: research)
    ├── models/  results/  reports/  audit/    (reserved: used by the portal phase; reports/ holds validation_report.md)
    └── manifests/                             ingest_manifest.csv, qc_summary.csv, status.csv, preprocess_summary_*.csv, ms_access_log.csv
```
IDs: patient = mias_pNNN (pair of films), image = img_<sha256[:12]>. Chest-wall side, laterality (even file number = left) and view (assumed MLO) are recorded with their source.
Later move to S3: `aws s3 sync workspace/raw s3://is360-breast-raw-...` (one bucket per top-level folder); only after the licence review in docs/04.

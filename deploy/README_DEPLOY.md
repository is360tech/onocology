# OncoSync / IS360 Breast PoC — deployment

## Scope (read this before applying anything)

This deployment is for two purposes, both confirmed explicitly before any of it was built:

1. **Internal research/QA service** — a containerized `bc.py` pipeline + a read-only inference API,
   for your team to run experiments, QA, and feed the OncoSync case-review UI with real (not mock)
   pipeline output.
2. **Staging environment for OncoSync** — infra behind the actual case-review / QA dashboard UI,
   reachable by real practitioners for review and feedback, still running on the current PoC models.

**This is explicitly NOT a production-clinical deployment.** It is not validated for, and is not
authorized for, supporting a real diagnosis. Every API response carries that disclaimer in a `scope`
field, and it isn't just a label: `severity_roi` (the malignancy task) is redacted server-side on every
response regardless of its raw probability, because under confirmatory testing it scores AUC 0.502 —
indistinguishable from chance (see `api/safety.py` and the manuscript's Section 3.7 / Appendix). If this
ever needs to become a production-clinical deployment, that is a new decision requiring a fresh safety,
regulatory, and access review — nothing here should be read as having pre-cleared that path.

## What's here

```
deploy/
  Dockerfile.pipeline       bc.py pipeline image (full ML stack: sklearn, scikit-image, etc.)
  Dockerfile.api            thin read-only API image (FastAPI + pandas only, vendors config.py)
  requirements-api.txt
  api/
    main.py                 endpoints: /health, /protocol/active, /governance/ledger, /qa/summary, /cases
    safety.py                server-side severity_roi redaction — see its docstring
  scripts/
    s3_sync_down.sh          pulls workspace/ prefixes from S3 before a run (no-op if unconfigured)
    s3_sync_up.sh             pushes generated outputs back to S3 after a run (no-op if unconfigured)
  docker-compose.yml         single-host staging (what was asked for first)
  k8s/                       cloud-native path (what was asked for second — same images, same scripts)
    configmap.yaml, secret-s3.yaml.template, pvc-workspace.yaml,
    deployment-api.yaml, cronjob-s3-sync-up.yaml, job-pipeline.yaml.template
```

## S3 layout

`config.py`'s own header comment states the intent: *"the workspace mirrors the S3 bucket layout of
the IS360 Breast Imaging Platform architecture."* The sync scripts implement exactly that — a 1:1
mirror of `config.py`'s `DIRS`, under one bucket/prefix:

```
s3://$BC_S3_BUCKET/$BC_S3_PREFIX/raw/tenant=is360/source=mias/...
                                  /processed/tenant=is360/source=mias/...
                                  /research/tenant=is360/...          (datasets, partitions, experiments)
                                  /models/tenant=is360/...
                                  /results/tenant=is360/...
                                  /reports/tenant=is360/...
                                  /audit/...
                                  /manifests/...                      (ms_access_log.csv — the governance ledger)
                                  /logs/...
```

`raw/` is pulled down but never pushed back up — a compute container should only ever read the source
dataset, never write to it. Everything else round-trips.

**Governance ledger note:** `manifests/ms_access_log.csv` is the append-only record of every MS
(manuscript-partition) access — confirmatory vs. exploratory, by protocol hash. `aws s3 sync` does not
make it tamper-evident on its own. For a real staging/production S3 bucket, turn on **S3 Versioning +
Object Lock (governance mode)** on that prefix at the bucket-policy level — that's an account/IAM
decision outside what these scripts can do, flagged here so it isn't missed.

If `BC_S3_BUCKET` is unset, both scripts no-op and everything runs against the local Docker volume /
PVC only — this keeps the exact local workflow already validated this session (the DEV smoke test that
caught the `paz_gate()` call-site break) working unchanged.

## Running it: Docker Compose (single host)

```bash
cd IS360_Breast_PoC
export BC_S3_BUCKET=your-bucket        # omit entirely to run local-only, no S3
export BC_S3_PREFIX=is360-breast-poc
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

docker compose -f deploy/docker-compose.yml up --build -d api s3-sync-up-cron
curl http://localhost:8000/health
curl http://localhost:8000/qa/summary
curl http://localhost:8000/cases?limit=5

# Run a pipeline step on demand (e.g. a fresh DEV QA sweep):
docker compose -f deploy/docker-compose.yml run --rm pipeline experiments --partition DEV --repeats 1 --boot 100
docker compose -f deploy/docker-compose.yml run --rm s3-sync-up   # push results back immediately, don't wait for the cron
```

Point the OncoSync canvas (`Main.dc.html` / `QA_Dashboard.dc.html`) at `http://<host>:8000` for the
case list and QA summary once you wire its fetch calls — it currently ships with inline mock data,
which this API is meant to replace with real pipeline output.

## Running it: Kubernetes

```bash
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/pvc-workspace.yaml
# fill in deploy/k8s/secret-s3.yaml.template -> secret-s3.yaml (gitignored) and apply, OR:
kubectl create secret generic oncosync-s3-creds \
  --from-literal=BC_S3_BUCKET=... --from-literal=AWS_ACCESS_KEY_ID=... --from-literal=AWS_SECRET_ACCESS_KEY=...
# push images first: docker build -f deploy/Dockerfile.pipeline -t REGISTRY/oncosync-pipeline:TAG .
#                    docker build -f deploy/Dockerfile.api -t REGISTRY/oncosync-api:TAG .
# then replace REGISTRY/oncosync-*:TAG in deployment-api.yaml / cronjob-s3-sync-up.yaml / job-pipeline.yaml.template
kubectl apply -f deploy/k8s/deployment-api.yaml
kubectl apply -f deploy/k8s/cronjob-s3-sync-up.yaml
```

On a cloud provider (EKS), prefer IAM Roles for Service Accounts (IRSA) over the static-key Secret
template — swap the `secretRef` for pod-level IAM and drop the AWS key fields. The static-secret
version is the portable baseline; it isn't the recommended production pattern.

## What was actually verified, and what wasn't (be precise about this)

- **The application logic was smoke-tested for real**, natively (not in Docker — the sandbox this was
  built in has no privilege to run a Docker daemon), with `BC_HOME` pointed at the actual pipeline tree
  and its real confirmatory-run data (`exp_20261009_003_MS`). All four endpoints were hit and returned
  correct, real numbers — including confirming the severity_roi redaction actually fires (raw `p_B`/`p_M`
  moved under `raw_probabilities`, `display_label` forced to `INDETERMINATE`,
  `requires_specialist_review: true`) regardless of the underlying probability.
- **`docker compose config` was run and validates the compose file's syntax and service graph.**
- **The k8s manifests were YAML-parse-validated**, not applied to a real cluster (none available here).
- **`docker build` was NOT run** — this sandbox's Docker daemon cannot start (no privilege for nested
  containerization: `ulimit: error setting limit (Operation not permitted)`). The Dockerfiles follow
  the same version pins as `provenance.py`'s recorded software stack and were written and reviewed
  carefully, but **building them on a real Docker host is the one step still outstanding** before
  calling this deployment-ready. Do that first, on whatever host will actually run this.

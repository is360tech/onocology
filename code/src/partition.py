"""Option A partition: seeded, patient-level split into DEV (~30%) and MS = manuscript (~70%), stratified by each patient's worst class.
The split is fully deterministic (hash ranking, no library RNG) and is frozen with SHA-256 hashes.
  python bc.py partition            create + freeze (refuses to overwrite)
  python bc.py partition --show     print the frozen composition"""
import os, json, hashlib, datetime, argparse
import pandas as pd
import config as C

RANK = {"normal": 0, "benign": 1, "malignant": 2}

def _h(s): return hashlib.sha256(s.encode()).hexdigest()
def _file_sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()

def patient_table(man):
    g = man.groupby("patient_id")
    return pd.DataFrame({"patient_id": list(g.groups), "worst": [max(v.image_class, key=RANK.get) for _, v in g]})

def make_split(man, seed=C.PARTITION_SEED, dev_frac=C.DEV_FRACTION):
    pt = patient_table(man); pt["rank_key"] = [_h(f"{seed}:{p}") for p in pt.patient_id]; pt["partition"] = "MS"
    for w, g in pt.groupby("worst"):
        n_dev = int(round(dev_frac * len(g))); dev_ids = g.sort_values("rank_key").patient_id.iloc[:n_dev]
        pt.loc[pt.patient_id.isin(dev_ids), "partition"] = "DEV"
    return pt.drop(columns="rank_key").sort_values("patient_id").reset_index(drop=True)

def freeze(force=False):
    man = pd.read_csv(os.path.join(C.MAN, "ingest_manifest.csv")); os.makedirs(os.path.join(C.DATASET_ROOT, "partitions"), exist_ok=True)
    fz = os.path.join(C.DATASET_ROOT, "partition_freeze.json")
    if os.path.exists(fz) and not force: raise SystemExit(f"partition already frozen: {fz} (use --force only with written justification)")
    pt = make_split(man); paths = {}
    for k in ("DEV", "MS"):
        p = os.path.join(C.DATASET_ROOT, "partitions", f"{k}.csv"); pt[pt.partition == k][["patient_id", "worst"]].to_csv(p, index=False); paths[k] = p
    m = man.merge(pt[["patient_id", "partition"]], on="patient_id")
    info = dict(dataset_id=C.DATASET_ID, rule="patient-level, stratified by worst class, hash-ranked (sha256 of seed:patient_id), DEV = first round(0.30*n) per stratum",
                seed=C.PARTITION_SEED, dev_fraction=C.DEV_FRACTION, created=datetime.datetime.now().isoformat(timespec="seconds"),
                ingest_manifest_sha256=_file_sha(os.path.join(C.MAN, "ingest_manifest.csv")), partition_files_sha256={k: _file_sha(p) for k, p in paths.items()},
                code_sha256=_file_sha(__file__), counts={k: dict(patients=int((pt.partition == k).sum()), images=int((m.partition == k).sum()),
                **{c: int(((pt.partition == k) & (pt.worst == c)).sum()) for c in RANK}) for k in ("DEV", "MS")})
    json.dump(info, open(fz, "w"), indent=2); return info

def load():
    """-> DataFrame patient_id, partition. Verifies the frozen hashes."""
    fz = os.path.join(C.DATASET_ROOT, "partition_freeze.json")
    if not os.path.exists(fz): raise SystemExit("no frozen partition: run `python bc.py partition` first")
    info = json.load(open(fz)); parts = []
    for k in ("DEV", "MS"):
        p = os.path.join(C.DATASET_ROOT, "partitions", f"{k}.csv")
        if _file_sha(p) != info["partition_files_sha256"][k]: raise SystemExit(f"partition file {k}.csv changed after freeze")
        d = pd.read_csv(p); d["partition"] = k; parts.append(d)
    return pd.concat(parts)[["patient_id", "partition"]]

def load_partition(name):
    """List of patient_ids in DEV or MS (verifies the frozen hashes)."""
    d = load(); return d[d.partition == name].patient_id.tolist()

def select(man, partition):
    """Filter a manifest-like DataFrame with a patient_id column by partition ('ALL', 'DEV' or 'MS')."""
    if partition == "ALL": return man
    ids = set(load().query("partition == @partition").patient_id); return man[man.patient_id.isin(ids)]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true"); ap.add_argument("--show", action="store_true"); a = ap.parse_args()
    C.ensure_dirs()
    info = json.load(open(os.path.join(C.DATASET_ROOT, "partition_freeze.json"))) if a.show else freeze(a.force)
    print(json.dumps(info["counts"], indent=1)); print("frozen:", os.path.join(C.DATASET_ROOT, "partition_freeze.json"))

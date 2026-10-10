"""Reproducibility record for every experiment: versions, seeds, settings, hashes of code and protocol."""
import os, sys, json, glob, hashlib, platform, datetime, importlib
import config as C

def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None

def record(extra=None):
    ver = {}
    for m in ["numpy", "scipy", "pandas", "sklearn", "skimage", "PIL", "joblib", "matplotlib"]:
        try: ver[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception: ver[m] = None
    src = os.path.dirname(os.path.abspath(__file__)); code = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(src, "*.py"))): code.update(open(f, "rb").read())
    return {"time": datetime.datetime.now().isoformat(timespec="seconds"), "python": sys.version.split()[0], "platform": platform.platform(),
            "versions": ver, "code_sha256": code.hexdigest(), "protocol_sha256": sha(C.PROTOCOL_FILE), "pipeline_version": C.PIPELINE_VERSION,
            "seed": C.SEED, "partition_seed": C.PARTITION_SEED, **(extra or {})}

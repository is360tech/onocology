"""T0 environment check. python bc.py check"""
import os, sys, shutil, importlib, platform
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config as C
rows = []
def add(s, item, note=""): rows.append(s); print(f"[{s:<4}] {item:<32} {note}")
add("PASS" if sys.version_info >= (3, 9) else "FAIL", "Python >= 3.9", platform.python_version())
for m in ["numpy", "scipy", "pandas", "sklearn", "skimage", "PIL", "joblib", "matplotlib"]:
    try: add("PASS", f"package {m}", getattr(importlib.import_module(m), "__version__", ""))
    except Exception: add("FAIL", f"package {m}", "pip install -r requirements.txt")
try: importlib.import_module("h5py"); add("PASS", "package h5py (optional)", "enables the HDF5 cross-check at ingest")
except Exception: add("WARN", "package h5py (optional)", "HDF5 cross-check will be skipped")
os.makedirs(C.WS, exist_ok=True); free = shutil.disk_usage(C.WS).free / 1e9; add("PASS" if free >= 10 else "WARN", "free disk (GB)", f"{free:.0f} (>= 10 needed for MIAS; 30 comfortable)")
add("PASS" if (os.cpu_count() or 1) >= 4 else "WARN", "CPU cores", str(os.cpu_count()))
try: ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9; add("PASS" if ram >= 7 else "WARN", "RAM (GB)", f"{ram:.0f}")
except Exception: add("WARN", "RAM (GB)", "not detected")
inbox = list(Path(C.HOME, "inbox").glob("*")); add("PASS" if any(x.suffix in (".zip", ".gz") or x.is_dir() for x in inbox) else "WARN", "MIAS archive in inbox/", ", ".join(x.name for x in inbox[:4]) or "put archive.zip or all-mias_tar.gz there")
add("PASS" if os.path.exists(os.path.join(C.HOME, "docs", "protocol.yaml")) else "FAIL", "docs/protocol.yaml")
print(f"\n{rows.count('FAIL')} FAIL, {rows.count('WARN')} WARN"); sys.exit(1 if "FAIL" in rows else 0)

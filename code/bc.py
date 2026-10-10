#!/usr/bin/env python3
"""IS360 Breast Imaging PoC - single entry point.   python bc.py <command> [options]
setup | check | ingest | partition | protocol | standardize | preprocess | patches | features | experiments | report | align | permute | validate | test
Typical order is in README_FIRST.md. Options are passed to the underlying script (add --help)."""
import os, sys, subprocess
from pathlib import Path
CODE = Path(__file__).resolve().parent; SRC, SCR = CODE / "src", CODE / "scripts"
CMDS = {"check": (SCR / "check_environment.py", []), "ingest": (SRC / "s0_ingest_mias.py", []), "partition": (SRC / "partition.py", []),
        "standardize": (SRC / "s1_standardize.py", []), "preprocess": (SRC / "s2_preprocess.py", []), "patches": (SRC / "s3_features.py", ["patches"]),
        "features": (SRC / "s3_features.py", ["features"]), "experiments": (SRC / "s4_experiments.py", []), "protocol": (SRC / "s4_experiments.py", ["--partition", "DEV", "--freeze-protocol"]),
        "report": (SRC / "s5_report.py", []), "align": (SCR / "alignment_check.py", []), "permute": (SCR / "permutation_test.py", []),
        "validate": (SCR / "validate_outputs.py", []), "test": (CODE / "tests" / "test_pipeline.py", [])}
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in list(CMDS) + ["setup"]: print(__doc__); sys.exit(0)
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd == "setup":
        sys.path.insert(0, str(SRC)); import config as C; C.ensure_dirs(); print("workspace ready:", C.WS); sys.exit(0)
    script, pre = CMDS[cmd]; env = dict(os.environ, PYTHONPATH=str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONWARNINGS="ignore")
    sys.exit(subprocess.call([sys.executable, str(script)] + pre + rest, env=env))

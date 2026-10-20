"""
gate0_setup.py
==============
Gate 0 — Environment Setup for cfDNA methylation cancer detection project.

This script is idempotent: it verifies that the cfDNA project's two Python
virtual environments are present and that all required packages import
successfully. It does NOT install packages (installs require user approval
and are done outside this script). Run this to confirm the environment is
healthy before starting later gates.

Venvs:
  - Main:        venvs/cfdna/          (numpy/pandas/torch/etc.)
  - methylprep:  venvs/cfdna-methylprep/  (idat->beta only)

Outputs:
  - Prints package versions to stdout.
  - Appends a verification line to work/logs/gates.log
    only when run with --log (so re-running for inspection doesn't pollute
    the gate log).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path("work")
RAW_ROOT = Path("work/raw")
GATE_LOG = PROJECT_ROOT / "logs" / "gates.log"

MAIN_VENV_PY = Path("venvs/cfdna/bin/python")
METHYLPREP_VENV_PY = Path("venvs/cfdna-methylprep/bin/python")

REQUIRED_DIRS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "processed",
    PROJECT_ROOT / "logs",
    PROJECT_ROOT / "results",
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "embeddings",
    RAW_ROOT,
]

MAIN_IMPORTS = [
    "numpy", "pandas", "scipy", "sklearn", "torch", "torchvision",
    "pyarrow", "tqdm", "requests", "GEOparse",
    "matplotlib", "seaborn", "statsmodels",
    "pyranges", "pyliftover", "pybedtools",
]

METHYLPREP_IMPORTS = ["methylprep"]


def ensure_dirs() -> None:
    for d in REQUIRED_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def check_venv(py_path: Path, modules: list[str]) -> dict[str, str | None]:
    """Return {module: version_or_None} or {module: 'ERROR: ...'} for each."""
    if not py_path.exists():
        return {m: f"ERROR: venv python missing at {py_path}" for m in modules}

    script = (
        "import importlib, json, sys\n"
        f"modules = {modules!r}\n"
        "result = {}\n"
        "for m in modules:\n"
        "    try:\n"
        "        mod = importlib.import_module(m)\n"
        "        result[m] = getattr(mod, '__version__', 'imported')\n"
        "    except Exception as e:\n"
        "        result[m] = f'ERROR: {type(e).__name__}: {e}'\n"
        "print(json.dumps(result))\n"
    )
    proc = subprocess.run(
        [str(py_path), "-c", script],
        capture_output=True, text=True, check=False,
    )
    if proc.returncode != 0:
        return {m: f"ERROR: subprocess failed: {proc.stderr.strip()[:200]}" for m in modules}
    import json
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="Append a verification line to gates.log")
    args = ap.parse_args()

    print("=" * 70)
    print("Gate 0 verification — cfDNA project")
    print(f"Timestamp: {_dt.datetime.now().isoformat(timespec='seconds')}")
    print("=" * 70)

    ensure_dirs()
    print("\n[1] Directories OK:")
    for d in REQUIRED_DIRS:
        print(f"    {d}  exists={d.exists()}")

    print("\n[2] Main venv imports:")
    main_results = check_venv(MAIN_VENV_PY, MAIN_IMPORTS)
    any_fail = False
    for m, v in main_results.items():
        marker = "FAIL" if str(v).startswith("ERROR") else "ok  "
        print(f"    [{marker}] {m:14s} {v}")
        if str(v).startswith("ERROR"):
            any_fail = True

    print("\n[3] methylprep venv imports:")
    mp_results = check_venv(METHYLPREP_VENV_PY, METHYLPREP_IMPORTS)
    for m, v in mp_results.items():
        marker = "FAIL" if str(v).startswith("ERROR") else "ok  "
        print(f"    [{marker}] {m:14s} {v}")
        if str(v).startswith("ERROR"):
            any_fail = True

    # bedtools binary check (informational; defer to Gate 2)
    print("\n[4] bedtools binary on PATH (needed at runtime for pybedtools):")
    try:
        proc = subprocess.run(["bedtools", "--version"], capture_output=True, text=True, timeout=5)
        if proc.returncode == 0:
            print(f"    OK: {proc.stdout.strip()}")
        else:
            print(f"    MISSING (returncode {proc.returncode}) — defer pybedtools use to Gate 2")
    except FileNotFoundError:
        print("    MISSING (binary not on PATH) — defer pybedtools use to Gate 2")

    # GPU/CUDA check (informational; cfDNA gates 1-4 are CPU-only)
    print("\n[5] GPU/CUDA status (informational):")
    try:
        proc = subprocess.run(
            [str(MAIN_VENV_PY), "-c",
             "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"],
            capture_output=True, text=True, timeout=20,
        )
        print(f"    {proc.stdout.strip()}")
    except Exception as e:
        print(f"    error: {e}")

    status = "FAIL" if any_fail else "PASS"
    print(f"\nGate 0 status: {status}")

    if args.log:
        line = f"[{_dt.datetime.now().isoformat(timespec='seconds')}] GATE_0: {status} verified_by=gate0_setup.py\n"
        with GATE_LOG.open("a") as f:
            f.write(line)
        print(f"Appended to {GATE_LOG}: {line.strip()}")

    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

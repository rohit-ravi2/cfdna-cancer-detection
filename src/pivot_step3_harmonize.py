#!/usr/bin/env python3
"""
PIVOT_PHASE1_GSE186458 STEP 3 — Harmonize GSE186458 cfDNA + WBC to atlas blocks.

For each .hg38.beta file in work/raw/GSE186458/pivot/:
  - Call `wgbstools beta_to_blocks` to collapse the per-CpG beta to the 9520
    atlas blocks (using atlas_blocks_hg38.bed which we built from the atlas
    TSV: chr, start, end, startCpG, endCpG).
  - Parse the resulting uint8 pair file (methylated_count, total_count per block).
  - Compute beta = methylated_count / total_count where total > 0.

Outputs two parquets:
  - GSE186458_cfdna_harmonized.parquet   [9520 blocks x 23 cfDNA samples]
  - GSE186458_wbc_harmonized.parquet     [9520 blocks x 23 WBC samples]

Also writes per-sample coverage (fraction of atlas blocks with reads >= 1)
and a metadata TSV with sample classification.

Hard-stop: any sample with <50% atlas-block coverage triggers failure.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------
PIVOT_DIR = Path("work/raw/GSE186458/pivot")
PROC_DIR = Path("work/processed")
LOG_DIR = Path("work/logs")
RES_DIR = Path("work/results")
PROC_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_BLOCKS_BED = PROC_DIR / "atlas_blocks_hg38.bed"
ATLAS_TSV = Path("work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
TMP_DIR = Path("/tmp/wgbs_harmonize")
TMP_DIR.mkdir(parents=True, exist_ok=True)

WGBSTOOLS = os.environ.get("WGBSTOOLS", "wgbstools")

GATES_LOG = LOG_DIR / "gates.log"

# -------------------------------------------------------------------------


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def parse_sample_id(filename: str) -> tuple[str, str]:
    """Return (gsm, kind) where kind in {cfDNA, WBC}."""
    m = re.match(r"(GSM\d+)_CNVS-NORM-(cfDNA|WBC)-", filename)
    if m:
        return m.group(1), m.group(2)
    return None, None


def list_beta_files() -> list[Path]:
    """Return all hg38.beta files in pivot dir for GSM6810003..GSM6810048."""
    files = []
    for p in sorted(PIVOT_DIR.glob("GSM*.hg38.beta")):
        gsm, kind = parse_sample_id(p.name)
        if gsm is None or kind is None:
            continue
        num = int(gsm.replace("GSM", ""))
        if 6810003 <= num <= 6810048:
            files.append(p)
    return files


def run_beta_to_blocks(beta_path: Path, out_dir: Path) -> Path:
    """Run wgbstools beta_to_blocks; return path to output .bin."""
    cmd = [
        WGBSTOOLS, "beta_to_blocks",
        "--blocks_file", str(ATLAS_BLOCKS_BED),
        "-o", str(out_dir),
        "-f",
        str(beta_path),
    ]
    env = os.environ.copy()
    env["PATH"] = os.environ.get("WGBS_BIN_DIR", "") + ":" + env.get("PATH", "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"beta_to_blocks failed for {beta_path.name}: {r.stderr}")
    # Output filename: beta basename minus .beta, plus .bin
    out_name = beta_path.stem + ".bin"  # strips .beta -> .bin
    out_path = out_dir / out_name
    if not out_path.exists():
        # Try the alt naming wgbstools may use
        alts = list(out_dir.glob(beta_path.name.replace(".beta", "*")))
        # Filter to .bin
        alts = [a for a in alts if a.suffix == ".bin"]
        if alts:
            out_path = alts[0]
        else:
            raise FileNotFoundError(f"output .bin not found for {beta_path}")
    return out_path


def load_block_bin(path: Path, n_blocks: int = 9520) -> np.ndarray:
    """Load the uint8 pair block file. Returns array [n_blocks, 2] (meth, total)."""
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size != 2 * n_blocks:
        raise ValueError(
            f"Unexpected size for {path}: {arr.size} bytes, expected {2*n_blocks}"
        )
    return arr.reshape(-1, 2)


def main():
    log("=== PIVOT_PHASE1_GSE186458 STEP 3 begin ===")

    # Make sure atlas blocks BED exists
    if not ATLAS_BLOCKS_BED.exists():
        log(f"building atlas blocks BED at {ATLAS_BLOCKS_BED}")
        # Build it from the atlas TSV
        df = pd.read_csv(ATLAS_TSV, sep="\t")
        df[["chr", "start", "end", "startCpG", "endCpG"]].to_csv(
            ATLAS_BLOCKS_BED, sep="\t", header=False, index=False
        )

    files = list_beta_files()
    log(f"found {len(files)} beta files")

    cfdna_samples = {}  # gsm -> dict
    wbc_samples = {}    # gsm -> dict
    metadata_rows = []
    coverage_rows = []

    for i, bp in enumerate(files, 1):
        gsm, kind = parse_sample_id(bp.name)
        log(f"[{i}/{len(files)}] {gsm} ({kind}): beta_to_blocks {bp.name}")
        out_bin = run_beta_to_blocks(bp, TMP_DIR)
        b = load_block_bin(out_bin)
        meth = b[:, 0].astype(np.float64)
        total = b[:, 1].astype(np.float64)
        covered = total > 0
        beta = np.full(b.shape[0], np.nan, dtype=np.float64)
        beta[covered] = meth[covered] / total[covered]
        cov_frac = float(covered.sum()) / b.shape[0]
        mean_depth = float(total[covered].mean()) if covered.any() else 0.0
        median_depth = float(np.median(total[covered])) if covered.any() else 0.0
        log(f"    cov={cov_frac:.3%}  mean_depth={mean_depth:.1f}  median_depth={median_depth:.1f}")

        metadata_rows.append({
            "gsm": gsm, "kind": kind, "filename": bp.name,
            "atlas_block_coverage": cov_frac,
            "mean_depth_covered": mean_depth,
            "median_depth_covered": median_depth,
        })
        coverage_rows.append({
            "gsm": gsm, "kind": kind,
            "coverage": cov_frac,
            "mean_depth": mean_depth,
            "median_depth": median_depth,
        })

        # Use trimmed GSM as the sample_id (column name)
        sid = gsm
        if kind == "cfDNA":
            cfdna_samples[sid] = beta
        elif kind == "WBC":
            wbc_samples[sid] = beta

        # Cleanup the bin
        try:
            out_bin.unlink()
        except OSError:
            pass

    # Build matrices: rows=block_id (0..9519), cols=samples
    n_blocks = 9520
    block_ids = np.arange(n_blocks)

    def build_matrix(samples_dict):
        df = pd.DataFrame({"block_id": block_ids})
        for sid in sorted(samples_dict):
            df[sid] = samples_dict[sid]
        return df

    cfdna_df = build_matrix(cfdna_samples)
    wbc_df = build_matrix(wbc_samples)
    log(f"cfDNA matrix: {cfdna_df.shape}  (n_samples={len(cfdna_samples)})")
    log(f"WBC matrix:   {wbc_df.shape}  (n_samples={len(wbc_samples)})")

    cfdna_path = PROC_DIR / "GSE186458_cfdna_harmonized.parquet"
    wbc_path = PROC_DIR / "GSE186458_wbc_harmonized.parquet"
    cfdna_df.to_parquet(cfdna_path, index=False)
    wbc_df.to_parquet(wbc_path, index=False)
    log(f"wrote {cfdna_path}")
    log(f"wrote {wbc_path}")

    # Coverage summary
    cov_df = pd.DataFrame(coverage_rows)
    cov_df.to_csv(PROC_DIR / "GSE186458_coverage.tsv", sep="\t", index=False)
    meta_df = pd.DataFrame(metadata_rows)
    meta_df.to_csv(PROC_DIR / "GSE186458_samples.tsv", sep="\t", index=False)

    cfdna_cov = cov_df.loc[cov_df["kind"] == "cfDNA", "coverage"]
    wbc_cov = cov_df.loc[cov_df["kind"] == "WBC", "coverage"]

    summary = {
        "n_cfdna": int(len(cfdna_cov)),
        "n_wbc": int(len(wbc_cov)),
        "cfdna_coverage_mean": float(cfdna_cov.mean()) if len(cfdna_cov) else 0.0,
        "cfdna_coverage_min": float(cfdna_cov.min()) if len(cfdna_cov) else 0.0,
        "cfdna_coverage_max": float(cfdna_cov.max()) if len(cfdna_cov) else 0.0,
        "wbc_coverage_mean": float(wbc_cov.mean()) if len(wbc_cov) else 0.0,
        "wbc_coverage_min": float(wbc_cov.min()) if len(wbc_cov) else 0.0,
        "wbc_coverage_max": float(wbc_cov.max()) if len(wbc_cov) else 0.0,
        "n_blocks": n_blocks,
        "hard_stop_threshold": 0.50,
        "any_sample_below_threshold": bool(
            (cfdna_cov < 0.50).any() or (wbc_cov < 0.50).any()
        ),
    }
    with open(RES_DIR / "pivot_phase1_step3.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"summary: {json.dumps(summary, indent=2)}")

    # Gate log
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if summary["any_sample_below_threshold"]:
        line = (
            f"[{ts}] PIVOT_PHASE1_GSE186458 STEP3_harmonize: HARD_STOP "
            f"cfdna_cov_min={summary['cfdna_coverage_min']:.3f} "
            f"wbc_cov_min={summary['wbc_coverage_min']:.3f}"
        )
        with open(GATES_LOG, "a") as f:
            f.write(line + "\n")
        log("HARD_STOP triggered")
        return 2

    line = (
        f"[{ts}] PIVOT_PHASE1_GSE186458 STEP3_harmonize: PASS "
        f"cfdna_n={summary['n_cfdna']} cov_mean={summary['cfdna_coverage_mean']:.3%} "
        f"wbc_n={summary['n_wbc']} cov_mean_wbc={summary['wbc_coverage_mean']:.3%}"
    )
    with open(GATES_LOG, "a") as f:
        f.write(line + "\n")
    log("STEP 3 PASS")
    n_cf = summary["n_cfdna"]
    cov_m = summary["cfdna_coverage_mean"]
    msg = f"cfdna step3 harmonize PASS: cfDNA n={n_cf} cov={cov_m:.1%}"
    os.system(f"~/bin/notify '{msg}' 'cfdna' 2>/dev/null")
    return 0


if __name__ == "__main__":
    sys.exit(main())

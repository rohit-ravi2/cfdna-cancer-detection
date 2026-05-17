#!/usr/bin/env python3
"""
Day 2 Phase B1.2 — Harmonize newly downloaded sorted-cell refs to atlas blocks.

Reads the .hg38.beta files in pivot/sorted_cells/ that we don't yet have
harmonized (compared to existing harmonized matrices), runs wgbstools
beta_to_blocks, and produces:

  /mnt/ssd/research/cfdna/processed/sorted_cells_harmonized.parquet
    [9520 blocks x (all sorted-cell samples)]
  /mnt/ssd/research/cfdna/processed/sorted_cells_metadata.tsv
    [gsm, cell_type, source_name, atlas_cov, mean_depth]
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HDD_RAW = Path("/mnt/hdd/research/cfdna/raw/GSE186458/pivot/sorted_cells")
PROC_DIR = Path("/mnt/ssd/research/cfdna/processed")
TMP_DIR = Path("/tmp/wgbs_harm_b1")
TMP_DIR.mkdir(parents=True, exist_ok=True)
ATLAS_BLOCKS_BED = PROC_DIR / "atlas_blocks_hg38.bed"
WGBSTOOLS = "/home/rohit/tools/wgbs_tools/wgbstools"
N_BLOCKS = 9520
OUT_PARQUET = PROC_DIR / "sorted_cells_harmonized.parquet"
OUT_META = PROC_DIR / "sorted_cells_metadata.tsv"
LOG_FILE = Path("/mnt/ssd/research/cfdna/logs/day2_b1_harm.log")


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def parse_filename(p: Path) -> tuple[str, str]:
    name = p.name
    # GSM5652313_Blood-Granulocytes-Z000000TZ.hg38.beta
    m = re.match(r"(GSM\d+)_(.+?)\.hg38\.beta$", name)
    if not m:
        return None, None
    gsm = m.group(1)
    tail = m.group(2)
    # Strip the donor-id suffix -Z000000XX
    cell_label = re.sub(r"-Z[0-9A-Z]+$", "", tail)
    return gsm, cell_label


def run_beta_to_blocks(beta_path: Path) -> Path:
    cmd = [
        WGBSTOOLS, "beta_to_blocks",
        "--blocks_file", str(ATLAS_BLOCKS_BED),
        "-o", str(TMP_DIR),
        "-f",
        str(beta_path),
    ]
    env = os.environ.copy()
    env["PATH"] = f"/home/rohit/miniconda3/envs/wgbs/bin:{env.get('PATH','')}"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"beta_to_blocks failed for {beta_path.name}: {r.stderr[:200]}")
    out = TMP_DIR / (beta_path.stem + ".bin")
    if not out.exists():
        alts = list(TMP_DIR.glob(beta_path.name.replace(".beta", "*.bin")))
        if not alts:
            raise FileNotFoundError(beta_path.name)
        out = alts[0]
    return out


def load_block_bin(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size != 2 * N_BLOCKS:
        raise ValueError(f"bad size for {path}: {arr.size}")
    return arr.reshape(-1, 2)


def main():
    log("=== Day2 B1.2 harmonize sorted-cell refs ===")
    if not ATLAS_BLOCKS_BED.exists():
        raise FileNotFoundError(ATLAS_BLOCKS_BED)

    files = sorted(HDD_RAW.glob("*.hg38.beta"))
    log(f"found {len(files)} .hg38.beta files")

    samples = {}
    meta_rows = []
    for i, p in enumerate(files, 1):
        gsm, cell_label = parse_filename(p)
        if gsm is None:
            continue
        log(f"[{i}/{len(files)}] {gsm} {cell_label}")
        try:
            out_bin = run_beta_to_blocks(p)
            b = load_block_bin(out_bin)
            meth = b[:, 0].astype(np.float64)
            total = b[:, 1].astype(np.float64)
            covered = total > 0
            beta = np.full(N_BLOCKS, np.nan, dtype=np.float64)
            beta[covered] = meth[covered] / total[covered]
            cov_frac = float(covered.sum()) / N_BLOCKS
            mean_depth = float(total[covered].mean()) if covered.any() else 0.0
            samples[gsm] = beta
            meta_rows.append({
                "gsm": gsm,
                "cell_label": cell_label,
                "atlas_cov": cov_frac,
                "mean_depth": mean_depth,
                "filename": p.name,
            })
            try:
                out_bin.unlink()
            except OSError:
                pass
        except Exception as e:
            log(f"  ERROR {gsm}: {e}")
            meta_rows.append({"gsm": gsm, "cell_label": cell_label, "atlas_cov": 0.0, "mean_depth": 0.0, "filename": p.name, "error": str(e)})

    # Build matrix
    df = pd.DataFrame({"block_id": np.arange(N_BLOCKS)})
    for sid in sorted(samples):
        df[sid] = samples[sid]
    log(f"matrix shape: {df.shape}")
    df.to_parquet(OUT_PARQUET, index=False)
    log(f"wrote {OUT_PARQUET}")

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(OUT_META, sep="\t", index=False)
    log(f"wrote {OUT_META}")

    print("\ncell_label distribution:")
    print(meta_df["cell_label"].value_counts())
    print(f"\nmean atlas_cov: {meta_df['atlas_cov'].mean():.3%}")
    print(f"min atlas_cov: {meta_df['atlas_cov'].min():.3%}")


if __name__ == "__main__":
    sys.exit(main() or 0)

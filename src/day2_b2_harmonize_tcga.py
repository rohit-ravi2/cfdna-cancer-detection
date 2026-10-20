#!/usr/bin/env python3
"""
Day 2 Phase B2.2 — Harmonize TCGA 450K methylation to atlas blocks.

Approach:
  1) Download Illumina HM450 manifest (probe-id -> chr, position, hg38).
  2) For each .txt file (probe_id, beta), map each probe to atlas block_id
     using bedtools-style interval intersection.
  3) Aggregate (mean beta) per block per sample.

Outputs:
  work/processed/tcga_<PROJECT>_harmonized.parquet
  work/processed/tcga_metadata.tsv

Caveat: 450K covers only ~485K CpGs vs 28M for WGBS. Atlas blocks may have
zero overlapping probes -> NaN. Expected per-block coverage ~30-50%.
"""
import os
import re
import sys
import time
import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROC_DIR = Path("work/processed")
ATLAS_BED = PROC_DIR / "atlas_blocks_hg38.bed"
TCGA_RAW = Path("work/raw/tcga_methylation")
MANIFEST_DIR = Path("work/raw/illumina_manifests")
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = Path("work/logs/day2_b2_harm.log")
PROJECTS = ["TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-BRCA", "TCGA-PAAD"]
N_BLOCKS = 9520


def log(m):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def ensure_manifest():
    """Use sesame/zhou-lab Infinium HM450 hg38 manifest from GitHub.

    URL: https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/HM450/
         HM450.hg38.manifest.tsv.gz
    """
    target = MANIFEST_DIR / "HM450.hg38.manifest.tsv.gz"
    if target.exists() and target.stat().st_size > 100_000:
        log(f"manifest exists: {target}")
        return target
    url = ("https://github.com/zhou-lab/InfiniumAnnotationV1/raw/main/Anno/"
           "HM450/HM450.hg38.manifest.tsv.gz")
    log(f"downloading manifest from {url}")
    with requests.get(url, timeout=120, stream=True) as r:
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)
    log(f"downloaded {target} ({target.stat().st_size/1e6:.1f} MB)")
    return target


def load_manifest(path):
    """Load Illumina manifest, return DataFrame with [probe, chr, start, end]."""
    log("loading manifest...")
    df = pd.read_csv(path, sep="\t", compression="gzip", low_memory=False)
    log(f"manifest columns: {list(df.columns)[:10]}")
    # Find probe id col + chr/start cols
    # The sesame manifest format: 'CpG_chrm', 'CpG_beg', 'CpG_end', 'probeID'
    probe_col = "probeID" if "probeID" in df.columns else "Probe_ID"
    chr_col = "CpG_chrm" if "CpG_chrm" in df.columns else "CHR_hg38"
    pos_col = "CpG_beg" if "CpG_beg" in df.columns else "Start_hg38"
    end_col = "CpG_end" if "CpG_end" in df.columns else "End_hg38"
    out = df[[probe_col, chr_col, pos_col, end_col]].copy()
    out.columns = ["probe", "chr", "start", "end"]
    out = out.dropna(subset=["chr", "start"])
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    log(f"loaded {len(out)} probes")
    return out


def build_probe_to_block_map(manifest, atlas):
    """For each atlas block (chr, start, end), find probes whose start is
    in [block_start, block_end)."""
    log("mapping probes to atlas blocks...")
    # Build per-chrom lookup
    block_map = []  # rows of (probe, block_id)
    blocks_by_chr = atlas.groupby("chr")
    for chrom, mdf in manifest.groupby("chr"):
        if chrom not in blocks_by_chr.groups:
            continue
        bdf = blocks_by_chr.get_group(chrom).sort_values("start").reset_index(drop=True)
        # For each probe, find block via searchsorted
        probe_pos = mdf["start"].values
        block_starts = bdf["start"].values
        block_ends = bdf["end"].values
        block_ids = bdf["block_id"].values
        # Find rightmost block_start <= probe_pos
        idx = np.searchsorted(block_starts, probe_pos, side="right") - 1
        valid = (idx >= 0) & (idx < len(block_starts))
        # Check probe within block range
        for p, i, probe in zip(probe_pos, idx, mdf["probe"].values):
            if not (0 <= i < len(block_starts)):
                continue
            if block_starts[i] <= p < block_ends[i]:
                block_map.append((probe, int(block_ids[i])))
    df = pd.DataFrame(block_map, columns=["probe", "block_id"])
    log(f"probe-block matches: {len(df)} (probes mapping to atlas blocks)")
    log(f"unique probes: {df['probe'].nunique()}, unique blocks: {df['block_id'].nunique()}")
    return df


def harmonize_project(project, probe2block):
    proj_dir = TCGA_RAW / project
    files = sorted(proj_dir.glob("*.txt"))
    log(f"{project}: {len(files)} files")
    if not files:
        return None, []

    # build sample matrix: block_id x sample
    n_blocks = N_BLOCKS
    matrix = np.full((n_blocks, len(files)), np.nan, dtype=np.float32)
    sample_ids = []
    meta = []

    # Lookup: probe -> block_id (one block per probe)
    p2b = dict(zip(probe2block["probe"], probe2block["block_id"]))

    for j, p in enumerate(files):
        sid = p.stem.split(".")[0]
        sample_ids.append(sid)
        df = pd.read_csv(p, sep="\t", header=None, names=["probe", "beta"], na_values=["NA", ""], dtype={"probe": str, "beta": float})
        df = df.dropna(subset=["beta"])
        df["block_id"] = df["probe"].map(p2b)
        df = df.dropna(subset=["block_id"])
        df["block_id"] = df["block_id"].astype(int)
        # Aggregate mean beta per block
        agg = df.groupby("block_id")["beta"].mean()
        # Fill matrix
        matrix[agg.index.values, j] = agg.values.astype(np.float32)
        n_probes = len(df)
        n_blocks_covered = int(np.isfinite(matrix[:, j]).sum())
        meta.append({"project": project, "sample_id": sid, "n_probes_mapped": n_probes, "n_blocks_covered": n_blocks_covered, "atlas_coverage": n_blocks_covered / n_blocks})
        if (j + 1) % 10 == 0:
            log(f"  [{j+1}/{len(files)}] {sid}: {n_blocks_covered}/{n_blocks} blocks ({100*n_blocks_covered/n_blocks:.1f}%)")

    df_out = pd.DataFrame({"block_id": np.arange(n_blocks)})
    for j, sid in enumerate(sample_ids):
        df_out[sid] = matrix[:, j]
    out_path = PROC_DIR / f"tcga_{project.replace('TCGA-','')}_harmonized.parquet"
    df_out.to_parquet(out_path, index=False)
    log(f"  wrote {out_path}: {df_out.shape}")
    return out_path, meta


def main():
    log("=== Day2 B2.2 harmonize TCGA ===")
    if not ATLAS_BED.exists():
        raise FileNotFoundError(ATLAS_BED)

    atlas = pd.read_csv(ATLAS_BED, sep="\t", header=None, names=["chr", "start", "end", "startCpG", "endCpG"])
    atlas = atlas.reset_index().rename(columns={"index": "block_id"})
    log(f"atlas blocks: {len(atlas)}")

    manifest_path = ensure_manifest()
    manifest = load_manifest(manifest_path)
    probe2block = build_probe_to_block_map(manifest, atlas)

    # Save probe2block map for later use
    probe2block.to_parquet(PROC_DIR / "hm450_probe_to_atlas_block.parquet", index=False)

    all_meta = []
    for proj in PROJECTS:
        if not (TCGA_RAW / proj).exists():
            log(f"{proj} not yet downloaded, skipping")
            continue
        _, meta = harmonize_project(proj, probe2block)
        all_meta.extend(meta)

    pd.DataFrame(all_meta).to_csv(PROC_DIR / "tcga_metadata.tsv", sep="\t", index=False)
    log(f"done; total samples harmonized: {len(all_meta)}")


if __name__ == "__main__":
    sys.exit(main() or 0)

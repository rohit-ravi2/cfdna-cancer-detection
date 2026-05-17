#!/usr/bin/env python3
"""
PIVOT_PHASE2 STEP 2b — Beta-space in-silico mixing (replaces flawed Step 2a).

Why beta-space mix:
    The wgbstools `mix_pat -L` workflow produces mixtures with realized
    tumor fractions ~10x below the requested fraction (root cause: per-read
    subsampling at -c 5 truncates the tumor reads at heavily-covered atlas
    marker blocks, while healthy reads are similarly truncated; the imbalanced
    truncation distorts the mix ratio). After running step 2a and observing
    realized liver-Hep contribution ~1.2% in a target-15% liver mixture, we
    pivoted to direct beta-space mixing, which is mathematically exact for
    the atlas-block-level NNLS deconvolution we use downstream.

Method:
    For each (healthy_donor h, tissue t, fraction f) cell:
        b_mix = f * b_tumor + (1 - f) * b_healthy
    where b_* are 9520-dim vectors of mean beta per atlas block.

    Tumor donor for each cell is sampled uniformly from the cell-type-subtype
    donors available (5 colon donors, 5 liver, 6 lung, 7 breast, 4 pancreas).
    For lung/breast which have two cell-type-subtypes, we randomly pick one of
    the subtypes (basal vs luminal, alveolar vs bronchial) then a donor of that
    subtype.

    The b_healthy is the existing harmonized cfDNA atlas-block-mean-beta vector
    from Phase 1 (GSE186458_cfdna_harmonized.parquet).

    The b_tumor is computed by running `wgbstools beta_to_blocks` on each
    sorted-cell .beta file once (we have the .beta on disk; no mixing needed).

Output (overwrites step 2a outputs):
    /mnt/ssd/research/cfdna/processed/GSE186458_mixtures_betas.parquet
    /mnt/ssd/research/cfdna/processed/GSE186458_mixtures_labels.parquet
    /mnt/hdd/research/cfdna/raw/GSE186458/pivot/mixtures/manifest.tsv
"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

WGBSTOOLS = "/home/rohit/tools/wgbs_tools/wgbstools"
PATH_PREFIX = "/home/rohit/miniconda3/envs/wgbs/bin:/home/rohit/tools/wgbs_tools"
ATLAS_BED = Path("/mnt/ssd/research/cfdna/processed/atlas_blocks_hg38.bed")

HEALTHY_HARM = Path("/mnt/ssd/research/cfdna/processed/GSE186458_cfdna_harmonized.parquet")
SORTED_DIR   = Path("/mnt/hdd/research/cfdna/raw/GSE186458/pivot/sorted_cells")
MIX_DIR      = Path("/mnt/hdd/research/cfdna/raw/GSE186458/pivot/mixtures")
LOG_DIR      = Path("/mnt/ssd/research/cfdna/logs")
GATES_LOG    = LOG_DIR / "gates.log"
STDERR_LOG   = LOG_DIR / "pivot_phase2_step2b.log"

TISSUE_CELLS = {
    "colon":    ["Colon-Left-Epithelium", "Colon-Right-Epithelium"],
    "liver":    ["Liver-Hepatocytes"],
    "lung":     ["Lung-Alveolar-Epithelium", "Lung-Bronchus-Epithelium"],
    "breast":   ["Breast-Basal-Epithelium", "Breast-Luminal-Epithelium"],
    "pancreas": ["Pancreas-Duct"],
}
FRACTIONS = [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
N_BLOCKS = 9520
SEED = 20260513


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with STDERR_LOG.open("a") as f:
        f.write(line + "\n")


def list_sorted_betas_by_subtype() -> dict[str, list[Path]]:
    """{subtype_name: [.beta paths]}"""
    out = {}
    for tissue, cells in TISSUE_CELLS.items():
        for cell in cells:
            out[cell] = sorted(SORTED_DIR.glob(f"GSM*_{cell}-Z*.hg38.beta"))
    return out


def run_beta_to_blocks(beta_path: Path, out_dir: Path) -> Path:
    """Run beta_to_blocks on a single .beta; return path to .bin."""
    env = os.environ.copy()
    env["PATH"] = PATH_PREFIX + ":" + env.get("PATH", "")
    cmd = [WGBSTOOLS, "beta_to_blocks",
           "--blocks_file", str(ATLAS_BED),
           "-o", str(out_dir),
           "-f", str(beta_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"beta_to_blocks failed for {beta_path.name}: {r.stderr[-300:]}")
    out_bin = out_dir / (beta_path.name.replace(".beta", ".bin"))
    if not out_bin.exists():
        raise FileNotFoundError(f"missing {out_bin}")
    return out_bin


def load_bin(path: Path) -> np.ndarray:
    arr = np.fromfile(path, dtype=np.uint8)
    if arr.size != 2 * N_BLOCKS:
        raise ValueError(f"wrong size {arr.size}")
    return arr.reshape(-1, 2)


def bin_to_beta(arr: np.ndarray) -> np.ndarray:
    m = arr[:, 0].astype(np.float64); t = arr[:, 1].astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(t > 0, m / t, np.nan)


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    MIX_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = random.Random(SEED)

    # 1) Load healthy harmonized atlas-block betas (Phase 1 output)
    log("Loading Phase 1 healthy cfDNA harmonized betas")
    harm = pd.read_parquet(HEALTHY_HARM)
    log(f"  harm shape: {harm.shape}  cols sample: {list(harm.columns[:5])}")
    healthy_donors = [c for c in harm.columns if c.startswith("GSM6810")]
    healthy_donors = sorted(healthy_donors)
    log(f"  healthy donors: {len(healthy_donors)} = {healthy_donors[:3]}...{healthy_donors[-3:]}")

    if N_BLOCKS != harm.shape[0]:
        log(f"FATAL: harm has {harm.shape[0]} rows, expected {N_BLOCKS}")
        sys.exit(1)

    # 2) Compute atlas-block beta for each sorted-cell donor (cache on disk)
    log("Computing atlas-block betas for each sorted-cell .beta")
    sorted_bins_dir = SORTED_DIR / "atlas_bins"
    sorted_bins_dir.mkdir(exist_ok=True)
    tumor_betas_by_subtype: dict[str, list[tuple[str, np.ndarray]]] = {}
    by_subtype = list_sorted_betas_by_subtype()
    for subtype, beta_files in by_subtype.items():
        recs = []
        for bp in beta_files:
            bin_path = sorted_bins_dir / bp.name.replace(".beta", ".bin")
            if not bin_path.exists():
                run_beta_to_blocks(bp, sorted_bins_dir)
            arr = load_bin(bin_path)
            beta_vec = bin_to_beta(arr)
            donor_id = bp.name.split("_")[0]   # GSM5652233 etc.
            recs.append((donor_id, beta_vec))
        tumor_betas_by_subtype[subtype] = recs
        log(f"  {subtype}: {len(recs)} donors  beta_mean(first donor)={np.nanmean(recs[0][1]):.3f}")

    # 3) Compute mixture beta vectors
    log("Generating in-silico mixtures (beta-space)")
    records = []  # for labels
    mix_betas = []  # list of (sample_id, beta_vec)
    # Unmixed healthy first
    for h_id in healthy_donors:
        b_h = harm[h_id].to_numpy(dtype=np.float64)
        sample_id = f"{h_id}_unmixed_healthy"
        mix_betas.append((sample_id, b_h))
        records.append({
            "sample_id": sample_id, "healthy_donor": h_id, "tumor_donor": None,
            "tumor_subtype": None, "tissue_of_origin": "none",
            "target_fraction": 0.0, "cancer_present": 0,
            "mean_beta": float(np.nanmean(b_h)),
        })

    # Mixed
    for h_id in healthy_donors:
        b_h = harm[h_id].to_numpy(dtype=np.float64)
        for tissue, subtypes in TISSUE_CELLS.items():
            for frac in FRACTIONS:
                # Pick a random subtype, then a random donor
                subtype = rng.choice(subtypes)
                t_donor_id, b_t = rng.choice(tumor_betas_by_subtype[subtype])
                # Beta-space mixture
                # Handle NaN: if either side is NaN at a block, fall back to whichever is non-NaN
                valid_h = ~np.isnan(b_h); valid_t = ~np.isnan(b_t)
                b_mix = np.where(valid_h & valid_t,
                                 frac * b_t + (1 - frac) * b_h,
                                 np.where(valid_h, b_h,
                                          np.where(valid_t, b_t, np.nan)))
                sample_id = f"{h_id}_{tissue}_f{frac:.4f}"
                mix_betas.append((sample_id, b_mix))
                records.append({
                    "sample_id": sample_id, "healthy_donor": h_id,
                    "tumor_donor": t_donor_id, "tumor_subtype": subtype,
                    "tissue_of_origin": tissue,
                    "target_fraction": float(frac), "cancer_present": 1,
                    "mean_beta": float(np.nanmean(b_mix)),
                })

    log(f"  total samples: {len(mix_betas)} ({len(healthy_donors)} unmixed + "
        f"{len(healthy_donors)} * {len(TISSUE_CELLS)} * {len(FRACTIONS)} mixed)")

    # 4) Assemble betas matrix
    sample_ids = [s for s, _ in mix_betas]
    beta_arr = np.stack([b for _, b in mix_betas], axis=1).astype(np.float32)
    log(f"  beta matrix: {beta_arr.shape}")

    # Block IDs from atlas BED
    bed = pd.read_csv(ATLAS_BED, sep="\t", header=None,
                      names=["chrom","start","end","cpg_start","cpg_end"])
    block_ids = [f"{r.chrom}:{r.start}-{r.end}" for r in bed.itertuples()]

    out_betas = pd.DataFrame(beta_arr, columns=sample_ids)
    out_betas.insert(0, "block_id", block_ids)
    out_betas.to_parquet("/mnt/ssd/research/cfdna/processed/GSE186458_mixtures_betas.parquet",
                         compression="zstd")
    log("  wrote betas parquet")

    # 5) Labels
    labels_df = pd.DataFrame(records)
    labels_df["tumor_fraction"] = labels_df["target_fraction"]
    out_labels = labels_df[["sample_id","cancer_present","tumor_fraction","tissue_of_origin",
                            "healthy_donor","tumor_donor","tumor_subtype","mean_beta"]]
    out_labels.to_parquet("/mnt/ssd/research/cfdna/processed/GSE186458_mixtures_labels.parquet",
                          compression="zstd")
    log(f"  wrote labels parquet: {len(out_labels)} rows")

    # 6) Manifest
    labels_df.to_csv(MIX_DIR / "manifest.tsv", sep="\t", index=False)
    log(f"  wrote {MIX_DIR/'manifest.tsv'}")

    elapsed_min = (time.time() - t0) / 60
    n_total = len(records)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status = "PASS" if n_total >= 800 else "FAIL"
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] PIVOT_PHASE2_GSE186458 STEP2b_beta_mix: {status} "
                f"n_total={n_total} method=beta-space-mixing "
                f"reason=mix_pat_realized_fraction_was_~10x_below_target "
                f"elapsed_min={elapsed_min:.2f}\n")


if __name__ == "__main__":
    main()

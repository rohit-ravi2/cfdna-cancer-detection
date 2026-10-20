#!/usr/bin/env python3
"""
PIVOT_PHASE2 STEP 3 — Deconvolve all 828 mixture samples (23 unmixed + 805 mixed)
via the validated Phase 1 NNLS pipeline.

Inputs:
    work/processed/GSE186458_mixtures_betas.parquet
    work/processed/GSE186458_mixtures_labels.parquet
    Loyfer atlas at work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv

Outputs:
    work/processed/GSE186458_mixtures_deconv.parquet
    work/results/mixture_calibration.png
    work/results/mixture_calibration.json

Sanity: for each tissue, plot tumor_fraction vs measured tissue cell-fraction
across the 7 fraction levels, and confirm Pearson r is meaningful (>= 0.5).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from joblib import Parallel, delayed

ATLAS_TSV  = Path("work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
META_COLS  = ["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"]
MIX_BETAS  = Path("work/processed/GSE186458_mixtures_betas.parquet")
MIX_LABELS = Path("work/processed/GSE186458_mixtures_labels.parquet")
OUT_DECONV = Path("work/processed/GSE186458_mixtures_deconv.parquet")
RES_DIR    = Path("work/results"); RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR    = Path("work/logs")
STDERR_LOG = LOG_DIR / "pivot_phase2_step3.log"
GATES_LOG  = LOG_DIR / "gates.log"

SUM_PENALTY = 20.0

# Mapping from our 5 tissue labels to the closest atlas cell type column(s)
# that represent the mixture-donor cell type
TISSUE_TO_ATLAS_CT = {
    "colon":    ["Colon-Ep"],     # canonical Loyfer atlas column for colon epithelium
    "liver":    ["Liver-Hep"],
    "lung":     ["Lung-Ep-Alveo", "Lung-Ep-Bron"],
    "breast":   ["Breast-Basal-Ep", "Breast-Luminal-Ep"],
    "pancreas": ["Pancreas-Duct"],
}


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with STDERR_LOG.open("a") as f:
        f.write(line + "\n")


def load_atlas():
    df = pd.read_csv(ATLAS_TSV, sep="\t")
    cts = [c for c in df.columns if c not in META_COLS]
    A_u = df[cts].to_numpy(dtype=np.float64)
    if np.isnan(A_u).any():
        cm = np.nanmean(A_u, axis=0)
        cm = np.where(np.isnan(cm), 0.5, cm)
        ix = np.where(np.isnan(A_u))
        A_u[ix] = np.take(cm, ix[1])
    return A_u, cts, df


def nnls_one(b, A_meth, w):
    mask = ~np.isnan(b)
    if mask.sum() == 0:
        return np.zeros(A_meth.shape[1]), np.nan, 0
    A_s = A_meth[mask]
    b_s = b[mask]
    sw = np.sqrt(np.clip(w[mask], 0, None))
    keep = sw > 0
    sw = sw[keep]; A_s = A_s[keep]; b_s = b_s[keep]
    A_w = A_s * sw[:, None]
    b_w = b_s * sw
    sum_row = np.full((1, A_meth.shape[1]), SUM_PENALTY, dtype=np.float64)
    A_aug = np.vstack([A_w, sum_row])
    b_aug = np.concatenate([b_w, np.array([SUM_PENALTY])])
    try:
        x, resid = nnls(A_aug, b_aug, maxiter=20000)
    except RuntimeError:
        return np.zeros(A_meth.shape[1]), np.nan, int(mask.sum())
    s = x.sum()
    if s > 0:
        x = x / s
    return x, float(resid), int(mask.sum())


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    log("Loading mixtures betas + labels")
    betas_df = pd.read_parquet(MIX_BETAS)
    labels_df = pd.read_parquet(MIX_LABELS)
    log(f"  betas shape: {betas_df.shape}  (rows=blocks, cols=block_id + samples)")
    log(f"  labels n={len(labels_df)} columns={list(labels_df.columns)}")

    sample_cols = [c for c in betas_df.columns if c != "block_id"]
    # Align order with labels
    sample_cols_in_labels = [s for s in sample_cols if s in labels_df["sample_id"].values]
    log(f"  samples present in both: {len(sample_cols_in_labels)}")
    sample_cols = sample_cols_in_labels

    log("Loading atlas")
    A_u, cts, atlas_df = load_atlas()
    log(f"  atlas U-fraction: {A_u.shape}  cell_types={len(cts)}")
    A_meth = 1.0 - A_u

    # Align by block_id (chr:start-end). Atlas rows correspond exactly to the
    # 9520 atlas blocks in the same order as atlas_blocks_hg38.bed used in mix step.
    # Both have same N_BLOCKS — verify.
    if betas_df.shape[0] != A_meth.shape[0]:
        log(f"FATAL: block count mismatch betas={betas_df.shape[0]} atlas={A_meth.shape[0]}")
        sys.exit(1)

    A_sorted = np.sort(A_u, axis=1)
    w_info = A_sorted[:, -1] - A_sorted[:, -2]
    log(f"  informativeness weight: mean={w_info.mean():.3f}  max={w_info.max():.3f}")

    B = betas_df[sample_cols].to_numpy(dtype=np.float64)
    log(f"  beta matrix B: {B.shape}")

    log(f"Running NNLS on {B.shape[1]} samples")
    t1 = time.time()
    results = Parallel(n_jobs=8, backend="loky")(
        delayed(nnls_one)(B[:, j], A_meth, w_info) for j in range(B.shape[1])
    )
    log(f"  NNLS done in {time.time()-t1:.1f}s")

    fracs = np.stack([r[0] for r in results], axis=0)
    deconv_df = pd.DataFrame(fracs, columns=cts)
    deconv_df.insert(0, "sample_id", sample_cols)
    deconv_df["nnls_resid"] = [r[1] for r in results]
    deconv_df["n_blocks_used"] = [r[2] for r in results]

    # Merge labels
    out_df = deconv_df.merge(labels_df, on="sample_id", how="left")
    out_df.to_parquet(OUT_DECONV, compression="zstd")
    log(f"  wrote {OUT_DECONV}  shape={out_df.shape}")

    # ------------- Calibration: tumor_fraction vs measured tissue fraction --
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    cal_summary = {}
    for ax_i, (tissue, atlas_cts_list) in enumerate(TISSUE_TO_ATLAS_CT.items()):
        ax = axes[ax_i]
        # Validate columns exist
        atlas_cols = [c for c in atlas_cts_list if c in deconv_df.columns]
        if not atlas_cols:
            log(f"  WARN: no atlas cell type matches {tissue}: tried {atlas_cts_list}")
            cal_summary[tissue] = {"missing_atlas_cell_types": atlas_cts_list}
            continue
        tissue_fraction = out_df[atlas_cols].sum(axis=1)
        sub = out_df.assign(measured_tissue_frac=tissue_fraction)
        sub = sub[(sub["tissue_of_origin"] == tissue) | (sub["tissue_of_origin"] == "none")]
        # Healthy as 0
        x = sub["tumor_fraction"].to_numpy()
        y = sub["measured_tissue_frac"].to_numpy()
        # Pearson on mixtures only (excluding healthy zero)
        nonzero_mask = x > 0
        if nonzero_mask.sum() >= 5:
            r = np.corrcoef(x[nonzero_mask], y[nonzero_mask])[0, 1]
        else:
            r = float("nan")
        # spearman on all
        from scipy.stats import spearmanr
        sp, _ = spearmanr(x, y)
        cal_summary[tissue] = {
            "atlas_cols_used": atlas_cols,
            "n_healthy": int((x == 0).sum()),
            "n_mixed": int(nonzero_mask.sum()),
            "pearson_r_mix_only": float(r),
            "spearman_r_all": float(sp),
            "healthy_mean_tissue_frac": float(y[~nonzero_mask].mean()),
            "mix_mean_tissue_frac_per_target": {
                f"{f:.4f}": float(y[(x == f)].mean()) if (x == f).any() else None
                for f in [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
            },
        }
        ax.scatter(x, y, alpha=0.5, s=12)
        ax.set_xlabel("target tumor fraction")
        ax.set_ylabel(f"measured tissue fraction\n({'+'.join(atlas_cols)})")
        ax.set_title(f"{tissue}  Pearson(mix)={r:.3f}  Spearman={sp:.3f}")
        ax.set_xscale("symlog", linthresh=1e-3)
        ax.set_yscale("symlog", linthresh=1e-3)
        # Diagonal
        ax.plot([0, 0.15], [0, 0.15], "k--", lw=0.5)
    # last subplot empty
    for ax in axes[5:]:
        ax.axis("off")
    plt.tight_layout()
    fig_path = RES_DIR / "mixture_calibration.png"
    plt.savefig(fig_path, dpi=130)
    plt.close()
    log(f"  saved {fig_path}")

    json_path = RES_DIR / "mixture_calibration.json"
    json.dump(cal_summary, json_path.open("w"), indent=2)
    log(f"  saved {json_path}")

    # Compute pooled pearson on mixtures
    all_pearson = [v.get("pearson_r_mix_only", np.nan) for v in cal_summary.values()
                   if isinstance(v.get("pearson_r_mix_only", None), float)]
    all_pearson = [r for r in all_pearson if not np.isnan(r)]
    mean_r = float(np.mean(all_pearson)) if all_pearson else float("nan")

    elapsed_min = (time.time() - t0) / 60
    status = "PASS" if mean_r >= 0.5 else "WARN" if mean_r >= 0.3 else "FAIL"
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] PIVOT_PHASE2_GSE186458 STEP3_deconv_mixtures: {status} "
                f"mean_pearson_target_vs_measured={mean_r:.3f} "
                f"per_tissue={[(t, round(v.get('pearson_r_mix_only', float('nan')),3)) for t,v in cal_summary.items()]} "
                f"elapsed_min={elapsed_min:.1f}\n")

    if status == "FAIL":
        log(f"FAIL: mean Pearson too low ({mean_r:.3f})")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Day-3 Phase A v2 — Distribution-Sampled Mixtures (TCGA-coverage-aware).

The fix from v1: TCGA 450K only covers ~8% of atlas blocks. Filling NaN
with per-block mean made tumor profiles indistinguishable from healthy.

v2 approach: a "tumor profile" is built block-wise:
  - At blocks where TCGA has real tumor measurements (>=50% of TCGA
    samples have data), sample beta from the GMM fit to those data.
  - At blocks where TCGA does NOT cover, **use the atlas tumor-tissue
    centroid** if the target cell type is the corresponding tumor tissue.
    Atlas U250 has per-cell-type beta for all 9520 blocks for ~40 types
    including Colon-Ep, Liver-Hep, etc.
  - This produces realistic tumor profiles that differ from healthy at
    cell-type-specific markers across the full atlas.

Per-cancer-type tumor profile:
  tumor_profile[b] =
    if TCGA covers block b:  sampled from TCGA-fit per-block Gaussian
    else:                    atlas tumor-tissue-centroid[b]

Mixing:
  b_mix = (1 - tf) * b_healthy + tf * b_tumor + noise

Healthy: sampled from GMM fit to GSE186458 healthy cfDNA (23 samples)
in PCA space, plus per-block measurement noise calibrated from GSE122126.
"""
import json
import time
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

PROC = Path("work/processed")
MODELS = Path("work/models")
LOG = Path("work/logs/day3_phase_a.log")
ATLAS_FULL = Path("work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")

N_BLOCKS = 9520
N_PER_COMBO = 100
TF_LEVELS = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
N_HEALTHY_SAMPLES = 1000
N_PCA_DIM = 18
RANDOM_STATE = 42

CANCER_TO_TISSUE = {
    "COAD": "colon",
    "LIHC": "liver",
    "LUAD": "lung",
    "BRCA": "breast",
    "PAAD": "pancreas",
}
CANCER_TO_ATLAS_CT = {
    "COAD": ["Colon-Ep"],
    "LIHC": ["Liver-Hep"],
    "LUAD": ["Lung-Ep-Alveo", "Lung-Ep-Bron"],
    "BRCA": ["Breast-Luminal-Ep", "Breast-Basal-Ep"],
    "PAAD": ["Pancreas-Acinar", "Pancreas-Duct"],
}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_block_matrix(parquet_path):
    df = pd.read_parquet(parquet_path)
    block_ids = df["block_id"].copy()
    cols = [c for c in df.columns if c != "block_id"]
    X = df[cols].to_numpy(dtype=np.float32).T  # [samples, blocks]
    return block_ids, X, cols


def fit_pca_gmm_full(X, ks=(1, 2, 3, 5), random_state=RANDOM_STATE):
    """Fit GMM after PCA pre-reduction. Returns (gmm, pca, K, BIC, cov_type)."""
    n_samples, n_dims = X.shape
    n_comp = min(N_PCA_DIM, n_samples - 1, n_dims)
    if n_comp <= 1:
        n_comp = max(2, min(8, n_samples - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(X)
    log(f"    PCA: kept {n_comp}/{n_dims}, expvar={pca.explained_variance_ratio_.sum():.3f}")
    best_bic = np.inf
    best = None
    for k in ks:
        if k > n_samples:
            continue
        try:
            cov_type = "full" if n_samples >= 5 * k * n_comp else "diag"
            gmm = GaussianMixture(
                n_components=k, covariance_type=cov_type,
                random_state=random_state, max_iter=300,
                reg_covar=1e-4, n_init=2,
            )
            gmm.fit(Z)
            bic = gmm.bic(Z)
            log(f"    K={k} cov={cov_type} BIC={bic:.1f}")
            if bic < best_bic:
                best_bic = bic
                best = (gmm, k, cov_type)
        except Exception as e:
            log(f"    K={k} fail: {e}")
    return best, pca, best_bic


def sample_pca_gmm(gmm, pca, n, rng):
    """Sample n profiles from GMM in PCA space, project to original space."""
    seed = rng.randint(0, 1 << 30)
    z, _ = gmm.sample(n)
    np.random.RandomState(seed).shuffle(z)
    X = pca.inverse_transform(z)
    return np.clip(X, 0.0, 1.0).astype(np.float32)


def calibrate_noise(parquet_path):
    """Per-block std across GSE122126 healthy-like samples (target test cohort)."""
    df = pd.read_parquet(parquet_path)
    cols_all = [c for c in df.columns if c != "block_id"]
    # Healthy cfDNA + CFDNA_OTHER pool samples
    cols = [
        c for c in cols_all
        if any(kw in c.lower() for kw in
               ["healthy", "young_male", "young_female", "old_male", "old_female"])
    ]
    if len(cols) < 3:
        cols = cols_all[: min(20, len(cols_all))]
    log(f"  noise calib on {len(cols)} GSE122126 samples")
    X = df[cols].to_numpy(dtype=np.float32)
    sigma = np.nanstd(X, axis=1)
    med = np.nanmedian(sigma)
    sigma = np.where(np.isnan(sigma), med, sigma)
    sigma = np.clip(sigma, 0.005, 0.20)
    log(f"  sigma: median={np.median(sigma):.4f} p90={np.percentile(sigma,90):.4f}")
    return sigma.astype(np.float32)


def fit_tcga_per_block_gaussian(X):
    """Per-block mean & std from TCGA tumor data (ignoring NaN).
    Returns (mu, sigma, observed_mask) per block."""
    n_samples, n_blocks = X.shape
    obs_count = (~np.isnan(X)).sum(axis=0)  # [n_blocks]
    obs_mask = obs_count >= max(3, int(0.3 * n_samples))   # need >=30% obs
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0)
    # if too few obs, mark NaN -> handle in mixer
    sigma = np.where(np.isnan(sigma) | (sigma < 1e-4), 0.05, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32), obs_mask


def load_atlas_centroids():
    """Load Loyfer atlas U250 cell-type beta values.
    Returns (block_index_in_order, dict of celltype -> [9520] vector)."""
    df = pd.read_csv(ATLAS_FULL, sep="\t")
    meta = {"chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"}
    ct_cols = [c for c in df.columns if c not in meta]
    centroids = {ct: df[ct].to_numpy(dtype=np.float32) for ct in ct_cols}
    targets = df["target"].astype(str).values
    return centroids, targets, ct_cols


def main():
    t0 = time.time()
    log("=== PHASE A v2: TCGA-coverage-aware Distribution-Sampled Mixtures ===")

    # Step 1: load atlas centroids (per-cell-type beta over all 9520 blocks)
    log("loading Loyfer atlas centroids...")
    atlas_centroids, atlas_targets, atlas_ct_cols = load_atlas_centroids()
    log(f"  atlas: {len(atlas_ct_cols)} cell types, {len(atlas_targets)} blocks")

    # Step 2: load healthy cfDNA and fit GMM
    log("loading GSE186458 healthy cfDNA...")
    cf_blocks, cf_X, _ = load_block_matrix(PROC / "GSE186458_cfdna_harmonized.parquet")
    log(f"  healthy: {cf_X.shape}")
    cf_col_means = np.nanmean(cf_X, axis=0)
    cf_col_means = np.where(np.isnan(cf_col_means), 0.5, cf_col_means)
    cf_filled = np.where(np.isnan(cf_X), cf_col_means[None, :], cf_X)
    # Replace any remaining NaN (should be none) with 0.5
    cf_filled = np.where(np.isnan(cf_filled), 0.5, cf_filled)

    log("fitting healthy GMM...")
    healthy_best, healthy_pca, healthy_bic = fit_pca_gmm_full(
        cf_filled, ks=(1, 2, 3)
    )
    healthy_gmm, healthy_K, healthy_cov = healthy_best
    log(f"  healthy: K={healthy_K} cov={healthy_cov} BIC={healthy_bic:.1f}")
    with open(MODELS / "healthy_gmm.pkl", "wb") as f:
        pickle.dump({"gmm": healthy_gmm, "pca": healthy_pca,
                     "K": healthy_K, "BIC": healthy_bic,
                     "col_means": cf_col_means,
                     "n_train": cf_X.shape[0]}, f)

    # Step 3: per-cancer-type tumor profile sampler
    log("processing TCGA tumor profiles...")
    tcga_info = {}
    for code in ["COAD", "LIHC", "LUAD", "BRCA", "PAAD"]:
        p = PROC / f"tcga_{code}_harmonized.parquet"
        if not p.exists():
            log(f"  {code}: missing")
            continue
        bb, X, _ = load_block_matrix(p)
        assert (bb.values == cf_blocks.values).all()
        mu, sigma, obs_mask = fit_tcga_per_block_gaussian(X)
        log(f"  {code}: TCGA n={X.shape[0]} obs_blocks={obs_mask.sum()} mean_obs_beta={mu[obs_mask].mean():.3f}")
        # For unobserved blocks, use atlas centroids of the tumor-tissue cell type
        atlas_cts = CANCER_TO_ATLAS_CT[code]
        centroid_vec = np.mean([atlas_centroids[ct] for ct in atlas_cts], axis=0)
        # Fallback std for unobserved blocks: small (tumor tissue is a relatively stable centroid)
        unobs_sigma_default = 0.08  # block-to-block tumor heterogeneity at unobserved blocks
        tcga_info[code] = {
            "mu": mu, "sigma": sigma, "obs_mask": obs_mask,
            "atlas_centroid": centroid_vec.astype(np.float32),
            "unobs_sigma": np.float32(unobs_sigma_default),
            "n_train": int(X.shape[0]),
        }
        with open(MODELS / f"tcga_tumor_gmm_{code}.pkl", "wb") as f:
            pickle.dump(tcga_info[code], f)

    # Step 4: noise calibration
    sigma_noise = calibrate_noise(PROC / "GSE122126_harmonized.parquet")

    # Step 5: generate distribution-sampled mixtures
    log("generating mixtures...")
    rng = np.random.RandomState(RANDOM_STATE)
    all_betas = []
    all_labels = []
    sample_idx = 0

    log(f"  {N_HEALTHY_SAMPLES} healthy...")
    h_synth = sample_pca_gmm(healthy_gmm, healthy_pca, N_HEALTHY_SAMPLES, rng)
    h_synth = np.clip(h_synth + rng.randn(*h_synth.shape) * sigma_noise[None, :], 0.0, 1.0)
    for i in range(N_HEALTHY_SAMPLES):
        sid = f"distsampled_healthy_{sample_idx:05d}"
        all_labels.append(dict(
            sample_id=sid, cancer_present=0, tumor_fraction=0.0,
            tissue_of_origin="none", cancer_type="none",
            stage_proxy=0, source="distsampled",
        ))
        all_betas.append(h_synth[i])
        sample_idx += 1

    for code, info in tcga_info.items():
        tissue = CANCER_TO_TISSUE[code]
        log(f"  {code} ({tissue})...")
        mu = info["mu"]; sigma_tcga = info["sigma"]; obs_mask = info["obs_mask"]
        centroid = info["atlas_centroid"]; unobs_sigma = info["unobs_sigma"]
        for tf in TF_LEVELS:
            # Background: healthy synth
            bg = sample_pca_gmm(healthy_gmm, healthy_pca, N_PER_COMBO, rng)
            # Tumor profile per sample: combine TCGA-observed blocks + atlas-centroid blocks
            tumor_mean = np.where(obs_mask, mu, centroid).astype(np.float32)
            tumor_sigma = np.where(
                obs_mask,
                np.clip(sigma_tcga, 0.005, 0.25),
                unobs_sigma,
            ).astype(np.float32)
            tumor = (
                tumor_mean[None, :]
                + rng.randn(N_PER_COMBO, N_BLOCKS).astype(np.float32) * tumor_sigma[None, :]
            )
            tumor = np.clip(tumor, 0.0, 1.0)
            mix = (1.0 - tf) * bg + tf * tumor
            mix = np.clip(mix + rng.randn(*mix.shape).astype(np.float32) * sigma_noise[None, :], 0.0, 1.0)
            if tf <= 0.005: stage = 1
            elif tf <= 0.03: stage = 2
            elif tf <= 0.10: stage = 3
            else: stage = 4
            for i in range(N_PER_COMBO):
                sid = f"distsampled_{code}_tf{int(tf*10000):05d}_{i:03d}"
                all_labels.append(dict(
                    sample_id=sid, cancer_present=1,
                    tumor_fraction=float(tf), tissue_of_origin=tissue,
                    cancer_type=code, stage_proxy=stage, source="distsampled",
                ))
                all_betas.append(mix[i])
                sample_idx += 1

    log(f"total: {len(all_labels)}")
    betas_arr = np.stack(all_betas, axis=0).astype(np.float32)
    labels_df = pd.DataFrame(all_labels)

    betas_df = pd.DataFrame(betas_arr.T, columns=labels_df["sample_id"].tolist())
    betas_df.insert(0, "block_id", cf_blocks.values)
    betas_df.to_parquet(PROC / "distsampled_mixtures_betas.parquet", index=False)
    labels_df.to_parquet(PROC / "distsampled_mixtures_labels.parquet", index=False)
    log(f"saved: betas {betas_df.shape} labels {labels_df.shape}")

    # Step 6: sanity check
    sanity = sanity_check(betas_arr, labels_df, atlas_centroids, atlas_targets, atlas_ct_cols)

    summary = {
        "healthy_K": int(healthy_K),
        "healthy_BIC": float(healthy_bic),
        "healthy_n_train": int(cf_X.shape[0]),
        "tcga_info": {
            code: {
                "n_train": info["n_train"],
                "obs_blocks": int(info["obs_mask"].sum()),
                "mean_obs_beta": float(info["mu"][info["obs_mask"]].mean()),
            }
            for code, info in tcga_info.items()
        },
        "n_total": int(len(all_labels)),
        "n_healthy_synth": int(N_HEALTHY_SAMPLES),
        "n_cancer_synth": int(len(all_labels) - N_HEALTHY_SAMPLES),
        "tf_levels": TF_LEVELS,
        "sanity": sanity,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(MODELS / "phase_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"=== PHASE A v2 DONE in {summary['elapsed_min']} min ===")
    log(json.dumps(summary, indent=2, default=str))
    return summary


def sanity_check(betas_arr, labels_df, atlas_centroids, atlas_targets, atlas_ct_cols):
    """For each cancer type, deconvolve via NNLS and check tumor-tissue
    fraction tracks target TF."""
    from scipy.optimize import nnls
    A = np.stack([atlas_centroids[ct] for ct in atlas_ct_cols], axis=1).astype(np.float32)
    # Impute NaN in atlas with 0.5 (some blocks may lack cell-type signal)
    A = np.where(np.isnan(A), 0.5, A)
    log(f"  NNLS atlas matrix: {A.shape} (NaN imputed at 0.5)")
    tissue_idx = {}
    for code, cts in CANCER_TO_ATLAS_CT.items():
        idxs = [atlas_ct_cols.index(c) for c in cts if c in atlas_ct_cols]
        tissue_idx[code] = idxs

    results = {}
    for code, idxs in tissue_idx.items():
        if not idxs:
            results[code] = None
            continue
        sub = labels_df[labels_df["cancer_type"] == code]
        if len(sub) == 0:
            results[code] = None
            continue
        picks = sub.groupby("tumor_fraction").head(2).index.values
        targets = sub.loc[picks, "tumor_fraction"].to_numpy()
        preds = []
        for i in picks:
            b = betas_arr[i]
            b_f = np.where(np.isnan(b), 0.5, b)
            x, _ = nnls(A, b_f)
            if x.sum() > 0:
                x = x / x.sum()
            preds.append(x[idxs].sum())
        preds = np.array(preds)
        r = float(np.corrcoef(targets, preds)[0, 1]) if (np.std(targets) > 0 and np.std(preds) > 0) else float("nan")
        # Also check tumor-tissue elevation at TF=0.2 vs TF=0.001
        mask_high = targets >= 0.15
        mask_low = targets <= 0.005
        elev = float(preds[mask_high].mean() - preds[mask_low].mean()) if mask_high.any() and mask_low.any() else float("nan")
        log(f"  {code} n={len(picks)} pearson={r:.3f} elev(high-low)={elev:.4f} mean@high={preds[mask_high].mean():.3f} mean@low={preds[mask_low].mean():.3f}")
        results[code] = {"n": int(len(picks)), "pearson": r,
                         "mean_at_high_tf": float(preds[mask_high].mean()),
                         "mean_at_low_tf": float(preds[mask_low].mean()),
                         "elevation": elev}
    return results


if __name__ == "__main__":
    main()

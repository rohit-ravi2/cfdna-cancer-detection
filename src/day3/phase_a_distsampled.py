#!/usr/bin/env python3
"""
Day-3 Phase A — Distribution-Sampled In-Silico Mixtures.

Fix Day-2 brittleness: the FIXED-donor mixture generator made the model
learn discrete centroids. Instead, sample tumor profiles from a GMM fit
to TCGA tumor methylation, and healthy profiles from a GMM fit to
GSE186458 cfDNA (+WBC) healthy samples.

Outputs:
  /mnt/ssd/research/cfdna/models/tcga_tumor_gmm_<TYPE>.pkl
  /mnt/ssd/research/cfdna/models/healthy_gmm.pkl
  /mnt/ssd/research/cfdna/processed/distsampled_mixtures_betas.parquet
  /mnt/ssd/research/cfdna/processed/distsampled_mixtures_labels.parquet
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

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
MODELS.mkdir(parents=True, exist_ok=True)
LOG = Path("/mnt/ssd/research/cfdna/logs/day3_phase_a.log")

N_BLOCKS = 9520
N_PER_COMBO = 100
TF_LEVELS = [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
N_HEALTHY_SAMPLES = 1000
N_PCA_DIM = 24   # dimensionality reduction for GMM (more stable than raw 9520)
RANDOM_STATE = 42

CANCER_TYPES = {
    "COAD": "colon",
    "LIHC": "liver",
    "LUAD": "lung",
    "BRCA": "breast",
    "PAAD": "pancreas",
}


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def load_block_matrix(parquet_path):
    """Return (block_ids: pd.Series, beta_matrix: ndarray [n_samples, n_blocks])."""
    df = pd.read_parquet(parquet_path)
    block_ids = df["block_id"].copy()
    cols = [c for c in df.columns if c != "block_id"]
    X = df[cols].to_numpy(dtype=np.float32).T  # [samples, blocks]
    return block_ids, X, cols


def fill_per_block_mean(X):
    """Fill NaN with per-block mean (column-wise mean), ignoring NaN."""
    X = X.copy()
    col_means = np.nanmean(X, axis=0)
    col_means = np.where(np.isnan(col_means), 0.5, col_means)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])
    return X, col_means


def fit_gmm_with_bic(X, ks=(1, 2, 3, 5, 8), random_state=RANDOM_STATE):
    """Fit GMM at multiple K and pick lowest BIC. Uses PCA pre-reduction."""
    n_samples, n_dims = X.shape
    # PCA pre-reduction for stability when n_samples << n_dims
    n_comp = min(N_PCA_DIM, n_samples - 1, n_dims)
    if n_comp <= 1:
        n_comp = max(2, min(8, n_samples - 1))
    pca = PCA(n_components=n_comp, random_state=random_state)
    Z = pca.fit_transform(X)
    log(f"    PCA: kept {n_comp}/{n_dims} dims, explained_var={pca.explained_variance_ratio_.sum():.3f}")

    best = None
    best_bic = np.inf
    for k in ks:
        if k > n_samples:
            continue
        try:
            cov_type = "full" if n_samples >= 5 * k * n_comp else "diag"
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=cov_type,
                random_state=random_state,
                max_iter=300,
                reg_covar=1e-4,
                n_init=2,
            )
            gmm.fit(Z)
            bic = gmm.bic(Z)
            log(f"    K={k} cov={cov_type} BIC={bic:.1f} converged={gmm.converged_}")
            if bic < best_bic:
                best_bic = bic
                best = (k, gmm, cov_type)
        except Exception as e:
            log(f"    K={k} FAIL: {e}")
    return best, pca, best_bic


def sample_from_pca_gmm(gmm, pca, n, random_state=None):
    """Sample n synthetic profiles in original space using the GMM in PCA space."""
    rng = np.random.RandomState(random_state)
    z, _ = gmm.sample(n)
    rng.shuffle(z)
    X = pca.inverse_transform(z)
    # clamp to [0,1] since these are beta values
    return np.clip(X, 0.0, 1.0).astype(np.float32)


def calibrate_per_block_noise(parquet_path, healthy_sample_ids=None):
    """Compute per-block std across healthy samples in GSE122126 to simulate
    measurement noise levels matching the target test cohort."""
    df = pd.read_parquet(parquet_path)
    cols_all = [c for c in df.columns if c != "block_id"]
    if healthy_sample_ids is None:
        cols = [c for c in cols_all if "healthy" in c.lower() or "young_male" in c.lower() or "young_female" in c.lower() or "old_male" in c.lower() or "old_female" in c.lower()]
        if not cols:
            cols = cols_all[: min(10, len(cols_all))]
    else:
        cols = [c for c in cols_all if c in healthy_sample_ids]
    log(f"    noise calibration on {len(cols)} GSE122126 healthy/CFDNA_OTHER samples")
    X = df[cols].to_numpy(dtype=np.float32)
    # std per block ignoring NaN
    sigma = np.nanstd(X, axis=1)
    # Replace NaN sigmas with median sigma
    med = np.nanmedian(sigma)
    sigma = np.where(np.isnan(sigma), med, sigma)
    sigma = np.clip(sigma, 0.005, 0.20)
    log(f"    per-block sigma: median={np.median(sigma):.4f} p90={np.percentile(sigma,90):.4f} max={sigma.max():.4f}")
    return sigma


def main():
    t0 = time.time()
    log("=== PHASE A: Distribution-Sampled Mixtures ===")

    # Step 1: load atlas block ids from first TCGA file
    log("loading TCGA matrices...")
    tcga_blocks = {}
    tcga_X = {}
    for code in ["COAD", "LIHC", "LUAD", "BRCA", "PAAD"]:
        p = PROC / f"tcga_{code}_harmonized.parquet"
        if not p.exists():
            log(f"  {code}: MISSING (PAAD may still be downloading)")
            continue
        block_ids, X, _ = load_block_matrix(p)
        tcga_blocks[code] = block_ids
        tcga_X[code] = X
        log(f"  {code}: shape={X.shape} cov_per_sample={(~np.isnan(X)).mean(axis=1).mean():.3f}")

    # Step 2: load healthy data
    log("loading GSE186458 cfdna (healthy) + WBC...")
    cf_blocks, cf_X, _ = load_block_matrix(PROC / "GSE186458_cfdna_harmonized.parquet")
    log(f"  cfdna: shape={cf_X.shape} cov={(~np.isnan(cf_X)).mean():.3f}")

    # WBC: not separately stored, but cfdna parquet has both populations.
    # Check sample IDs: GSE186458_cfdna_uxm_family.tsv lists labels
    fam_path = PROC / "GSE186458_cfdna_uxm_family.tsv"
    if fam_path.exists():
        fam = pd.read_csv(fam_path, sep="\t")
        log(f"  family file: {fam.shape} cols={list(fam.columns)[:5]}")

    # Step 3: align all matrices to a SHARED block set (intersection of
    # block-coverage with healthy cfdna; TCGA already has full atlas IDs)
    # All matrices use same atlas; we just need the index aligned.
    base_blocks = cf_blocks.tolist()
    assert all(tcga_blocks[c].tolist() == base_blocks for c in tcga_X), "block id mismatch"

    # For each (cancer type, block) keep blocks where TCGA has at least 50% coverage
    # AND healthy cfDNA has at least 1 observation
    cf_cov = (~np.isnan(cf_X)).mean(axis=0)  # per block fraction of samples with data
    cf_block_mask = cf_cov >= 0.5
    log(f"  healthy block keep mask: {cf_block_mask.sum()}/{N_BLOCKS}")

    # Step 4: noise model calibrated on GSE122126 healthy cfDNA samples
    sigma = calibrate_per_block_noise(PROC / "GSE122126_harmonized.parquet")

    # Step 5: fit healthy GMM
    log("fitting healthy GMM...")
    # Fill NaN with per-block mean (computed from healthy only)
    cf_filled, cf_means = fill_per_block_mean(cf_X)
    healthy_best, healthy_pca, healthy_bic = fit_gmm_with_bic(
        cf_filled, ks=(1, 2, 3)
    )
    healthy_k, healthy_gmm, healthy_cov = healthy_best
    log(f"  healthy GMM: K={healthy_k} cov={healthy_cov} BIC={healthy_bic:.1f}")
    with open(MODELS / "healthy_gmm.pkl", "wb") as f:
        pickle.dump(
            {
                "gmm": healthy_gmm,
                "pca": healthy_pca,
                "K": healthy_k,
                "BIC": healthy_bic,
                "col_means": cf_means,
                "n_train": cf_X.shape[0],
            },
            f,
        )

    # Step 6: fit TCGA tumor GMMs
    log("fitting TCGA tumor GMMs...")
    tumor_gmms = {}
    for code, X in tcga_X.items():
        log(f"  {code}: filling NaN, fitting GMM (n={X.shape[0]})...")
        Xf, means = fill_per_block_mean(X)
        best, pca_t, bic = fit_gmm_with_bic(Xf, ks=(1, 2, 3, 5))
        if best is None:
            log(f"    {code}: GMM fit FAILED, falling back to single Gaussian via K=1")
            continue
        k, gmm_t, cov_t = best
        log(f"    {code}: K={k} cov={cov_t} BIC={bic:.1f}")
        tumor_gmms[code] = {
            "gmm": gmm_t,
            "pca": pca_t,
            "K": k,
            "BIC": bic,
            "cov_type": cov_t,
            "col_means": means,
            "n_train": X.shape[0],
        }
        with open(MODELS / f"tcga_tumor_gmm_{code}.pkl", "wb") as f:
            pickle.dump(tumor_gmms[code], f)

    # Step 7: generate distribution-sampled mixtures
    log("generating distribution-sampled mixtures...")
    rng = np.random.RandomState(RANDOM_STATE)
    all_betas = []
    all_labels = []
    sample_idx = 0

    # Healthy synthetic samples
    log(f"  generating {N_HEALTHY_SAMPLES} synthetic healthy...")
    h_synthetic = sample_from_pca_gmm(
        healthy_gmm, healthy_pca, N_HEALTHY_SAMPLES, random_state=RANDOM_STATE
    )
    # add measurement noise
    h_synthetic = np.clip(h_synthetic + rng.randn(*h_synthetic.shape) * sigma, 0.0, 1.0)
    for i in range(N_HEALTHY_SAMPLES):
        sid = f"distsampled_healthy_{sample_idx:05d}"
        all_labels.append({
            "sample_id": sid,
            "cancer_present": 0,
            "tumor_fraction": 0.0,
            "tissue_of_origin": "none",
            "cancer_type": "none",
            "stage_proxy": 0,
            "source": "distsampled",
        })
        all_betas.append(h_synthetic[i])
        sample_idx += 1

    # Cancer-type mixtures
    for code, info in tumor_gmms.items():
        tissue = CANCER_TYPES[code]
        log(f"  {code} ({tissue}): generating mixtures across {len(TF_LEVELS)} TF levels...")
        for tf in TF_LEVELS:
            # sample N_PER_COMBO healthy backgrounds and N_PER_COMBO tumor profiles
            h_bg = sample_from_pca_gmm(
                healthy_gmm, healthy_pca, N_PER_COMBO, random_state=rng.randint(0, 1 << 30)
            )
            t_prof = sample_from_pca_gmm(
                info["gmm"], info["pca"], N_PER_COMBO, random_state=rng.randint(0, 1 << 30)
            )
            mix = (1.0 - tf) * h_bg + tf * t_prof
            mix = np.clip(mix + rng.randn(*mix.shape) * sigma, 0.0, 1.0)
            for i in range(N_PER_COMBO):
                sid = f"distsampled_{code}_tf{int(tf*10000):05d}_{i:03d}"
                # stage proxy: small TF = early, large TF = late
                if tf <= 0.005: stage = 1
                elif tf <= 0.03: stage = 2
                elif tf <= 0.10: stage = 3
                else: stage = 4
                all_labels.append({
                    "sample_id": sid,
                    "cancer_present": 1,
                    "tumor_fraction": float(tf),
                    "tissue_of_origin": tissue,
                    "cancer_type": code,
                    "stage_proxy": stage,
                    "source": "distsampled",
                })
                all_betas.append(mix[i])
                sample_idx += 1

    log(f"total distsampled samples: {len(all_labels)}")

    # Stack and save
    betas_arr = np.stack(all_betas, axis=0).astype(np.float32)  # [N, blocks]
    labels_df = pd.DataFrame(all_labels)

    # Save as parquet (rows=blocks, cols=samples like the v2 mixtures)
    betas_df = pd.DataFrame(
        betas_arr.T,
        columns=labels_df["sample_id"].tolist(),
    )
    betas_df.insert(0, "block_id", cf_blocks.values)
    betas_df.to_parquet(PROC / "distsampled_mixtures_betas.parquet", index=False)
    labels_df.to_parquet(PROC / "distsampled_mixtures_labels.parquet", index=False)
    log(f"saved betas: {betas_df.shape}, labels: {labels_df.shape}")

    # Step 8: sanity check via NNLS deconv on 10 samples per cancer type
    log("sanity check: NNLS deconv on subset...")
    sanity_results = sanity_check_deconv(betas_arr, labels_df, cf_blocks)

    # Final summary
    summary = {
        "healthy_GMM_K": int(healthy_k),
        "healthy_GMM_BIC": float(healthy_bic),
        "healthy_n_train": int(cf_X.shape[0]),
        "tumor_GMMs": {
            code: {
                "K": int(info["K"]),
                "BIC": float(info["BIC"]),
                "cov_type": info["cov_type"],
                "n_train": int(info["n_train"]),
            }
            for code, info in tumor_gmms.items()
        },
        "n_distsampled_total": int(len(all_labels)),
        "n_distsampled_healthy": int(N_HEALTHY_SAMPLES),
        "n_distsampled_cancer": int(len(all_labels) - N_HEALTHY_SAMPLES),
        "tf_levels": TF_LEVELS,
        "sanity_check": sanity_results,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(MODELS / "phase_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"=== PHASE A DONE in {summary['elapsed_min']} min ===")
    log(f"summary: {json.dumps(summary, indent=2)}")
    return summary


def sanity_check_deconv(betas_arr, labels_df, block_ids):
    """Run NNLS deconv on a subset and check tumor_fraction prediction
    correlates with target TF per cancer type."""
    from scipy.optimize import nnls
    # Load Loyfer atlas (cell-type beta values per block)
    atlas_p = Path("/mnt/hdd/research/cfdna/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
    if not atlas_p.exists():
        log("  Loyfer atlas not found; skipping sanity")
        return {}

    atlas_df = pd.read_csv(atlas_p, sep="\t")
    log(f"  atlas shape: {atlas_df.shape}")
    meta_cols = {"chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"}
    ct_cols = [c for c in atlas_df.columns if c not in meta_cols]
    log(f"  cell types: {len(ct_cols)}")
    A = atlas_df[ct_cols].to_numpy(dtype=np.float32)  # [blocks, cell_types]
    # impute atlas missing with 0.5
    A = np.where(np.isnan(A), 0.5, A)

    # Identify tissue-of-origin column for each cancer type
    tissue_keywords = {
        "COAD": ["Colon-Ep"],
        "LIHC": ["Liver-Hep"],
        "LUAD": ["Lung-Ep-Alveo", "Lung-Ep-Bron"],
        "BRCA": ["Breast-Luminal-Ep", "Breast-Basal-Ep"],
        "PAAD": ["Pancreas-Acinar", "Pancreas-Duct", "Pancreas-Alpha", "Pancreas-Beta", "Pancreas-Delta"],
    }
    tissue_idx = {}
    for code, kws in tissue_keywords.items():
        idxs = []
        for kw in kws:
            for j, ct in enumerate(ct_cols):
                if kw == ct:
                    idxs.append(j)
        idxs = list(set(idxs))
        tissue_idx[code] = idxs
        log(f"  {code} target cell-type cols: {[ct_cols[j] for j in idxs]}")

    results = {}
    for code, idxs in tissue_idx.items():
        if not idxs:
            results[code] = None
            continue
        # pick 10 samples per TF level for this code
        sub = labels_df[labels_df["cancer_type"] == code]
        if len(sub) == 0:
            results[code] = None
            continue
        # 2 per TF level
        picks = sub.groupby("tumor_fraction").head(2).index.values
        targets = sub.loc[picks, "tumor_fraction"].to_numpy()
        preds = []
        for i in picks:
            b = betas_arr[i]
            # impute NaN -> 0.5
            b_f = np.where(np.isnan(b), 0.5, b)
            x, _ = nnls(A, b_f)
            if x.sum() > 0:
                x = x / x.sum()
            pred = x[idxs].sum()
            preds.append(pred)
        preds = np.array(preds)
        if len(preds) >= 3 and np.std(targets) > 0 and np.std(preds) > 0:
            r = np.corrcoef(targets, preds)[0, 1]
        else:
            r = float("nan")
        log(f"  {code} sanity: n={len(picks)} pearson(target,pred)={r:.3f} target_range=[{targets.min():.3f},{targets.max():.3f}] pred_range=[{preds.min():.3f},{preds.max():.3f}]")
        results[code] = {
            "n_picks": int(len(picks)),
            "pearson": float(r) if not np.isnan(r) else None,
            "target_mean_at_tf005": float(np.mean(preds[targets > 0.04])) if (targets > 0.04).any() else None,
            "target_mean_at_tf001": float(np.mean(preds[targets < 0.005])) if (targets < 0.005).any() else None,
        }
    return results


if __name__ == "__main__":
    main()

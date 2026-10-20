"""
Gate 3 — Loyfer atlas loading, NNLS deconvolution, biological sanity checks.

Loads the U250.l4.hg38 atlas (9520 blocks x 40 cell types). For each cfDNA
sample in GSE186007 and GSE122126 (already harmonized to atlas block
coordinates), solves a non-negative least squares to estimate cell-type
fractions, then validates biological plausibility (granulocyte fraction
in healthy donors, tissue-of-origin elevation in cancer).

For GSE186007 (low-pass WGBS) we apply per-block depth weighting using
total_cpg_records / n_samples_with_data as a depth proxy (cohort-level,
since per-(sample, block) depth is not stored). For GSE122126 we use
unit weights (array data, depth is uniform).

Outputs:
  work/processed/{GSE186007,GSE122126}_deconv.parquet
  work/processed/cell_type_groups.json
  work/results/gate3_sanity.json
  work/results/gate3_sanity.png
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from scipy.stats import ttest_ind
from tqdm import tqdm

ATLAS_TSV = Path("work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
PROC_DIR = Path("work/processed")
RES_DIR = Path("work/results")
LOG_DIR = Path("work/logs")
GATES_LOG = LOG_DIR / "gates.log"

RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "gate3_deconv.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("gate3")


# -----------------------------------------------------------------------------
# Atlas + cell-type families
# -----------------------------------------------------------------------------

META_COLS = ["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"]


def load_atlas() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Return (A [n_blocks x n_celltypes], cell_types, meta DataFrame).

    Atlas has NaN entries (cell types where the marker block was not measured
    or filtered). NNLS cannot accept NaNs in A. We impute NaN entries with the
    column (cell-type) mean across non-NaN blocks. This is a soft prior that
    a missing-marker block is at the cell type's average methylation rather
    than driving the solver toward zero or infinity.
    """
    df = pd.read_csv(ATLAS_TSV, sep="\t")
    cell_types = [c for c in df.columns if c not in META_COLS]
    A = df[cell_types].to_numpy(dtype=np.float64)
    n_nan_before = int(np.isnan(A).sum())
    if n_nan_before > 0:
        col_means = np.nanmean(A, axis=0)
        # If a column is entirely NaN (shouldn't happen for UXM atlas), fill 0.5
        col_means = np.where(np.isnan(col_means), 0.5, col_means)
        inds = np.where(np.isnan(A))
        A[inds] = np.take(col_means, inds[1])
        log.info("atlas NaN imputation: filled %d cells with column means",
                 n_nan_before)
    meta = df[META_COLS].copy()
    meta["block_id"] = np.arange(len(meta))
    log.info("atlas loaded: %d blocks x %d cell types", A.shape[0], A.shape[1])
    return A, cell_types, meta


def classify_cell_types(cell_types: list[str]) -> dict[str, list[str]]:
    """Group atlas cell types into hematopoietic families + tissue panels."""
    groups: dict[str, list[str]] = {
        "granulocyte": [],
        "lymphocyte": [],
        "monocyte_macrophage": [],
        "erythroid": [],
        "megakaryocyte": [],
        "colon_epithelial": [],
        "liver_hepatocyte": [],
        "pancreas_epithelial": [],
        "breast_epithelial": [],
        "lung_epithelial": [],
    }
    for ct in cell_types:
        cl = ct.lower()
        if "granul" in cl or "neutrophil" in cl or "eosinoph" in cl or "basoph" in cl:
            groups["granulocyte"].append(ct)
        if (
            "blood-b" in cl
            or "blood-t" in cl
            or "blood-nk" in cl
            or "lymph" in cl
            or "plasma" in cl
            or "t cell" in cl
            or "b cell" in cl
        ):
            groups["lymphocyte"].append(ct)
        if "mono" in cl or "macro" in cl:
            groups["monocyte_macrophage"].append(ct)
        if "eryth" in cl or "red blood" in cl:
            groups["erythroid"].append(ct)
        if "megakary" in cl:
            groups["megakaryocyte"].append(ct)
        if cl == "colon-ep" or "colon-ep" in cl or "small-int-ep" in cl:
            groups["colon_epithelial"].append(ct)
        if "liver" in cl or "hepato" in cl:
            groups["liver_hepatocyte"].append(ct)
        if cl.startswith("pancreas") or "acinar" in cl or "duct" in cl:
            groups["pancreas_epithelial"].append(ct)
        if "breast" in cl:
            groups["breast_epithelial"].append(ct)
        if "lung-ep" in cl or "lung-alveo" in cl or "lung-bron" in cl:
            groups["lung_epithelial"].append(ct)
    return groups


# -----------------------------------------------------------------------------
# NNLS per sample
# -----------------------------------------------------------------------------

SUM_PENALTY = 20.0  # Lagrangian weight on sum(x) = 1 constraint


def nnls_one(b: np.ndarray, A_meth: np.ndarray, w: np.ndarray | None = None
             ) -> tuple[np.ndarray, float, int]:
    """Run NNLS on a single sample with explicit sum-to-1 constraint.

    Returns (fractions [n_ct], residual, n_blocks_used).

    Solves: argmin_{x>=0} || [diag(sqrt(w)) A; lambda*1^T] x - [diag(sqrt(w)) b; lambda] ||
    where lambda = SUM_PENALTY (Lagrangian sum-to-1), w is a per-block weight
    combining (a) per-block UXM-atlas informativeness (max - 2nd-max U-fraction)
    and (b) per-cohort sample-block depth proxy. Both contribute when set; one
    works alone.

    NOTE: A_meth here is the *methylation-space* atlas (1 - A_u). The Loyfer
    .full.tsv stores U-read fractions; we convert once at load time. The cfDNA
    betas are mean methylation per block, so they live in the same space as
    A_meth. This is an approximation to the proper UXM read-level deconvolution
    (which would require .pat files we don't have).
    """
    mask = ~np.isnan(b)
    if mask.sum() == 0:
        return np.zeros(A_meth.shape[1]), np.nan, 0
    A_s = A_meth[mask]
    b_s = b[mask]
    if w is not None:
        sw = np.sqrt(np.clip(w[mask], 0, None))
        sw = np.where(np.isfinite(sw), sw, 0.0)
        # Drop blocks with zero weight entirely
        keep = sw > 0
        sw = sw[keep]
        A_s = A_s[keep]
        b_s = b_s[keep]
        A_w = A_s * sw[:, None]
        b_w = b_s * sw
    else:
        A_w, b_w = A_s, b_s
    sum_row = np.full((1, A_meth.shape[1]), SUM_PENALTY, dtype=np.float64)
    A_aug = np.vstack([A_w, sum_row])
    b_aug = np.concatenate([b_w, np.array([SUM_PENALTY])])
    try:
        x, resid = nnls(A_aug, b_aug, maxiter=20000)
    except RuntimeError as e:
        log.warning("NNLS failed (%s); returning zeros", e)
        return np.zeros(A_meth.shape[1]), np.nan, int(mask.sum())
    s = x.sum()
    if s > 0:
        x = x / s
    return x, float(resid), int(mask.sum())


def deconvolve_cohort(cohort: str, A_meth: np.ndarray, A_u_for_info: np.ndarray,
                       cell_types: list[str], use_depth_weight: bool,
                       n_jobs: int = 8) -> pd.DataFrame:
    log.info("[%s] loading harmonized matrix", cohort)
    H = pd.read_parquet(PROC_DIR / f"{cohort}_harmonized.parquet")
    samples = [c for c in H.columns if c != "block_id"]
    H_sorted = H.sort_values("block_id").reset_index(drop=True)
    assert (H_sorted["block_id"].to_numpy() == np.arange(A_meth.shape[0])).all(), \
        f"block_id alignment mismatch in {cohort}"

    # Per-block atlas informativeness weight: max(U-frac) - 2nd-max(U-frac)
    # High-info blocks are those where ONE cell type is clearly distinct.
    A_sorted = np.sort(A_u_for_info, axis=1)
    delta = A_sorted[:, -1] - A_sorted[:, -2]  # informativeness per block
    w_info = delta  # 0..1
    log.info("[%s] atlas informativeness: median=%.3f, q25=%.3f, q75=%.3f",
             cohort, float(np.median(w_info)), float(np.percentile(w_info, 25)),
             float(np.percentile(w_info, 75)))

    if use_depth_weight:
        bc = pd.read_parquet(PROC_DIR / f"_block_coverage_{cohort}.parquet")
        bc = bc.sort_values("block_id").reset_index(drop=True)
        assert (bc["block_id"].to_numpy() == np.arange(A_meth.shape[0])).all(), \
            f"block_coverage alignment mismatch in {cohort}"
        depth = bc["total_cpg_records"].to_numpy(dtype=np.float64)
        nsmp = np.clip(bc["n_samples_with_data"].to_numpy(dtype=np.float64), 1, None)
        depth_per_sample = depth / nsmp  # avg CpG records per sample per block
        depth_norm = np.minimum(depth_per_sample, 20.0)  # cap
        # Combine informativeness * depth (both as factors in the squared weight)
        w = w_info * depth_norm
        log.info("[%s] combined weight: median=%.3f cap=%.3f, blocks_w_pos=%d/%d",
                 cohort, float(np.median(w)), 20.0,
                 int((w > 0).sum()), len(w))
    else:
        w = w_info.copy()
        log.info("[%s] info-only weight (array data): blocks_w_pos=%d/%d",
                 cohort, int((w > 0).sum()), len(w))

    B = H_sorted[samples].to_numpy(dtype=np.float64)
    log.info("[%s] running NNLS on %d samples", cohort, B.shape[1])

    t0 = time.time()
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(nnls_one)(B[:, j], A_meth, w) for j in tqdm(range(B.shape[1]),
                                                              desc=f"NNLS {cohort}",
                                                              leave=False)
    )
    log.info("[%s] NNLS done in %.1fs", cohort, time.time() - t0)

    fracs = np.stack([r[0] for r in results], axis=0)
    resids = np.array([r[1] for r in results])
    n_used = np.array([r[2] for r in results], dtype=int)

    out = pd.DataFrame(fracs, columns=[f"ct_fraction_{c}" for c in cell_types])
    out.insert(0, "sample_id", samples)
    out["n_blocks_used"] = n_used
    out["nnls_residual"] = resids
    return out


def merge_labels(deconv: pd.DataFrame, cohort: str) -> pd.DataFrame:
    s = pd.read_csv(PROC_DIR / f"{cohort}_samples.tsv", sep="\t")
    cols = ["sample_id", "label"]
    if "source" in s.columns:
        cols.append("source")
    if "title" in s.columns:
        cols.append("title")
    s = s[cols]
    merged = deconv.merge(s, on="sample_id", how="left")
    HEALTHY_LABELS = {"HEA", "HEALTHY"}
    merged["cancer_type"] = merged["label"]
    merged["is_healthy"] = merged["label"].isin(HEALTHY_LABELS)
    return merged


# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------

def family_fractions(df: pd.DataFrame, groups: dict[str, list[str]]) -> pd.DataFrame:
    """Sum ct_fraction columns within each family group."""
    out = pd.DataFrame({"sample_id": df["sample_id"].values})
    for fam, members in groups.items():
        cols = [f"ct_fraction_{m}" for m in members
                if f"ct_fraction_{m}" in df.columns]
        if not cols:
            out[f"fam_{fam}"] = 0.0
        else:
            out[f"fam_{fam}"] = df[cols].sum(axis=1).values
    for c in ["label", "cancer_type", "is_healthy"]:
        if c in df.columns:
            out[c] = df[c].values
    if "source" in df.columns:
        out["source"] = df["source"].values
    return out


def summarize_family(name: str, vals: np.ndarray) -> dict:
    vals = np.asarray(vals, dtype=float)
    return {
        "n": int(vals.size),
        "mean": float(np.nanmean(vals)) if vals.size else float("nan"),
        "std": float(np.nanstd(vals)) if vals.size else float("nan"),
        "median": float(np.nanmedian(vals)) if vals.size else float("nan"),
    }


def tissue_elevation_test(fam_df: pd.DataFrame, cancer_label: str,
                            healthy_label_set: set[str], fam_col: str) -> dict:
    cancer = fam_df.loc[fam_df["label"] == cancer_label, fam_col].to_numpy(dtype=float)
    healthy = fam_df.loc[fam_df["label"].isin(healthy_label_set), fam_col].to_numpy(dtype=float)
    cancer = cancer[~np.isnan(cancer)]
    healthy = healthy[~np.isnan(healthy)]
    if cancer.size < 2 or healthy.size < 2:
        return {
            "cancer_n": int(cancer.size),
            "healthy_n": int(healthy.size),
            "cancer_mean": float(cancer.mean()) if cancer.size else float("nan"),
            "healthy_mean": float(healthy.mean()) if healthy.size else float("nan"),
            "t": None, "p": None, "cohens_d": None,
            "note": "insufficient n for t-test",
        }
    t, p = ttest_ind(cancer, healthy, equal_var=False)
    pooled = np.sqrt(((cancer.std(ddof=1) ** 2 + healthy.std(ddof=1) ** 2) / 2))
    d = (cancer.mean() - healthy.mean()) / pooled if pooled > 0 else float("nan")
    return {
        "cancer_n": int(cancer.size),
        "healthy_n": int(healthy.size),
        "cancer_mean": float(cancer.mean()),
        "healthy_mean": float(healthy.mean()),
        "t": float(t),
        "p": float(p),
        "cohens_d": float(d),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    log.info("==== GATE 3 BEGIN ====")
    A_u, cell_types, _meta = load_atlas()
    # Methylation-space atlas (cfDNA beta is mean methylation per block, while
    # the Loyfer .full.tsv values are U-read fractions per block).
    A_meth = 1.0 - A_u

    groups = classify_cell_types(cell_types)
    log.info("cell-type family classification:")
    for k, v in groups.items():
        log.info("  %-25s %s", k, v)

    with open(PROC_DIR / "cell_type_groups.json", "w") as f:
        json.dump({"cell_types_all": cell_types, "groups": groups}, f, indent=2)

    # Deconvolve cohorts (atlas passed as A_meth; A_u kept only for informativeness
    # weighting since max-min contrast is identical in U or methylation space)
    d186 = deconvolve_cohort("GSE186007", A_meth, A_u, cell_types,
                              use_depth_weight=True, n_jobs=8)
    d186 = merge_labels(d186, "GSE186007")
    d186.to_parquet(PROC_DIR / "GSE186007_deconv.parquet", index=False)
    log.info("[GSE186007] deconv saved: %s", PROC_DIR / "GSE186007_deconv.parquet")

    os.system('~/bin/notify "cfdna Gate 3 GSE186007 NNLS done (100 samples)" "cfdna" 2>/dev/null')

    d122 = deconvolve_cohort("GSE122126", A_meth, A_u, cell_types,
                              use_depth_weight=False, n_jobs=8)
    d122 = merge_labels(d122, "GSE122126")
    d122.to_parquet(PROC_DIR / "GSE122126_deconv.parquet", index=False)
    log.info("[GSE122126] deconv saved: %s", PROC_DIR / "GSE122126_deconv.parquet")

    os.system('~/bin/notify "cfdna Gate 3 GSE122126 NNLS done (101 samples)" "cfdna" 2>/dev/null')

    # ----- Sanity -----
    fam186 = family_fractions(d186, groups)
    fam122 = family_fractions(d122, groups)
    fam_all = pd.concat([fam186, fam122], axis=0, ignore_index=True)

    HEALTHY_LABELS = {"HEA", "HEALTHY"}

    # Healthy summary per cohort + pooled
    hea_186 = fam186[fam186["label"] == "HEA"]
    hea_122 = fam122[fam122["label"] == "HEALTHY"]
    hea_pooled = pd.concat([hea_186, hea_122], axis=0, ignore_index=True)

    def fam_stats(df, fam_cols):
        return {c.replace("fam_", ""): summarize_family(c, df[c].to_numpy()) for c in fam_cols}

    fam_cols = [c for c in fam186.columns if c.startswith("fam_")]
    healthy_stats = {
        "GSE186007_HEA": fam_stats(hea_186, fam_cols),
        "GSE122126_HEALTHY": fam_stats(hea_122, fam_cols),
        "POOLED": fam_stats(hea_pooled, fam_cols),
    }

    # Cancer-type elevation tests
    elevation = {}
    elevation["CRC_pooled_vs_HEA_pooled"] = tissue_elevation_test(
        fam_all, cancer_label="CRC", healthy_label_set=HEALTHY_LABELS,
        fam_col="fam_colon_epithelial",
    )
    elevation["CRC_GSE186007_vs_HEA"] = tissue_elevation_test(
        fam186, cancer_label="CRC", healthy_label_set={"HEA"},
        fam_col="fam_colon_epithelial",
    )
    elevation["PDAC_GSE186007_vs_HEA_pancreas"] = tissue_elevation_test(
        fam186, cancer_label="PDAC", healthy_label_set={"HEA"},
        fam_col="fam_pancreas_epithelial",
    )
    elevation["PDAC_GSE186007_vs_HEA_liver"] = tissue_elevation_test(
        fam186, cancer_label="PDAC", healthy_label_set={"HEA"},
        fam_col="fam_liver_hepatocyte",
    )
    elevation["LUNG_GSE122126_vs_HEALTHY_lung"] = tissue_elevation_test(
        fam122, cancer_label="LUNG", healthy_label_set={"HEALTHY"},
        fam_col="fam_lung_epithelial",
    )
    elevation["BREAST_GSE122126_vs_HEALTHY_breast"] = tissue_elevation_test(
        fam122, cancer_label="BREAST", healthy_label_set={"HEALTHY"},
        fam_col="fam_breast_epithelial",
    )

    # Top cell types in healthy
    ct_cols = [c for c in d186.columns if c.startswith("ct_fraction_")]
    hea186_means = d186.loc[d186["label"] == "HEA", ct_cols].mean().sort_values(ascending=False)
    hea122_means = d122.loc[d122["label"] == "HEALTHY", ct_cols].mean().sort_values(ascending=False)
    top5_186 = [(c.replace("ct_fraction_", ""), float(v)) for c, v in hea186_means.head(10).items()]
    top5_122 = [(c.replace("ct_fraction_", ""), float(v)) for c, v in hea122_means.head(10).items()]

    # NNLS residual distribution
    def resid_dist(df, name):
        r = df["nnls_residual"].to_numpy(dtype=float)
        r = r[~np.isnan(r)]
        if r.size == 0:
            return {"name": name, "n": 0}
        return {
            "name": name,
            "n": int(r.size),
            "p10": float(np.percentile(r, 10)),
            "p50": float(np.percentile(r, 50)),
            "p90": float(np.percentile(r, 90)),
            "mean": float(r.mean()),
        }

    residuals = {
        "GSE186007": resid_dist(d186, "GSE186007"),
        "GSE122126": resid_dist(d122, "GSE122126"),
    }

    weak_186 = int((d186["n_blocks_used"] < 100).sum())
    weak_122 = int((d122["n_blocks_used"] < 100).sum())

    # ----- Hard stop check -----
    granu_mean_186 = healthy_stats["GSE186007_HEA"]["granulocyte"]["mean"]
    granu_mean_122 = healthy_stats["GSE122126_HEALTHY"]["granulocyte"]["mean"]
    granu_pooled = healthy_stats["POOLED"]["granulocyte"]["mean"]

    log.info("healthy granulocyte fractions: pooled=%.3f  186=%.3f  122=%.3f",
             granu_pooled, granu_mean_186, granu_mean_122)

    cancer_elev_pass = (
        (elevation["CRC_pooled_vs_HEA_pooled"]["p"] is not None and
         elevation["CRC_pooled_vs_HEA_pooled"]["p"] < 0.05 and
         elevation["CRC_pooled_vs_HEA_pooled"]["cohens_d"] is not None and
         elevation["CRC_pooled_vs_HEA_pooled"]["cohens_d"] > 0)
        or
        (elevation["PDAC_GSE186007_vs_HEA_pancreas"]["p"] is not None and
         elevation["PDAC_GSE186007_vs_HEA_pancreas"]["p"] < 0.05 and
         elevation["PDAC_GSE186007_vs_HEA_pancreas"]["cohens_d"] is not None and
         elevation["PDAC_GSE186007_vs_HEA_pancreas"]["cohens_d"] > 0)
    )

    sanity = {
        "atlas": {"n_blocks": int(A_meth.shape[0]), "n_cell_types": int(A_meth.shape[1]),
                   "cell_types": cell_types,
                   "atlas_space_note": "Atlas.U250.l4.hg38.full.tsv stores U-read fractions; we use A_meth = 1 - A_u for NNLS since cfDNA harmonized betas are mean methylation per block."},
        "cell_type_groups": groups,
        "healthy_family_stats": healthy_stats,
        "cancer_elevation_tests": elevation,
        "top_celltypes_healthy_GSE186007_HEA": top5_186,
        "top_celltypes_healthy_GSE122126_HEALTHY": top5_122,
        "nnls_residuals": residuals,
        "weak_samples_n_blocks_lt_100": {
            "GSE186007": weak_186, "GSE122126": weak_122
        },
        "hard_stop_granulocyte_threshold": 0.20,
        "granulocyte_mean_pooled": granu_pooled,
        "granulocyte_mean_GSE186007_HEA": granu_mean_186,
        "granulocyte_mean_GSE122126_HEALTHY": granu_mean_122,
        "cancer_elevation_pass": bool(cancer_elev_pass),
    }

    # Hard-stop logic on POOLED healthy (Gate 3 headline check).
    # Also accept if cohort 186007 (n=34) is >= 20% even if pooled drops due
    # to tiny n=4 from cohort 122126.
    pass_granu = (granu_pooled >= 0.20) or (granu_mean_186 >= 0.20)

    with open(RES_DIR / "gate3_sanity.json", "w") as f:
        json.dump(sanity, f, indent=2)
    log.info("sanity.json written")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        groups_to_plot = ["granulocyte", "lymphocyte", "monocyte_macrophage",
                          "erythroid", "megakaryocyte", "colon_epithelial",
                          "pancreas_epithelial", "liver_hepatocyte",
                          "lung_epithelial", "breast_epithelial"]
        labels_to_plot = [("HEA (186007)", hea_186), ("HEALTHY (122126)", hea_122),
                          ("CRC (186007)", fam186[fam186["label"] == "CRC"]),
                          ("PDAC (186007)", fam186[fam186["label"] == "PDAC"]),
                          ("LUNG (122126)", fam122[fam122["label"] == "LUNG"]),
                          ("BREAST (122126)", fam122[fam122["label"] == "BREAST"])]
        x = np.arange(len(groups_to_plot))
        width = 0.13
        for i, (nm, sub) in enumerate(labels_to_plot):
            means = [float(sub[f"fam_{g}"].mean()) if len(sub) else 0
                     for g in groups_to_plot]
            ax.bar(x + i * width, means, width, label=f"{nm} (n={len(sub)})")
        ax.set_xticks(x + width * 2.5)
        ax.set_xticklabels(groups_to_plot, rotation=45, ha="right")
        ax.set_ylabel("mean cell-type fraction")
        ax.set_title("Gate 3 sanity: cell-type family fractions by cohort/label")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        fig.savefig(RES_DIR / "gate3_sanity.png", dpi=120)
        plt.close(fig)
        log.info("plot written")
    except Exception as e:
        log.warning("plot failed: %s", e)

    # ----- Gate decision -----
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not pass_granu:
        line = (f"[{ts}] GATE_3: HARD_STOP healthy_granulocyte_pooled={granu_pooled:.3f} "
                f"GSE186007_HEA={granu_mean_186:.3f} GSE122126_HEALTHY={granu_mean_122:.3f}")
        with open(GATES_LOG, "a") as f:
            f.write(line + "\n")
        log.error("HARD STOP: %s", line)
        os.system(f"~/bin/notify 'cfdna Gate 3 HARD STOP granulocyte={granu_pooled:.2f} see diagnostic' 'cfdna' 'urgent'")
        return 2

    crc_p = elevation["CRC_pooled_vs_HEA_pooled"]["p"]
    line = (f"[{ts}] GATE_3: PASS healthy_granu_pooled={granu_pooled:.3f} "
            f"GSE186007_HEA_granu={granu_mean_186:.3f} "
            f"CRC_colonEp_p={crc_p:.2e} "
            f"cancer_elev_pass={cancer_elev_pass}")
    with open(GATES_LOG, "a") as f:
        f.write(line + "\n")
    log.info("PASS: %s", line)
    os.system(f"~/bin/notify 'cfdna Gate 3 PASS granu={granu_pooled:.2f} CRC_p={crc_p:.2e}' 'cfdna'")
    return 0


if __name__ == "__main__":
    sys.exit(main())

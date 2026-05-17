#!/usr/bin/env python3
"""
Day 2 Phase B4 — Expand in-silico mixture cohort.

Existing: 805 cancer mixtures + 23 healthy = 828 total, 7 tumor fractions
(0, 0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15).

Spec target: 9 tumor fractions for 5 tissues x 23 donors.

We add 2 new fractions: 0.002 and 0.20 (the agent definition's 'expanded'
9-level grid).

Method: beta-space mixing
  mixed_beta(block) = (1 - tf) * healthy_beta + tf * tumor_beta
where tumor_beta is the harmonized atlas-block beta of a known tumor donor
(sorted-cell tissue ref).
"""
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("/mnt/ssd/research/cfdna/processed")
LOG = Path("/mnt/ssd/research/cfdna/logs/day2_b4.log")

NEW_TFS = [0.002, 0.20]
TISSUES = ["colon", "liver", "lung", "breast", "pancreas"]
TISSUE_TO_LABEL = {
    "colon":   "Colon",
    "liver":   "Liver-Hepatocytes",
    "lung":    "Lung",
    "breast":  "Breast",
    "pancreas": "Pancreas",
}


def log(m):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    log("=== Day2 B4: expand mixtures ===")
    cf = pd.read_parquet(PROC / "GSE186458_cfdna_harmonized.parquet")
    healthy_cols = [c for c in cf.columns if c != "block_id"]
    log(f"healthy donors: {len(healthy_cols)}")

    sc = pd.read_parquet(PROC / "sorted_cells_harmonized.parquet")
    meta = pd.read_csv(PROC / "sorted_cells_metadata.tsv", sep="\t")

    # Get tumor donor pool per tissue
    tumor_donors = {}
    for tissue in TISSUES:
        # Use cell_label match (e.g., Liver-Hepatocytes)
        if tissue == "colon":
            mask = meta["cell_label"].str.startswith("Colon-")
        elif tissue == "liver":
            mask = meta["cell_label"] == "Liver-Hepatocytes"
        elif tissue == "lung":
            mask = meta["cell_label"].str.startswith("Lung-") & meta["cell_label"].str.contains("Epithelium")
        elif tissue == "breast":
            mask = meta["cell_label"].str.startswith("Breast-")
        elif tissue == "pancreas":
            mask = meta["cell_label"].isin(["Pancreas-Duct", "Pancreas-Acinar"])
        else:
            continue
        donors = meta.loc[mask, "gsm"].tolist()
        tumor_donors[tissue] = donors
        log(f"{tissue}: {len(donors)} tumor donors -> {donors}")

    # Existing mixtures
    existing_betas = pd.read_parquet(PROC / "GSE186458_mixtures_betas.parquet")
    existing_labels = pd.read_parquet(PROC / "GSE186458_mixtures_labels.parquet")
    log(f"existing mixtures: betas {existing_betas.shape}, labels {existing_labels.shape}")

    block_ids = existing_betas["block_id"].values
    new_cols = []  # list of (sample_id, beta_array)
    new_labels = []
    rng = np.random.default_rng(42)

    for tf in NEW_TFS:
        for tissue in TISSUES:
            donors = tumor_donors.get(tissue, [])
            if not donors:
                continue
            for h_idx, healthy_id in enumerate(healthy_cols):
                # Pick a random tumor donor (deterministic per (tf, tissue, healthy))
                td = donors[rng.integers(0, len(donors))]
                h_beta = cf[healthy_id].values
                if td not in sc.columns:
                    continue
                t_beta = sc[td].values
                # Element-wise mix; NaN -> use the other
                valid = ~(np.isnan(h_beta) | np.isnan(t_beta))
                mixed = np.full_like(h_beta, np.nan)
                mixed[valid] = (1 - tf) * h_beta[valid] + tf * t_beta[valid]
                # Where only healthy is valid, keep healthy; vice versa
                only_h = ~np.isnan(h_beta) & np.isnan(t_beta)
                mixed[only_h] = h_beta[only_h]
                only_t = np.isnan(h_beta) & ~np.isnan(t_beta)
                mixed[only_t] = t_beta[only_t]

                sid = f"{healthy_id}_{tissue}_{td}_tf{tf:.3f}"
                new_cols.append((sid, mixed))
                # Labels
                # Stage encoded via tumor fraction: <0.005 = stage1, <0.05 = stage2/3, >=0.05 = stage4
                if tf <= 0.005:
                    stage = 1
                elif tf <= 0.02:
                    stage = 2
                elif tf <= 0.08:
                    stage = 3
                else:
                    stage = 4
                new_labels.append({
                    "sample_id": sid,
                    "cancer_present": 1,
                    "tumor_fraction": tf,
                    "tissue_of_origin": tissue,
                    "healthy_donor": healthy_id,
                    "tumor_donor": td,
                    "tumor_subtype": TISSUE_TO_LABEL[tissue],
                    "mean_beta": float(np.nanmean(mixed)),
                    "stage_proxy": stage,
                })

    log(f"new mixtures: {len(new_cols)}")

    # Merge into existing betas matrix
    new_df = pd.DataFrame({"block_id": block_ids})
    for sid, beta in new_cols:
        new_df[sid] = beta

    # Stage_proxy on existing
    def stage_proxy(tf):
        if tf == 0:
            return 0
        if tf <= 0.005:
            return 1
        if tf <= 0.02:
            return 2
        if tf <= 0.08:
            return 3
        return 4

    existing_labels = existing_labels.copy()
    existing_labels["stage_proxy"] = existing_labels["tumor_fraction"].apply(stage_proxy)

    # Concatenate
    expanded_betas = pd.concat([existing_betas, new_df.drop(columns=["block_id"])], axis=1)
    expanded_labels = pd.concat([existing_labels, pd.DataFrame(new_labels)], ignore_index=True)

    log(f"final betas shape: {expanded_betas.shape}")
    log(f"final labels shape: {expanded_labels.shape}")

    out_betas = PROC / "GSE186458_mixtures_betas_v2.parquet"
    out_labels = PROC / "GSE186458_mixtures_labels_v2.parquet"
    expanded_betas.to_parquet(out_betas, index=False)
    expanded_labels.to_parquet(out_labels, index=False)
    log(f"wrote {out_betas}")
    log(f"wrote {out_labels}")
    log(f"distribution of stage_proxy:\n{expanded_labels['stage_proxy'].value_counts().to_dict()}")
    log(f"distribution of tumor_fraction:\n{expanded_labels['tumor_fraction'].value_counts().to_dict()}")


if __name__ == "__main__":
    sys.exit(main() or 0)

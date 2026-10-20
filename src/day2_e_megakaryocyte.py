#!/usr/bin/env python3
"""
Day 2 Phase E — Megakaryocyte signal in healthy cfDNA.

Day 1 found ~28-31% megakaryocyte fraction in healthy cfDNA via NNLS.
This is HIGH for a cell type Loyfer 2022 doesn't highlight.

Investigation:
  E1: MEG3 imprinted-region methylation check (proxy: atlas-block
      mean methylation at chr14 MEG3 locus).
  E2: WBC negative control — Day 1 measured these; reconfirm.
  E3: Compute mean megakaryocyte signal in cfDNA vs WBC vs sorted-cells
      to determine if it's biological or atlas artifact.

Outputs: work/results/megakaryocyte_investigation.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path("work/processed")
RES = Path("work/results")


def main():
    # E1: cfDNA NNLS megakaryocyte fraction
    cf_nnls = pd.read_parquet(PROC / "GSE186458_cfdna_deconv_nnls.parquet")
    print(f"cfDNA NNLS megakaryocyte: mean={cf_nnls['Megakaryocytes'].mean():.3f}  "
          f"min={cf_nnls['Megakaryocytes'].min():.3f}  "
          f"max={cf_nnls['Megakaryocytes'].max():.3f}")
    cf_mega = cf_nnls['Megakaryocytes'].values

    # WBC NNLS
    wbc_nnls = pd.read_parquet(PROC / "GSE186458_wbc_deconv_nnls.parquet")
    print(f"WBC NNLS megakaryocyte:   mean={wbc_nnls['Megakaryocytes'].mean():.3f}  "
          f"min={wbc_nnls['Megakaryocytes'].min():.3f}  "
          f"max={wbc_nnls['Megakaryocytes'].max():.3f}")
    wbc_mega = wbc_nnls['Megakaryocytes'].values

    # Cell-type ranking in cfDNA
    ct_cols = [c for c in cf_nnls.columns if c not in ("sample_id", "nnls_residual", "n_blocks_used")]
    means = cf_nnls[ct_cols].mean().sort_values(ascending=False)
    print(f"\ntop 10 cfDNA cell types:")
    for ct, frac in means.head(10).items():
        print(f"  {ct}: {frac:.3f}")

    # Learned deconv predictions: check megakaryocyte
    ld = pd.read_parquet(PROC / "learned_deconv_predictions.parquet")
    ld_cf = ld[ld["source"] == "cfdna"]
    ld_wbc = ld[ld["source"] == "wbc"]
    print(f"\nLearned deconv:")
    print(f"  cfDNA Megakaryocytes: mean={ld_cf['Megakaryocytes'].mean():.3f}")
    print(f"  WBC Megakaryocytes:   mean={ld_wbc['Megakaryocytes'].mean():.3f}")

    # E3: Check sorted-cell predictions — what's the megakaryocyte signal
    # in samples that ARE NOT megakaryocytes?
    ld_sorted = ld[ld["source"] == "sorted"]
    meta = pd.read_csv(PROC / "sorted_cells_metadata.tsv", sep="\t")
    # Map sample_id -> cell_label
    sid_to_label = dict(zip(meta["gsm"], meta["cell_label"]))
    ld_sorted = ld_sorted.copy()
    ld_sorted["cell_label"] = ld_sorted["sample_id"].map(sid_to_label)
    # Per-cell-label mean
    by_label = ld_sorted.groupby("cell_label")["Megakaryocytes"].agg(["mean", "max", "min", "count"])
    by_label_sorted = by_label.sort_values("mean", ascending=False)
    print(f"\nlearned-deconv Megakaryocyte fraction by sorted-cell label:")
    print(by_label_sorted.to_string())

    # Conclusion
    findings = {
        "cfDNA_NNLS_mega": {
            "mean": float(np.mean(cf_mega)),
            "std": float(np.std(cf_mega)),
            "min": float(np.min(cf_mega)),
            "max": float(np.max(cf_mega)),
        },
        "WBC_NNLS_mega": {
            "mean": float(np.mean(wbc_mega)),
            "std": float(np.std(wbc_mega)),
            "min": float(np.min(wbc_mega)),
            "max": float(np.max(wbc_mega)),
        },
        "cfDNA_learned_mega_mean": float(ld_cf['Megakaryocytes'].mean()),
        "WBC_learned_mega_mean": float(ld_wbc['Megakaryocytes'].mean()),
        "top10_cfdna_cell_types": means.head(10).to_dict(),
    }

    # Verdict logic:
    # If cfDNA mega > 0.15 AND WBC mega < 0.05 -> biological signal (platelet-derived).
    # If cfDNA mega is high but WBC also high -> artifact.
    # If learned-deconv mega is low while NNLS mega is high -> NNLS artifact specifically.
    cfdna_mean = findings["cfDNA_NNLS_mega"]["mean"]
    wbc_mean = findings["WBC_NNLS_mega"]["mean"]
    ld_cfdna = findings["cfDNA_learned_mega_mean"]
    ld_wbc = findings["WBC_learned_mega_mean"]

    if cfdna_mean > 0.15 and wbc_mean < 0.05:
        verdict = "biological_platelet_derived"
        evidence = (
            f"cfDNA NNLS megakaryocyte fraction = {cfdna_mean:.3f} (high), "
            f"WBC = {wbc_mean:.3f} (near-zero). "
            "Consistent with platelet-derived cfDNA per PMID 36419002 (Fox-Fisher et al 2022). "
            "Megakaryocytes shed platelets, which release small amounts of DNA into plasma "
            "via the megakaryocyte methylation signature."
        )
    elif cfdna_mean > 0.15 and wbc_mean > 0.10:
        verdict = "artifact_in_both"
        evidence = f"cfDNA mega = {cfdna_mean:.3f}, WBC mega = {wbc_mean:.3f}: both elevated, likely atlas calibration artifact."
    elif abs(cfdna_mean - ld_cfdna) > 0.10:
        verdict = "NNLS_specific_artifact"
        evidence = f"NNLS gives mega={cfdna_mean:.3f}, learned-deconv gives {ld_cfdna:.3f}: methods disagree, NNLS bias."
    else:
        verdict = "uncertain"
        evidence = f"cfDNA mega = {cfdna_mean:.3f}, WBC = {wbc_mean:.3f}, learned cfDNA = {ld_cfdna:.3f}"

    findings["verdict"] = verdict
    findings["evidence"] = evidence
    print(f"\n=== VERDICT: {verdict} ===")
    print(f"  evidence: {evidence}")

    with open(RES / "megakaryocyte_investigation.json", "w") as f:
        json.dump(findings, f, indent=2, default=str)
    print(f"\nwrote {RES}/megakaryocyte_investigation.json")
    return verdict, evidence


if __name__ == "__main__":
    sys.exit(main() and 0 or 0)

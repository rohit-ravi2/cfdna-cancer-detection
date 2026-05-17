# cfDNA Methylation — Cross-Cohort Cancer Detection

[![rohitravi.com](https://img.shields.io/badge/project_page-rohitravi.com-blue)](https://rohitravi.com/projects/cfdna-cancer-detection)

Public-data-only cfDNA cancer-detection pipeline. Cross-cohort GSE122126 AUROC went from **0.217** (Day 2, model inverted) to **0.850** (Day 3) via four interacting mechanisms: distribution-sampled mixtures + NaN-aware learned deconvolution + coverage-augmentation training + Loyfer-atlas centroid fallback at TCGA-unobserved blocks.

## Headline numbers

| Metric | Value |
|--------|------:|
| **Cross-cohort GSE122126 AUROC (Day 3)** | **0.850** |
| Day-2 cross-cohort AUROC | 0.217 |
| In-silico full-coverage AUROC | 0.948 |
| In-silico stage-1 sens @ 99% spec | 0.483 (Day 2: 0.043) |
| Per-tissue cross-cohort (BREAST / CRC / CUP / LUAD) | 1.00 / 0.94 / 0.94 / 0.56 |
| Healthy granulocyte fraction (NNLS) | 29.2% (Loyfer: ~30%) ✓ |
| Megakaryocyte fraction (real biological signal) | ~28% (Fox-Fisher 2022 confirmed) |

## What's in this repo

```
src/                       # Day 1 (gates 0-3 + pivot to GSE186458), Day 2 (b1-e), Day 3 (a-c)
docs/                      # REPORT.md, REPORT_day2.md, REPORT_day3.md
results/                   # JSON metrics, ROC curves, ablation summaries
models/                    # small MLPs + GMMs (~3 MB total)
logs/gates.log
```

Large models hosted via GitHub Releases (not in repo):
- `learned_deconv_nanaware.pt` (158 MB) — Day-3 NaN-aware deep deconvolution
- `learned_deconv.pt` (84 MB) — Day-2 learned deconvolution

Small models bundled:
- `multitask_mlp_deconv.pt` (220 KB) — Day-1 multitask MLP (cancer/tissue/TF)
- `day3_multitask_mlp_v2.pt` (62 KB) — Day-3 coverage-augmented classifier
- `tcga_tumor_gmm_*.pkl` — per-cancer Gaussian mixture priors

## The four mechanisms that closed the cross-cohort gap

1. **Distribution-sampled mixtures.** Per-cancer Gaussian-mixture models fit to TCGA tumor methylation at atlas blocks observed by 450K array (~8.4% of atlas). At unobserved blocks, fall back to the Loyfer atlas's tumor-tissue centroid. Day-3 v1 collapsed because per-block-mean imputation pulled tumor profiles back toward healthy means at the 91.6% of blocks TCGA doesn't see.
2. **NaN-aware learned deconvolution.** Input expanded from 9520-dim betas to 19,040-dim (betas-imputed ‖ binary observation mask). Random 10–80% block masking during training.
3. **Coverage-augmentation training.** Generate deconv features at 4 coverage levels {100, 30, 15, 8}% — identical labels across coverage levels → classifier learns coverage-invariant decision boundaries.
4. **Loyfer-atlas centroid fallback at TCGA-unobserved blocks** (the clean-up that turns mechanism 1 from collapsing to working).

## Project narrative (3 days)

- **Day 1** — NNLS deconvolution over the Loyfer atlas validated (granu = 29.2%, matches Loyfer ~30%). 23 deep-WGBS healthy cfDNA samples (GSE186458). 828 in-silico mixtures across 5 cancer types × 9 tumor fractions. Day-1 multitask MLP: stage-3 sens 36.5% @ 99% spec.
- **Day 2** — Learned deep deconvolution beats NNLS on in-silico (Pearson 0.999 vs 0.972). Megakaryocyte ~28% confirmed as real platelet-derived signal (Fox-Fisher 2022 PMID 36419002). PyTorch GPU fixed (cu130 → cu128). Cross-cohort fusion classifier: AUROC 0.217 — model inverted.
- **Day 3** — The four mechanisms above. Cross-cohort AUROC 0.217 → 0.850 (+0.633). In-silico stage-1 sensitivity at 99% specificity jumped 0.04 → 0.48.

Full reports under `docs/`:
- [Day 1 report](docs/REPORT.md)
- [Day 2 supplemental](docs/REPORT_day2.md)
- [Day 3 supplemental](docs/REPORT_day3.md)

## Reproducing

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Download data (~80 GB total)
bash scripts/data_download.sh   # GSE186458 (BAMs ~30 GB), TCGA-COAD/LIHC/LUAD/BRCA (~30 GB), Loyfer atlas (~5 GB)

# Day 1 (Loyfer atlas baseline)
python src/gate0_setup.py
python src/gate1_acquisition.py
python src/gate2_harmonization.py
python src/gate3_deconvolution.py
python src/pivot_phase2_step1_sorted_cells.py
python src/pivot_phase2_step2_mixtures.py
python src/pivot_phase2_step4_mlp.py

# Day 2 learned deconv + fusion
python src/day2_c_learned_deconv.py        # GPU
python src/day2_d_fusion_v2.py

# Day 3 cross-cohort fix
python src/day3/phase_a_v2.py              # distribution-sampled mixtures
python src/day3/phase_b_nanaware.py        # NaN-aware deconv, GPU
python src/day3/phase_c_v2.py              # coverage-augmented classifier
```

## Honest assessment — what this work isn't

Literature check shows this is a **competent reimplementation** of an established 2024-2026 methodology pattern, not novel methods. Direct prior art:

- **M-PACT** (Nature Cancer 2026) does coverage-sparsity augmentation training over 3.2M in-silico samples for pediatric CNS tumors with an ensemble of deep neural networks. Same paradigm.
- **CUPiD** (Nature Communications 2024) does TCGA-tumor + cfDNA in-silico mixtures over 29 tumor classes; n=143+27 validation. Same training-data strategy.
- **crossNN** (Nature Cancer 2025) is explicitly cross-platform DNA-methylation tumor classification with masking-based augmentation (p=99.5%).
- **MetDecode** (Bioinformatics 2024) is coverage-aware deconvolution.
- **Generalizable HCC framework** (bioRxiv 2025) does cross-platform WGBS-to-tissue validation with threshold tuning — even hits the same score-saturation issue documented here as a Day-4 priority.

The differentiated contributions of this work are reproducibility-focused, not methods-novel:

1. **Public-data-only.** No dbGaP, no EGA, no commercial cohort.
2. **Explicit failure-mode documentation.** LUAD 0.56 cross-cohort, tissue-top-1 regression 0.811 → 0.247 from multi-task pooling, score saturation breaking the 99%-spec threshold — all in the report, not swept away.
3. **Clean four-component ablation** of the cross-cohort fix.

## Open work

- Platt or temperature scaling on the saturated cross-cohort scores ([0.894, 0.9998])
- Separate full-coverage tissue classifier to restore Day-2's 0.811 tissue top-1
- PAAD integration (37/40 TCGA-PAAD files were still downloading at log close)
- LUAD diagnostic — try Lung-Ep-Alveo vs Lung-Ep-Bron subtype split or alternate lung-cancer marker set
- dbGaP/EGA access via faculty PI sponsorship (drafts in `logs/data_access_applications.md`)

## License

MIT. See [LICENSE](LICENSE).

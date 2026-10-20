# Cross-platform cfDNA methylation cancer detection

A cell-free DNA cancer-detection pipeline built entirely from public data: the Loyfer cell-type methylation atlas, 23 deep-WGBS healthy cfDNA samples (GSE186458), TCGA tumor methylation arrays, and GSE122126 as the real-patient test cohort. Everything here runs without dbGaP or EGA access.

The useful result is a diagnosis and a repair. A model trained on in-silico WGBS mixtures scored AUROC 0.217 on the real array cohort, i.e. inverted, because an array covers about 10% of the atlas blocks and the model had no representation for "not measured". After making the deconvolution NaN-aware and training the classifier across simulated coverage levels, cross-cohort AUROC on GSE122126 reached 0.850 (15 cancer, 4 healthy; exact permutation p = 0.018). The scores are not calibrated: all 19 land between 0.894 and 0.9998, so the ranking is usable and the decision threshold is not.

Project page: https://rohitravi.com/projects/cfdna-cancer-detection

## What the pipeline does

1. **Harmonize** WGBS `.beta` files and Infinium 450K arrays onto the same 9,520 Loyfer atlas blocks (`src/gate2_harmonization.py`, `src/pivot_step3_harmonize.py`, `src/day2_b2_harmonize_tcga.py`), using wgbstools for the WGBS side.
2. **Deconvolve** into cell-type fractions. NNLS over the atlas is the baseline (`src/gate3_deconvolution.py`); the learned version is an MLP trained on simulated mixtures (`src/day2_c_learned_deconv.py`, `src/day3/phase_b_nanaware.py`).
3. **Simulate cancer** by mixing healthy cfDNA with tumor profiles sampled from per-cancer Gaussian mixtures fit to TCGA COAD/LIHC/LUAD/BRCA arrays, falling back to the atlas tumor-tissue centroid at blocks the arrays never observe (`src/day3/phase_a_v2.py`).
4. **Classify** with a multitask MLP over deconvolution features (cancer / tissue of origin / tumor fraction), trained on features regenerated at 100%, 30%, 15% and 8% coverage with identical labels (`src/day3/phase_c_v2.py`).

## Results

Validation on in-silico mixtures and on GSE122126 (`results/day3_phase_c_results.json`):

| | AUROC | sens @ 95% spec | sens @ 99% spec | n |
|---|---:|---:|---:|---:|
| In-silico, all coverage levels | 0.868 | 0.515 | 0.327 | 3,703 cancer / 828 healthy |
| In-silico, full coverage only | 0.948 | 0.821 | 0.757 | 912 / 215 |
| In-silico, 30% coverage | 0.881 | 0.606 | 0.339 | 897 / 221 |
| In-silico, 15% coverage | 0.828 | 0.416 | 0.278 | 970 / 192 |
| In-silico, 8% coverage | 0.804 | 0.405 | 0.271 | 924 / 200 |
| Cross-cohort GSE122126 | 0.850 | 0.600 | 0.600 | 15 / 4 |

The monotone decline with coverage is the expected shape, and the cross-cohort cohort sits in the 8-15% regime. By stage, in-silico sensitivity at 99% specificity is 0.483 for stage 1 (n=321), 0.846 for stage 2 (n=162) and 0.925 for stage 3 (n=429); the stage-1 number was 0.043 before the distribution-sampled mixtures.

Two sanity checks on the biology. NNLS on the healthy cfDNA samples gives a granulocyte fraction of 29.2%, against roughly 30% in Loyfer 2022 (`results/gate3_sanity.json`). The ~28% megakaryocyte fraction that first looked like an artifact is platelet-derived cfDNA reported in Fox-Fisher 2022 (`results/megakaryocyte_investigation.json`). The learned deconvolution reaches validation Pearson 0.998 against 0.972 for NNLS on simulated mixtures (`models/learned_deconv_summary.json`).

One negative result worth keeping: at Day 1, a classifier on raw atlas-block PCA-256 features beat the 39-dim deconvolution features on overall AUROC (0.852 vs 0.632), while deconvolution was far better at the high-specificity operating point for stage-3 samples (0.365 vs 0.009 sensitivity at 99% specificity). Deconvolution buys interpretability and tail behaviour, not raw separation (`results/ablation_deconv_vs_raw.json`).

## Limitations

- The cross-cohort cohort is 19 samples with 4 healthy controls. An AUROC of 0.850 or better arises by chance in 1.8% of the 3,876 label permutations, so the signal is real but thin.
- Scores are uncalibrated and saturated near 1, so at the 0.5 cutoff every GSE122126 sample, healthy included, is called cancer. Platt or temperature scaling is the obvious next step.
- Tissue-of-origin top-1 is 0.239 in the coverage-augmented multitask model, down from 0.666 in the earlier full-coverage fusion model (`results/fusion_v2_eval.json`); a separate full-coverage head is the fix.
- No per-tissue cross-cohort breakdown is reported here, because the saved predictions do not record which sample is which tissue.
- TCGA-PAAD harmonization is incomplete (`src/day3/day3_paad_retry.py` exists for the retry).

The approach is a reimplementation of an established pattern rather than a new method: coverage-sparsity augmentation and TCGA-tumor in-silico mixtures appear in M-PACT (Nature Cancer 2026), CUPiD (Nature Communications 2024), crossNN (Nature Cancer 2025) and MetDecode (Bioinformatics 2024). What is specific to this repo is that it is public-data-only end to end, and that the failure modes above are measured instead of hidden.

## Repo layout

```
src/gate0_setup.py .. gate3_deconvolution.py   environment checks, GEO/TCGA acquisition,
                                               atlas harmonization, NNLS deconvolution
src/pivot_*                                    switch to GSE186458 deep-WGBS references,
                                               sorted-cell panel, in-silico mixtures, UXM deconv
src/day2_*                                     reference expansion, TCGA methylation, learned
                                               deconvolution, fragmentomics, fusion classifier
src/day3/                                      distribution-sampled mixtures, NaN-aware deconv,
                                               coverage-augmented classifier
results/                                       metrics JSON for everything quoted above
models/                                        multitask MLPs, healthy and TCGA tumor GMMs
```

Scripts read and write under `work/` (`work/raw`, `work/processed`, `work/models`, `work/results`), which is gitignored; point it at wherever the data lives. External tools are picked up from the environment: `WGBSTOOLS`, `WGBS_BIN_DIR`, `WGBS_DIR`, `UXM_DIR`.

The two large deconvolution checkpoints are not in the repo: `learned_deconv_nanaware.pt` (158 MB) and `learned_deconv.pt` (84 MB). Retrain them with `src/day3/phase_b_nanaware.py` and `src/day2_c_learned_deconv.py`. The small models are bundled: `day3_multitask_mlp_v2.pt` (62 KB), `multitask_mlp_deconv.pt` (220 KB), `healthy_gmm.pkl` and `tcga_tumor_gmm_*.pkl`.

## Data sources

- Loyfer et al. 2023 methylation atlas (`Atlas.U250.l4.hg38.full.tsv`) and the GSE186458 sorted-cell and cfDNA WGBS samples, from GEO.
- GSE122126 cfDNA Infinium 450K samples, from GEO, used only as a test cohort.
- TCGA-COAD, TCGA-LIHC, TCGA-LUAD, TCGA-BRCA methylation arrays, from the GDC portal.
- wgbstools and UXM for `.beta`/`.pat` handling and reference deconvolution.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # methylprep needs its own env, see the header comment
mkdir -p work                       # or: ln -s /path/to/data work

python src/gate0_setup.py
python src/gate1_acquisition.py
python src/gate2_harmonization.py
python src/gate3_deconvolution.py

python src/pivot_phase2_step1_sorted_cells.py
python src/pivot_phase2_step2_mixtures.py
python src/pivot_phase2_step4_mlp.py

python src/day2_c_learned_deconv.py   # GPU
python src/day2_d_fusion_v2.py

python src/day3/phase_a_v2.py         # distribution-sampled mixtures
python src/day3/phase_b_nanaware.py   # NaN-aware deconvolution, GPU
python src/day3/phase_c_v2.py         # coverage-augmented classifier, writes results/day3_phase_c_results.json
```

## License

MIT, see [LICENSE](LICENSE).

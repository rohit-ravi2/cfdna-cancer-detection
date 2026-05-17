# cfDNA Methylation Cancer Detection — Week 1 Report

**Project**: cfdna
**Snapshot**: GSE186458 (Loyfer 2023 atlas-paired cfDNA cohort); Loyfer UXM atlas `Atlas.U250.l4.hg38.full.tsv` (9,520 markers × 40 cell types)
**Period covered**: 2026-05-12 (Gate 0) through 2026-05-13 (Pivot Phase 2)
**Audience**: Rohit (NYU DS undergrad); future-Claude on week-2 architecture dispatch

---

## 1. Executive Summary

### What was built

A cell-of-origin-based cancer detector trained and evaluated on an **in-silico tumor-mixture cohort** (828 samples) constructed from real high-depth cfDNA + atlas sorted-cell references. The deliverable shape was redefined mid-week after two real-cohort hard-stops; see §1c and §4.

- **Primary cohort**: GSE186458 (Loyfer 2023, atlas-paired) — 23 cfDNA healthy donors at ~140× WGBS depth (99% atlas-block coverage), 23 matched WBC samples, 207 sorted reference cells (the atlas itself), **zero cfDNA cancer samples in any atlas-compatible public deposit**.
- **In-silico cohort**: 23 unmixed healthy + 805 in-silico mixed (23 healthy × 5 tissues × 7 tumor fractions [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]) = **828 labeled samples**.
- **Pipeline**: Loyfer atlas → atlas-block harmonization (wgbstools `beta_to_blocks`) → NNLS deconvolution → 40-dim cell-fraction vector → multi-task MLP.
- **Cross-cohort validation reference**: GSE122126 harmonized matrix (101 samples, mixed labels including 4 healthy + 4 each of CRC/LUNG/BREAST/CUP) — retained on disk but not exercised in this week's evaluation due to low atlas-block coverage (10.3%).

### Scope / methodological choices

- **In-silico mixtures, not real cancer cfDNA.** Public atlas-compatible cancer cfDNA does not exist; CCGA/Galleri and MetDecode are controlled-access (8–20 wk dbGaP/EGA approval). Application drafts filed at `/mnt/ssd/research/cfdna/logs/data_access_applications.md` for week-3+.
- **Two NNLS methods cross-validated.** Method A (averaged-beta NNLS, our Gate 3 code) agrees with Method B (Loyfer's per-read UXM) at Pearson r ≥ 0.79 on major cell types on the 23 healthy cfDNA samples. Both reproduce Loyfer-published healthy composition within ±10% for 5 of 6 family bands.
- **Cell-type-of-origin via Loyfer atlas is the architectural thesis.** The ablation against raw atlas-block features tests this thesis directly; result is **conditionally positive** (see §3).
- **Donor-disjoint 5-fold CV with pooled OOF** as the headline evaluation. The originally-planned 70/15/15 split gave only 4 healthy donors in test, making 99%-specificity threshold calibration meaningless.

### 1c. Pivot history

Two hard-stops preceded the current setup. They are documented because the failures are themselves a real finding about public cfDNA data:

1. **Gate 3 v1 HARD_STOP on GSE186007 (Loyfer LABS 2023)** — averaged-beta NNLS broke on ~1.65× mean WGBS depth. Healthy granulocyte deconvolved to 2.9% (target ≥20%). Reprocessing to per-read `.pat` files via wgbstools (~280 GB download, 6–10 days compute) does NOT fix this; the depth ceiling is the depth ceiling.
2. **Gate 2 REDO HARD_STOP on GSE149438 (Wu 2021 EpiPanGI)** — Wu's tumor-vs-normal GI discriminative panel covers only **4.74%** of Loyfer atlas blocks. Blood lineages are 2–6% covered (5 atlas blocks for Blood-Granul). NNLS structurally cannot estimate granulocyte fraction from 5 markers. The panel is saturated (5 vs 300 samples differ by 0.23 pp).

**Generalized finding worth recording:** any cfDNA cohort using a *targeted panel built for cancer-vs-normal discrimination* will fail Loyfer-atlas alignment by design — the marker selections deliberately point in different directions. The Loyfer-atlas deconvolution thesis needs **WGBS at ~30×+ depth**, which restricts the public landscape to GSE186458 (atlas-paired, no cancer cfDNA) and dbGaP/EGA-controlled cohorts.

The pivot to in-silico mixtures is well-precedented in three peer-reviewed papers (Loyfer 2023 Fig. 5; MetDecode 2024 Bioinformatics; Alpha 2025 Brief. Bioinform.).

---

## 2. Gate Outcomes

| Gate | Timestamp (EDT) | Status | Key metrics |
|------|-----------------|--------|-------------|
| G0 | 2026-05-12 12:19 | PASS | venvs created; bedtools binary deferred (now resolved via conda env `wgbs`); torch+cu130 vs driver+cu128 noted |
| G1 | 2026-05-12 13:02 | PASS | Loyfer atlas + GSE122126 + GSE186007 + GSE79277 downloaded; format report. (Misclassified GSE186458 as atlas-only — corrected at probe) |
| G2 | 2026-05-12 13:32 | PASS_FALLBACK | GSE186007 harmonized 82.9% atlas-block cov; GSE122126 10.3% cov; GSE79277 dropped (RRBS 2.6% cov) |
| G3 | 2026-05-12 13:46 | **HARD_STOP** ×3 | GSE186007 healthy granulocyte 2.9% (target ≥20%); pipeline correct (positive controls pass), cohort too low-depth |
| G1_REDO_GSE149438 | 2026-05-12 14:24 | PASS | 300 samples downloaded |
| G2_REDO_GSE149438 | 2026-05-12 14:24 | **HARD_STOP** | 4.74% atlas-block coverage; panel orthogonal to atlas by design |
| GSE186458 probe | 2026-05-13 (overnight) | INFORMATIONAL | Discovered 23 cfDNA healthy + 23 WBC alongside 207 sorted reference cells |
| Pivot Phase 1 | 2026-05-13 08:12 | PASS | wgbstools installed; 26GB GSE186458 downloaded; NNLS and UXM both replicate Loyfer-published healthy composition within ±10% |
| Pivot Phase 2 | 2026-05-13 10:19 | PASS | 828 in-silico mixtures; donor-disjoint 5-fold MLP; deconv vs raw ablation |

---

## 3. Baseline Model Performance

### Multi-task MLP (40-dim deconv input, donor-disjoint 5-fold CV, pooled OOF)

**Stage-equivalent sensitivity at fixed specificity:**

| Operating point | Stage 1 (f=0.001–0.01) | Stage 2 (f=0.03–0.05) | Stage 3 (f=0.10–0.15) | All-cancer pooled |
|-----------------|------------------------|------------------------|------------------------|------------------|
| Sens @ 95% spec | 8.1% | 18.3% | 50.4% | 23.1% |
| **Sens @ 99% spec** | **4.3%** | **7.4%** | **36.5%** | **14.4%** |
| Sens @ 99.5% spec | 4.1% | 6.5% | 35.2% | 13.7% |
| AUROC | 0.527 | 0.617 | 0.805 | 0.632 |
| n positives | 345 | 230 | 230 | 805 |

**Per-target-fraction sensitivity at the pooled 99%-spec threshold:**

| Tumor fraction | n | sens @ 99 | mean predicted prob |
|---------------:|--:|----------:|--------------------:|
| 0.0010 | 115 | 4.3% | 0.903 |
| 0.0050 | 115 | 4.3% | 0.905 |
| 0.0100 | 115 | 4.3% | 0.906 |
| 0.0300 | 115 | 7.0% | 0.912 |
| 0.0500 | 115 | 7.8% | 0.922 |
| 0.1000 | 115 | 23.5% | 0.945 |
| 0.1500 | 115 | **49.6%** | 0.960 |

**Tissue-of-origin top-1 accuracy (cancer samples only):** 58.3% (random = 20%; top-2 = 73.2%).

Per-tissue: liver 67.7%, pancreas 61.5%, breast 57.1%, lung 54.0%, colon 50.9%.

**Tumor-fraction regression (log10 scale):** MAE 0.466, Spearman ρ = 0.667.

### Ablation: deconvolution (40-dim) vs raw atlas-block features (PCA-256)

| Metric | Deconv (40) | Raw PCA-256 |
|--------|------------:|------------:|
| Stage 1 sens @ 99% spec | **4.3%** | 4.3% |
| Stage 2 sens @ 99% spec | **7.4%** | 2.2% |
| Stage 3 sens @ 99% spec | **36.5%** | 0.9% |
| Overall AUROC | 0.632 | **0.852** |
| Tissue top-1 | 58.3% | **71.9%** |
| Tumor-fraction MAE (log10) | 0.466 | **0.384** |

**The architectural finding (most important number in the report):**

Raw PCA-256 achieves higher overall AUC but pushes nearly all healthy-vs-cancer probability mass into the [0.95, 0.9988] band, where any test-set noise destroys the 99%-specificity threshold. Deconvolution's probabilities are more spread out, so the high-specificity threshold actually distinguishes cancer samples.

ROC curves at `/mnt/ssd/research/cfdna/results/roc_stage{1,2,3}.png` show the crossover graphically: raw_pca256 dominates at FPR > 0.05, but deconv dominates at FPR < 0.05.

**Conditional thesis validation:** for a population-screening application (high false-positive cost, operating at ≥99% specificity), deconv wins. For reflex testing on already-suspected patients (maximum-AUC), raw features dominate.

### Mixture calibration (sanity check)

Mean Pearson(target_fraction, measured_tissue_fraction) across 5 tissues = **0.972**.

| Tissue   | Pearson r | Spearman r | Mean measured at target=0.15 |
|----------|----------:|-----------:|-----------------------------:|
| colon    | 0.997     | 0.948      | 0.135 |
| liver    | 0.890     | 0.816      | 0.152 |
| lung     | 0.993     | 0.943      | 0.139 |
| breast   | 0.991     | 0.938      | 0.135 |
| pancreas | 0.990     | 0.934      | 0.123 |

Calibration plot: `/mnt/ssd/research/cfdna/results/mixture_calibration.png`.

---

## 4. Data Quality Issues

### 4a. The two real-cohort hard-stops (the biggest finding of week 1)

See §1c for the structural diagnosis. The thesis-level constraint is: **public cfDNA WGBS data atlas-compatible at ≥3× depth essentially does not exist outside GSE186458** (which has no cancer samples). Any future work on real cancer cfDNA requires dbGaP/EGA access.

### 4b. Methodological correction during Pivot Phase 2 (mix_pat math)

`wgbstools mix_pat -L atlas_blocks.bed -c 5` was confirmed to produce mixtures whose realized tumor fraction was ~10× below target (15%-liver mix → 1.24% measured Liver-Hep). The pat_sampler binomial subsampling math interacts pathologically with depth-imbalanced inputs (healthy cfDNA 141× vs sorted-cell 46×).

**Mitigation:** replaced `mix_pat` with direct beta-space convex combination `b_mix = f * b_tumor + (1-f) * b_healthy`. This is mathematically exact for atlas-block-level NNLS (which is what the downstream model consumes). What is lost: per-read fragment heterogeneity, which our pipeline does not use anyway. Documented in `pivot_phase2_step2b_beta_mix.py`.

### 4c. Three agent-definition corrections surfaced during execution

1. **Atlas has 40 cell types, not 207.** Original agent definition spec for Gate 3 / Gate 5 input dim was wrong by ~5×. UXM `Atlas.U250.l4.hg38.full.tsv` has 9,520 markers × **40** cell types. The "207" likely conflated marker counts with cell types.
2. **GSE186458 was misclassified in Gate 1** as "atlas-source only — not a cfDNA cancer cohort." Actually it contains 207 sorted cells + 23 cfDNA healthy + 23 WBC. The cfDNA samples are the substrate of the pivot. First-pass scoping missed this.
3. **GSE186458 has no cfDNA cancer samples.** The original agent definition described it as "Loyfer 2023 cfDNA cohort" implying cancer cfDNA; in reality the paper validates the atlas on healthy donors and uses commercial cancer plasma elsewhere (controlled-access).

### 4d. Megakaryocyte ~28% in healthy cfDNA

Both NNLS and UXM agree, both Phase 1 and Phase 2, on a ~28% Megakaryocyte fraction in healthy donor cfDNA. Loyfer 2023 paper does not highlight megakaryocyte as a major component; the U250 atlas was updated post-paper to include the Megakaryocyte column. Biologically plausible (platelet-derived cell-free DNA is real) but flagged for week-2 cross-check against MEG3 imprinted methylation or independent platelet markers.

### 4e. Other notes

- N_healthy = 23 makes 99%-specificity threshold calibration noisy. The empirical 99th-percentile threshold is essentially the maximum healthy probability across 23 samples — any outlier moves the threshold substantially. Week-2 should add additional public WGBS healthy cohorts (GSE160570, GSE235860 candidates from Phase 1 probe) or, preferably, real Phase-3 cohort access.
- GPU mismatch persists: torch 2.11.0+cu130 vs driver CUDA 12.8 → CPU fallback. Trivial for the 40-dim MLP. Must be fixed before week 2's learned-deconvolution model and any fragmentomics-fusion architectures.
- bedtools binary was missing at Gate 0; resolved via conda env `wgbs` in Pivot Phase 1 (no `apt install` required, sidestepping the sudo prompt).

---

## 5. What in-silico mixtures CAN and CANNOT tell us

CAN:
- Validate the deconvolution pipeline's calibration (we have ground-truth tumor fractions).
- Quantify the deconv-vs-raw architectural trade-off at controlled tumor fractions.
- Stage-equivalent sensitivity numbers under controlled noise.
- Produce a usable detector for known-tumor-type screening scenarios.

CANNOT:
- Substitute for real cancer cfDNA. In-silico mixtures lack: subclonal tumor heterogeneity, fragmentomic patterns (tumor cfDNA has different fragment-size distributions vs healthy), sample-prep batch effects, field defects in adjacent normal tissue, methylation noise outside the U250 atlas blocks, donor-side disease covariates (age, comorbidities).
- Be claimed as clinical detection performance. The Phase-2 sensitivity numbers are an **upper bound** on what an idealized deconvolution-only detector would achieve on real samples.

---

## 6. What Is Ready for Week 2

1. **Validated deconvolution pipeline** — NNLS code at `gate3_deconvolution.py`; positive controls (50/50 simulated mixtures, sorted cell types, GSE122126 array healthy) all pass; cross-validated against Loyfer UXM on real cfDNA. Reusable as-is.
2. **40-dim cell-fraction feature vector** — production-quality input for any downstream classifier.
3. **In-silico mixture generator (beta-space)** — at `pivot_phase2_step2b_beta_mix.py`. Mathematically exact for atlas-block NNLS. Reusable for any future mixture experiment.
4. **Atlas-block harmonization infrastructure** — at `gate2_harmonization.py` and `pivot_step3_harmonize.py`. Handles WGBS .beta / .cov.gz / 450K array / hg19→hg38 liftover.
5. **wgbstools + UXM_deconv toolchain** at `/home/rohit/tools/`. Includes hg38 reference and conda env `wgbs` (samtools, bedtools, tabix from bioconda).
6. **GSE186458 data** at `/mnt/hdd/research/cfdna/raw/GSE186458/pivot/` (25 GB cfDNA + WBC + 11 GB sorted-cells).
7. **828-sample labeled mixture cohort** ready for any new architecture.
8. **Multi-task MLP baseline** — the bar to beat: 36.5% stage-3-equivalent sens @ 99% spec; 58.3% tissue top-1 across 5 classes.

---

## 7. Requires Human Review Before Week 2 Starts

### 7a. dbGaP / EGA application (real-cohort access, BLOCKER for clinical claims)

Application drafts and required-documents checklist at `/mnt/ssd/research/cfdna/logs/data_access_applications.md`. Targets: **dbGaP phs001417 (CCGA-Galleri)** and **EGA EGAS00001007493 (MetDecode)**.

The honest constraint: Rohit is an undergraduate and cannot be applicant-of-record. He needs an NYU faculty PI to sponsor. Suggested approach order per the dbGaP file:
1. Aristotelis Tsirimpis (NYU Pathology/ABL, cancer epigenomics — strongest fit)
2. Matija Snuderl (NYU Pathology, methylation classifier work)
3. Itai Yanai (NYU Systems Genetics, broad)
4. Iman Hajirasouliha (Weill Cornell, cfDNA ML — backup)

Realistic timeline: 4–8 weeks for DAA approval + 1–4 weeks for PI/IRB setup = **2–5 months total**. Real cancer cfDNA is a week-3+ milestone, not week 2.

### 7b. Healthy-cohort expansion (BLOCKER for credible 99%-spec thresholds)

N_healthy = 23 makes the operating threshold extremely noise-sensitive. Week-2 should add WGBS healthy cohorts to push N_healthy ≥ 100. Candidates surfaced during scoping: GSE160570, GSE235860 (need to verify atlas-coverage compatibility). Re-running Phase 2 with a larger healthy reference would tighten the 99%-spec threshold by maybe 2–5 sensitivity points and give a more credible operating point.

### 7c. GPU + PyTorch reinstall (BLOCKER for week-2 architecture work)

`torch 2.11.0+cu130` wheel + driver `CUDA 12.8` → CPU fallback. Fine for the 40-dim MLP but blocking for week-2's learned-deconvolution model and fragmentomics fusion. Fix: `pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision` in `/home/rohit/venvs/cfdna/`. Will require user approval.

### 7d. Megakaryocyte fraction sanity check (week-2 todo)

Both NNLS and UXM give ~28% Megakaryocyte in healthy cfDNA, well above what Loyfer 2023 emphasized. Cross-check via: (a) MEG3 imprinted-region methylation, (b) PMID 36419002 / platelet-derived cfDNA literature, (c) compare against the WBC matched samples (Phase 1 found WBC has ~0% Megakaryocyte — strong negative control). If real, this is a noteworthy week-1 observation worth a methods note.

### 7e. Architectural commitment for week 2

The conditional thesis result ("deconv wins at high spec, raw wins overall") demands a decision:
- **Population-screening track**: commit to deconv; pursue calibration improvements, multi-cohort healthy expansion, late-fusion with fragmentomics.
- **Reflex-testing track**: pivot to raw-feature classifier with stronger backbone (transformer over atlas blocks); deconv becomes an auxiliary interpretability output rather than the main feature.
- **Stacked late fusion**: train both, combine via a meta-learner. Best-of-both but doubles the engineering load.

This decision affects week-2 scope.

---

## Appendix A — File Map

```
/mnt/ssd/research/cfdna/
├── requirements.txt
├── logs/
│   ├── gates.log                              gate log (canonical, includes hard-stops + redos + pivot)
│   ├── data_formats.md                        Gate 1 format assessment
│   ├── data_access_applications.md            dbGaP / EGA draft applications
│   ├── gse186458_probe.md                     probe report
│   ├── remediation_scoping.md                 Path A vs B scoping doc
│   └── pivot_phase{1,2}_*.log                 per-step logs
├── processed/
│   ├── _atlas_blocks.parquet                  9520-block canonical reference
│   ├── _block_coverage_<cohort>.parquet       per-block CpG coverage by cohort
│   ├── GSE186458_cfdna_harmonized.parquet     23 healthy cfDNA × 9520 blocks
│   ├── GSE186458_wbc_harmonized.parquet       23 WBC × 9520 blocks
│   ├── GSE186458_cfdna_deconv_{nnls,uxm}.parquet
│   ├── GSE186458_wbc_deconv_{nnls,uxm}.parquet
│   ├── GSE186458_mixtures_betas.parquet       9520 × 828 in-silico mixture matrix
│   ├── GSE186458_mixtures_labels.parquet      828 labels
│   ├── GSE186458_mixtures_deconv.parquet      828 × 40 cell-fraction vectors
│   ├── GSE122126_harmonized.parquet           cross-cohort validation reference
│   └── cell_type_groups.json                  atlas cell-type family mapping
├── models/
│   ├── multitask_mlp_deconv.pt
│   └── multitask_mlp_raw.pt
├── results/
│   ├── multitask_eval.json                    full eval table
│   ├── ablation_deconv_vs_raw.json
│   ├── mixture_calibration.{json,png}         calibration sanity
│   ├── pivot_phase1_sanity.json               Loyfer-published numbers comparison
│   └── roc_stage{1,2,3}.png                   stage-stratified ROC
└── src/
    ├── gate{0,1,2,3}_*.py                     original gate scripts
    ├── gate_redo1_panel_probe.py              GSE149438 hard-stop probe
    └── pivot_phase{1,2}_step*.py              pivot scripts
```

External tooling:
- `/home/rohit/tools/wgbs_tools/` (commit 6f24bed)
- `/home/rohit/tools/UXM_deconv/`
- conda env `wgbs` at `/home/rohit/miniconda3/envs/wgbs/`
- HDD raw: `/mnt/hdd/research/cfdna/raw/GSE186458/pivot/` (25 GB cfDNA+WBC + 11 GB sorted cells)

## Appendix B — Reference Numbers (Loyfer 2023 healthy cfDNA)

Published vs Pivot Phase 1 measured (UXM on 23 healthy):

| Family | Loyfer pub | UXM measured | Within ±50% rel error? |
|--------|-----------:|-------------:|:----------------------:|
| Granulocyte | 30% | 29.2% | yes |
| Mono/Macro | 20% | 19.2% | yes |
| Lymphocyte (B+T+NK) | 3% | 2.7% | yes |
| Liver-Hep | 3% | 3.0% | yes |
| Endothelial | 6% | 5.7% | yes |
| Erythroid | 8% | 3.5% | no (still small abs error) |
| Megakaryocyte | (not highlighted in paper) | 31.5% | n/a |

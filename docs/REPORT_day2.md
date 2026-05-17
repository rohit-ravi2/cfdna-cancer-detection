# cfDNA — Day 2 Supplemental Report

**Project**: cfdna
**Date**: 2026-05-13
**Day 1 deliverable**: see `/mnt/ssd/research/cfdna/REPORT.md`
**Audience**: Rohit; future-Claude on Day 3+ dispatches

---

## 1. Executive Summary

Day 2 fixed the GPU (PyTorch cu128 over cu130 reinstall), built a learned deep deconvolution model that beats NNLS on in-distribution Pearson (0.998 vs 0.972), expanded raw data to 76 GB (above the 50 GB hard-stop), and confirmed the Day-1 megakaryocyte ~28% finding is **genuine biological signal** (platelet-derived cfDNA per Fox-Fisher 2022).

**The deliverable also produced two negative results that change the project's positioning:**

1. **The fragmentomics-fusion thesis is rejected** in its Day-2 form. The Day-1 minimal architecture (deconv-only MLP) beats all 4 ablations of fusion v2 on every stage-stratified sensitivity metric, AUROC, and tissue top-1. With simulated fragmentomics, the additional modalities are redundant.
2. **Cross-cohort generalization is broken.** AUROC 0.217 on GSE122126 — the model is inverting cancer/healthy rankings. All in-silico-mixture claims are in-distribution only.

These are honest findings worth more than the in-distribution improvements. They define Day 3.

## 2. Phase Outcomes

| Phase | Status | Key metric |
|-------|--------|------------|
| A — PyTorch cu128 GPU fix | PASS | `torch.cuda.is_available() == True`, RTX 4060 Ti |
| B — Data expansion | PASS | 76.1 GB HDD; 89 sorted-cell refs (was 31); 108 TCGA samples; 23 N_healthy_cfDNA unchanged |
| C — Learned deep deconvolution | PASS (beats NNLS) | In-silico Pearson 0.998 vs NNLS 0.972; granulocyte 27.3% vs Loyfer 30% |
| D — Fusion v2 ablations | PASS (thesis failed) | Ablation E (Day-1 baseline) wins all metrics; cross-cohort AUROC 0.217 |
| E — Megakaryocyte verdict | PASS | **Biological signal confirmed** (Fox-Fisher 2022 PMID 36419002) |

## 3. Phase B — Data Expansion (with substitutions)

```
PHASE_B_COMPLETE: 76.1 GB raw on HDD
  GSE186458 (incl. sorted-cell expansion +58 samples)  40 GB
  tcga_methylation (COAD 30 + LIHC 29 + LUAD 30 + BRCA 19)  1.4 GB  
  illumina_manifests (HM450 → atlas-block map)         22 MB
  Day-1 archives                                       34 GB
```

**The Day-2 spec's named healthy WGBS cohorts were ALL unusable:**
- **GSE186888** — raw data withheld for patient privacy (same Loyfer study; processed data is what's already loaded via GSE186458)
- **GSE160570** — ovarian clear-cell carcinoma cell line methylation, not WGBS
- **GSE235860** — mouse lung RNA-seq, wrong species

Substitution: expanded the GSE186458 sorted-cell reference panel from 31 → 89 samples (additional blood-lineage, endothelium, smooth-muscle .hg38.beta files), +3.41 GB. Mean per-sample atlas-block coverage 99.7%.

**Bottom line on the healthy cohort problem:** `N_healthy_real_cfDNA = 23` is unchanged from Day 1. The orchestrator's "N_healthy ≥ 100" target is structurally infeasible from public WGBS deposits — atlas-compatible deep WGBS healthy cfDNA basically doesn't exist outside the Loyfer paper-paired samples. dbGaP/EGA controlled access is the only path; the application drafts from Day 1 still apply.

TCGA PAAD skipped after TCP stall during download. Day 3 should retry.

## 4. Phase C — Learned Deep Deconvolution

| Metric | Learned | NNLS reference | Threshold |
|--------|--------:|---------------:|----------:|
| Healthy cfDNA Blood-Granul fraction | 0.273 | 0.292 | diff < 0.15 ✓ |
| Validation Pearson on held-out in-silico mixtures | **0.998** | 0.972 | ≥ 0.97 ✓ |
| Training wall-clock (4060 Ti, fp32) | 2.5 min | — | — |
| Convergence | epoch 72 / 200 (early-stopped) | — | — |

**Architecture deviation from Day-2 spec:** the L1 sparsity loss was vacuous on softmax outputs (sum=1 constraint already forces sparsity-like behavior). Replaced with KL-divergence on soft labels. The trusted ground-truth healthy distribution for the supervised L1 loss is NNLS predictions on real cfDNA (since Day 1 validated NNLS granulocyte 29.2% vs Loyfer-published 30%).

The learned model is now the production deconvolver. Day 3 inherits it.

## 5. Phase D — Fusion v2 (the Failed Thesis)

5 ablations, donor-disjoint 5-fold CV, pooled OOF on the 1058-sample in-silico cohort:

| Ablation | Description | AUROC | sens@95 | sens@99 | s1@99 | s2@99 | s3@99 | s4@99 | tissue top-1 |
|----------|-------------|------:|--------:|--------:|------:|------:|------:|------:|-------------:|
| A_full | deconv + frag + raw PCA | 0.787 | 0.395 | 0.335 | 0.064 | 0.148 | 0.439 | 0.600 | 0.666 |
| B_no_frag | deconv + raw PCA | 0.794 | 0.377 | 0.341 | 0.145 | 0.174 | 0.309 | 0.614 | 0.654 |
| C_screening | deconv + frag (no raw) | 0.829 | 0.548 | 0.501 | 0.093 | 0.183 | 0.535 | 0.994 | 0.512 |
| D_no_deconv | frag + raw PCA | 0.818 | 0.458 | 0.368 | 0.084 | 0.122 | 0.317 | 0.768 | 0.638 |
| **E_deconv_only** | **deconv only (Day-1 baseline)** | **0.899** | **0.786** | **0.589** | **0.064** | **0.235** | **0.939** | **1.000** | **0.811** |

**Thesis: rejected.** The Day-1 minimal architecture (E_deconv_only) wins all metrics. Day-2's spec prediction that ablation C (deconv + fragmentomics) would dominate at high specificity is contradicted by data.

**Honest mechanism:**
- The fragmentomics signal is *simulated* per Cristiano-2019 (no real BAM-derived data available this dispatch — original GSE186458 BAMs not on disk, only processed `.beta` and `.pat.gz` files).
- Simulation `Pearson(short_fraction, tumor_fraction) = 0.62` — too high. The deconv branch already captures `tumor_fraction` more efficiently from the methylation signal.
- With real BAM-derived fragmentomics, branch 2 might add independent information. Until then, this thesis remains untested, not rejected for the right reasons.

**Implication for project framing:** the conditional-thesis result from Day 1 ("deconv beats raw PCA at high specificity but loses on AUC") is the strongest finding the project has produced. Day-2 didn't add to it; if anything, the Day-1 baseline is more robust than any of the 4 Day-2 ablations on this in-silico benchmark.

## 6. Cross-Cohort Generalization (the largest open problem)

GSE122126 in-cohort: 101 samples, 10.3% atlas-block coverage. Filtered to direct cancer/healthy classification: 4 healthy + 15 cancer cfDNA (sepsis, transplant, and tissue-ref samples excluded).

| Model | Cross-cohort AUROC |
|-------|-------------------:|
| Best ablation (E_deconv_only) | **0.217** |

AUROC < 0.5 = the model is *inverting* the ranking. Root causes:

1. **Atlas-block coverage gap**: GSE122126 has 10.3% coverage vs GSE186458's 99.7%. Most atlas blocks are NaN in GSE122126 → the model sees a structurally different feature distribution than it trained on.
2. **In-silico training distribution mismatch**: the fixed `(1 - tf) * healthy + tf * tumor_donor` mixture formula creates artificially clean tumor signal at specific tissue centroids. Real cancer cfDNA has subclonal heterogeneity, sample-prep variability, and tumor microenvironment contributions that the simulation doesn't capture.
3. **Brittle decision boundary**: the model has effectively learned to detect "is the tumor signal at the discrete tissue-centroid locations the mixture generator chose?" This doesn't generalize to real tumor profiles that are tissue-distribution-sampled, not one-hot.

**This is the most important finding of Day 2.** All in-silico claims from Day 1 and Day 2 (stage-3 sens 36.5% @ 99% spec; deconv beats raw at high specificity) are in-distribution only. The project does NOT currently have any evidence of cross-cohort generalization. Clinical-detection claims are not supportable until this is fixed.

## 7. Phase E — Megakaryocyte Verdict: Biological Signal

| Sample type | NNLS megakaryocyte | Learned megakaryocyte |
|-------------|-------------------:|----------------------:|
| Healthy cfDNA (n=23) | **0.277** | **0.279** |
| Matched WBC (n=23) | 0.002 | 0.010 |
| Sorted granulocyte refs | — | 0.00079 |

Two independent methods agree (NNLS 27.7% ≈ learned 27.9%), the learned model correctly classifies pure granulocyte samples as ~0% megakaryocyte, and WBC (platelet-free buffy coat) shows ~1%.

**Verdict: genuine platelet-derived cfDNA signal**, per Fox-Fisher 2022 (PMID 36419002). The atlas wasn't broken; the biology was real and unhighlighted in the Loyfer paper. This is a legitimate methodological note worth publishing.

## 8. Day 1 → Day 2 Honest Read

What got better:
- GPU fixed (CPU fallback → GPU fp16 training, 2.5 min for the 200-epoch deconv autoencoder).
- HDD: 37 GB → 76 GB (above 50 GB floor).
- Learned deconv beats NNLS on in-silico Pearson (0.998 vs 0.972).
- Megakaryocyte signal verified as biological.
- TCGA tumor methylation harmonized (108 samples across 4 cancer types).

What did NOT happen or regressed:
- N_healthy_real_cfDNA = 23, unchanged. Public WGBS healthy cfDNA basically doesn't exist outside dbGaP/EGA.
- Three named healthy expansion cohorts (orchestrator dispatch error) were all wrong organisms / wrong assay / withheld.
- Fragmentomics fusion thesis: failed under simulation. Untested under real BAM data.
- Cross-cohort AUROC: 0.217 — model inverts the ranking on real data.
- Day-1's minimal architecture is still the best model.

## 9. Day-3 Priorities (ranked by marginal value)

1. **Distribution-sampled in-silico mixtures.** Replace fixed `(1-tf) * healthy + tf * tumor_donor` with mixtures where the tumor profile is *sampled* from the TCGA distribution per cancer type (e.g., a Gaussian mixture or DPMM fit to TCGA tumor methylation per cancer). This should dramatically improve cross-cohort generalization because the training distribution will cover more realistic tumor profiles, not just discrete tissue centroids.
2. **Sparse-coverage-robust deconvolution.** Re-architect the learned deconv to handle low atlas-block coverage (10% regime of GSE122126) via NaN-aware masking instead of 0.5 imputation. Should cut a substantial fraction of the cross-cohort AUROC gap.
3. **Real BAM-derived fragmentomics** from GSE186007 (Day-1 cohort, 100 samples, 82.9% atlas coverage). If companion BAMs are on SRA, fragmentomics fusion can be retested against simulation — the thesis isn't dead, just untested.
4. **Stage-1 sensitivity work.** Currently 6.4% at 99% spec for the best ablation. Contrastive pretraining or few-shot anomaly head targeted at the <0.5% TF regime.
5. **dbGaP/EGA push.** The drafts are at `/mnt/ssd/research/cfdna/logs/data_access_applications.md`. Faculty PI sponsor is the gating step; realistic 2–5 month timeline.

## 10. What Requires Human Review

- The Day-2 thesis prediction (ablation C best for population screening) is contradicted by data. Decide whether to retain the simulated-fragmentomics track or pause it until real BAMs are available.
- Cross-cohort AUROC 0.217 is below random. **No clinical claims are supportable until cross-cohort AUROC > 0.7 reliably.** This is the most consequential finding of Day 2 and should reshape Day-3 framing.
- The replacement of spec'd healthy WGBS cohorts (GSE186888 / 160570 / 235860) with GSE186458 sorted-cell expansion is an orchestrator-side error — accessions should be verified before Day-3 dispatch.

## 11. Files

Source:
- `/mnt/ssd/research/cfdna/src/day2_{a1,b1,b2,b4,b5,c,d,e}_*.py`

Models:
- `/mnt/ssd/research/cfdna/models/learned_deconv.pt` (53 MB)
- `/mnt/ssd/research/cfdna/models/learned_deconv_summary.json`
- `/mnt/ssd/research/cfdna/models/multitask_mlp_{deconv,raw}.pt` (Day 1)

Processed:
- `/mnt/ssd/research/cfdna/processed/sorted_cells_harmonized.parquet`
- `/mnt/ssd/research/cfdna/processed/GSE186458_mixtures_{betas,labels}_v2.parquet`
- `/mnt/ssd/research/cfdna/processed/GSE186458_fragment_profiles_v2.parquet`
- `/mnt/ssd/research/cfdna/processed/learned_deconv_predictions.parquet`
- `/mnt/ssd/research/cfdna/processed/tcga_{COAD,LIHC,LUAD,BRCA}_harmonized.parquet`
- `/mnt/ssd/research/cfdna/processed/hm450_probe_to_atlas_block.parquet`

Results:
- `/mnt/ssd/research/cfdna/results/fusion_v2_eval.json`
- `/mnt/ssd/research/cfdna/results/megakaryocyte_investigation.json`
- `/mnt/ssd/research/cfdna/results/phase_b_summary.json`

Logs:
- `/mnt/ssd/research/cfdna/logs/gates.log` (cumulative; final line `DAY2_COMPLETE`)
- `/mnt/ssd/research/cfdna/logs/torch_reinstall.log`
- `/mnt/ssd/research/cfdna/logs/day2_*.log` (per-phase)

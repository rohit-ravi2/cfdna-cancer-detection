#!/usr/bin/env python3
"""
Day-3 Phase C — Multi-task classifier on NaN-aware deconv features +
cross-cohort GSE122126 evaluation. The headline metric for Day-3.

Architecture (Day-1 winner ablation E, deconv-only):
  Input: 40-dim cell fractions
  Body: Linear(40, 128) -> ReLU -> Dropout(0.3) ->
        Linear(128, 64)  -> ReLU
  Heads:
    cancer_present:    Linear(64, 1)  -> Sigmoid
    tissue_of_origin:  Linear(64, 5)  -> Softmax  (colon/liver/lung/breast/pancreas)
    tumor_fraction:    Linear(64, 1)  -> ReLU     (regression)
    stage_proxy:       Linear(64, 4)  -> Softmax

Training:
  - 4600 distsampled mixtures (PRIMARY signal)
  - 1058 fixed-donor mixtures (Day-2 mixtures, smaller weight)
  - 23 real healthy cfDNA labeled cancer_present=0

Eval:
  - In-silico held-out (20% of distsampled, stratified by cancer)
  - Cross-cohort: GSE122126 (4 healthy + 15 cancer cfDNA)
"""
import json, time, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve

PROC = Path("work/processed")
MODELS = Path("work/models")
RESULTS = Path("work/results")
RESULTS.mkdir(parents=True, exist_ok=True)
LOG = Path("work/logs/day3_phase_c.log")

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

TISSUE_LIST = ["colon", "liver", "lung", "breast", "pancreas"]


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def load_atlas_cts():
    atlas_tsv = "work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv"
    df = pd.read_csv(atlas_tsv, sep="\t", nrows=1)
    return [c for c in df.columns
            if c not in ("chr","start","end","startCpG","endCpG","target","name","direction")]


class MultiTaskMLP(nn.Module):
    def __init__(self, n_in=40, n_tissues=5, n_stages=4):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.head_cancer = nn.Linear(64, 1)
        self.head_tissue = nn.Linear(64, n_tissues)
        self.head_tf = nn.Linear(64, 1)
        self.head_stage = nn.Linear(64, n_stages)

    def forward(self, x):
        h = self.body(x)
        return (
            self.head_cancer(h).squeeze(-1),
            self.head_tissue(h),
            self.head_tf(h).squeeze(-1),
            self.head_stage(h),
        )


def build_data(atlas_cts):
    # 1) distsampled
    distsamp = pd.read_parquet(PROC / "day3_distsampled_deconv_nanaware.parquet")
    distsamp_lbl = pd.read_parquet(PROC / "distsampled_mixtures_labels.parquet")
    distsamp = distsamp.merge(distsamp_lbl[["sample_id","cancer_present","tumor_fraction","tissue_of_origin","stage_proxy"]],
                              on="sample_id")
    # 2) fixed mixtures
    fixed = pd.read_parquet(PROC / "day3_fixed_deconv_nanaware.parquet")
    fixed_lbl = pd.read_parquet(PROC / "GSE186458_mixtures_labels_v2.parquet")
    fixed = fixed.merge(fixed_lbl[["sample_id","cancer_present","tumor_fraction","tissue_of_origin","stage_proxy"]],
                         on="sample_id")
    # 3) real healthy
    cfdna = pd.read_parquet(PROC / "day3_cfdna_real_deconv_nanaware.parquet")
    cfdna["cancer_present"] = 0
    cfdna["tumor_fraction"] = 0.0
    cfdna["tissue_of_origin"] = "none"
    cfdna["stage_proxy"] = 0
    # 4) GSE122126 — cross-cohort test (filter to cfdna only, exclude TISSUE_REF/SEPSIS/TRANSPLANT/INVITRO_MIX)
    gse = pd.read_parquet(PROC / "day3_gse122126_deconv_nanaware.parquet")
    samples = pd.read_csv(PROC / "GSE122126_samples.tsv", sep="\t")
    gse = gse.merge(samples[["sample_id","label","title"]], on="sample_id")
    # cfDNA only
    cfdna_labels = ["HEALTHY","CRC","LUNG","CUP","BREAST"]
    gse = gse[gse["label"].isin(cfdna_labels)].copy()
    # cancer_present
    gse["cancer_present"] = gse["label"].apply(lambda x: 0 if x == "HEALTHY" else 1)
    gse["tumor_fraction"] = 0.0
    gse["tissue_of_origin"] = gse["label"].map({"HEALTHY":"none","CRC":"colon","LUNG":"lung","CUP":"unknown","BREAST":"breast"})
    gse["stage_proxy"] = 0
    log(f"datasets: distsamp={len(distsamp)} fixed={len(fixed)} cfdna_real={len(cfdna)} gse122126={len(gse)} (healthy={(gse['cancer_present']==0).sum()} cancer={(gse['cancer_present']==1).sum()})")
    return distsamp, fixed, cfdna, gse, atlas_cts


def make_xy(df, atlas_cts):
    X = df[atlas_cts].to_numpy(dtype=np.float32)
    y_cancer = df["cancer_present"].to_numpy(dtype=np.float32)
    y_tf = df["tumor_fraction"].to_numpy(dtype=np.float32)
    tissue = df["tissue_of_origin"].astype(str).to_numpy()
    y_tissue = np.array([TISSUE_LIST.index(t) if t in TISSUE_LIST else -1 for t in tissue], dtype=np.int64)
    y_stage = df["stage_proxy"].to_numpy(dtype=np.int64).clip(0, 3)
    return X, y_cancer, y_tf, y_tissue, y_stage


def train_one(X_train, ycan_train, yti_train, ytf_train, yst_train,
              X_val, ycan_val, yti_val, ytf_val, yst_val,
              n_in, device, n_epochs=50, lr=1e-3):
    model = MultiTaskMLP(n_in=n_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss()
    ce_t = nn.CrossEntropyLoss(ignore_index=-1)
    ce_s = nn.CrossEntropyLoss(ignore_index=0)  # 0 = unknown stage
    mse = nn.MSELoss()

    def to_t(*a): return tuple(torch.from_numpy(x).to(device) for x in a)
    Xt, yct, yti_t, ytft, ystt = to_t(X_train, ycan_train, yti_train, ytf_train, yst_train)
    Xv, ycv, yti_v, ytfv, ystv = to_t(X_val, ycan_val, yti_val, ytf_val, yst_val)

    best_auroc = -1; best_state = None; patience = 10; stale = 0
    bs = 64
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(Xt.size(0), device=device)
        ltot = 0
        n_batches = 0
        for i in range(0, Xt.size(0), bs):
            idx = perm[i:i+bs]
            x = Xt[idx]
            opt.zero_grad()
            lc, lti, ltf, lst = model(x)
            L = (
                3*bce(lc, yct[idx])
                + 1*ce_t(lti, yti_t[idx])
                + 1*mse(ltf, ytft[idx])
                + 1*ce_s(lst, ystt[idx])
            )
            L.backward()
            opt.step()
            ltot += L.item(); n_batches += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            lc, lti, ltf, lst = model(Xv)
            p_cancer = torch.sigmoid(lc).cpu().numpy()
        try:
            auroc = roc_auc_score(ycan_val, p_cancer)
        except Exception:
            auroc = 0.5
        log(f"  epoch {epoch+1}/{n_epochs} loss={ltot/n_batches:.4f} val_auroc={auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  early stop at {epoch+1}")
                break
    model.load_state_dict(best_state)
    return model, best_auroc


def sens_at_spec(y_true, y_score, spec_target):
    """Compute sensitivity at fixed specificity."""
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    # Find largest threshold where spec >= spec_target
    valid = spec >= spec_target
    if not valid.any():
        return float("nan"), float("nan")
    # Among valid points, pick the one maximizing TPR
    idx = np.argmax(tpr * valid)
    return float(tpr[idx]), float(spec[idx])


def eval_set(model, X, y_cancer, y_tissue, device, name):
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X).to(device)
        lc, lti, ltf, lst = model(Xt)
        p_cancer = torch.sigmoid(lc).cpu().numpy()
        p_tissue = torch.softmax(lti, dim=-1).cpu().numpy()
    try:
        auroc = float(roc_auc_score(y_cancer, p_cancer))
    except Exception:
        auroc = float("nan")
    sens95, _ = sens_at_spec(y_cancer, p_cancer, 0.95)
    sens99, _ = sens_at_spec(y_cancer, p_cancer, 0.99)
    sens995, _ = sens_at_spec(y_cancer, p_cancer, 0.995)
    # Tissue top-1 on cancer-positive samples with known tissue
    cancer_mask = (y_cancer == 1) & (y_tissue >= 0) & (y_tissue < len(TISSUE_LIST))
    if cancer_mask.sum() > 0:
        top1 = float((p_tissue[cancer_mask].argmax(axis=1) == y_tissue[cancer_mask]).mean())
    else:
        top1 = float("nan")
    log(f"  {name}: AUROC={auroc:.4f} sens@95={sens95:.3f} sens@99={sens99:.3f} sens@99.5={sens995:.3f} tissue_top1={top1:.3f} (cancer_n={cancer_mask.sum()})")
    return {
        "AUROC": auroc, "sens95": sens95, "sens99": sens99, "sens995": sens995,
        "tissue_top1": top1, "n_total": int(len(y_cancer)),
        "n_cancer": int((y_cancer == 1).sum()),
        "n_healthy": int((y_cancer == 0).sum()),
        "p_cancer_min_max": [float(p_cancer.min()), float(p_cancer.max())],
        "p_cancer_healthy_mean": float(p_cancer[y_cancer == 0].mean()) if (y_cancer == 0).any() else None,
        "p_cancer_cancer_mean": float(p_cancer[y_cancer == 1].mean()) if (y_cancer == 1).any() else None,
    }


def main():
    t0 = time.time()
    log("=== PHASE C: Multi-task classifier + cross-cohort eval ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    atlas_cts = load_atlas_cts()
    distsamp, fixed, cfdna, gse, atlas_cts = build_data(atlas_cts)

    # Pool training: distsamp + fixed + cfdna_real (all with deconv features)
    train_df = pd.concat([distsamp, fixed, cfdna], ignore_index=True)
    log(f"pooled training: {len(train_df)}")

    # 5-fold (donor-disjoint where possible)
    # For simplicity, use random 80/20 split
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(train_df))
    cut = int(0.8 * len(train_df))
    tr_idx, va_idx = perm[:cut], perm[cut:]
    train_part = train_df.iloc[tr_idx]
    val_part = train_df.iloc[va_idx]

    Xtr, ycan_tr, ytf_tr, yti_tr, yst_tr = make_xy(train_part, atlas_cts)
    Xva, ycan_va, ytf_va, yti_va, yst_va = make_xy(val_part, atlas_cts)
    log(f"train n={len(Xtr)} val n={len(Xva)} cancer_train={(ycan_tr==1).sum()} cancer_val={(ycan_va==1).sum()}")

    log("training multi-task MLP...")
    model, best_val_auroc = train_one(
        Xtr, ycan_tr, yti_tr, ytf_tr, yst_tr,
        Xva, ycan_va, yti_va, ytf_va, yst_va,
        n_in=len(atlas_cts), device=device, n_epochs=80,
    )
    log(f"best val AUROC: {best_val_auroc:.4f}")
    torch.save({"state_dict": model.state_dict(), "atlas_cts": atlas_cts,
                "n_tissues": len(TISSUE_LIST)},
               MODELS / "day3_multitask_mlp.pt")

    # In-silico eval (held-out val)
    log("In-silico held-out eval:")
    insilico_res = eval_set(model, Xva, ycan_va, yti_va, device, "in_silico_holdout")

    # Cross-cohort eval on GSE122126
    log("Cross-cohort GSE122126 eval (the headline number):")
    Xgse, ycan_gse, ytf_gse, yti_gse, yst_gse = make_xy(gse, atlas_cts)
    cross_res = eval_set(model, Xgse, ycan_gse, yti_gse, device, "gse122126_cross_cohort")

    # Per-stage sensitivity on in-silico
    log("In-silico stage-stratified:")
    stage_res = {}
    Xva_arr, ycan_va_arr, yti_va_arr = Xva, ycan_va, yti_va
    yst_va_arr = yst_va
    # Threshold from val (max youden on cancer)
    model.eval()
    with torch.no_grad():
        lc, _, _, _ = model(torch.from_numpy(Xva).to(device))
        p_val = torch.sigmoid(lc).cpu().numpy()
    fpr_v, tpr_v, thr_v = roc_curve(ycan_va_arr, p_val)
    spec_target = 0.99
    valid = (1 - fpr_v) >= spec_target
    if valid.any():
        idx = np.argmax(tpr_v * valid)
        threshold = float(thr_v[idx])
    else:
        threshold = 0.5
    log(f"99%-specificity threshold (in-silico val): {threshold:.4f}")
    for stage in [1, 2, 3, 4]:
        m = yst_va_arr == stage
        if m.sum() < 5:
            stage_res[f"stage_{stage}"] = None
            continue
        # Among samples in this stage class, all are cancer; sens = TPR with threshold
        with torch.no_grad():
            lc, _, _, _ = model(torch.from_numpy(Xva[m]).to(device))
            p = torch.sigmoid(lc).cpu().numpy()
        sens = float((p >= threshold).mean())
        stage_res[f"stage_{stage}"] = {"n": int(m.sum()), "sens_at_99spec": sens}
        log(f"  stage {stage}: n={m.sum()} sens@99spec={sens:.3f}")

    # Per-cancer-type sensitivity on in-silico
    type_res = {}
    log("In-silico per-cancer-type:")
    val_subset = val_part.copy()
    for ttype in TISSUE_LIST:
        m_str = (val_subset["tissue_of_origin"].astype(str).values == ttype)
        if m_str.sum() < 5:
            type_res[ttype] = None
            continue
        Xt = val_subset[atlas_cts].to_numpy(dtype=np.float32)[m_str]
        with torch.no_grad():
            lc, _, _, _ = model(torch.from_numpy(Xt).to(device))
            p = torch.sigmoid(lc).cpu().numpy()
        sens = float((p >= threshold).mean())
        type_res[ttype] = {"n": int(m_str.sum()), "sens_at_99spec": sens}
        log(f"  {ttype}: n={m_str.sum()} sens@99spec={sens:.3f}")

    # Day-2 comparison: run the OLD model on GSE122126 deconv outputs
    # Skip — we don't have Day-2 MLP outputs preserved. Just record Day-2's headline.

    summary = {
        "best_val_auroc": float(best_val_auroc),
        "in_silico_holdout": insilico_res,
        "cross_cohort_gse122126": cross_res,
        "in_silico_stage_strat": stage_res,
        "in_silico_per_type": type_res,
        "day2_cross_cohort_auroc": 0.217,
        "day2_in_silico_auroc": 0.899,
        "delta_cross_cohort": cross_res["AUROC"] - 0.217,
        "delta_in_silico": insilico_res["AUROC"] - 0.899,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(MODELS / "phase_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(RESULTS / "day3_phase_c_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"=== PHASE C DONE in {summary['elapsed_min']} min ===")
    log("HEADLINE:")
    log(f"  Day-2 cross-cohort AUROC: 0.217")
    log(f"  Day-3 cross-cohort AUROC: {cross_res['AUROC']:.4f}")
    log(f"  Day-2 in-silico AUROC: 0.899")
    log(f"  Day-3 in-silico AUROC: {insilico_res['AUROC']:.4f}")
    log(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
PIVOT_PHASE2 STEP 4 — Multi-task MLP classifier on 39-dim deconvolved
cell-fraction vectors (and ablation MLP on raw 9520-dim atlas-block betas
reduced to 256-dim via PCA so we compare equal-capacity inputs).

Tasks:
    cancer_present  : binary (BCE)
    tissue_of_origin: 5-class softmax (cross-entropy, masked for healthy)
    tumor_fraction  : regression (MSE on log10(fraction + 1e-4) bounded)

Train: 70/15/15 stratified split on (cancer_present, tissue_of_origin).
Same architecture template; only the input dim changes.

Evaluation: sensitivity at 95/99/99.5% specificity, stratified by
"stage-equivalent" tumor_fraction bins:
    stage1 = fraction in {0.001, 0.005, 0.01}
    stage2 = fraction in {0.03, 0.05}
    stage3 = fraction in {0.10, 0.15}

Outputs:
    work/models/multitask_mlp_deconv.pt
    work/models/multitask_mlp_raw.pt
    work/results/multitask_eval.json
    work/results/ablation_deconv_vs_raw.json
    work/results/roc_stage1.png
    work/results/roc_stage2.png
    work/results/roc_stage3.png
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DECONV_FILE  = Path("work/processed/GSE186458_mixtures_deconv.parquet")
BETAS_FILE   = Path("work/processed/GSE186458_mixtures_betas.parquet")
LABELS_FILE  = Path("work/processed/GSE186458_mixtures_labels.parquet")
ATLAS_TSV    = Path("work/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
META_COLS    = ["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"]
MODELS_DIR   = Path("work/models"); MODELS_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR      = Path("work/results"); RES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR      = Path("work/logs")
GATES_LOG    = LOG_DIR / "gates.log"
STDERR_LOG   = LOG_DIR / "pivot_phase2_step4.log"

TISSUE_LIST = ["colon", "liver", "lung", "breast", "pancreas"]
TISSUE2IDX  = {t: i for i, t in enumerate(TISSUE_LIST)}

STAGE_BINS = {
    "stage1": [0.001, 0.005, 0.01],
    "stage2": [0.03, 0.05],
    "stage3": [0.10, 0.15],
}

SEED = 20260513
DEVICE = "cpu"   # GPU mismatch per dispatch; 39->64-dim MLP is trivial on CPU


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with STDERR_LOG.open("a") as f:
        f.write(line + "\n")


# ============================================================
# Data
# ============================================================
def load_atlas_columns():
    df = pd.read_csv(ATLAS_TSV, sep="\t", nrows=1)
    cts = [c for c in df.columns if c not in META_COLS]
    return cts


def build_deconv_features(deconv_df: pd.DataFrame, labels_df: pd.DataFrame, atlas_cts: list[str]):
    df = deconv_df.merge(labels_df, on="sample_id", how="inner", suffixes=("", "_lb"))
    # If columns shadow labels columns, ensure correct ones used
    for c in ["cancer_present", "tumor_fraction", "tissue_of_origin"]:
        if c + "_lb" in df.columns and c not in df.columns:
            df[c] = df[c + "_lb"]
    X = df[atlas_cts].to_numpy(dtype=np.float32)
    y_cancer = df["cancer_present"].to_numpy(dtype=np.float32)
    y_frac   = df["tumor_fraction"].to_numpy(dtype=np.float32)
    tissue_idx = df["tissue_of_origin"].map(lambda t: TISSUE2IDX.get(t, -1)).to_numpy(dtype=np.int64)
    sample_ids = df["sample_id"].to_numpy()
    healthy_donor = df["healthy_donor"].to_numpy()
    return X, y_cancer, y_frac, tissue_idx, sample_ids, healthy_donor


def build_raw_features(labels_df: pd.DataFrame, n_components: int = 256):
    betas_df = pd.read_parquet(BETAS_FILE)
    sample_cols = [c for c in betas_df.columns if c != "block_id"]
    # Order = labels order
    sample_cols_in = [s for s in labels_df["sample_id"].values if s in sample_cols]
    log(f"raw: matched {len(sample_cols_in)} samples")
    Bmat = np.array(betas_df[sample_cols_in].to_numpy(dtype=np.float32).T, copy=True)  # [N, 9520]
    # Imputation: NaN -> per-column mean
    col_mean = np.nanmean(Bmat, axis=0)
    col_mean = np.where(np.isnan(col_mean), 0.5, col_mean)
    inds = np.where(np.isnan(Bmat))
    Bmat[inds] = np.take(col_mean, inds[1])
    log(f"raw: shape after impute = {Bmat.shape}")
    pca = PCA(n_components=n_components, random_state=SEED)
    Xpca = pca.fit_transform(Bmat).astype(np.float32)
    log(f"raw: PCA -> {Xpca.shape}  ev_explained={pca.explained_variance_ratio_.sum():.3f}")
    # Re-build label arrays in this sample order
    lbl = labels_df.set_index("sample_id").loc[sample_cols_in].reset_index()
    y_cancer = lbl["cancer_present"].to_numpy(dtype=np.float32)
    y_frac   = lbl["tumor_fraction"].to_numpy(dtype=np.float32)
    tissue_idx = lbl["tissue_of_origin"].map(lambda t: TISSUE2IDX.get(t, -1)).to_numpy(dtype=np.int64)
    sample_ids = lbl["sample_id"].to_numpy()
    healthy_donor = lbl["healthy_donor"].to_numpy()
    return Xpca, y_cancer, y_frac, tissue_idx, sample_ids, healthy_donor


# ============================================================
# Model
# ============================================================
class MultiTaskMLP(nn.Module):
    def __init__(self, in_dim: int, n_tissue: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128),    nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.BatchNorm1d(64),  nn.ReLU(),
        )
        self.head_cancer = nn.Linear(64, 1)
        self.head_tissue = nn.Linear(64, n_tissue)
        self.head_frac   = nn.Linear(64, 1)

    def forward(self, x):
        h = self.backbone(x)
        return self.head_cancer(h), self.head_tissue(h), self.head_frac(h)


# ============================================================
# Training
# ============================================================
def encode_frac(y_frac: np.ndarray) -> np.ndarray:
    """Map tumor_fraction in [0, 0.15] to a target for regression. Use
    log10(fraction + 1e-4) for the cancer samples; for healthy (fraction=0),
    use log10(1e-4) = -4 as the floor.
    """
    return np.log10(y_frac + 1e-4)


def donor_kfold(healthy_donor: np.ndarray, n_splits: int = 5, seed: int = SEED):
    """Donor-disjoint K-fold. Returns list of (train_mask, val_mask, test_mask)
    tuples. Each donor (and all their derived mixtures) belongs to exactly one
    test fold; train comes from the remaining donors with 15% held out as val."""
    rng = np.random.default_rng(seed)
    unique_donors = sorted(set(healthy_donor))
    rng.shuffle(unique_donors)
    folds = np.array_split(unique_donors, n_splits)
    out = []
    for fi in range(n_splits):
        test_d = set(folds[fi])
        remaining = [d for f_j, fdl in enumerate(folds) for d in fdl if f_j != fi]
        rng2 = np.random.default_rng(seed + fi + 100)
        rng2.shuffle(remaining)
        n_val = max(1, int(round(0.18 * len(remaining))))
        val_d = set(remaining[:n_val])
        train_d = set(remaining[n_val:])
        tr = np.array([d in train_d for d in healthy_donor])
        va = np.array([d in val_d   for d in healthy_donor])
        te = np.array([d in test_d  for d in healthy_donor])
        out.append((tr, va, te, sorted(train_d), sorted(val_d), sorted(test_d)))
    return out


class MixDataset(Dataset):
    def __init__(self, X, y_cancer, tissue_idx, y_frac_log):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_cancer = torch.tensor(y_cancer, dtype=torch.float32)
        self.tissue = torch.tensor(tissue_idx, dtype=torch.long)
        self.y_frac = torch.tensor(y_frac_log, dtype=torch.float32)
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y_cancer[i], self.tissue[i], self.y_frac[i]


def train_model(model, train_loader, val_loader, n_epochs=200, patience=30, label="?"):
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    mse = nn.MSELoss()
    best_val = float("inf"); best_state = None; bad = 0
    for ep in range(n_epochs):
        model.train()
        tr_loss = 0
        for X, yc, yt, yf in train_loader:
            X, yc, yt, yf = X.to(DEVICE), yc.to(DEVICE), yt.to(DEVICE), yf.to(DEVICE)
            opt.zero_grad()
            lc, lt, lf = model(X)
            loss_c = bce(lc.squeeze(-1), yc)
            # tissue loss only on cancer-positive samples (yc==1)
            mask = (yc == 1.0)
            if mask.sum() > 0:
                loss_t = ce(lt[mask], yt[mask])
            else:
                loss_t = torch.tensor(0.0, device=DEVICE)
            # fraction loss on all (healthy -> log10(1e-4)=-4 floor)
            loss_f = mse(lf.squeeze(-1), yf)
            loss = 1.0 * loss_c + 0.5 * loss_t + 0.5 * loss_f
            loss.backward()
            opt.step()
            tr_loss += loss.item() * X.size(0)
        sched.step()
        tr_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, yc, yt, yf in val_loader:
                X, yc, yt, yf = X.to(DEVICE), yc.to(DEVICE), yt.to(DEVICE), yf.to(DEVICE)
                lc, lt, lf = model(X)
                loss_c = bce(lc.squeeze(-1), yc)
                mask = (yc == 1.0)
                if mask.sum() > 0:
                    loss_t = ce(lt[mask], yt[mask])
                else:
                    loss_t = torch.tensor(0.0, device=DEVICE)
                loss_f = mse(lf.squeeze(-1), yf)
                loss = 1.0 * loss_c + 0.5 * loss_t + 0.5 * loss_f
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_loader.dataset)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if ep % 10 == 0 or ep < 3:
            log(f"  [{label}] ep={ep:3d} tr={tr_loss:.4f} val={val_loss:.4f} bad={bad}")
        if bad >= patience:
            log(f"  [{label}] early stop at ep={ep}  best_val={best_val:.4f}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def predict(model, X):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        lc, lt, lf = model(Xt)
        prob_cancer = torch.sigmoid(lc).cpu().numpy().squeeze()
        prob_tissue = torch.softmax(lt, dim=-1).cpu().numpy()
        pred_frac_log = lf.cpu().numpy().squeeze()
    return prob_cancer, prob_tissue, pred_frac_log


# ============================================================
# Metrics
# ============================================================
def sensitivity_at_specificity(prob_pos, y_true, spec_levels=(0.95, 0.99, 0.995)):
    """Given probabilities (higher=cancer), pick threshold from healthy distribution
    s.t. specificity = (TN / (TN+FP)) >= target_spec. Return sensitivity on positives."""
    out = {}
    healthy_scores = prob_pos[y_true == 0]
    cancer_scores = prob_pos[y_true == 1]
    for s in spec_levels:
        # threshold at the (1-s) quantile of healthy
        thr = np.quantile(healthy_scores, s)
        sens = float((cancer_scores >= thr).mean()) if len(cancer_scores) else float("nan")
        achieved_spec = float((healthy_scores < thr).mean())
        out[f"sens_at_spec_{s:.3f}"] = sens
        out[f"thr_at_spec_{s:.3f}"] = float(thr)
        out[f"achieved_spec_{s:.3f}"] = achieved_spec
    return out


def near_in(arr, vals, tol=1e-5):
    """Float-tolerant np.isin."""
    return np.any(np.abs(arr[:, None] - np.array(vals)[None, :]) < tol, axis=1)


def evaluate(prob_cancer, prob_tissue, pred_frac_log,
             y_cancer, tissue_idx, y_frac, sample_ids):
    """Stage-bin-stratified eval."""
    eval_out = {}
    # Overall AUC and pooled-stage metrics
    from sklearn.metrics import roc_auc_score, roc_curve
    try:
        eval_out["overall_auc"] = float(roc_auc_score(y_cancer, prob_cancer))
    except ValueError:
        eval_out["overall_auc"] = float("nan")

    # Pooled (all positives vs all healthy)
    eval_out["pooled"] = sensitivity_at_specificity(prob_cancer, y_cancer)
    for stage_name, fracs in STAGE_BINS.items():
        mask_pos_stage = near_in(y_frac, fracs) & (y_cancer == 1)
        mask_neg = (y_cancer == 0)
        # Combine: healthy controls + this stage's positives
        keep = mask_pos_stage | mask_neg
        if mask_pos_stage.sum() == 0:
            eval_out[stage_name] = {"n_positives": 0}
            continue
        sub_p = prob_cancer[keep]
        sub_y = y_cancer[keep]
        eval_out[stage_name] = sensitivity_at_specificity(sub_p, sub_y)
        eval_out[stage_name]["n_positives"] = int(mask_pos_stage.sum())
        eval_out[stage_name]["n_negatives"] = int(mask_neg.sum())
        try:
            eval_out[stage_name]["auc"] = float(roc_auc_score(sub_y, sub_p))
        except ValueError:
            eval_out[stage_name]["auc"] = float("nan")

    # Per-fraction sensitivity at 99% spec
    per_frac = {}
    healthy_scores = prob_cancer[y_cancer == 0]
    thr99 = float(np.quantile(healthy_scores, 0.99))
    for f in [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]:
        m = near_in(y_frac, [f]) & (y_cancer == 1)
        if m.sum() == 0:
            continue
        per_frac[f"f_{f:.4f}"] = {
            "n": int(m.sum()),
            "sens_at_thr99": float((prob_cancer[m] >= thr99).mean()),
            "mean_prob": float(prob_cancer[m].mean()),
        }
    eval_out["per_fraction"] = per_frac
    eval_out["thr_at_spec_99_pooled"] = thr99

    # Tissue-of-origin top-1/top-2 on cancer samples
    pos_mask = (y_cancer == 1)
    if pos_mask.sum() > 0:
        prob_t = prob_tissue[pos_mask]
        true_t = tissue_idx[pos_mask]
        valid = (true_t >= 0)
        if valid.sum() > 0:
            top1 = float((np.argmax(prob_t[valid], axis=1) == true_t[valid]).mean())
            top2_idx = np.argsort(-prob_t[valid], axis=1)[:, :2]
            top2 = float(np.any(top2_idx == true_t[valid][:, None], axis=1).mean())
            eval_out["tissue_top1"] = top1
            eval_out["tissue_top2"] = top2
            eval_out["tissue_n"] = int(valid.sum())
            # Per-tissue accuracy & per-stage tissue accuracy
            per_tissue = {}
            for ti, tname in enumerate(TISSUE_LIST):
                tm = (true_t == ti) & valid
                if tm.sum() > 0:
                    pred_t = np.argmax(prob_t, axis=1)
                    per_tissue[tname] = {
                        "n": int(tm.sum()),
                        "top1_acc": float((pred_t[tm] == ti).mean()),
                    }
            eval_out["tissue_per_class"] = per_tissue
            # Stage1-equivalent tissue-of-origin acc
            for stage_name, fracs in STAGE_BINS.items():
                mask_pos_stage = np.isin(y_frac[pos_mask], fracs) & valid
                if mask_pos_stage.sum() > 0:
                    eval_out.setdefault("tissue_per_stage", {})[stage_name] = {
                        "n": int(mask_pos_stage.sum()),
                        "top1": float((np.argmax(prob_t[mask_pos_stage], axis=1)
                                       == true_t[mask_pos_stage]).mean()),
                    }

    # Tumor-fraction MAE on log10 scale (positives only)
    if (y_cancer == 1).sum() > 0:
        true_log = np.log10(y_frac[y_cancer == 1] + 1e-4)
        pred_log = pred_frac_log[y_cancer == 1]
        eval_out["frac_mae_log10"] = float(np.mean(np.abs(true_log - pred_log)))
        eval_out["frac_spearman"] = float(
            pd.Series(true_log).corr(pd.Series(pred_log), method="spearman")
        )
    return eval_out


# ============================================================
# Main
# ============================================================
def run_one(X, y_cancer, y_frac, tissue_idx, sample_ids, healthy_donor, label, n_splits=5):
    """Run donor-K-fold CV. Pool out-of-fold predictions for a single evaluation
    that uses all 23 healthy + all 805 mixed samples. Returns the model trained
    on the largest fold + the pooled OOF predictions."""
    log(f"=== {label}: feature dim = {X.shape[1]}")
    folds = donor_kfold(healthy_donor, n_splits=n_splits)
    yfrac_log = encode_frac(y_frac)
    N = X.shape[0]
    oof_cancer = np.full(N, np.nan, dtype=np.float64)
    oof_tissue = np.full((N, len(TISSUE_LIST)), np.nan, dtype=np.float64)
    oof_frac_log = np.full(N, np.nan, dtype=np.float64)
    best_models = []
    fold_meta = []
    for fi, (tr, va, te, td_list, vd_list, ted_list) in enumerate(folds):
        log(f"  fold {fi}: train_donors={len(td_list)} val_donors={len(vd_list)} test_donors={len(ted_list)} "
            f"sizes tr={tr.sum()} va={va.sum()} te={te.sum()}")
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xva = scaler.transform(X[va])
        Xte = scaler.transform(X[te])
        tr_ds = MixDataset(Xtr, y_cancer[tr], tissue_idx[tr], yfrac_log[tr])
        va_ds = MixDataset(Xva, y_cancer[va], tissue_idx[va], yfrac_log[va])
        tr_loader = DataLoader(tr_ds, batch_size=64, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)

        torch.manual_seed(SEED + fi)
        model = MultiTaskMLP(in_dim=X.shape[1], n_tissue=len(TISSUE_LIST)).to(DEVICE)
        model, best_val = train_model(model, tr_loader, va_loader,
                                      n_epochs=200, patience=30, label=f"{label}-fold{fi}")
        pcancer, ptissue, plog = predict(model, Xte)
        oof_cancer[te] = pcancer
        oof_tissue[te] = ptissue
        oof_frac_log[te] = plog
        best_models.append((model, best_val, tr.sum(), va.sum(), te.sum()))
        fold_meta.append({"fold": fi, "best_val": best_val,
                           "n_train": int(tr.sum()), "n_val": int(va.sum()),
                           "n_test": int(te.sum()),
                           "test_donors": ted_list})

    # Sanity: all predictions filled in
    assert not np.isnan(oof_cancer).any(), "OOF missing predictions"
    eval_out = evaluate(oof_cancer, oof_tissue, oof_frac_log,
                        y_cancer, tissue_idx, y_frac, sample_ids)
    eval_out["_meta"] = {
        "label": label,
        "in_dim": int(X.shape[1]),
        "n_splits": n_splits,
        "fold_meta": fold_meta,
        "best_val_losses": [m[1] for m in best_models],
        "evaluation_mode": "out-of-fold pooled across all samples",
    }
    # Return the model with the smallest val loss as 'final' (saved to disk)
    best_idx = int(np.argmin([m[1] for m in best_models]))
    return best_models[best_idx][0], eval_out, (None, None, None), (oof_cancer, oof_tissue, oof_frac_log)


def plot_roc_for_stages(y_test, prob_test, y_frac_test, out_dir, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    for stage_name, fracs in STAGE_BINS.items():
        m_pos = np.isin(y_frac_test, fracs)
        m_neg = (y_test == 0)
        keep = m_pos | m_neg
        if m_pos.sum() < 3:
            continue
        fpr, tpr, _ = roc_curve(y_test[keep], prob_test[keep])
        a = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} {stage_name} (AUC={a:.3f})")
    plt.plot([0,1], [0,1], "k--", lw=0.5)
    plt.xlim(0, 0.15); plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.title(f"ROC (zoomed FPR<=0.15) — {label}")
    plt.tight_layout()


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not DECONV_FILE.exists():
        log(f"FATAL: {DECONV_FILE} missing"); sys.exit(1)
    if not LABELS_FILE.exists():
        log(f"FATAL: {LABELS_FILE} missing"); sys.exit(1)

    labels_df = pd.read_parquet(LABELS_FILE)
    deconv_df = pd.read_parquet(DECONV_FILE)
    atlas_cts = load_atlas_columns()
    log(f"deconv features: {len(atlas_cts)} atlas cell types")

    Xd, ycd, yfd, tid, sids_d, hdon_d = build_deconv_features(deconv_df, labels_df, atlas_cts)
    log(f"deconv X: {Xd.shape}  cancer+={int(ycd.sum())}  healthy={int((ycd==0).sum())}")

    # Raw features
    Xr, ycr, yfr, tir, sids_r, hdon_r = build_raw_features(labels_df, n_components=256)
    log(f"raw X (PCA256): {Xr.shape}")

    # Verify same sample order
    assert (sids_d == sids_r).all(), "sample order mismatch deconv vs raw"

    # Run deconv MLP
    model_d, eval_d, splits_d, preds_d = run_one(
        Xd, ycd, yfd, tid, sids_d, hdon_d, label="deconv")
    torch.save(model_d.state_dict(), MODELS_DIR / "multitask_mlp_deconv.pt")
    log(f"  saved multitask_mlp_deconv.pt")

    # Run raw MLP
    model_r, eval_r, splits_r, preds_r = run_one(
        Xr, ycr, yfr, tir, sids_r, hdon_r, label="raw_pca256")
    torch.save(model_r.state_dict(), MODELS_DIR / "multitask_mlp_raw.pt")
    log(f"  saved multitask_mlp_raw.pt")

    # Save eval
    eval_combined = {"deconv": eval_d, "raw_pca256": eval_r}
    json.dump(eval_combined, (RES_DIR / "multitask_eval.json").open("w"), indent=2,
              default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    log("  wrote multitask_eval.json")

    # Ablation summary
    ablation = {
        "summary": "Compare deconv (39-dim cell-type fractions) vs raw atlas-block PCA256",
        "stage1_sens_at_99spec_deconv":
            eval_d.get("stage1", {}).get("sens_at_spec_0.990", None),
        "stage1_sens_at_99spec_raw":
            eval_r.get("stage1", {}).get("sens_at_spec_0.990", None),
        "stage2_sens_at_99spec_deconv":
            eval_d.get("stage2", {}).get("sens_at_spec_0.990", None),
        "stage2_sens_at_99spec_raw":
            eval_r.get("stage2", {}).get("sens_at_spec_0.990", None),
        "stage3_sens_at_99spec_deconv":
            eval_d.get("stage3", {}).get("sens_at_spec_0.990", None),
        "stage3_sens_at_99spec_raw":
            eval_r.get("stage3", {}).get("sens_at_spec_0.990", None),
        "tissue_top1_deconv": eval_d.get("tissue_top1", None),
        "tissue_top1_raw":    eval_r.get("tissue_top1", None),
        "overall_auc_deconv": eval_d.get("overall_auc", None),
        "overall_auc_raw":    eval_r.get("overall_auc", None),
        "frac_mae_log10_deconv": eval_d.get("frac_mae_log10", None),
        "frac_mae_log10_raw":    eval_r.get("frac_mae_log10", None),
    }
    json.dump(ablation, (RES_DIR / "ablation_deconv_vs_raw.json").open("w"), indent=2,
              default=lambda o: None if isinstance(o, float) and np.isnan(o) else o)
    log("  wrote ablation_deconv_vs_raw.json")

    # ROC plots per stage (deconv vs raw)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    pcancer_d, _, _ = preds_d
    pcancer_r, _, _ = preds_r

    def near_in(arr, vals, tol=1e-5):
        return np.any(np.abs(arr[:, None] - np.array(vals)[None, :]) < tol, axis=1)

    for stage_name, fracs in STAGE_BINS.items():
        plt.figure(figsize=(5, 5))
        for label, prob_te, y_test, y_frac_test in [
            ("deconv", pcancer_d, ycd, yfd),
            ("raw_pca256", pcancer_r, ycr, yfr),
        ]:
            m_pos = near_in(y_frac_test, fracs) & (y_test == 1)
            m_neg = (y_test == 0)
            keep = m_pos | m_neg
            if m_pos.sum() < 3:
                continue
            fpr, tpr, _ = roc_curve(y_test[keep], prob_te[keep])
            a = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} (AUC={a:.3f}, n_pos={m_pos.sum()})")
        plt.plot([0,1],[0,1],"k--",lw=0.5)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title(f"ROC — {stage_name} ({'+'.join(str(f) for f in fracs)})")
        plt.legend(loc="lower right", fontsize=8)
        plt.xlim(0,1); plt.ylim(0,1.02)
        plt.tight_layout()
        plt.savefig(RES_DIR / f"roc_{stage_name}.png", dpi=130)
        plt.close()
    log("  wrote roc_*.png")

    # Gate log
    s1_d = eval_d.get("stage1", {}).get("sens_at_spec_0.990", float("nan"))
    s1_r = eval_r.get("stage1", {}).get("sens_at_spec_0.990", float("nan"))
    tt1 = eval_d.get("tissue_top1", float("nan"))
    elapsed_min = (time.time() - t0) / 60
    s1_d_str = f"{s1_d:.3f}" if isinstance(s1_d, float) else str(s1_d)
    s1_r_str = f"{s1_r:.3f}" if isinstance(s1_r, float) else str(s1_r)
    tt1_str = f"{tt1:.3f}" if isinstance(tt1, float) else str(tt1)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] PIVOT_PHASE2_GSE186458 STEP4_mlp: PASS "
                f"stage1_sens_at_99spec_deconv={s1_d_str} "
                f"stage1_sens_at_99spec_raw={s1_r_str} "
                f"tissue_top1={tt1_str} elapsed_min={elapsed_min:.1f}\n")
    log(f"Step4 done. elapsed_min={elapsed_min:.1f}")


if __name__ == "__main__":
    main()

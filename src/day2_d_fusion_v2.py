#!/usr/bin/env python3
"""
Day 2 Phase D — Fusion v2: three-branch model with deconv + fragmentomics + raw PCA.

Architecture:
  Branch 1: cell-type fraction (40 dims) -> Linear(40->128) -> LN -> ReLU
  Branch 2: fragment profile (8 dims) -> Linear(8->32) -> LN -> ReLU -> Linear(32->64)
  Branch 3: raw atlas-block PCA-64 -> Linear(64->128) -> LN -> ReLU

  Fusion: cross-attention (Branch1 as Q, Branch2 as K/V), concat with Branch3.
  Linear(128+64+128 -> 256) -> LN -> ReLU -> Dropout(0.3) -> Linear(256 -> 128)

Heads:
  - cancer_pred: Linear(128 -> 1) -> Sigmoid    (BCE)
  - tissue: Linear(128 -> 5) -> Softmax        (cross-entropy, only cancer samples)
  - stage: Linear(128 -> 4) -> Softmax         (cross-entropy, only cancer samples)
  - tumor_fraction: Linear(128 -> 1) -> Sigmoid (MSE, only cancer)

Training:
  Donor-disjoint 5-fold CV pooled OOF (same as Day 1).
  Cross-cohort eval: train on GSE186458-mixtures, test on GSE122126 healthy controls.

Ablations:
  A: full (all 3 branches)
  B: no fragmentomics (branch 2 zeroed)
  C: no raw PCA (branch 3 zeroed)  # population-screening track
  D: no deconv (branch 1 zeroed)
  E: deconv only (branches 2,3 zeroed)  # Day-1 baseline
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
RES = Path("/mnt/ssd/research/cfdna/results")
RES.mkdir(parents=True, exist_ok=True)
LOG = Path("/mnt/ssd/research/cfdna/logs/day2_d.log")

N_BLOCKS = 9520
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

TISSUE_LIST = ["colon", "liver", "lung", "breast", "pancreas"]
TISSUE_IDX = {t: i for i, t in enumerate(TISSUE_LIST)}


def log(m):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_data():
    """Load all features for in-silico mixtures + healthy."""
    log("loading data...")
    betas = pd.read_parquet(PROC / "GSE186458_mixtures_betas_v2.parquet")
    labels = pd.read_parquet(PROC / "GSE186458_mixtures_labels_v2.parquet")
    deconv = pd.read_parquet(PROC / "learned_deconv_predictions.parquet")
    frag = pd.read_parquet(PROC / "GSE186458_fragment_profiles_v2.parquet")

    # Get block-feature matrix [9520 x n_samples] -> transpose to [N, 9520]
    sample_cols = [c for c in betas.columns if c != "block_id"]
    block_mat = betas[sample_cols].values.T.astype(np.float32)
    block_mat = np.nan_to_num(block_mat, nan=0.5)
    log(f"block matrix shape: {block_mat.shape}")

    # Match label rows
    labels = labels.set_index("sample_id").loc[sample_cols].reset_index()

    # Deconv predictions: filter to mixture-source rows
    deconv_mix = deconv[deconv["source"] == "mix"].set_index("sample_id")
    # Reindex to match
    missing = [c for c in sample_cols if c not in deconv_mix.index]
    if missing:
        log(f"WARNING: {len(missing)} samples missing from deconv (using zeros)")
    ct_cols = [c for c in deconv_mix.columns if c not in ("sample_id", "source")]
    deconv_mat = np.zeros((len(sample_cols), len(ct_cols)), dtype=np.float32)
    for i, sid in enumerate(sample_cols):
        if sid in deconv_mix.index:
            deconv_mat[i] = deconv_mix.loc[sid, ct_cols].values
    log(f"deconv matrix: {deconv_mat.shape}, ct_cols={len(ct_cols)}")

    # Fragmentomics: 8 features in order
    frag_cols = ["modal_length", "mean", "std", "short_frac", "nucl_frac",
                 "entropy", "skewness", "kurtosis"]
    frag = frag.set_index("sample_id")
    frag_mat = np.zeros((len(sample_cols), 8), dtype=np.float32)
    for i, sid in enumerate(sample_cols):
        if sid in frag.index:
            frag_mat[i] = frag.loc[sid, frag_cols].values
    log(f"frag matrix: {frag_mat.shape}")

    # Labels
    y_cancer = labels["cancer_present"].values.astype(np.int64)
    y_tissue = np.array([TISSUE_IDX.get(t, -1) for t in labels["tissue_of_origin"]], dtype=np.int64)
    y_stage = labels["stage_proxy"].values.astype(np.int64)  # 0-4
    y_tf = labels["tumor_fraction"].values.astype(np.float32)

    # Groups for donor-disjoint folds: use healthy_donor
    groups = labels["healthy_donor"].fillna("unknown").values

    return block_mat, deconv_mat, frag_mat, y_cancer, y_tissue, y_stage, y_tf, groups, sample_cols, ct_cols, frag_cols


# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

class FusionV2(nn.Module):
    def __init__(self, n_ct=40, n_frag=8, n_pca=64, n_hidden=128, n_tissues=5, n_stages=5):
        super().__init__()
        self.b1_ct = nn.Sequential(
            nn.Linear(n_ct, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        self.b2_frag = nn.Sequential(
            nn.Linear(n_frag, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 64),
        )
        self.b3_pca = nn.Sequential(
            nn.Linear(n_pca, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.ReLU(),
        )
        # Cross-attention: methylation (b1) attends to fragmentomics (b2).
        # Q dim: 128 (b1), K/V dim: 64 (b2)
        self.qk_proj = nn.Linear(n_hidden, 64)  # bring Q to same dim as K/V
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.b1_to_64 = nn.Linear(n_hidden, 64)
        self.fuse = nn.Sequential(
            nn.Linear(n_hidden + 64 + n_hidden, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
        )
        self.head_cancer = nn.Linear(128, 1)
        self.head_tissue = nn.Linear(128, n_tissues)
        self.head_stage = nn.Linear(128, n_stages)
        self.head_tf = nn.Linear(128, 1)

    def forward(self, x_ct, x_frag, x_pca, gate=(1.0, 1.0, 1.0)):
        # gate selects which branches are on for ablation
        g1, g2, g3 = gate
        b1 = self.b1_ct(x_ct) * g1                  # [B, 128]
        b2 = self.b2_frag(x_frag) * g2              # [B, 64]
        b3 = self.b3_pca(x_pca) * g3                # [B, 128]

        # Cross-attention: B1 attends to B2. Need [B, 1, 64]
        q = self.b1_to_64(b1).unsqueeze(1)          # [B,1,64]
        kv = b2.unsqueeze(1)                        # [B,1,64]
        attn_out, _ = self.attn(q, kv, kv)          # [B,1,64]
        attn_out = attn_out.squeeze(1)               # [B,64]

        h = torch.cat([b1, attn_out, b3], dim=-1)   # [B, 320]
        h = self.fuse(h)                             # [B, 128]
        return {
            "cancer_logit": self.head_cancer(h).squeeze(-1),
            "tissue_logit": self.head_tissue(h),
            "stage_logit": self.head_stage(h),
            "tf": torch.sigmoid(self.head_tf(h).squeeze(-1)),
        }


# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

def sens_at_spec(y_true, y_score, target_spec=0.95):
    """Sensitivity at fixed specificity threshold."""
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    if neg_mask.sum() == 0 or pos_mask.sum() == 0:
        return float("nan"), float("nan")
    # threshold = (1-target_spec) quantile of negative scores
    neg_scores = y_score[neg_mask]
    thr = np.quantile(neg_scores, target_spec)
    pred_pos = y_score > thr
    sens = pred_pos[pos_mask].mean()
    return float(sens), float(thr)


def train_eval_fold(model, train_loader, val_loader, device, gate, max_epochs=80, patience=10):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    best_val_auroc = -1.0
    best_state = None
    no_improve = 0
    for epoch in range(max_epochs):
        model.train()
        for batch in train_loader:
            x_ct, x_frag, x_pca, yc, yt, ys, ytf = [b.to(device) for b in batch]
            out = model(x_ct, x_frag, x_pca, gate=gate)
            L_cancer = F.binary_cross_entropy_with_logits(out["cancer_logit"], yc.float())
            # tissue & stage only for cancer samples
            pos = yc == 1
            if pos.any():
                L_tissue = F.cross_entropy(out["tissue_logit"][pos], yt[pos])
                L_stage = F.cross_entropy(out["stage_logit"][pos], ys[pos])
                L_tf = F.mse_loss(out["tf"][pos], ytf[pos])
            else:
                L_tissue = L_stage = L_tf = torch.tensor(0.0, device=device)
            loss = 2 * L_cancer + L_tissue + L_stage + 0.5 * L_tf
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        # Val
        model.eval()
        scores = []
        labels = []
        with torch.no_grad():
            for batch in val_loader:
                x_ct, x_frag, x_pca, yc, yt, ys, ytf = [b.to(device) for b in batch]
                out = model(x_ct, x_frag, x_pca, gate=gate)
                scores.append(torch.sigmoid(out["cancer_logit"]).cpu().numpy())
                labels.append(yc.cpu().numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        from sklearn.metrics import roc_auc_score
        if len(set(labels)) > 1:
            auroc = roc_auc_score(labels, scores)
        else:
            auroc = float("nan")
        if auroc > best_val_auroc:
            best_val_auroc = auroc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_auroc


def predict_full(model, loader, device, gate):
    model.eval()
    cancer_scores = []
    tissue_pred = []
    stage_pred = []
    tf_pred = []
    y_c = []; y_t = []; y_s = []; y_tf = []
    with torch.no_grad():
        for batch in loader:
            x_ct, x_frag, x_pca, yc, yt, ys, ytf = [b.to(device) for b in batch]
            out = model(x_ct, x_frag, x_pca, gate=gate)
            cancer_scores.append(torch.sigmoid(out["cancer_logit"]).cpu().numpy())
            tissue_pred.append(out["tissue_logit"].argmax(dim=-1).cpu().numpy())
            stage_pred.append(out["stage_logit"].argmax(dim=-1).cpu().numpy())
            tf_pred.append(out["tf"].cpu().numpy())
            y_c.append(yc.cpu().numpy()); y_t.append(yt.cpu().numpy())
            y_s.append(ys.cpu().numpy()); y_tf.append(ytf.cpu().numpy())
    return {
        "scores": np.concatenate(cancer_scores),
        "tissue_pred": np.concatenate(tissue_pred),
        "stage_pred": np.concatenate(stage_pred),
        "tf_pred": np.concatenate(tf_pred),
        "y_cancer": np.concatenate(y_c),
        "y_tissue": np.concatenate(y_t),
        "y_stage": np.concatenate(y_s),
        "y_tf": np.concatenate(y_tf),
    }


def make_loaders(X_ct, X_frag, X_pca, y_c, y_t, y_s, y_tf, train_idx, val_idx, batch_size=64, balance_train=True):
    """Class-balanced sampler on train; raw on val."""
    def make_train(idxs):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_ct[idxs], dtype=torch.float32),
            torch.tensor(X_frag[idxs], dtype=torch.float32),
            torch.tensor(X_pca[idxs], dtype=torch.float32),
            torch.tensor(y_c[idxs], dtype=torch.long),
            torch.tensor(y_t[idxs], dtype=torch.long),
            torch.tensor(y_s[idxs], dtype=torch.long),
            torch.tensor(y_tf[idxs], dtype=torch.float32),
        )
        if balance_train:
            yc_train = y_c[idxs]
            n_pos = (yc_train == 1).sum()
            n_neg = (yc_train == 0).sum()
            if n_neg > 0 and n_pos > 0:
                weights = np.where(yc_train == 0, 1.0 / n_neg, 1.0 / n_pos)
                sampler = torch.utils.data.WeightedRandomSampler(weights.tolist(), num_samples=int(2 * max(int(n_pos), int(n_neg))), replacement=True)
                return torch.utils.data.DataLoader(ds, batch_size=batch_size, sampler=sampler)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    def make_val(idxs):
        ds = torch.utils.data.TensorDataset(
            torch.tensor(X_ct[idxs], dtype=torch.float32),
            torch.tensor(X_frag[idxs], dtype=torch.float32),
            torch.tensor(X_pca[idxs], dtype=torch.float32),
            torch.tensor(y_c[idxs], dtype=torch.long),
            torch.tensor(y_t[idxs], dtype=torch.long),
            torch.tensor(y_s[idxs], dtype=torch.long),
            torch.tensor(y_tf[idxs], dtype=torch.float32),
        )
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    return make_train(train_idx), make_val(val_idx)


def evaluate_predictions(pred):
    """Compute headline metrics from out-of-fold predictions."""
    res = {}
    from sklearn.metrics import roc_auc_score, accuracy_score
    y_c = pred["y_cancer"]; scores = pred["scores"]
    res["auroc_cancer"] = roc_auc_score(y_c, scores) if len(set(y_c)) > 1 else float("nan")
    # Sens at specs
    for spec in [0.95, 0.99, 0.995]:
        s, t = sens_at_spec(y_c, scores, target_spec=spec)
        res[f"sens_at_{int(spec*100)}_spec"] = s
        res[f"thr_at_{int(spec*100)}_spec"] = t

    # Stratified by stage
    y_s = pred["y_stage"]
    for stage in [1, 2, 3, 4]:
        # Pos = cancer samples at this stage, Neg = healthy (stage=0)
        mask = (y_s == stage) | (y_s == 0)
        if mask.sum() == 0:
            continue
        y_sub = (y_s[mask] == stage).astype(int)
        s_sub = scores[mask]
        for spec in [0.95, 0.99, 0.995]:
            s, t = sens_at_spec(y_sub, s_sub, target_spec=spec)
            res[f"stage{stage}_sens_at_{int(spec*100)}_spec"] = s

    # Tissue top-1 (cancer only)
    pos = y_c == 1
    if pos.any():
        y_t = pred["y_tissue"][pos]
        t_pred = pred["tissue_pred"][pos]
        res["tissue_top1"] = accuracy_score(y_t, t_pred)
    # Stage top-1
    pos_stage = (y_s > 0)
    if pos_stage.any():
        res["stage_top1"] = accuracy_score(y_s[pos_stage], pred["stage_pred"][pos_stage])
    # TF MSE
    if pos.any():
        res["tf_mse"] = float(((pred["tf_pred"][pos] - pred["y_tf"][pos]) ** 2).mean())
    return res


def main():
    log("=== Day2 Phase D: Fusion v2 ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    block_mat, deconv_mat, frag_mat, y_c, y_t, y_s, y_tf, groups, sample_cols, ct_cols, frag_cols = load_data()
    log(f"total samples: {len(sample_cols)}")
    log(f"cancer rate: {y_c.mean():.3f}")

    # Standardize fragmentomics
    sc_frag = StandardScaler().fit(frag_mat)
    frag_z = sc_frag.transform(frag_mat).astype(np.float32)

    # PCA on raw blocks
    log("computing PCA-64 on raw atlas-blocks...")
    sc_blocks = StandardScaler().fit(block_mat)
    block_z = sc_blocks.transform(block_mat).astype(np.float32)
    pca = PCA(n_components=64, random_state=SEED)
    block_pca = pca.fit_transform(block_z).astype(np.float32)
    log(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # Donor-disjoint 5-fold CV
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(block_pca, y_c, groups=groups))
    log(f"folds: {len(folds)}")
    for i, (tr, va) in enumerate(folds):
        log(f"  fold {i}: train_n={len(tr)} val_n={len(va)} val_cancer_rate={y_c[va].mean():.3f}")

    # Ablations
    ABLATIONS = {
        "A_full":       (1.0, 1.0, 1.0),
        "B_no_frag":    (1.0, 0.0, 1.0),
        "C_screening":  (1.0, 1.0, 0.0),  # deconv + frag (no raw PCA) -- thesis claim
        "D_no_deconv":  (0.0, 1.0, 1.0),
        "E_deconv_only":(1.0, 0.0, 0.0),  # Day-1 baseline
    }

    all_results = {}
    pooled_preds_per_ablation = {}

    for abl_name, gate in ABLATIONS.items():
        log(f"=== ablation {abl_name} gate={gate} ===")
        pooled = {"scores": [], "tissue_pred": [], "stage_pred": [], "tf_pred": [],
                  "y_cancer": [], "y_tissue": [], "y_stage": [], "y_tf": [],
                  "sample_idx": []}
        for fold, (tr, va) in enumerate(folds):
            train_loader, val_loader = make_loaders(deconv_mat, frag_z, block_pca, y_c, y_t, y_s, y_tf, tr, va)
            model = FusionV2(n_ct=deconv_mat.shape[1], n_frag=frag_z.shape[1], n_pca=block_pca.shape[1]).to(device)
            model, fold_auroc = train_eval_fold(model, train_loader, val_loader, device, gate)
            preds = predict_full(model, val_loader, device, gate)
            for k in pooled:
                if k == "sample_idx":
                    pooled[k].append(va)
                else:
                    pooled[k].append(preds[k])
            log(f"  fold {fold}: val_auroc={fold_auroc:.3f}")
        pooled_final = {k: np.concatenate(v) for k, v in pooled.items()}
        res = evaluate_predictions(pooled_final)
        log(f"{abl_name} metrics: {json.dumps(res, default=str)}")
        all_results[abl_name] = res
        pooled_preds_per_ablation[abl_name] = pooled_final

    # ------------------------------------------------------------------
    # Cross-cohort eval: train on GSE186458 mixtures, test on GSE122126
    # ------------------------------------------------------------------
    log("=== cross-cohort eval: GSE122126 ===")
    try:
        cross = pd.read_parquet(PROC / "GSE122126_harmonized.parquet")
        sample_cols_c = [c for c in cross.columns if c != "block_id"]
        block_c = cross[sample_cols_c].values.T.astype(np.float32)
        block_c = np.nan_to_num(block_c, nan=0.5)
        log(f"GSE122126 samples: {len(sample_cols_c)}")
        # Load GSE122126 sample labels
        gse_samples = pd.read_csv(PROC / "GSE122126_samples.tsv", sep="\t")
        log(f"GSE122126_samples cols: {gse_samples.columns.tolist()}")
        # We need cancer/healthy classifications; use existing harmonized list
        # Map: sample_id -> binary label
        # Day 1 used 'group' column or similar — inspect.
        # If no clear label column, infer from filename pattern
        # We'll compute predictions then score on healthy subset only.

        block_c_z = sc_blocks.transform(block_c).astype(np.float32)
        block_c_pca = pca.transform(block_c_z).astype(np.float32)

        # For deconvolution: use the Day-1 NNLS cross-cohort output, which
        # was computed with proper handling of the 10.3% atlas coverage
        # (only observed blocks contribute). The learned model imputes NaN
        # as 0.5 and produces near-constant outputs across samples in this
        # low-coverage regime.
        nnls_cross = pd.read_parquet(PROC / "GSE122126_deconv.parquet")
        # Day-1 NNLS columns have 'ct_fraction_' prefix
        nnls_cols = [c for c in nnls_cross.columns if c.startswith("ct_fraction_")]
        ct_map = {c.replace("ct_fraction_", ""): c for c in nnls_cols}
        atlas_cts_local = ckpt_atlas if False else None  # placeholder
        # Need atlas cell-type order matching the training-time deconv_mat
        # In training we used learned_deconv predictions which had 40 cell types.
        # We rebuild deconv_c by reindexing nnls_cross columns
        deconv_train_ct_cols = ct_cols  # the deconv ct cols used in training
        deconv_c = np.zeros((len(sample_cols_c), len(deconv_train_ct_cols)), dtype=np.float32)
        nnls_cross_idx = nnls_cross.set_index("sample_id")
        for i, sid in enumerate(sample_cols_c):
            if sid in nnls_cross_idx.index:
                row = nnls_cross_idx.loc[sid]
                for j, ct in enumerate(deconv_train_ct_cols):
                    col_name = f"ct_fraction_{ct}"
                    if col_name in nnls_cross_idx.columns:
                        deconv_c[i, j] = float(row[col_name])
                # Normalize to sum 1
                s = deconv_c[i].sum()
                if s > 0:
                    deconv_c[i] /= s
        log(f"NNLS-deconv for GSE122126 cross: {deconv_c.shape}, granu mean={deconv_c[:, ct_cols.index('Blood-Granul') if 'Blood-Granul' in ct_cols else 0].mean():.3f}")

        # No fragmentomics for real GSE122126 (no BAMs). Use zero-vector
        # (the standardizer already removed the mean, so zero = healthy mean
        # in standardized space). We acknowledge this limits the C/B ablations.
        frag_c = np.zeros((block_c.shape[0], frag_z.shape[1]), dtype=np.float32)

        # Try to classify based on sample labels; use cancer/healthy column if present
        gse_samples = gse_samples.set_index(gse_samples.columns[0])
        # Try to find the labels
        labels_c = None
        for col_try in ['label', 'group', 'sample_type', 'condition', 'disease', 'type', 'state']:
            if col_try in gse_samples.columns:
                labels_c = gse_samples[col_try].reindex(sample_cols_c).values
                break

        # Train both ablation-A (full) and ablation-C (screening) models on the
        # full pool and evaluate cross-cohort. The PCA-64 isn't transferable
        # because GSE122126 has only 10.3% atlas coverage so we'll evaluate
        # the screening (deconv+frag) gate which the thesis predicts.
        from sklearn.metrics import roc_auc_score
        labels_c_bin = None
        if labels_c is not None:
            # GSE122126 labels: HEALTHY (0), CRC/LUNG/BREAST/CUP (1), others = exclude
            cancer_labels = {"CRC", "LUNG", "BREAST", "CUP", "HCC", "GBM"}
            healthy_labels = {"HEALTHY"}
            valid = np.array([str(l).upper() in cancer_labels | healthy_labels for l in labels_c])
            log(f"GSE122126 labels: {pd.Series(labels_c).value_counts().to_dict()}")
            log(f"valid (cancer/healthy only) samples: {valid.sum()} of {len(labels_c)}")
            labels_c_bin = np.full(len(labels_c), -1, dtype=int)
            for i, l in enumerate(labels_c):
                u = str(l).upper()
                if u in healthy_labels:
                    labels_c_bin[i] = 0
                elif u in cancer_labels:
                    labels_c_bin[i] = 1
        cross_results = {}
        for cross_abl, cross_gate in [("A_full", (1.0, 1.0, 1.0)),
                                       ("C_screening", (1.0, 1.0, 0.0)),
                                       ("E_deconv_only", (1.0, 0.0, 0.0))]:
            log(f"training full-pool {cross_abl} for cross-cohort...")
            train_loader = make_loaders(deconv_mat, frag_z, block_pca, y_c, y_t, y_s, y_tf,
                                       np.arange(len(y_c)), np.arange(len(y_c)), batch_size=64)[0]
            model = FusionV2(n_ct=deconv_mat.shape[1], n_frag=frag_z.shape[1], n_pca=block_pca.shape[1]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for epoch in range(40):
                model.train()
                for batch in train_loader:
                    x_ct, x_frag, x_pca, yc, yt, ys, ytf = [b.to(device) for b in batch]
                    out = model(x_ct, x_frag, x_pca, gate=cross_gate)
                    L_cancer = F.binary_cross_entropy_with_logits(out["cancer_logit"], yc.float())
                    pos = yc == 1
                    if pos.any():
                        L_tissue = F.cross_entropy(out["tissue_logit"][pos], yt[pos])
                        L_stage = F.cross_entropy(out["stage_logit"][pos], ys[pos])
                        L_tf = F.mse_loss(out["tf"][pos], ytf[pos])
                    else:
                        L_tissue = L_stage = L_tf = torch.tensor(0.0, device=device)
                    loss = 2 * L_cancer + L_tissue + L_stage + 0.5 * L_tf
                    opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                scores_c = torch.sigmoid(model(
                    torch.tensor(deconv_c, dtype=torch.float32).to(device),
                    torch.tensor(frag_c, dtype=torch.float32).to(device),
                    torch.tensor(block_c_pca, dtype=torch.float32).to(device),
                    gate=cross_gate,
                )["cancer_logit"]).cpu().numpy()
            cross_auroc = float("nan")
            if labels_c_bin is not None:
                # Use only valid (cancer + healthy) samples for AUROC
                valid_mask = labels_c_bin >= 0
                if valid_mask.sum() > 1 and len(set(labels_c_bin[valid_mask])) > 1:
                    cross_auroc = float(roc_auc_score(labels_c_bin[valid_mask], scores_c[valid_mask]))
            log(f"  cross-cohort {cross_abl}: AUROC={cross_auroc:.3f} score_mean={scores_c.mean():.3f}")
            cross_results[cross_abl] = {
                "auroc": cross_auroc,
                "scores_mean": float(scores_c.mean()),
                "scores_std": float(scores_c.std()),
                "scores_min": float(scores_c.min()),
                "scores_max": float(scores_c.max()),
            }
        all_results["cross_cohort_GSE122126"] = {
            "n_samples": int(len(sample_cols_c)),
            "n_cancer": int(labels_c_bin.sum()) if labels_c_bin is not None else None,
            "n_healthy": int((labels_c_bin == 0).sum()) if labels_c_bin is not None else None,
            "ablations": cross_results,
        }
    except Exception as e:
        import traceback
        log(f"cross-cohort eval failed: {e}")
        log(traceback.format_exc())
        all_results["cross_cohort_GSE122126"] = {"error": str(e)}

    out_path = RES / "fusion_v2_eval.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"wrote {out_path}")

    # Print summary table
    log("\n=== ABLATION SUMMARY ===")
    rows = []
    for abl, res in all_results.items():
        if abl == "cross_cohort_GSE122126":
            continue
        rows.append({
            "ablation": abl,
            "auroc": res.get("auroc_cancer"),
            "sens@95spec": res.get("sens_at_95_spec"),
            "sens@99spec": res.get("sens_at_99_spec"),
            "sens@995spec": res.get("sens_at_995_spec"),
            "s1@99": res.get("stage1_sens_at_99_spec"),
            "s2@99": res.get("stage2_sens_at_99_spec"),
            "s3@99": res.get("stage3_sens_at_99_spec"),
            "s4@99": res.get("stage4_sens_at_99_spec"),
            "tissue_top1": res.get("tissue_top1"),
        })
    df = pd.DataFrame(rows)
    log(df.to_string())


if __name__ == "__main__":
    sys.exit(main() or 0)

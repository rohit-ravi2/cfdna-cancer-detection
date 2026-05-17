#!/usr/bin/env python3
"""
Day-3 Phase B — NaN-Aware Sparse-Coverage Learned Deconvolution.

Day-2 architecture used 0.5 imputation for NaN, which makes missing positions
indistinguishable from genuine 0.5 methylation. Fix: pass a mask as second
input so the model knows which positions are real measurements.

Architecture:
  Input:
    betas: [B, 9520]  (NaN replaced with 0)
    mask : [B, 9520]  (1 if observed, 0 if NaN)
  Encoder:
    cat([betas, mask], dim=-1) -> [B, 19040]
    Linear(19040, 2048) -> BN -> ReLU -> Dropout
    Linear(2048, 512)   -> BN -> ReLU -> Dropout
    Linear(512, 128)    -> BN -> ReLU
  Heads:
    deconv: Linear(128, 40), softmax
    recon : Linear(40, 128) -> ReLU -> Linear(128, 9520), sigmoid

Loss = 2*L_frac (MSE on labeled samples)
     + 1*L_recon (MSE on OBSERVED positions only, masked recon)
     + 0.1*L_sparse (L1 on fractions)
     + 1*L_dom (cross-entropy on dominant cell type)

Training data:
  - distsampled mixtures (4600 samples, full coverage by construction)
  - Day-2 fixed-donor mixtures (1058 samples)
  - 23 real healthy cfDNA (~99.7% coverage)
  - 23 WBC samples
  - 89 sorted-cell refs (full coverage)
  - 108 TCGA tumor tissue samples (~8% coverage; ideal NaN test case)

Augmentation: randomly mask 10-80% of blocks per sample during training.

Output:
  /mnt/ssd/research/cfdna/models/learned_deconv_nanaware.pt
  /mnt/ssd/research/cfdna/models/phase_b_summary.json
  /mnt/ssd/research/cfdna/processed/distsampled_deconv_nanaware.parquet
"""
import json, time, pickle, sys, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
LOG = Path("/mnt/ssd/research/cfdna/logs/day3_phase_b.log")

N_BLOCKS = 9520
SEED = 42

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def load_atlas_cts():
    atlas_tsv = "/mnt/hdd/research/cfdna/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv"
    df = pd.read_csv(atlas_tsv, sep="\t", nrows=1)
    return [c for c in df.columns
            if c not in ("chr","start","end","startCpG","endCpG","target","name","direction")]


def load_block_matrix(parquet_path):
    df = pd.read_parquet(parquet_path)
    block_ids = df["block_id"].copy()
    cols = [c for c in df.columns if c != "block_id"]
    X = df[cols].to_numpy(dtype=np.float32).T  # [N, blocks]
    return block_ids, X, cols


class NaNAwareDeconv(nn.Module):
    def __init__(self, n_blocks=9520, n_ct=40, hidden=2048):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_blocks * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.deconv = nn.Linear(128, n_ct)
        self.recon = nn.Sequential(
            nn.Linear(n_ct, 128), nn.ReLU(),
            nn.Linear(128, n_blocks), nn.Sigmoid(),
        )

    def forward(self, betas_filled, mask):
        x = torch.cat([betas_filled, mask], dim=-1)
        h = self.enc(x)
        logits = self.deconv(h)
        frac = F.softmax(logits, dim=-1)
        recon = self.recon(frac)
        return frac, recon, logits


def build_training_data(atlas_cts):
    """Return list of (X_np[N,9520], mask_np[N,9520], Y_np[N,40], has_label[N], source_str, sample_ids)."""
    all_X, all_M, all_Y, all_HL, all_src, all_sids = [], [], [], [], [], []

    # 1) Distribution-sampled mixtures
    log("loading distsampled mixtures...")
    betas = pd.read_parquet(PROC / "distsampled_mixtures_betas.parquet")
    labels = pd.read_parquet(PROC / "distsampled_mixtures_labels.parquet")
    cols = [c for c in betas.columns if c != "block_id"]
    X = betas[cols].to_numpy(dtype=np.float32).T  # [N, blocks]
    mask = (~np.isnan(X)).astype(np.float32)
    X = np.where(np.isnan(X), 0.0, X)
    # Build Y: tissue_of_origin and tumor_fraction
    # baseline composition: use NNLS healthy mean
    nnls = pd.read_parquet(PROC / "GSE186458_cfdna_deconv_nnls.parquet")
    nnls_cols = [c for c in nnls.columns if c not in ("sample_id","nnls_residual","n_blocks_used")]
    baseline_dict = nnls[nnls_cols].mean().to_dict()
    baseline_vec = np.array([baseline_dict.get(ct, 0.0) for ct in atlas_cts], dtype=np.float32)
    baseline_vec = baseline_vec / max(baseline_vec.sum(), 1e-6)
    tissue_to_ct = {"colon":"Colon-Ep","liver":"Liver-Hep","lung":"Lung-Ep-Alveo","breast":"Breast-Luminal-Ep","pancreas":"Pancreas-Duct"}
    Y = np.zeros((len(cols), len(atlas_cts)), dtype=np.float32)
    sample_to_label = labels.set_index("sample_id")
    has_label = np.zeros(len(cols), dtype=bool)
    for i, sid in enumerate(cols):
        if sid not in sample_to_label.index:
            continue
        row = sample_to_label.loc[sid]
        tf = float(row["tumor_fraction"]); tissue = row["tissue_of_origin"]
        target = baseline_vec * (1.0 - tf)
        if tissue in tissue_to_ct and tf > 0:
            j = atlas_cts.index(tissue_to_ct[tissue])
            target[j] += tf
        s = target.sum()
        if s > 0:
            target = target / s
        Y[i] = target
        has_label[i] = True
    all_X.append(X); all_M.append(mask); all_Y.append(Y); all_HL.append(has_label)
    all_src.append("distsampled"); all_sids.append(cols)
    log(f"  distsampled: {X.shape}, labeled={has_label.sum()}")

    # 2) Day-2 fixed-donor mixtures
    log("loading fixed-donor mixtures (v2)...")
    betas = pd.read_parquet(PROC / "GSE186458_mixtures_betas_v2.parquet")
    labels = pd.read_parquet(PROC / "GSE186458_mixtures_labels_v2.parquet")
    cols = [c for c in betas.columns if c != "block_id"]
    X = betas[cols].to_numpy(dtype=np.float32).T
    mask = (~np.isnan(X)).astype(np.float32)
    X = np.where(np.isnan(X), 0.0, X)
    Y = np.zeros((len(cols), len(atlas_cts)), dtype=np.float32)
    sample_to_label = labels.set_index("sample_id")
    has_label = np.zeros(len(cols), dtype=bool)
    for i, sid in enumerate(cols):
        if sid not in sample_to_label.index: continue
        row = sample_to_label.loc[sid]
        tf = float(row["tumor_fraction"]); tissue = row["tissue_of_origin"]
        target = baseline_vec * (1.0 - tf)
        if tissue in tissue_to_ct and tf > 0:
            j = atlas_cts.index(tissue_to_ct[tissue])
            target[j] += tf
        s = target.sum()
        if s > 0: target = target / s
        Y[i] = target
        has_label[i] = True
    all_X.append(X); all_M.append(mask); all_Y.append(Y); all_HL.append(has_label)
    all_src.append("fixed"); all_sids.append(cols)
    log(f"  fixed: {X.shape}, labeled={has_label.sum()}")

    # 3) Real healthy cfDNA
    log("loading real healthy cfDNA...")
    betas = pd.read_parquet(PROC / "GSE186458_cfdna_harmonized.parquet")
    cols = [c for c in betas.columns if c != "block_id"]
    X = betas[cols].to_numpy(dtype=np.float32).T
    mask = (~np.isnan(X)).astype(np.float32)
    X = np.where(np.isnan(X), 0.0, X)
    nnls = pd.read_parquet(PROC / "GSE186458_cfdna_deconv_nnls.parquet")
    nnls_idx = nnls.set_index("sample_id")
    Y = np.zeros((len(cols), len(atlas_cts)), dtype=np.float32)
    has_label = np.zeros(len(cols), dtype=bool)
    for i, sid in enumerate(cols):
        if sid in nnls_idx.index:
            row = nnls_idx.loc[sid]
            for j, ct in enumerate(atlas_cts):
                if ct in nnls.columns:
                    Y[i, j] = float(row.get(ct, 0.0))
            s = Y[i].sum()
            if s > 0:
                Y[i] /= s
                has_label[i] = True
    all_X.append(X); all_M.append(mask); all_Y.append(Y); all_HL.append(has_label)
    all_src.append("cfdna_real"); all_sids.append(cols)
    log(f"  cfdna: {X.shape}, labeled={has_label.sum()}")

    # 4) TCGA tumor tissue samples — DO NOT label as supervised (collapse risk)
    # We include them only as reconstruction-only signal: NaN-aware encoder learns
    # to handle the high-NaN coverage pattern matching GSE122126.
    log("loading TCGA tumor matrices (recon-only)...")
    for code in ["COAD", "LIHC", "LUAD", "BRCA"]:
        p = PROC / f"tcga_{code}_harmonized.parquet"
        if not p.exists(): continue
        betas = pd.read_parquet(p)
        cols = [c for c in betas.columns if c != "block_id"]
        X = betas[cols].to_numpy(dtype=np.float32).T
        mask = (~np.isnan(X)).astype(np.float32)
        X = np.where(np.isnan(X), 0.0, X)
        # Y is unused since has_label=False; recon loss still applies
        Y = np.zeros((len(cols), len(atlas_cts)), dtype=np.float32)
        has_label = np.zeros(len(cols), dtype=bool)
        all_X.append(X); all_M.append(mask); all_Y.append(Y); all_HL.append(has_label)
        all_src.append(f"tcga_{code}"); all_sids.append(cols)
        log(f"  tcga {code}: {X.shape}, mean_obs={mask.mean():.3f} (RECON-ONLY)")

    return all_X, all_M, all_Y, all_HL, all_src, all_sids


def augment_mask(mask_np, drop_min=0.10, drop_max=0.80, p_aug=0.5):
    """Randomly drop a fraction of observed blocks per sample.
    Only operates on currently-observed positions (don't un-drop NaN ones)."""
    N, B = mask_np.shape
    out = mask_np.copy()
    for i in range(N):
        if np.random.rand() < p_aug:
            drop = np.random.uniform(drop_min, drop_max)
            obs_idx = np.where(out[i] > 0.5)[0]
            n_drop = int(drop * len(obs_idx))
            if n_drop > 0:
                drop_idx = np.random.choice(obs_idx, n_drop, replace=False)
                out[i, drop_idx] = 0
    return out


def train(model, train_loader, val_loader, device, n_epochs=40, lr=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    best_val = -np.inf
    best_state = None
    patience = 8; stale = 0
    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            betas, mask, Y, hl = [b.to(device) for b in batch]
            opt.zero_grad()
            frac, recon, logits = model(betas, mask)
            # L_frac on labeled (MSE + KL-div for distribution matching)
            if hl.sum() > 0:
                Yh = Y[hl.bool()]
                Fh = frac[hl.bool()]
                Lf = F.mse_loss(Fh, Yh)
                # KL divergence: model -> target (with tiny smoothing on target)
                Yh_smooth = Yh + 1e-6
                Yh_smooth = Yh_smooth / Yh_smooth.sum(dim=-1, keepdim=True)
                Fh_smooth = Fh + 1e-6
                Fh_smooth = Fh_smooth / Fh_smooth.sum(dim=-1, keepdim=True)
                Lk = F.kl_div(torch.log(Fh_smooth), Yh_smooth, reduction='batchmean')
            else:
                Lf = torch.tensor(0.0, device=device); Lk = torch.tensor(0.0, device=device)
            # L_recon masked
            diff = (recon - betas) ** 2
            Lr = (diff * mask).sum() / (mask.sum() + 1e-6)
            # L_entropy: -H(frac) push fractions to be peaky (anti-uniform)
            # But also avoid collapse: ent regularizer pushes toward target entropy
            ent = -(frac * torch.log(frac + 1e-8)).sum(dim=-1).mean()
            # Light L1 sparsity
            Ls = (frac ** 2).sum(dim=-1).mean()  # ~uniform when small
            loss = 10*Lf + 5*Lk + 0.5*Lr + 0.01*Ls
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        sched.step()

        # Val: pearson on labeled mixtures
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                betas, mask, Y, hl = [b.to(device) for b in batch]
                frac, _, _ = model(betas, mask)
                if hl.sum() > 0:
                    preds_all.append(frac[hl.bool()].cpu().numpy())
                    targets_all.append(Y[hl.bool()].cpu().numpy())
        if preds_all:
            P = np.concatenate(preds_all); T = np.concatenate(targets_all)
            r = np.corrcoef(P.flatten(), T.flatten())[0, 1]
            # Diversity diagnostics: how variable is the prediction across samples?
            col_std = P.std(axis=0).mean()
            max_mean_col = P.mean(axis=0).max()
        else:
            r = -1; col_std = 0; max_mean_col = 0
        log(f"  epoch {epoch+1}/{n_epochs} train_loss={np.mean(train_losses):.4f} val_pearson={r:.4f} col_std={col_std:.4f} max_col_mean={max_mean_col:.3f}")
        if r > best_val:
            best_val = r
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                log(f"  early stop at {epoch+1}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


def cross_coverage_eval(model, atlas_cts, device):
    """Evaluate model on synthetic mixtures at 3 coverage levels: 99%, 20%, 10%.
    We use the distsampled mixtures (full coverage) and randomly mask down."""
    betas = pd.read_parquet(PROC / "distsampled_mixtures_betas.parquet")
    labels = pd.read_parquet(PROC / "distsampled_mixtures_labels.parquet")
    cols = [c for c in betas.columns if c != "block_id"]
    X = betas[cols].to_numpy(dtype=np.float32).T
    # Filter to cancer-only samples with known tissue and tf > 0
    sample_to_label = labels.set_index("sample_id")
    keep_idx = []
    targets_tf = []
    for i, sid in enumerate(cols):
        if sid in sample_to_label.index:
            row = sample_to_label.loc[sid]
            if row["cancer_present"] == 1 and row["tissue_of_origin"] in ["colon","liver","lung","breast"]:
                keep_idx.append(i)
                targets_tf.append(float(row["tumor_fraction"]))
    keep_idx = np.array(keep_idx); targets_tf = np.array(targets_tf)
    X = X[keep_idx]
    log(f"  cross-coverage eval set: {X.shape}")
    tissue_to_ct = {"colon":"Colon-Ep","liver":"Liver-Hep","lung":"Lung-Ep-Alveo","breast":"Breast-Luminal-Ep"}
    tissues = [str(sample_to_label.loc[cols[i]]["tissue_of_origin"]) for i in keep_idx]

    results = {}
    for keep_frac in [1.0, 0.20, 0.10]:
        rng = np.random.RandomState(123)
        mask = np.zeros_like(X)
        for i in range(X.shape[0]):
            if keep_frac < 1.0:
                keep_count = int(keep_frac * N_BLOCKS)
                idx = rng.choice(N_BLOCKS, keep_count, replace=False)
                mask[i, idx] = 1.0
            else:
                mask[i] = 1.0
        X_masked = X * mask  # zero where masked
        # Predict
        model.eval()
        with torch.no_grad():
            batch = 512
            preds = []
            for s in range(0, X.shape[0], batch):
                b = torch.from_numpy(X_masked[s:s+batch]).to(device)
                m = torch.from_numpy(mask[s:s+batch]).to(device)
                frac, _, _ = model(b, m)
                preds.append(frac.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
        # Per-tissue pearson: target tissue fraction vs target tumor fraction
        per_tissue = {}
        for tissue, ct in tissue_to_ct.items():
            mask_t = np.array([t == tissue for t in tissues])
            if mask_t.sum() < 3: continue
            j = atlas_cts.index(ct)
            tf_pred = preds[mask_t, j]
            tf_targ = targets_tf[mask_t]
            r = float(np.corrcoef(tf_pred, tf_targ)[0, 1]) if np.std(tf_pred) > 0 else float("nan")
            per_tissue[tissue] = r
        mean_r = float(np.nanmean(list(per_tissue.values())))
        results[f"keep_{int(keep_frac*100)}"] = {"per_tissue": per_tissue, "mean_pearson": mean_r}
        log(f"  coverage {int(keep_frac*100)}%: mean_pearson={mean_r:.3f} per_tissue={per_tissue}")
    return results


def main():
    t0 = time.time()
    log("=== PHASE B: NaN-Aware Learned Deconvolution ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}")

    atlas_cts = load_atlas_cts()
    log(f"atlas: {len(atlas_cts)} cell types")

    # Build data
    Xs, Ms, Ys, HLs, srcs, sids = build_training_data(atlas_cts)

    # Concatenate
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    M = np.concatenate(Ms, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    HL = np.concatenate(HLs, axis=0)
    src_arr = np.concatenate([np.array([s]*len(x)) for s, x in zip(srcs, Xs)])
    sid_arr = np.concatenate([np.array(s) for s in sids])
    log(f"total training: {X.shape}, labeled={HL.sum()}")

    # 80/20 train/val split, stratified by source
    rng = np.random.RandomState(SEED)
    train_idx, val_idx = [], []
    for src in set(srcs):
        idx = np.where(src_arr == src)[0]
        rng.shuffle(idx)
        cut = int(0.8 * len(idx))
        train_idx.extend(idx[:cut]); val_idx.extend(idx[cut:])
    train_idx = np.array(train_idx); val_idx = np.array(val_idx)
    log(f"train n={len(train_idx)} val n={len(val_idx)}")

    # Augmentation: apply random mask drop to training data
    class AugDataset(torch.utils.data.Dataset):
        def __init__(self, X, M, Y, HL, augment=True):
            self.X = X; self.M = M; self.Y = Y; self.HL = HL; self.augment = augment
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            x = self.X[i]; m = self.M[i]
            if self.augment and np.random.rand() < 0.5:
                drop = np.random.uniform(0.10, 0.80)
                obs = np.where(m > 0.5)[0]
                n_drop = int(drop * len(obs))
                if n_drop > 0:
                    didx = np.random.choice(obs, n_drop, replace=False)
                    m = m.copy(); m[didx] = 0
                    x = x * m  # zero out dropped
            return torch.from_numpy(x), torch.from_numpy(m), torch.from_numpy(self.Y[i]), torch.tensor(self.HL[i])

    train_ds = AugDataset(X[train_idx], M[train_idx], Y[train_idx], HL[train_idx], augment=True)
    val_ds = AugDataset(X[val_idx], M[val_idx], Y[val_idx], HL[val_idx], augment=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = NaNAwareDeconv(n_blocks=N_BLOCKS, n_ct=len(atlas_cts)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model params: {n_params:,}")

    log("training...")
    model, best_val = train(model, train_loader, val_loader, device, n_epochs=40)
    log(f"best val pearson: {best_val:.4f}")

    # Save
    torch.save({
        "state_dict": model.state_dict(),
        "atlas_cts": atlas_cts,
        "n_blocks": N_BLOCKS,
        "best_val_pearson": float(best_val),
    }, MODELS / "learned_deconv_nanaware.pt")

    # Cross-coverage eval
    log("cross-coverage eval...")
    cov_results = cross_coverage_eval(model, atlas_cts, device)

    # Compute deconv on real healthy cfDNA + GSE122126 cfDNA + distsampled
    log("deconvolving real-cohort data for downstream Phase C...")
    def deconv_parquet(name, path):
        if not Path(path).exists():
            return None
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if c != "block_id"]
        X = df[cols].to_numpy(dtype=np.float32).T
        mask = (~np.isnan(X)).astype(np.float32)
        Xf = np.where(np.isnan(X), 0.0, X)
        model.eval()
        with torch.no_grad():
            batch = 256
            preds = []
            for s in range(0, Xf.shape[0], batch):
                b = torch.from_numpy(Xf[s:s+batch]).to(device)
                m = torch.from_numpy(mask[s:s+batch]).to(device)
                frac, _, _ = model(b, m)
                preds.append(frac.cpu().numpy())
            preds = np.concatenate(preds, axis=0)
        out_df = pd.DataFrame(preds, columns=atlas_cts)
        out_df.insert(0, "sample_id", cols)
        out_df.to_parquet(PROC / f"day3_{name}_deconv_nanaware.parquet", index=False)
        log(f"  {name}: {preds.shape} mean granu={preds[:, atlas_cts.index('Blood-Granul')].mean():.3f}")
        return preds

    deconv_parquet("distsampled", PROC / "distsampled_mixtures_betas.parquet")
    deconv_parquet("fixed", PROC / "GSE186458_mixtures_betas_v2.parquet")
    deconv_parquet("cfdna_real", PROC / "GSE186458_cfdna_harmonized.parquet")
    deconv_parquet("gse122126", PROC / "GSE122126_harmonized.parquet")

    summary = {
        "model_path": str(MODELS / "learned_deconv_nanaware.pt"),
        "best_val_pearson": float(best_val),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_params": int(n_params),
        "cross_coverage": cov_results,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(MODELS / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"=== PHASE B DONE in {summary['elapsed_min']} min ===")
    log(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

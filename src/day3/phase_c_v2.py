#!/usr/bin/env python3
"""
Day-3 Phase C v2 — Multi-task classifier with COVERAGE AUGMENTATION.

Day-3 v1 (vanilla classifier on NaN-aware deconv features) got AUROC=0.5
on GSE122126 because the classifier never saw sparse-coverage features
during training. v2 adds coverage-augmented features: for each training
mixture, also generate copies at 20% and 10% coverage and run the
deconv model on them. The classifier sees the full coverage distribution.

This forces the classifier to learn "Colon-Ep at low coverage may be
elevated but still consistent with healthy".
"""
import json, time, sys, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
RESULTS = Path("/mnt/ssd/research/cfdna/results")
LOG = Path("/mnt/ssd/research/cfdna/logs/day3_phase_c.log")
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

TISSUE_LIST = ["colon", "liver", "lung", "breast", "pancreas"]
N_BLOCKS = 9520


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


def load_atlas_cts():
    df = pd.read_csv("/mnt/hdd/research/cfdna/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv", sep="\t", nrows=1)
    return [c for c in df.columns
            if c not in ("chr","start","end","startCpG","endCpG","target","name","direction")]


class NaNAwareDeconv(nn.Module):
    def __init__(self, n_blocks=9520, n_ct=40, hidden=2048):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_blocks * 2, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.deconv = nn.Linear(128, n_ct)
        self.recon = nn.Sequential(nn.Linear(n_ct, 128), nn.ReLU(), nn.Linear(128, n_blocks), nn.Sigmoid())
    def forward(self, b, m):
        h = self.enc(torch.cat([b, m], dim=-1))
        l = self.deconv(h); f = F.softmax(l, dim=-1)
        return f, self.recon(f), l


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
        return (self.head_cancer(h).squeeze(-1),
                self.head_tissue(h),
                self.head_tf(h).squeeze(-1),
                self.head_stage(h))


def deconv_with_mask(model, X, mask, device, bs=256):
    """Run deconv on X with custom mask."""
    model.eval()
    preds = []
    Xf = X.copy()
    Xf[mask < 0.5] = 0.0
    with torch.no_grad():
        for s in range(0, X.shape[0], bs):
            b = torch.from_numpy(Xf[s:s+bs].astype(np.float32)).to(device)
            m = torch.from_numpy(mask[s:s+bs].astype(np.float32)).to(device)
            f, _, _ = model(b, m)
            preds.append(f.cpu().numpy())
    return np.concatenate(preds, axis=0)


def build_aug_dataset(deconv_model, atlas_cts, device,
                       coverage_levels=(1.0, 0.30, 0.15, 0.08)):
    """For training data, generate deconv features at multiple coverage levels.
    Returns features X [N_aug, 40], labels and source tags."""
    log(f"Generating coverage-augmented features at {coverage_levels}...")
    rng = np.random.RandomState(SEED)

    # Load betas + labels for each source
    sources = []
    distsamp_b = pd.read_parquet(PROC / "distsampled_mixtures_betas.parquet")
    distsamp_l = pd.read_parquet(PROC / "distsampled_mixtures_labels.parquet")
    sources.append(("distsampled", distsamp_b, distsamp_l))

    fixed_b = pd.read_parquet(PROC / "GSE186458_mixtures_betas_v2.parquet")
    fixed_l = pd.read_parquet(PROC / "GSE186458_mixtures_labels_v2.parquet")
    sources.append(("fixed", fixed_b, fixed_l))

    # Real healthy cfDNA - keep at full coverage only (it's already real cfDNA)
    cfdna_b = pd.read_parquet(PROC / "GSE186458_cfdna_harmonized.parquet")
    # Build synthetic labels
    cfdna_l = pd.DataFrame({
        "sample_id": [c for c in cfdna_b.columns if c != "block_id"],
        "cancer_present": 0, "tumor_fraction": 0.0,
        "tissue_of_origin": "none", "stage_proxy": 0,
    })
    sources.append(("cfdna_real", cfdna_b, cfdna_l))

    all_X, all_meta = [], []
    for name, b_df, l_df in sources:
        cols = [c for c in b_df.columns if c != "block_id"]
        X = b_df[cols].to_numpy(dtype=np.float32).T  # [N, 9520]
        base_mask = (~np.isnan(X)).astype(np.float32)
        X = np.where(np.isnan(X), 0.0, X)
        sample_to_label = l_df.set_index("sample_id")
        log(f"  {name}: X={X.shape} base_cov_mean={base_mask.mean():.3f}")

        for cov in coverage_levels:
            if cov < 1.0 and name == "cfdna_real":
                # Skip cfdna_real at low coverage (real samples are already real)
                continue
            if cov >= 1.0:
                m = base_mask.copy()
            else:
                # Subsample coverage uniformly
                n_keep = int(cov * N_BLOCKS)
                m = np.zeros_like(base_mask)
                for i in range(X.shape[0]):
                    idx = rng.choice(N_BLOCKS, n_keep, replace=False)
                    m[i, idx] = 1.0
                m = m * base_mask  # AND with original mask
            preds = deconv_with_mask(deconv_model, X, m, device)
            for i, sid in enumerate(cols):
                if sid in sample_to_label.index:
                    r = sample_to_label.loc[sid]
                    all_X.append(preds[i])
                    all_meta.append({
                        "sample_id": f"{sid}_cov{int(cov*100):03d}",
                        "cancer_present": int(r["cancer_present"]),
                        "tumor_fraction": float(r["tumor_fraction"]),
                        "tissue_of_origin": str(r["tissue_of_origin"]),
                        "stage_proxy": int(r["stage_proxy"]),
                        "source": name,
                        "coverage": cov,
                    })

    X = np.stack(all_X, axis=0).astype(np.float32)
    meta = pd.DataFrame(all_meta)
    log(f"Augmented dataset: {X.shape}")
    log(f"By coverage: {meta.groupby('coverage').size().to_dict()}")
    log(f"By source: {meta.groupby('source').size().to_dict()}")
    log(f"Cancer balance: {meta['cancer_present'].value_counts().to_dict()}")
    return X, meta


def make_y(meta, atlas_cts):
    y_cancer = meta["cancer_present"].to_numpy(dtype=np.float32)
    y_tf = meta["tumor_fraction"].to_numpy(dtype=np.float32)
    tissue = meta["tissue_of_origin"].astype(str).to_numpy()
    y_tissue = np.array([TISSUE_LIST.index(t) if t in TISSUE_LIST else -1 for t in tissue], dtype=np.int64)
    y_stage = meta["stage_proxy"].to_numpy(dtype=np.int64).clip(0, 3)
    return y_cancer, y_tf, y_tissue, y_stage


def train_mlp(X_tr, y_tr_can, y_tr_ti, y_tr_tf, y_tr_st,
              X_va, y_va_can, y_va_ti, y_va_tf, y_va_st,
              n_in, device, n_epochs=80, lr=1e-3):
    model = MultiTaskMLP(n_in=n_in).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    bce = nn.BCEWithLogitsLoss()
    ce_t = nn.CrossEntropyLoss(ignore_index=-1)
    ce_s = nn.CrossEntropyLoss(ignore_index=0)
    mse = nn.MSELoss()

    # Explicit conversions
    Xt = torch.from_numpy(X_tr).float().to(device)
    yct = torch.from_numpy(y_tr_can).float().to(device)
    ytit = torch.from_numpy(y_tr_ti).long().to(device)
    ytft = torch.from_numpy(y_tr_tf).float().to(device)
    ystt = torch.from_numpy(y_tr_st).long().to(device)
    Xv = torch.from_numpy(X_va).float().to(device)
    ycv = torch.from_numpy(y_va_can).float().to(device)
    ytiv = torch.from_numpy(y_va_ti).long().to(device)
    ytfv = torch.from_numpy(y_va_tf).float().to(device)
    ystv = torch.from_numpy(y_va_st).long().to(device)

    best_auroc = -1; best_state = None; patience = 12; stale = 0
    bs = 128
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(Xt.size(0), device=device)
        ltot = 0; nb = 0
        for i in range(0, Xt.size(0), bs):
            idx = perm[i:i+bs]
            x = Xt[idx]
            opt.zero_grad()
            lc, lti, ltf, lst = model(x)
            L = (3*bce(lc, yct[idx]) + 1*ce_t(lti, ytit[idx])
                 + 0.5*mse(ltf, ytft[idx]) + 0.5*ce_s(lst, ystt[idx]))
            L.backward()
            opt.step()
            ltot += L.item(); nb += 1
        sched.step()
        model.eval()
        with torch.no_grad():
            lc, _, _, _ = model(Xv)
            pv = torch.sigmoid(lc).cpu().numpy()
        try: auroc = roc_auc_score(y_va_can, pv)
        except Exception: auroc = 0.5
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            log(f"  epoch {epoch+1}/{n_epochs} loss={ltot/nb:.4f} val_auroc={auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience: log(f"  early stop @ {epoch+1}"); break
    model.load_state_dict(best_state)
    return model, best_auroc


def sens_at_spec(y_true, y_score, spec_target):
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1 - fpr
    valid = spec >= spec_target
    if not valid.any(): return float("nan")
    idx = np.argmax(tpr * valid)
    return float(tpr[idx])


def eval_mlp(model, X, y_cancer, y_tissue, device, name):
    model.eval()
    with torch.no_grad():
        lc, lti, _, _ = model(torch.from_numpy(X).to(device))
        p = torch.sigmoid(lc).cpu().numpy()
        pt = torch.softmax(lti, dim=-1).cpu().numpy()
    try: auroc = float(roc_auc_score(y_cancer, p))
    except Exception: auroc = float("nan")
    s95 = sens_at_spec(y_cancer, p, 0.95)
    s99 = sens_at_spec(y_cancer, p, 0.99)
    s995 = sens_at_spec(y_cancer, p, 0.995)
    cm = (y_cancer == 1) & (y_tissue >= 0) & (y_tissue < len(TISSUE_LIST))
    top1 = float((pt[cm].argmax(axis=1) == y_tissue[cm]).mean()) if cm.sum() > 0 else float("nan")
    healthy_p = float(p[y_cancer==0].mean()) if (y_cancer==0).any() else None
    cancer_p = float(p[y_cancer==1].mean()) if (y_cancer==1).any() else None
    log(f"  {name}: AUROC={auroc:.4f} sens95={s95:.3f} sens99={s99:.3f} sens995={s995:.3f} top1={top1:.3f} (n_can={int((y_cancer==1).sum())} n_h={int((y_cancer==0).sum())}) healthy_p={healthy_p:.3f} cancer_p={cancer_p:.3f}")
    return {
        "AUROC": auroc, "sens95": s95, "sens99": s99, "sens995": s995,
        "tissue_top1": top1, "n_cancer": int((y_cancer==1).sum()),
        "n_healthy": int((y_cancer==0).sum()),
        "healthy_p_mean": healthy_p, "cancer_p_mean": cancer_p,
        "p_min_max": [float(p.min()), float(p.max())],
        "p_scores": p.tolist(),
    }


def main():
    t0 = time.time()
    log("=== PHASE C v2: COVERAGE-AUGMENTED multi-task classifier ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    atlas_cts = load_atlas_cts()

    # Load NaN-aware deconv
    ckpt = torch.load(MODELS / "learned_deconv_nanaware.pt", map_location=device, weights_only=False)
    deconv = NaNAwareDeconv(N_BLOCKS, len(atlas_cts)).to(device)
    deconv.load_state_dict(ckpt["state_dict"])
    deconv.eval()
    log(f"loaded NaN-aware deconv (val_pearson={ckpt['best_val_pearson']:.4f})")

    # Build augmented training dataset
    X, meta = build_aug_dataset(deconv, atlas_cts, device,
                                 coverage_levels=(1.0, 0.30, 0.15, 0.08))

    # Build GSE122126 test set (cfDNA only)
    log("loading GSE122126 cfDNA test set...")
    gse_b = pd.read_parquet(PROC / "GSE122126_harmonized.parquet")
    samples = pd.read_csv(PROC / "GSE122126_samples.tsv", sep="\t")
    cols_g = [c for c in gse_b.columns if c != "block_id"]
    Xg = gse_b[cols_g].to_numpy(dtype=np.float32).T
    mg = (~np.isnan(Xg)).astype(np.float32)
    Xg = np.where(np.isnan(Xg), 0.0, Xg)
    preds_g = deconv_with_mask(deconv, Xg, mg, device)
    gse_df = pd.DataFrame(preds_g, columns=atlas_cts)
    gse_df["sample_id"] = cols_g
    gse_df = gse_df.merge(samples[["sample_id","label","title"]], on="sample_id")
    cfdna_labels = ["HEALTHY","CRC","LUNG","CUP","BREAST"]
    gse_test = gse_df[gse_df["label"].isin(cfdna_labels)].copy()
    gse_test["cancer_present"] = gse_test["label"].apply(lambda x: 0 if x == "HEALTHY" else 1)
    gse_test["tissue_of_origin"] = gse_test["label"].map(
        {"HEALTHY":"none","CRC":"colon","LUNG":"lung","CUP":"unknown","BREAST":"breast"}).fillna("unknown")
    Xg_test = gse_test[atlas_cts].to_numpy(dtype=np.float32)
    yg_can = gse_test["cancer_present"].to_numpy(dtype=np.float32)
    yg_ti = np.array([TISSUE_LIST.index(t) if t in TISSUE_LIST else -1
                      for t in gse_test["tissue_of_origin"].astype(str).to_numpy()], dtype=np.int64)
    log(f"  GSE122126 test: n={len(gse_test)} healthy={int((yg_can==0).sum())} cancer={int((yg_can==1).sum())}")

    # Stratified train/val split (random by index)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    tr_idx, va_idx = perm[:cut], perm[cut:]
    X_tr, X_va = X[tr_idx], X[va_idx]
    meta_tr, meta_va = meta.iloc[tr_idx].reset_index(drop=True), meta.iloc[va_idx].reset_index(drop=True)
    y_tr = make_y(meta_tr, atlas_cts)
    y_va = make_y(meta_va, atlas_cts)
    log(f"train n={len(X_tr)} val n={len(X_va)}")

    log("training multi-task MLP...")
    model, best_val = train_mlp(X_tr, *y_tr, X_va, *y_va, n_in=len(atlas_cts), device=device)
    log(f"best val AUROC: {best_val:.4f}")
    torch.save({"state_dict": model.state_dict(), "atlas_cts": atlas_cts}, MODELS / "day3_multitask_mlp_v2.pt")

    # Eval
    log("In-silico held-out:")
    insilico = eval_mlp(model, X_va, y_va[0], y_va[2], device, "in_silico")

    log("Cross-cohort GSE122126:")
    cross = eval_mlp(model, Xg_test, yg_can, yg_ti, device, "gse122126")

    # Per-coverage breakdown on val
    log("Val per-coverage:")
    cov_results = {}
    for cov in (1.0, 0.30, 0.15, 0.08):
        mask = (meta_va["coverage"] == cov)
        if mask.sum() < 5:
            cov_results[cov] = None; continue
        sub_X = X_va[mask.values]
        sub_y = y_va[0][mask.values]
        sub_t = y_va[2][mask.values]
        cov_results[cov] = eval_mlp(model, sub_X, sub_y, sub_t, device, f"cov_{cov}")

    # Per-stage on in-silico val (only full coverage)
    log("In-silico stage-strat (full-cov only):")
    fc_mask = (meta_va["coverage"] == 1.0).values
    stage_res = {}
    fc_X = X_va[fc_mask]; fc_y = y_va[0][fc_mask]; fc_st = y_va[3][fc_mask]
    fc_t = y_va[2][fc_mask]
    # Threshold @ 99% spec on full-cov val
    model.eval()
    with torch.no_grad():
        lc, _, _, _ = model(torch.from_numpy(fc_X).to(device))
        pv = torch.sigmoid(lc).cpu().numpy()
    fpr_v, tpr_v, thr_v = roc_curve(fc_y, pv)
    valid = (1 - fpr_v) >= 0.99
    threshold = float(thr_v[np.argmax(tpr_v * valid)]) if valid.any() else 0.5
    for s in (1, 2, 3, 4):
        m = fc_st == s
        if m.sum() < 5: stage_res[f"stage_{s}"] = None; continue
        with torch.no_grad():
            lc, _, _, _ = model(torch.from_numpy(fc_X[m]).to(device))
            pp = torch.sigmoid(lc).cpu().numpy()
        sens = float((pp >= threshold).mean())
        stage_res[f"stage_{s}"] = {"n": int(m.sum()), "sens_at_99spec": sens}
        log(f"  stage {s}: n={m.sum()} sens@99spec={sens:.3f}")

    summary = {
        "best_val_auroc": float(best_val),
        "in_silico": insilico,
        "cross_cohort_gse122126": cross,
        "per_coverage": {str(k): v for k, v in cov_results.items()},
        "stage_strat": stage_res,
        "n_train": int(len(X_tr)), "n_val": int(len(X_va)),
        "day2_cross_cohort_auroc": 0.217,
        "day2_in_silico_auroc": 0.899,
        "delta_cross": cross["AUROC"] - 0.217 if not np.isnan(cross["AUROC"]) else None,
        "delta_insilico": insilico["AUROC"] - 0.899 if not np.isnan(insilico["AUROC"]) else None,
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(MODELS / "phase_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(RESULTS / "day3_phase_c_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log(f"=== PHASE C v2 DONE in {summary['elapsed_min']} min ===")
    log("HEADLINE:")
    log(f"  Day-2 cross-cohort AUROC: 0.217")
    log(f"  Day-3 cross-cohort AUROC: {cross['AUROC']:.4f}")
    log(f"  Day-2 in-silico AUROC: 0.899")
    log(f"  Day-3 in-silico AUROC: {insilico['AUROC']:.4f}")


if __name__ == "__main__":
    main()

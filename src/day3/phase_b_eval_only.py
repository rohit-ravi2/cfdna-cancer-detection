#!/usr/bin/env python3
"""Run cross-coverage eval on saved model + write Phase B summary."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
LOG = Path("/mnt/ssd/research/cfdna/logs/day3_phase_b.log")
N_BLOCKS = 9520


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")


class NaNAwareDeconv(nn.Module):
    def __init__(self, n_blocks=9520, n_ct=40, hidden=2048):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_blocks * 2, hidden),
            nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU(),
        )
        self.deconv = nn.Linear(128, n_ct)
        self.recon = nn.Sequential(
            nn.Linear(n_ct, 128), nn.ReLU(),
            nn.Linear(128, n_blocks), nn.Sigmoid(),
        )

    def forward(self, b, m):
        x = torch.cat([b, m], dim=-1)
        h = self.enc(x)
        logits = self.deconv(h)
        frac = F.softmax(logits, dim=-1)
        recon = self.recon(frac)
        return frac, recon, logits


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS / "learned_deconv_nanaware.pt", map_location=device, weights_only=False)
    atlas_cts = ckpt["atlas_cts"]
    model = NaNAwareDeconv(N_BLOCKS, len(atlas_cts)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    log(f"loaded model: val_pearson={ckpt['best_val_pearson']:.4f}")

    betas = pd.read_parquet(PROC / "distsampled_mixtures_betas.parquet")
    labels = pd.read_parquet(PROC / "distsampled_mixtures_labels.parquet")
    cols = [c for c in betas.columns if c != "block_id"]
    X = betas[cols].to_numpy(dtype=np.float32).T
    # Distsampled betas may have NaN from atlas centroid lookups; treat those
    # as unobserved at eval time (mask=0, beta=0)
    base_mask = (~np.isnan(X)).astype(np.float32)
    X = np.where(np.isnan(X), 0.0, X)

    # Cancer-only with tissue
    sample_to_label = labels.set_index("sample_id")
    keep_idx, tissues, targets_tf = [], [], []
    for i, sid in enumerate(cols):
        if sid in sample_to_label.index:
            r = sample_to_label.loc[sid]
            t = str(r["tissue_of_origin"])
            if int(r["cancer_present"]) == 1 and t in ("colon","liver","lung","breast"):
                keep_idx.append(i); tissues.append(t); targets_tf.append(float(r["tumor_fraction"]))
    keep_idx = np.array(keep_idx); targets_tf = np.array(targets_tf)
    X = X[keep_idx]
    base_mask = base_mask[keep_idx]
    log(f"eval set: {X.shape}, tissue counts:")
    from collections import Counter
    log(f"  {Counter(tissues)}")
    tissue_to_ct = {"colon":"Colon-Ep","liver":"Liver-Hep","lung":"Lung-Ep-Alveo","breast":"Breast-Luminal-Ep"}

    results = {}
    for keep_frac in [1.0, 0.20, 0.10]:
        rng = np.random.RandomState(123)
        mask = np.zeros_like(X)
        if keep_frac < 1.0:
            n_keep = int(keep_frac * N_BLOCKS)
            for i in range(X.shape[0]):
                idx = rng.choice(N_BLOCKS, n_keep, replace=False)
                mask[i, idx] = 1.0
        else:
            mask[:] = 1.0
        # Final mask = original observation mask AND the random sub-sampling mask
        mask = mask * base_mask
        X_m = (X * mask).astype(np.float32)
        with torch.no_grad():
            preds_list = []
            bs = 512
            for s in range(0, X_m.shape[0], bs):
                b = torch.from_numpy(X_m[s:s+bs]).to(device)
                m = torch.from_numpy(mask[s:s+bs]).to(device)
                f, _, _ = model(b, m)
                preds_list.append(f.cpu().numpy())
            preds = np.concatenate(preds_list, axis=0)
        log(f"  coverage {int(keep_frac*100)}% preds: mean={preds.mean():.4f} std_per_col_mean={preds.std(axis=0).mean():.4f}")
        per_tissue = {}
        for tissue, ct in tissue_to_ct.items():
            mt = np.array([t == tissue for t in tissues])
            if mt.sum() < 3: continue
            j = atlas_cts.index(ct)
            p = preds[mt, j]; q = targets_tf[mt]
            r = float(np.corrcoef(p, q)[0,1]) if np.std(p) > 1e-8 else float("nan")
            per_tissue[tissue] = r
            log(f"    {tissue} ({ct}): n={mt.sum()} pred_std={np.std(p):.4f} target_std={np.std(q):.4f} pearson={r:.4f}")
        mean_r = float(np.nanmean(list(per_tissue.values())))
        results[f"keep_{int(keep_frac*100)}"] = {"per_tissue": per_tissue, "mean_pearson": mean_r}
        log(f"  ---> coverage {int(keep_frac*100)}% mean_pearson={mean_r:.4f}")

    # Save summary
    summary = {
        "model_path": str(MODELS / "learned_deconv_nanaware.pt"),
        "best_val_pearson": float(ckpt["best_val_pearson"]),
        "cross_coverage": results,
    }
    with open(MODELS / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log("DONE")
    log(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

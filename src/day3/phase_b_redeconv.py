#!/usr/bin/env python3
"""Re-run deconvolution outputs using the saved NaN-aware model, with
proper handling of NaN in distsampled betas (some come from atlas
centroid NaN; treat as unobserved)."""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

PROC = Path("/mnt/ssd/research/cfdna/processed")
MODELS = Path("/mnt/ssd/research/cfdna/models")
N_BLOCKS = 9520


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
        h = self.enc(x); logits = self.deconv(h)
        frac = F.softmax(logits, dim=-1)
        return frac, self.recon(frac), logits


def deconvolve(parquet_path, model, atlas_cts, device):
    df = pd.read_parquet(parquet_path)
    cols = [c for c in df.columns if c != "block_id"]
    X = df[cols].to_numpy(dtype=np.float32).T
    mask = (~np.isnan(X)).astype(np.float32)
    Xf = np.where(np.isnan(X), 0.0, X)
    model.eval()
    with torch.no_grad():
        bs = 256
        preds = []
        for s in range(0, Xf.shape[0], bs):
            b = torch.from_numpy(Xf[s:s+bs]).to(device)
            m = torch.from_numpy(mask[s:s+bs]).to(device)
            f, _, _ = model(b, m)
            preds.append(f.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
    out = pd.DataFrame(preds, columns=atlas_cts)
    out.insert(0, "sample_id", cols)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(MODELS / "learned_deconv_nanaware.pt", map_location=device, weights_only=False)
    atlas_cts = ckpt["atlas_cts"]
    model = NaNAwareDeconv(N_BLOCKS, len(atlas_cts)).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"loaded model: val_pearson={ckpt['best_val_pearson']:.4f}")

    sources = {
        "distsampled": PROC / "distsampled_mixtures_betas.parquet",
        "fixed":       PROC / "GSE186458_mixtures_betas_v2.parquet",
        "cfdna_real":  PROC / "GSE186458_cfdna_harmonized.parquet",
        "gse122126":   PROC / "GSE122126_harmonized.parquet",
    }
    summary = {}
    for name, p in sources.items():
        out = deconvolve(p, model, atlas_cts, device)
        out.to_parquet(PROC / f"day3_{name}_deconv_nanaware.parquet", index=False)
        # Summary stats
        arr = out[atlas_cts].to_numpy()
        granu = arr[:, atlas_cts.index("Blood-Granul")].mean()
        lymph = arr[:, atlas_cts.index("Blood-T")].mean() + arr[:, atlas_cts.index("Blood-B")].mean() + arr[:, atlas_cts.index("Blood-NK")].mean()
        liver = arr[:, atlas_cts.index("Liver-Hep")].mean()
        colon = arr[:, atlas_cts.index("Colon-Ep")].mean()
        print(f"{name}: n={out.shape[0]} granu={granu:.3f} lymph={lymph:.3f} liver={liver:.3f} colon={colon:.3f}")
        summary[name] = {
            "n": int(out.shape[0]), "granu": float(granu), "lymph": float(lymph),
            "liver": float(liver), "colon": float(colon),
        }
    with open(MODELS / "phase_b_deconv_outputs.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done")


if __name__ == "__main__":
    main()

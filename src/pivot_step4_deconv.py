#!/usr/bin/env python3
"""
PIVOT_PHASE1_GSE186458 STEP 4 — Deconvolution + biological sanity validation.

Runs TWO deconvolution methods on the 23 GSE186458 healthy cfDNA samples
(and compares to the matched WBC samples as a side check):

Method A — Beta-NNLS (our validated Gate 3 code, but on properly-deep WGBS):
    For each sample, solve
       argmin_{x>=0} || diag(sqrt(w)) (A x - b) ||  +  lambda * |sum(x) - 1|
    where A is the methylation atlas (1 - U-frac), b is the mean beta per
    atlas block from beta_to_blocks, w is per-block informativeness.

Method B — UXM (Loyfer's per-read fragment-level method):
    Runs `wgbstools homog` per sample on the .pat.gz to count (U, X, M) reads
    per block, then NNLS on the U-fraction matrix against the published UXM
    atlas. We invoke the UXM_deconv CLI directly.

Outputs:
    /mnt/ssd/research/cfdna/processed/GSE186458_cfdna_deconv_nnls.parquet
    /mnt/ssd/research/cfdna/processed/GSE186458_cfdna_deconv_uxm.parquet
    /mnt/ssd/research/cfdna/processed/GSE186458_wbc_deconv_nnls.parquet  (sanity)
    /mnt/ssd/research/cfdna/results/pivot_phase1_sanity.json

Pass criterion: granulocyte family fraction (mean across 23 cfDNA samples)
is within [15%, 45%] using Method B (UXM). If both methods fail, hard-stop.
We also compare each cell-type family to Loyfer 2023 published numbers
(granulocyte ~30%, monocyte/macrophage ~20%, lymphocyte ~3%, hepatocyte ~3%,
endothelial ~6%) with relative-error tolerance ±50%.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import nnls
from tqdm import tqdm

# --- Paths -------------------------------------------------------------
PROC_DIR = Path("/mnt/ssd/research/cfdna/processed")
LOG_DIR = Path("/mnt/ssd/research/cfdna/logs")
RES_DIR = Path("/mnt/ssd/research/cfdna/results")
PROC_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

ATLAS_TSV = Path("/mnt/hdd/research/cfdna/raw/loyfer_atlas/Atlas.U250.l4.hg38.full.tsv")
# Use U25 for UXM (the canonical UXM atlas: top 25 markers per cell type,
# ~1000 rows total). U250 (top 250) is for averaged-beta NNLS.
ATLAS_FOR_UXM = Path("/mnt/hdd/research/cfdna/raw/loyfer_atlas/Atlas.U25.l4.hg38.tsv")
GROUPS_JSON = PROC_DIR / "cell_type_groups.json"
GATES_LOG = LOG_DIR / "gates.log"

PIVOT_RAW = Path("/mnt/hdd/research/cfdna/raw/GSE186458/pivot")

UXM_BIN = "/home/rohit/tools/UXM_deconv/uxm"
META_COLS = ["chr", "start", "end", "startCpG", "endCpG", "target", "name", "direction"]

# Loyfer 2023 published cfDNA family fractions (Figure 5; healthy cfDNA mean)
LOYFER_PUBLISHED = {
    "granulocyte": 0.30,
    "monocyte_macrophage": 0.20,
    "lymphocyte": 0.03,
    "liver_hepatocyte": 0.03,
    "endothelial": 0.06,
    "erythroid": 0.08,
}

# Pass band on granulocyte (the gate)
GRANU_PASS_MIN = 0.15
GRANU_PASS_MAX = 0.45


def log(msg):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ----------------------------------------------------------------------
# Method A: averaged-beta NNLS
# ----------------------------------------------------------------------

SUM_PENALTY = 20.0


def load_atlas():
    df = pd.read_csv(ATLAS_TSV, sep="\t")
    cts = [c for c in df.columns if c not in META_COLS]
    A_u = df[cts].to_numpy(dtype=np.float64)
    # Fill NaN cells with column means
    if np.isnan(A_u).any():
        cm = np.nanmean(A_u, axis=0)
        cm = np.where(np.isnan(cm), 0.5, cm)
        ix = np.where(np.isnan(A_u))
        A_u[ix] = np.take(cm, ix[1])
    return A_u, cts, df


def nnls_one(b, A_meth, w):
    mask = ~np.isnan(b)
    if mask.sum() == 0:
        return np.zeros(A_meth.shape[1]), np.nan, 0
    A_s = A_meth[mask]
    b_s = b[mask]
    sw = np.sqrt(np.clip(w[mask], 0, None))
    keep = sw > 0
    sw = sw[keep]
    A_s = A_s[keep]
    b_s = b_s[keep]
    A_w = A_s * sw[:, None]
    b_w = b_s * sw
    sum_row = np.full((1, A_meth.shape[1]), SUM_PENALTY, dtype=np.float64)
    A_aug = np.vstack([A_w, sum_row])
    b_aug = np.concatenate([b_w, np.array([SUM_PENALTY])])
    try:
        x, resid = nnls(A_aug, b_aug, maxiter=20000)
    except RuntimeError:
        return np.zeros(A_meth.shape[1]), np.nan, int(mask.sum())
    s = x.sum()
    if s > 0:
        x = x / s
    return x, float(resid), int(mask.sum())


def run_nnls_on_matrix(matrix_path: Path, label: str):
    log(f"NNLS [{label}]: loading {matrix_path.name}")
    H = pd.read_parquet(matrix_path)
    samples = [c for c in H.columns if c != "block_id"]
    H = H.sort_values("block_id").reset_index(drop=True)
    A_u, cts, _ = load_atlas()
    A_meth = 1.0 - A_u
    # informativeness weight: max - 2nd max in U-fraction space
    A_sorted = np.sort(A_u, axis=1)
    w_info = A_sorted[:, -1] - A_sorted[:, -2]
    B = H[samples].to_numpy(dtype=np.float64)
    log(f"NNLS [{label}]: running on {B.shape[1]} samples, {B.shape[0]} blocks")
    t0 = time.time()
    results = Parallel(n_jobs=8, backend="loky")(
        delayed(nnls_one)(B[:, j], A_meth, w_info) for j in range(B.shape[1])
    )
    log(f"NNLS [{label}]: done in {time.time()-t0:.1f}s")
    fracs = np.stack([r[0] for r in results], axis=0)
    out = pd.DataFrame(fracs, columns=cts)
    out.insert(0, "sample_id", samples)
    out["nnls_residual"] = [r[1] for r in results]
    out["n_blocks_used"] = [r[2] for r in results]
    return out, cts


# ----------------------------------------------------------------------
# Method B: UXM (per-read)
# ----------------------------------------------------------------------

def list_pat_files(kind: str) -> list[Path]:
    files = []
    for p in sorted(PIVOT_RAW.glob("GSM*.hg38.pat.gz")):
        m = re.match(r"(GSM\d+)_CNVS-NORM-(cfDNA|WBC)-", p.name)
        if m and m.group(2) == kind:
            num = int(m.group(1).replace("GSM", ""))
            if 6810003 <= num <= 6810048:
                files.append(p)
    return files


def run_uxm(pat_files: list[Path], out_csv: Path, atlas_path: Path,
            label: str, threads: int = 8):
    log(f"UXM [{label}]: running on {len(pat_files)} samples (atlas={atlas_path.name})")
    env = os.environ.copy()
    env["PATH"] = (
        "/home/rohit/miniconda3/envs/wgbs/bin:"
        "/home/rohit/tools/wgbs_tools:"
        "/home/rohit/tools/UXM_deconv:"
        + env.get("PATH", "")
    )
    cmd = [
        UXM_BIN, "deconv",
        "--atlas", str(atlas_path),
        "-o", str(out_csv),
        "--threads", str(threads),
        "--rlen", "4",
        "--tmp_dir", "/tmp/uxm_tmp",
    ] + [str(p) for p in pat_files]
    log(f"UXM [{label}]: cmd = {' '.join(cmd[:8])} ... ({len(pat_files)} pats)")
    t0 = time.time()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    log(f"UXM [{label}]: rc={r.returncode}  elapsed={time.time()-t0:.1f}s")
    if r.returncode != 0:
        log(f"UXM stderr (last 30 lines):\n" + "\n".join(r.stderr.splitlines()[-30:]))
        raise RuntimeError(f"UXM failed: {r.returncode}")
    log(f"UXM [{label}]: wrote {out_csv}")
    return r.stderr


def parse_uxm_output(csv_path: Path) -> pd.DataFrame:
    """UXM writes a CSV with cell-type rows and sample columns (or vice versa)."""
    df = pd.read_csv(csv_path)
    # First col is cell-type
    df = df.rename(columns={df.columns[0]: "cell_type"})
    # Transpose so rows = samples, cols = cell types
    out = df.set_index("cell_type").T.reset_index().rename(columns={"index": "sample_full"})
    # Extract GSM from sample_full
    out["sample_id"] = out["sample_full"].apply(
        lambda s: re.match(r"(GSM\d+)_", str(s)).group(1)
        if re.match(r"(GSM\d+)_", str(s)) else str(s)
    )
    cols = ["sample_id"] + [c for c in out.columns if c not in ("sample_id", "sample_full")]
    return out[cols]


# ----------------------------------------------------------------------
# Family fractions
# ----------------------------------------------------------------------

def load_groups():
    with open(GROUPS_JSON) as f:
        g = json.load(f)
    return g["groups"]


def add_endothelial_group(groups: dict, cell_types: list[str]):
    """Add an endothelial group if not present (Loyfer published includes endothel)."""
    if "endothelial" in groups:
        return groups
    endo = [c for c in cell_types if "endothel" in c.lower()]
    groups["endothelial"] = endo
    return groups


def family_fractions_df(deconv_df: pd.DataFrame, groups: dict,
                         cell_types: list[str]) -> pd.DataFrame:
    """deconv_df cols: sample_id + (subset of) cell_types. Returns
    sample_id + fam_<family> sums.
    """
    out = pd.DataFrame({"sample_id": deconv_df["sample_id"].values})
    for fam, members in groups.items():
        cols = [m for m in members if m in deconv_df.columns]
        if not cols:
            out[f"fam_{fam}"] = 0.0
        else:
            vals = deconv_df[cols].astype(float).fillna(0).sum(axis=1).values
            out[f"fam_{fam}"] = vals
    return out


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    log("=== STEP 4 begin ===")

    # --- Method A: NNLS on cfDNA and WBC harmonized betas ---
    cf_nnls_df, cell_types = run_nnls_on_matrix(
        PROC_DIR / "GSE186458_cfdna_harmonized.parquet", "cfDNA"
    )
    cf_nnls_df.to_parquet(PROC_DIR / "GSE186458_cfdna_deconv_nnls.parquet",
                          index=False)
    log(f"wrote GSE186458_cfdna_deconv_nnls.parquet  shape={cf_nnls_df.shape}")

    wbc_nnls_df, _ = run_nnls_on_matrix(
        PROC_DIR / "GSE186458_wbc_harmonized.parquet", "WBC"
    )
    wbc_nnls_df.to_parquet(PROC_DIR / "GSE186458_wbc_deconv_nnls.parquet",
                            index=False)
    log(f"wrote GSE186458_wbc_deconv_nnls.parquet  shape={wbc_nnls_df.shape}")

    # --- Method B: UXM on cfDNA and WBC pat files ---
    cf_pats = list_pat_files("cfDNA")
    wbc_pats = list_pat_files("WBC")
    log(f"cfDNA pat files: {len(cf_pats)}")
    log(f"WBC pat files: {len(wbc_pats)}")

    Path("/tmp/uxm_tmp").mkdir(parents=True, exist_ok=True)

    cf_uxm_csv = PROC_DIR / "GSE186458_cfdna_uxm_raw.csv"
    wbc_uxm_csv = PROC_DIR / "GSE186458_wbc_uxm_raw.csv"

    if cf_pats:
        run_uxm(cf_pats, cf_uxm_csv, ATLAS_FOR_UXM, "cfDNA", threads=8)
        cf_uxm = parse_uxm_output(cf_uxm_csv)
        cf_uxm.to_parquet(PROC_DIR / "GSE186458_cfdna_deconv_uxm.parquet",
                          index=False)
        log(f"wrote GSE186458_cfdna_deconv_uxm.parquet  shape={cf_uxm.shape}")
    else:
        cf_uxm = None

    if wbc_pats:
        run_uxm(wbc_pats, wbc_uxm_csv, ATLAS_FOR_UXM, "WBC", threads=8)
        wbc_uxm = parse_uxm_output(wbc_uxm_csv)
        wbc_uxm.to_parquet(PROC_DIR / "GSE186458_wbc_deconv_uxm.parquet",
                           index=False)
        log(f"wrote GSE186458_wbc_deconv_uxm.parquet  shape={wbc_uxm.shape}")
    else:
        wbc_uxm = None

    # --- Family-fraction roll-up & sanity ---
    groups = load_groups()
    groups = add_endothelial_group(groups, cell_types)

    cf_nnls_fam = family_fractions_df(cf_nnls_df, groups, cell_types)
    wbc_nnls_fam = family_fractions_df(wbc_nnls_df, groups, cell_types)

    summary = {
        "atlas": str(ATLAS_TSV.name),
        "n_cfdna": int(cf_nnls_df.shape[0]),
        "n_wbc": int(wbc_nnls_df.shape[0]),
        "loyfer_published": LOYFER_PUBLISHED,
        "method_A_nnls_cfdna": {},
        "method_A_nnls_wbc": {},
        "method_B_uxm_cfdna": {},
        "method_B_uxm_wbc": {},
        "method_agreement": {},
        "verdict": {},
    }

    def fam_stats(df, fam_cols=None):
        if fam_cols is None:
            fam_cols = [c for c in df.columns if c.startswith("fam_")]
        d = {}
        for c in fam_cols:
            vals = df[c].astype(float).dropna().values
            d[c.replace("fam_", "")] = {
                "n": int(vals.size),
                "mean": float(vals.mean()) if vals.size else float("nan"),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "median": float(np.median(vals)) if vals.size else float("nan"),
            }
        return d

    summary["method_A_nnls_cfdna"] = fam_stats(cf_nnls_fam)
    summary["method_A_nnls_wbc"] = fam_stats(wbc_nnls_fam)

    # UXM family fractions
    if cf_uxm is not None:
        cf_uxm_fam = family_fractions_df(cf_uxm, groups, cell_types)
        summary["method_B_uxm_cfdna"] = fam_stats(cf_uxm_fam)
        cf_uxm_fam.to_csv(PROC_DIR / "GSE186458_cfdna_uxm_family.tsv",
                          sep="\t", index=False)
    else:
        cf_uxm_fam = None
    if wbc_uxm is not None:
        wbc_uxm_fam = family_fractions_df(wbc_uxm, groups, cell_types)
        summary["method_B_uxm_wbc"] = fam_stats(wbc_uxm_fam)
        wbc_uxm_fam.to_csv(PROC_DIR / "GSE186458_wbc_uxm_family.tsv",
                           sep="\t", index=False)
    else:
        wbc_uxm_fam = None

    # Method agreement (per-cell-type per-sample correlation Method A vs B)
    if cf_uxm is not None:
        # Align cell type columns
        shared_cts = [c for c in cell_types if c in cf_uxm.columns]
        agree = {}
        for ct in shared_cts:
            a = cf_nnls_df.set_index("sample_id")[ct]
            b = cf_uxm.set_index("sample_id")[ct]
            join = pd.concat([a, b], axis=1, keys=["A", "B"]).dropna()
            if len(join) >= 5:
                r = float(join["A"].corr(join["B"]))
                agree[ct] = {
                    "pearson_r": r,
                    "A_mean": float(join["A"].mean()),
                    "B_mean": float(join["B"].mean()),
                }
        summary["method_agreement"]["per_celltype"] = agree

    # --- Verdict ---
    # Headline: UXM granulocyte fraction in [15%, 45%]
    granu_A = summary["method_A_nnls_cfdna"].get("granulocyte", {}).get("mean", float("nan"))
    granu_B = summary["method_B_uxm_cfdna"].get("granulocyte", {}).get("mean", float("nan"))

    pass_B = (not np.isnan(granu_B)) and GRANU_PASS_MIN <= granu_B <= GRANU_PASS_MAX
    pass_A = (not np.isnan(granu_A)) and GRANU_PASS_MIN <= granu_A <= GRANU_PASS_MAX

    # Loyfer agreement: relative error of UXM means vs published, mean across families
    def rel_err(observed, published):
        if observed is None or np.isnan(observed):
            return float("nan")
        if published == 0:
            return float("nan")
        return abs(observed - published) / published

    loyfer_rel_err = {}
    if summary["method_B_uxm_cfdna"]:
        for fam, pub in LOYFER_PUBLISHED.items():
            obs = summary["method_B_uxm_cfdna"].get(fam, {}).get("mean", float("nan"))
            loyfer_rel_err[fam] = {
                "published": pub, "observed_uxm": obs,
                "rel_err": rel_err(obs, pub),
            }
    summary["loyfer_relative_error_uxm"] = loyfer_rel_err

    # Pass if Method B granulocyte in band. Also report agreement.
    verdict = {
        "granu_method_A": granu_A,
        "granu_method_B": granu_B,
        "granu_pass_band": [GRANU_PASS_MIN, GRANU_PASS_MAX],
        "method_A_pass": bool(pass_A),
        "method_B_pass": bool(pass_B),
        "overall_pass": bool(pass_B),
    }
    summary["verdict"] = verdict

    with open(RES_DIR / "pivot_phase1_sanity.json", "w") as f:
        json.dump(summary, f, indent=2)
    log(f"wrote pivot_phase1_sanity.json")

    log("=== STEP 4 verdict ===")
    log(f"granu Method A NNLS:  {granu_A:.3f}  (band {GRANU_PASS_MIN}-{GRANU_PASS_MAX})")
    log(f"granu Method B UXM :  {granu_B:.3f}  (band {GRANU_PASS_MIN}-{GRANU_PASS_MAX})")
    log(f"overall pass = {verdict['overall_pass']}")

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if verdict["overall_pass"]:
        line = (
            f"[{ts}] PIVOT_PHASE1_GSE186458 STEP4_deconv: PASS "
            f"granu_uxm={granu_B:.3f} granu_nnls={granu_A:.3f} "
            f"matches_loyfer_granu=Y agreement_methods_A_B_recorded"
        )
        with open(GATES_LOG, "a") as f:
            f.write(line + "\n")
        os.system(
            f"~/bin/notify 'cfdna step4 deconv PASS UXM granu={granu_B:.2f} "
            f"NNLS granu={granu_A:.2f}' 'cfdna' 2>/dev/null"
        )
        return 0
    else:
        line = (
            f"[{ts}] PIVOT_PHASE1_GSE186458 STEP4_deconv: HARD_STOP "
            f"granu_uxm={granu_B:.3f} granu_nnls={granu_A:.3f} "
            f"band={GRANU_PASS_MIN}-{GRANU_PASS_MAX}"
        )
        with open(GATES_LOG, "a") as f:
            f.write(line + "\n")
        os.system(
            f"~/bin/notify 'cfdna step4 deconv HARD STOP granu_uxm={granu_B:.2f}' "
            f"'cfdna' 'urgent'"
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

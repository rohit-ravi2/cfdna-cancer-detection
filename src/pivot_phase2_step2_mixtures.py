#!/usr/bin/env python3
"""
PIVOT_PHASE2 STEP 2 — In-silico mixture generation via `wgbstools mix_pat`.

Design:
    healthy donors: 23  (GSM6810003..GSM6810025) -- already on disk at
                       work/raw/GSE186458/pivot/
    tissue-of-origin: 5 categories
        colon    : Colon-Left-Epithelium  +  Colon-Right-Epithelium
        liver    : Liver-Hepatocytes
        lung     : Lung-Alveolar-Epithelium + Lung-Bronchus-Epithelium
        breast   : Breast-Basal-Epithelium + Breast-Luminal-Epithelium
        pancreas : Pancreas-Duct
    fractions:  [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
    replicate per cell: stochastic — pick one donor at random per (h,t,f) cell.
    Total mixtures: 23 * 5 * 7 = 805
    Plus 23 unmixed healthy in label set.

Performance: with `-L atlas_blocks.bed`, mix_pat runs ~11s; pat2beta ~2s;
beta_to_blocks ~1s. With 6 workers in parallel, ~30 min total.

Outputs:
    work/processed/mixtures_beta/*.bin  (one per mix; 19KB each)
    work/raw/GSE186458/pivot/mixtures/manifest.tsv
    work/processed/GSE186458_mixtures_betas.parquet   [9520 x 828]
    work/processed/GSE186458_mixtures_labels.parquet  [828 x 4]
"""
import json
import os
import random
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

WGBSTOOLS = os.environ.get("WGBSTOOLS", "wgbstools")
PATH_PREFIX = os.environ.get("WGBS_BIN_DIR", "") + ":" + os.environ.get("WGBS_DIR", "")
ATLAS_BED = Path("work/processed/atlas_blocks_hg38.bed")

HEALTHY_DIR = Path("work/raw/GSE186458/pivot")
SORTED_DIR  = Path("work/raw/GSE186458/pivot/sorted_cells")
MIX_DIR     = Path("work/raw/GSE186458/pivot/mixtures")
BIN_DIR     = Path("work/processed/mixtures_beta")
LOG_DIR     = Path("work/logs")
GATES_LOG   = LOG_DIR / "gates.log"
STDERR_LOG  = LOG_DIR / "pivot_phase2_step2.log"

MIX_DIR.mkdir(parents=True, exist_ok=True)
BIN_DIR.mkdir(parents=True, exist_ok=True)

TISSUE_CELLS = {
    "colon":    ["Colon-Left-Epithelium", "Colon-Right-Epithelium"],
    "liver":    ["Liver-Hepatocytes"],
    "lung":     ["Lung-Alveolar-Epithelium", "Lung-Bronchus-Epithelium"],
    "breast":   ["Breast-Basal-Epithelium", "Breast-Luminal-Epithelium"],
    "pancreas": ["Pancreas-Duct"],
}

FRACTIONS = [0.001, 0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
N_BLOCKS = 9520
SEED = 20260513

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with STDERR_LOG.open("a") as f:
        f.write(line + "\n")


def list_healthy_pats() -> list[Path]:
    # Filter to cfDNA samples only (GSM6810003..GSM6810025).
    # WBC samples (GSM6810026..) are kept on disk but not used as mixture donors.
    all_pats = sorted(HEALTHY_DIR.glob("GSM6810*.hg38.pat.gz"))
    return [p for p in all_pats if "cfDNA" in p.name]


def list_sorted_pats_by_tissue() -> dict[str, list[Path]]:
    out = {t: [] for t in TISSUE_CELLS}
    for tissue, cells in TISSUE_CELLS.items():
        for cell in cells:
            out[tissue].extend(sorted(SORTED_DIR.glob(f"GSM*_{cell}-Z*.hg38.pat.gz")))
    return out


def get_sample_id(pat_path: Path) -> str:
    """Return GSM6810003 from full path basename."""
    name = pat_path.name
    return name.split("_")[0]


def get_donor_label(pat_path: Path) -> str:
    """Return human-readable label for the mix output filename."""
    return pat_path.name.replace(".hg38.pat.gz", "")


def compute_unmixed_block_beta(healthy_pat: Path, tmp_dir: Path) -> np.ndarray:
    """For unmixed-healthy samples, just run beta_to_blocks on the existing .beta."""
    beta_path = healthy_pat.with_suffix("").with_suffix("")  # strip .pat.gz
    beta_path = healthy_pat.parent / (healthy_pat.name.replace(".pat.gz", ".beta"))
    if not beta_path.exists():
        raise FileNotFoundError(f"expected beta: {beta_path}")
    env = os.environ.copy()
    env["PATH"] = PATH_PREFIX + ":" + env.get("PATH", "")
    cmd = [WGBSTOOLS, "beta_to_blocks",
           "--blocks_file", str(ATLAS_BED),
           "-o", str(tmp_dir),
           "-f",
           str(beta_path)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"beta_to_blocks failed: {r.stderr[:300]}")
    out_bin = tmp_dir / (beta_path.name.replace(".beta", ".bin"))
    if not out_bin.exists():
        raise FileNotFoundError(f"missing output {out_bin}")
    arr = np.fromfile(out_bin, dtype=np.uint8)
    if arr.size != 2 * N_BLOCKS:
        raise ValueError(f"wrong size {arr.size} expected {2*N_BLOCKS}")
    return arr.reshape(-1, 2)  # [meth, total]


def make_one_mix(args: tuple) -> dict:
    """Worker: generate one mixture and return its 9520x2 block bin."""
    (healthy_pat, tumor_pat, tumor_fraction, tissue, out_prefix, tmp_root) = args
    tmp_dir = Path(tmp_root) / f"mix_{os.getpid()}_{int(time.time()*1e6)%1_000_000}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        env = os.environ.copy()
        env["PATH"] = PATH_PREFIX + ":" + env.get("PATH", "")

        # 1) mix_pat
        cmd_mix = [
            WGBSTOOLS, "mix_pat",
            "--rates", str(tumor_fraction), str(1 - tumor_fraction),
            "-c", "5",
            "--reps", "1",
            "-L", str(ATLAS_BED),
            "--labels", tissue, "healthy",
            "-p", str(tmp_dir / "mix"),
            "-f",
            str(tumor_pat),
            str(healthy_pat),
        ]
        r = subprocess.run(cmd_mix, capture_output=True, text=True, env=env, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"mix_pat failed: {r.stderr[-500:]}")

        # mix_pat output: tmp_dir/mix_1.pat.gz
        mix_pat_path = tmp_dir / "mix_1.pat.gz"
        if not mix_pat_path.exists():
            # alternative naming
            alt = list(tmp_dir.glob("mix*.pat.gz"))
            if not alt:
                raise FileNotFoundError(f"mix output not found in {tmp_dir}")
            mix_pat_path = alt[0]

        # 2) pat2beta
        cmd_pat2beta = [WGBSTOOLS, "pat2beta", "-o", str(tmp_dir), "-f", str(mix_pat_path)]
        r = subprocess.run(cmd_pat2beta, capture_output=True, text=True, env=env, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pat2beta failed: {r.stderr[-500:]}")
        beta_path = mix_pat_path.with_suffix("").with_suffix(".beta")
        if not beta_path.exists():
            # alt naming
            alts = list(tmp_dir.glob("*.beta"))
            if not alts:
                raise FileNotFoundError(f"no .beta in {tmp_dir}")
            beta_path = alts[0]

        # 3) beta_to_blocks
        cmd_b2b = [WGBSTOOLS, "beta_to_blocks",
                   "--blocks_file", str(ATLAS_BED),
                   "-o", str(tmp_dir),
                   "-f",
                   str(beta_path)]
        r = subprocess.run(cmd_b2b, capture_output=True, text=True, env=env, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"beta_to_blocks failed: {r.stderr[-500:]}")
        bin_path = tmp_dir / beta_path.name.replace(".beta", ".bin")
        if not bin_path.exists():
            alts = list(tmp_dir.glob("*.bin"))
            if not alts:
                raise FileNotFoundError(f"no .bin in {tmp_dir}")
            bin_path = alts[0]

        # 4) read bin to numpy
        arr = np.fromfile(bin_path, dtype=np.uint8)
        if arr.size != 2 * N_BLOCKS:
            raise ValueError(f"wrong size {arr.size}")
        block_meth_total = arr.reshape(-1, 2)

        # 5) persist final bin to SSD with descriptive name
        out_bin = BIN_DIR / f"{out_prefix}.bin"
        block_meth_total.tofile(out_bin)

        # 6) Compute realized fraction from rate-adjust output? We can't read it
        # easily from stderr; record the requested fraction.
        # Compute mean-beta and per-block coverage stats for QC.
        m = block_meth_total[:, 0].astype(np.float64)
        t = block_meth_total[:, 1].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            betas = np.where(t > 0, m / t, np.nan)
        result = {
            "sample_id": out_prefix,
            "healthy_donor": get_sample_id(healthy_pat),
            "tumor_donor": get_sample_id(tumor_pat),
            "tumor_donor_label": get_donor_label(tumor_pat),
            "tissue": tissue,
            "target_fraction": float(tumor_fraction),
            "mean_beta": float(np.nanmean(betas)),
            "block_cov_frac": float(np.mean(t > 0)),
            "mean_block_depth": float(np.mean(t)),
            "ok": True,
        }
        return result
    except Exception as e:
        return {"sample_id": out_prefix, "ok": False, "error": str(e),
                "healthy_donor": get_sample_id(healthy_pat),
                "tumor_donor": get_sample_id(tumor_pat),
                "tissue": tissue, "target_fraction": float(tumor_fraction)}
    finally:
        # Clean tmp dir
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


def main():
    random.seed(SEED)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    healthy = list_healthy_pats()
    by_tissue = list_sorted_pats_by_tissue()
    log(f"healthy n={len(healthy)}")
    for t, lst in by_tissue.items():
        log(f"  tissue={t}: n_donors={len(lst)}")

    if len(healthy) < 20 or any(len(v) == 0 for v in by_tissue.values()):
        log("FATAL: insufficient inputs")
        sys.exit(1)

    # Build mixture plan
    plan = []
    for h_pat in healthy:
        h_id = get_sample_id(h_pat)
        for tissue, donors in by_tissue.items():
            for frac in FRACTIONS:
                # Choose a random tumor donor for this (h,t,f) cell
                t_pat = random.choice(donors)
                out_prefix = f"{h_id}_{tissue}_f{frac:.4f}"
                plan.append((h_pat, t_pat, frac, tissue, out_prefix, "/tmp"))
    log(f"Total mixture plan: {len(plan)}")
    # Also process unmixed healthy
    unmixed_plan = [(h_pat,) for h_pat in healthy]
    log(f"Unmixed healthy: {len(unmixed_plan)}")

    # --- Process unmixed healthy (single-threaded, very fast) ---
    unmixed_records = []
    for (h_pat,) in unmixed_plan:
        h_id = get_sample_id(h_pat)
        out_prefix = f"{h_id}_unmixed_healthy"
        # Run beta_to_blocks once
        block_arr = compute_unmixed_block_beta(h_pat, Path("/tmp"))
        out_bin = BIN_DIR / f"{out_prefix}.bin"
        block_arr.tofile(out_bin)
        m = block_arr[:, 0].astype(np.float64)
        t = block_arr[:, 1].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            betas = np.where(t > 0, m / t, np.nan)
        unmixed_records.append({
            "sample_id": out_prefix,
            "healthy_donor": h_id,
            "tumor_donor": None,
            "tumor_donor_label": None,
            "tissue": None,
            "target_fraction": 0.0,
            "mean_beta": float(np.nanmean(betas)),
            "block_cov_frac": float(np.mean(t > 0)),
            "mean_block_depth": float(np.mean(t)),
            "ok": True,
        })
    log(f"Unmixed healthy: {len(unmixed_records)} done")

    # --- Process mixtures in parallel ---
    n_workers = 6
    records = []
    n_ok = 0; n_fail = 0
    log(f"Starting parallel mixture generation with {n_workers} workers")
    last_report = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(make_one_mix, item) for item in plan]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
                records.append(rec)
                if rec.get("ok"):
                    n_ok += 1
                else:
                    n_fail += 1
                    log(f"  FAIL {rec.get('sample_id')}: {rec.get('error','?')[:150]}")
            except Exception as e:
                n_fail += 1
                log(f"  EXC: {e}")
            completed += 1
            if time.time() - last_report > 30:
                elapsed_min = (time.time() - t0) / 60
                rate = completed / max(elapsed_min, 0.01)
                remain = (len(plan) - completed) / max(rate, 0.01)
                log(f"  progress: {completed}/{len(plan)}  ok={n_ok} fail={n_fail}  "
                    f"rate={rate:.1f}/min  ETA={remain:.1f}min")
                last_report = time.time()

    elapsed_min = (time.time() - t0) / 60
    log(f"All mixtures done: ok={n_ok}/{len(plan)} fail={n_fail} elapsed={elapsed_min:.1f}min")

    # Combine unmixed + mixtures
    all_records = unmixed_records + records
    df = pd.DataFrame(all_records)
    df.to_csv(MIX_DIR / "manifest.tsv", sep="\t", index=False)
    log(f"  wrote {MIX_DIR/'manifest.tsv'}: {len(df)} rows")

    # Build block-beta matrix
    log("Assembling block-beta matrix from per-sample .bin files...")
    sample_ids = []
    beta_cols = []
    meth_cols = []
    total_cols = []
    for rec in all_records:
        if not rec.get("ok"):
            continue
        sid = rec["sample_id"]
        path = BIN_DIR / f"{sid}.bin"
        if not path.exists():
            log(f"  missing bin: {path}")
            continue
        arr = np.fromfile(path, dtype=np.uint8).reshape(-1, 2)
        m = arr[:, 0].astype(np.float64); t = arr[:, 1].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(t > 0, m / t, np.nan)
        sample_ids.append(sid)
        beta_cols.append(beta.astype(np.float32))
        meth_cols.append(arr[:, 0])
        total_cols.append(arr[:, 1])

    if not sample_ids:
        log("FATAL: no successful mixtures")
        sys.exit(1)

    beta_matrix = np.stack(beta_cols, axis=1)   # [9520, N_samples]
    meth_matrix = np.stack(meth_cols, axis=1).astype(np.uint16)
    total_matrix = np.stack(total_cols, axis=1).astype(np.uint16)
    log(f"  beta matrix: {beta_matrix.shape}")

    # Save as parquet (rows = blocks, columns = sample_ids)
    block_idx = pd.read_csv(ATLAS_BED, sep="\t", header=None,
                            names=["chrom","start","end","cpg_start","cpg_end"])
    beta_df = pd.DataFrame(beta_matrix, columns=sample_ids)
    beta_df.insert(0, "block_id", [f"{r.chrom}:{r.start}-{r.end}" for r in block_idx.itertuples()])
    beta_df.to_parquet("work/processed/GSE186458_mixtures_betas.parquet",
                       compression="zstd")
    log(f"  wrote betas parquet")

    # Save labels
    labels = pd.DataFrame(all_records)
    # Filter to only successful samples in matrix order
    labels = labels[labels["sample_id"].isin(sample_ids)].set_index("sample_id").loc[sample_ids].reset_index()
    labels["cancer_present"] = (labels["target_fraction"] > 0).astype(int)
    labels["tissue_of_origin"] = labels["tissue"].fillna("none")
    labels[["sample_id","cancer_present","tumor_fraction","tissue_of_origin",
            "healthy_donor","tumor_donor","mean_beta","block_cov_frac","mean_block_depth"]] = labels[[
            "sample_id","cancer_present","target_fraction","tissue_of_origin",
            "healthy_donor","tumor_donor","mean_beta","block_cov_frac","mean_block_depth"]]
    out_labels = labels[["sample_id","cancer_present","tumor_fraction","tissue_of_origin",
            "healthy_donor","tumor_donor","mean_beta","block_cov_frac","mean_block_depth"]]
    out_labels.to_parquet("work/processed/GSE186458_mixtures_labels.parquet",
                          compression="zstd")
    log(f"  wrote labels parquet: {len(out_labels)} rows")

    # Disk check
    bin_size = sum(p.stat().st_size for p in BIN_DIR.glob("*.bin"))
    bin_gb = bin_size / (1024**3)

    # Persist gate log
    n_total = len(all_records)
    n_ok_total = sum(1 for r in all_records if r.get("ok"))
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status = "PASS" if n_ok_total >= 800 and n_fail < 5 else "WARN" if n_ok_total >= 750 else "FAIL"
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] PIVOT_PHASE2_GSE186458 STEP2_mixtures: {status} "
                f"n_mixtures={n_ok}/{len(plan)} n_unmixed={len(unmixed_records)} "
                f"n_total={n_ok_total} fail={n_fail} bin_disk_gb={bin_gb:.2f} "
                f"elapsed_min={elapsed_min:.1f}\n")

    if status == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()

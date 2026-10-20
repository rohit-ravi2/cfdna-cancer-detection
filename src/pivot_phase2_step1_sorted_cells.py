#!/usr/bin/env python3
"""
PIVOT_PHASE2 STEP 1 — Download Loyfer atlas sorted-cell .pat.gz files for the
5 tissue-of-origin categories used in the in-silico mixture generation:

    colon    -> Colon-Ep
    liver    -> Liver-Hep   (Liver-Hepatocytes)
    lung     -> Lung-Ep-Alveo + Lung-Ep-Bron
    breast   -> Breast-Basal-Ep + Breast-Luminal-Ep
    pancreas -> Pancreas-Duct

Source: GSE186458 (the atlas itself), supplementary .pat.gz + .beta files
for the sorted cell-type GSMs.

Outputs:
    work/raw/GSE186458/pivot/sorted_cells/<GSM>_<title>.{pat.gz,beta,pat.gz.csi}
    work/raw/GSE186458/pivot/sorted_cells/manifest.json
"""
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

SERIES_MATRIX = Path("work/raw/GSE186458/GSE186458_series_matrix.txt.gz")
SORTED_DIR    = Path("work/raw/GSE186458/pivot/sorted_cells")
GATES_LOG     = Path("work/logs/gates.log")
LOG_DIR       = Path("work/logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
STDERR_LOG    = LOG_DIR / "pivot_phase2_step1.log"

# Cell-type title regexes (anchored on Sample_title prefix). Loyfer uses
# patterns like "Liver-Hepatocytes-Z..." or "Colon-Epithelium-Z..." etc.
# We match by *prefix* before the donor id (Z...).
# Verified against the GSE186458 series_matrix Sample_title field.
TISSUE_PREFIXES = {
    "colon":    ["Colon-Left-Epithelium", "Colon-Right-Epithelium"],
    "liver":    ["Liver-Hepatocytes"],
    "lung":     ["Lung-Alveolar-Epithelium", "Lung-Bronchus-Epithelium"],
    "breast":   ["Breast-Basal-Epithelium", "Breast-Luminal-Epithelium"],
    "pancreas": ["Pancreas-Duct"],
}

def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with STDERR_LOG.open("a") as f:
        f.write(line + "\n")


def parse_series_matrix() -> list[dict]:
    """Return list of {gsm, title, supplementary_files: [urls]} from series matrix."""
    titles = []
    gsms = []
    supps = []
    with gzip.open(SERIES_MATRIX, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!Sample_title\t"):
                titles = [t.strip().strip('"') for t in line.split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession\t"):
                gsms = [t.strip().strip('"') for t in line.split("\t")[1:]]
            elif line.startswith("!Sample_supplementary_file"):
                row = [t.strip().strip('"') for t in line.split("\t")[1:]]
                supps.append(row)
    # supps is a list of rows, each row indexed by sample. Combine into list per sample.
    n = len(titles)
    per_sample_supps = []
    for i in range(n):
        files = []
        for row in supps:
            if i < len(row) and row[i].lower() != "none":
                files.append(row[i])
        per_sample_supps.append(files)
    out = []
    for gsm, title, sf in zip(gsms, titles, per_sample_supps):
        out.append({"gsm": gsm, "title": title, "files": sf})
    return out


def filter_to_target_cells(samples: list[dict]) -> dict[str, list[dict]]:
    """Return {tissue: [sample, ...]} filtered to ones whose title starts with a TISSUE_PREFIXES entry."""
    out = {t: [] for t in TISSUE_PREFIXES}
    for s in samples:
        title = s["title"]
        for tissue, prefixes in TISSUE_PREFIXES.items():
            for pref in prefixes:
                # Match: "<pref>-<donor-id>" or exact prefix at start
                if title.startswith(pref + "-") or title == pref:
                    out[tissue].append({**s, "matched_prefix": pref})
                    break
            else:
                continue
            break
    return out


def download_one(url: str, out_path: Path, max_tries: int = 3, timeout: int = 600) -> bool:
    """Download url to out_path with curl. Return True on success."""
    if out_path.exists() and out_path.stat().st_size > 1024:
        return True
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, max_tries + 1):
        try:
            r = subprocess.run(
                ["curl", "-fsSL", "--retry", "2", "--retry-delay", "5",
                 "--connect-timeout", "30", "-o", str(out_path), url],
                capture_output=True, text=True, timeout=timeout,
            )
            if r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1024:
                return True
            log(f"  attempt {attempt} failed (rc={r.returncode}): {r.stderr[:200]}")
        except subprocess.TimeoutExpired:
            log(f"  attempt {attempt} timed out")
        time.sleep(2 * attempt)
    return False


def main():
    SORTED_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Parsing series matrix: {SERIES_MATRIX}")
    samples = parse_series_matrix()
    log(f"  total samples in GSE186458: {len(samples)}")

    by_tissue = filter_to_target_cells(samples)
    summary = {t: len(v) for t, v in by_tissue.items()}
    log(f"  matched by tissue: {summary}")

    # Build per-sample download list (.pat.gz + .pat.gz.csi + .beta if present)
    plan = []
    for tissue, lst in by_tissue.items():
        # Sort by GSM id for reproducibility, cap at 5 donors per cell-type-subtype
        # per prefix to stay within ~20GB
        for prefix in TISSUE_PREFIXES[tissue]:
            sub = [s for s in lst if s["matched_prefix"] == prefix]
            sub.sort(key=lambda x: x["gsm"])
            sub = sub[:5]   # cap to 5 replicates per cell-type-subtype
            for s in sub:
                pat_url = None; beta_url = None; csi_url = None
                for f in s["files"]:
                    if f.endswith(".pat.gz"):
                        pat_url = f
                    elif f.endswith(".pat.gz.csi"):
                        csi_url = f
                    elif f.endswith(".hg38.beta") or (f.endswith(".beta") and "hg38" in f):
                        beta_url = f
                if pat_url is None:
                    log(f"  WARN: {s['gsm']} {s['title']} — no .pat.gz file in supplementary")
                    continue
                plan.append({
                    "gsm": s["gsm"], "title": s["title"], "tissue": tissue,
                    "prefix": prefix, "pat_url": pat_url,
                    "csi_url": csi_url, "beta_url": beta_url,
                })
    log(f"  total download plan: {len(plan)} samples across {len(by_tissue)} tissues")

    # Download in parallel (4 threads — moderate)
    manifest = []
    fail_count = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {}
        for item in plan:
            futs[ex.submit(_download_item, item)] = item
        for fut in as_completed(futs):
            item = futs[fut]
            try:
                rec = fut.result()
                manifest.append(rec)
                if not rec["ok"]:
                    fail_count += 1
                    log(f"  FAIL: {rec['gsm']} {rec['title']}")
            except Exception as e:
                fail_count += 1
                log(f"  EXC : {item['gsm']} {item['title']}: {e}")

    # Persist manifest
    manifest.sort(key=lambda x: (x["tissue"], x["prefix"], x["gsm"]))
    with (SORTED_DIR / "manifest.json").open("w") as f:
        json.dump({"plan": plan, "manifest": manifest,
                   "summary": summary, "n_failed": fail_count}, f, indent=2)

    # Compute disk usage
    total_size = 0
    for p in SORTED_DIR.glob("*.pat.gz"):
        total_size += p.stat().st_size
    for p in SORTED_DIR.glob("*.beta"):
        total_size += p.stat().st_size
    size_gb = total_size / (1024**3)

    n_ok = sum(1 for m in manifest if m["ok"])
    n_cell_types = len(set(m["prefix"] for m in manifest if m["ok"]))
    elapsed_min = (time.time() - t0) / 60

    log(f"Step1 done: ok={n_ok}/{len(plan)}  fail={fail_count}  "
        f"cell_type_subtypes={n_cell_types}  size_gb={size_gb:.2f}  elapsed_min={elapsed_min:.1f}")

    # Append to gates log
    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    status = "PASS" if n_ok >= 14 and fail_count == 0 else "WARN" if n_ok >= 10 else "FAIL"
    with GATES_LOG.open("a") as f:
        f.write(f"[{ts}] PIVOT_PHASE2_GSE186458 STEP1_sorted_cells: {status} "
                f"n_cell_type_subtypes={n_cell_types} n_samples={n_ok} "
                f"size_gb={size_gb:.2f} fail={fail_count}\n")

    if status == "FAIL":
        sys.exit(1)


def _download_item(item: dict) -> dict:
    gsm = item["gsm"]; title = item["title"]
    pat_path = SORTED_DIR / f"{gsm}_{title}.hg38.pat.gz"
    csi_path = SORTED_DIR / f"{gsm}_{title}.hg38.pat.gz.csi"
    beta_path = SORTED_DIR / f"{gsm}_{title}.hg38.beta"

    pat_ok = download_one(item["pat_url"], pat_path)
    csi_ok = True
    if item.get("csi_url"):
        csi_ok = download_one(item["csi_url"], csi_path)
    beta_ok = True
    if item.get("beta_url"):
        beta_ok = download_one(item["beta_url"], beta_path)

    # If no csi_url, build index
    if pat_ok and not csi_path.exists():
        try:
            env = os.environ.copy()
            env["PATH"] = os.environ.get("WGBS_BIN_DIR", "") + ":" + os.environ.get("WGBS_DIR", "") + ":" + env.get("PATH", "")
            r = subprocess.run(
                [os.environ.get("WGBSTOOLS", "wgbstools"), "index", str(pat_path)],
                env=env, capture_output=True, text=True, timeout=120,
            )
            if r.returncode != 0:
                log(f"  WARN: index failed for {pat_path.name}: {r.stderr[:200]}")
        except Exception as e:
            log(f"  index exception {pat_path.name}: {e}")

    sizes = {}
    for p in [pat_path, csi_path, beta_path]:
        sizes[p.name] = p.stat().st_size if p.exists() else 0

    return {
        "gsm": gsm, "title": title, "tissue": item["tissue"], "prefix": item["prefix"],
        "pat_path": str(pat_path), "csi_path": str(csi_path), "beta_path": str(beta_path),
        "sizes_bytes": sizes,
        "ok": pat_ok and csi_ok and beta_ok,
    }


if __name__ == "__main__":
    main()

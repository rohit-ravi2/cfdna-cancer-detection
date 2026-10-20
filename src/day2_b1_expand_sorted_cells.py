#!/usr/bin/env python3
"""
Day 2 Phase B1 — Expand sorted-cell tissue + blood-lineage reference panel.

Current state: 31 tissue sorted-cell .beta files (5 cancer-relevant tissues only).
We need to additionally pull all blood-lineage sorted samples
(granulocytes, monocytes, NK, B cells, T-cell subsets, erythrocyte progenitors,
megakaryocytes if present) from GSE186458 to improve learned-deconvolution
training and resolve the Day-1 megakaryocyte question.

Download via NCBI FTP. Skip files already on disk.
"""
import os
import re
import sys
import time
import json
import urllib.request
from pathlib import Path

import requests

HDD_RAW = Path("work/raw/GSE186458/pivot/sorted_cells")
HDD_RAW.mkdir(parents=True, exist_ok=True)
LOG = Path("work/logs/day2_b1.log")

# These cell types are the blood lineages we need on top of what's there.
TARGET_KEYWORDS = [
    "Blood-Granulocytes",
    "Blood-Monocytes",
    "Blood-NK",
    "Blood-B-Cells",
    "Blood-B-Memory",
    "Blood-T",
    "Eryth-prog",
    "Erythrocyte",
    "Megakaryocyte",
    "Smooth-Musc",  # for atlas calibration
    "Endothel",
]


def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def fetch_sample_list():
    """Pull the GSE186458 quick listing and parse all sample (gsm, title) pairs."""
    url = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE186458&targ=gsm&form=text&view=quick"
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    text = r.text
    blocks = text.split("^SAMPLE = ")[1:]
    out = []
    for b in blocks:
        gsm = b.split("\n")[0].strip()
        title_m = re.search(r"!Sample_title = (.+)", b)
        title = title_m.group(1).strip() if title_m else ""
        ct = None
        for c in re.findall(r"!Sample_characteristics_ch1 = (.+)", b):
            if c.startswith("cell type:"):
                ct = c.split("cell type:")[1].strip()
                break
        out.append((gsm, title, ct))
    return out


def gsm_supplementary_files(gsm):
    """Use GEO supplementary file listing API to get .hg38.beta URLs for a GSM."""
    # GEO ftp pattern: ftp.ncbi.nlm.nih.gov/geo/samples/GSM<thousands>nnn/GSM<gsm>/suppl/
    num = gsm.replace("GSM", "")
    if len(num) < 5:
        return []
    bucket = "GSM" + num[:-3] + "nnn"
    base = f"https://ftp.ncbi.nlm.nih.gov/geo/samples/{bucket}/{gsm}/suppl/"
    try:
        r = requests.get(base, timeout=30)
        r.raise_for_status()
        files = re.findall(r'href="([^"?][^"]+)"', r.text)
        # keep .hg38.beta and skip everything else
        beta_files = [f for f in files if f.endswith(".hg38.beta")]
        return [base + f for f in beta_files]
    except Exception as e:
        log(f"  gsm_supplementary_files error for {gsm}: {e}")
        return []


def download_url(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return "already"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
            while True:
                buf = r.read(1024 * 256)
                if not buf:
                    break
                f.write(buf)
        tmp.rename(dest)
        return "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return f"err: {e}"


def main():
    log("=== Day2 B1: expand sorted-cell blood lineages ===")
    samples = fetch_sample_list()
    log(f"total samples in GSE186458: {len(samples)}")

    # Filter to blood / additional lineages we want.
    targets = []
    for gsm, title, ct in samples:
        for k in TARGET_KEYWORDS:
            if k.lower() in title.lower():
                targets.append((gsm, title, ct))
                break
    # Dedup
    seen = set()
    targets_d = []
    for g, t, c in targets:
        if g in seen:
            continue
        seen.add(g)
        targets_d.append((g, t, c))
    log(f"targets matched: {len(targets_d)}")

    # Already on disk
    existing = {p.name for p in HDD_RAW.glob("*.hg38.beta")}
    log(f"existing on disk: {len(existing)}")

    results = []
    n_downloaded = 0
    n_already = 0
    n_failed = 0
    total_bytes_pre = sum(p.stat().st_size for p in HDD_RAW.glob("*.hg38.beta"))

    for i, (gsm, title, ct) in enumerate(targets_d, 1):
        log(f"[{i}/{len(targets_d)}] {gsm}  title='{title[:60]}' ct={ct}")
        urls = gsm_supplementary_files(gsm)
        # Want only the *.hg38.beta (not pat.gz etc.)
        beta_urls = [u for u in urls if u.endswith(".hg38.beta")]
        if not beta_urls:
            log(f"  no hg38.beta files for {gsm} (probably old hg19 only)")
            results.append({"gsm": gsm, "title": title, "ct": ct, "status": "no_hg38"})
            n_failed += 1
            continue
        for u in beta_urls:
            fname = u.rsplit("/", 1)[-1]
            dest = HDD_RAW / fname
            if dest.name in existing:
                log(f"  already have {fname}")
                n_already += 1
                results.append({"gsm": gsm, "title": title, "ct": ct, "file": fname, "status": "already"})
                continue
            status = download_url(u, dest)
            if status == "ok":
                size_mb = dest.stat().st_size / 1e6
                log(f"  downloaded {fname} ({size_mb:.1f} MB)")
                n_downloaded += 1
                results.append({"gsm": gsm, "title": title, "ct": ct, "file": fname, "status": "ok", "mb": round(size_mb, 1)})
            else:
                log(f"  FAILED {fname}: {status}")
                n_failed += 1
                results.append({"gsm": gsm, "title": title, "ct": ct, "file": fname, "status": status})

    total_bytes_post = sum(p.stat().st_size for p in HDD_RAW.glob("*.hg38.beta"))
    delta_gb = (total_bytes_post - total_bytes_pre) / 1e9
    log(f"=== done.  downloaded={n_downloaded} already={n_already} failed={n_failed} +{delta_gb:.2f}GB ===")

    out_json = Path("work/processed/day2_b1_sorted_cells_download.json")
    with open(out_json, "w") as f:
        json.dump({
            "n_targets": len(targets_d),
            "n_downloaded": n_downloaded,
            "n_already": n_already,
            "n_failed": n_failed,
            "delta_gb": delta_gb,
            "results": results,
        }, f, indent=2)
    log(f"wrote {out_json}")


if __name__ == "__main__":
    sys.exit(main() or 0)

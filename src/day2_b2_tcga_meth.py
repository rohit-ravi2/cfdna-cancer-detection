#!/usr/bin/env python3
"""
Day 2 Phase B2 — TCGA methylation (450K) for 5 priority cancer types.

GDC API filter:
  cases.project.project_id IN [TCGA-COAD, TCGA-LIHC, TCGA-LUAD, TCGA-BRCA, TCGA-PAAD]
  data_type = "Methylation Beta Value"
  access = "open"

Downloads a *sample* (capped at MAX_PER_PROJECT) of methylation files per
project. The 450K beta value files are large (~120 MB) so we cap to 30 per
project for ~600 MB per project. Total budget ~3 GB.

Stores under /mnt/hdd/research/cfdna/raw/tcga_methylation/<project>/.

Each TCGA 450K file: columns = [Composite Element REF, Beta_value, ...]
where Composite Element REF = Illumina probe ID.
We then map to atlas blocks via UCSC genome coordinates from the
Illumina manifest in a separate harmonization step.
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import requests
import pandas as pd

OUT_DIR = Path("/mnt/hdd/research/cfdna/raw/tcga_methylation")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG = Path("/mnt/ssd/research/cfdna/logs/day2_b2_tcga.log")

PROJECTS = ["TCGA-COAD", "TCGA-LIHC", "TCGA-LUAD", "TCGA-BRCA", "TCGA-PAAD"]
MAX_PER_PROJECT = 30  # tumor only

GDC_FILES = "https://api.gdc.cancer.gov/files"
GDC_DATA = "https://api.gdc.cancer.gov/data"


def log(m):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def query_files_for_project(project):
    """Query GDC for Methylation Beta Value files (Illumina 450K) for a project."""
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}},
            {"op": "in", "content": {"field": "access", "value": ["open"]}},
            {"op": "in", "content": {"field": "platform", "value": ["Illumina Human Methylation 450"]}},
            {"op": "in", "content": {"field": "cases.samples.sample_type", "value": ["Primary Tumor"]}},
        ],
    }
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.case_id,cases.samples.sample_type,cases.project.project_id",
        "format": "JSON",
        "size": "200",
    }
    r = requests.get(GDC_FILES, params=params, timeout=60)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def download_file(file_id, dest):
    if dest.exists() and dest.stat().st_size > 1000:
        return "already"
    url = f"{GDC_DATA}/{file_id}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with requests.get(url, timeout=120, stream=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        tmp.rename(dest)
        return "ok"
    except Exception as e:
        if tmp.exists():
            tmp.unlink()
        return f"err: {e}"


def main():
    log("=== Day2 B2: TCGA methylation download ===")
    summary = {}
    for proj in PROJECTS:
        log(f"Querying {proj}...")
        try:
            hits = query_files_for_project(proj)
        except Exception as e:
            log(f"  query error: {e}")
            summary[proj] = {"error": str(e)}
            continue
        log(f"  {len(hits)} files available")
        proj_dir = OUT_DIR / proj
        proj_dir.mkdir(parents=True, exist_ok=True)

        # Subsample MAX_PER_PROJECT distinct case_ids
        by_case = {}
        for h in hits:
            cid = h["cases"][0]["case_id"]
            if cid not in by_case:
                by_case[cid] = h
        picks = list(by_case.values())[:MAX_PER_PROJECT]
        log(f"  picking {len(picks)} unique cases")

        ok = 0
        fail = 0
        total_mb = 0
        manifest = []
        for i, h in enumerate(picks, 1):
            fid = h["file_id"]
            fname = h["file_name"]
            dest = proj_dir / fname
            status = download_file(fid, dest)
            if status in ("ok", "already"):
                ok += 1
                size_mb = dest.stat().st_size / 1e6
                total_mb += size_mb
                manifest.append({
                    "file_id": fid,
                    "file_name": fname,
                    "case_id": h["cases"][0]["case_id"],
                    "project": proj,
                    "size_mb": round(size_mb, 1),
                    "status": status,
                })
                if i % 5 == 0:
                    log(f"  [{i}/{len(picks)}] {ok} ok, {total_mb:.0f} MB so far")
            else:
                fail += 1
                log(f"  fail {fname}: {status}")
        summary[proj] = {"ok": ok, "fail": fail, "total_mb": total_mb}
        log(f"  {proj}: ok={ok} fail={fail} total_mb={total_mb:.0f}")

        # Write project manifest
        pd.DataFrame(manifest).to_csv(proj_dir / "manifest.tsv", sep="\t", index=False)

    log(f"summary: {json.dumps(summary, indent=2)}")
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)

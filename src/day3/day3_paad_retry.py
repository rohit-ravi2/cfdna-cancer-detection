#!/usr/bin/env python3
"""Day-3: PAAD TCGA retry. Day-2 stalled (3/195 done). Resume."""
import json, time, sys
from pathlib import Path
import requests
import os

MANIFEST = Path("/mnt/hdd/research/cfdna/raw/TCGA/manifests/TCGA-PAAD_manifest.json")
RAW_DIR = Path("/mnt/hdd/research/cfdna/raw/TCGA/TCGA-PAAD")
RAW_DIR.mkdir(parents=True, exist_ok=True)
LOG = Path("/mnt/ssd/research/cfdna/logs/day3_paad.log")
N_TARGET = 40   # we only need ~30 like other types; download 40 for buffer
GDC_DATA = "https://api.gdc.cancer.gov/data"


def log(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def main():
    with open(MANIFEST) as f:
        manifest = json.load(f)
    # Only Primary Tumor
    primary = [
        e for e in manifest
        if any(s.get("sample_type") == "Primary Tumor"
               for c in e.get("cases", [])
               for s in c.get("samples", []))
    ]
    log(f"primary tumor manifest entries: {len(primary)}")

    # Already downloaded
    have = {f.name for f in RAW_DIR.glob("*.txt")}
    log(f"already on disk: {len(have)}")

    target = [e for e in primary if e["file_name"] not in have]
    target = target[:N_TARGET - len(have)] if N_TARGET > len(have) else []
    log(f"to download: {len(target)}")

    ok, fail = 0, 0
    for i, e in enumerate(target):
        url = f"{GDC_DATA}/{e['file_id']}"
        dest = RAW_DIR / e["file_name"]
        if dest.exists() and dest.stat().st_size > 1000:
            ok += 1
            continue
        tmp = dest.with_suffix(".part")
        try:
            with requests.get(url, timeout=180, stream=True) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        if chunk:
                            f.write(chunk)
            tmp.rename(dest)
            ok += 1
            if (i + 1) % 5 == 0:
                log(f"  downloaded {i+1}/{len(target)} ({dest.stat().st_size/1e6:.1f} MB) ok={ok} fail={fail}")
        except Exception as ex:
            log(f"  FAIL {e['file_name']}: {ex}")
            if tmp.exists():
                tmp.unlink()
            fail += 1
    log(f"DONE ok={ok} fail={fail} total_files={len(list(RAW_DIR.glob('*.txt')))}")


if __name__ == "__main__":
    main()

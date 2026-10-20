#!/usr/bin/env python3
"""
PIVOT_PHASE1_GSE186458 STEP 2 — Download cfDNA + matched WBC arms.

Targets: GSM6810003..GSM6810048 (23 cfDNA + 23 WBC).
Files per sample: hg38 .beta, hg38 .pat.gz, hg38 .pat.gz.csi.
Source: GEO FTP via the URL pattern derived from filelist.txt.
"""
import os
import sys
import re
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

OUT_DIR = Path("work/raw/GSE186458/pivot")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FILELIST = "work/raw/GSE186458/filelist.txt"

# GEO supplementary URL pattern:
# https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM6810nnn/GSMxxxxxxx/suppl/<filename>
GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/samples"

# Sample range
GSM_START = 6810003
GSM_END = 6810048  # inclusive

# Wanted suffixes per sample (hg38 only)
WANTED_SUFFIXES = ["hg38.beta", "hg38.pat.gz", "hg38.pat.gz.csi"]


def parse_filelist():
    """Return dict[gsm] -> dict[suffix] -> (filename, size)."""
    samples = {}
    with open(FILELIST) as f:
        for line in f:
            if not line.startswith("File\t"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            fname = parts[1]
            try:
                size = int(parts[3])
            except ValueError:
                continue
            m = re.match(r"(GSM\d+)_", fname)
            if not m:
                continue
            gsm = m.group(1)
            try:
                num = int(gsm.replace("GSM", ""))
            except ValueError:
                continue
            if not (GSM_START <= num <= GSM_END):
                continue
            for sfx in WANTED_SUFFIXES:
                if fname.endswith("." + sfx):
                    samples.setdefault(gsm, {})[sfx] = (fname, size)
                    break
    return samples


def gsm_url(gsm: str, fname: str) -> str:
    """Build GEO FTP url."""
    # GSM6810003 -> samples/GSM6810nnn/GSM6810003/suppl/<fname>
    bucket = gsm[:-3] + "nnn"
    return f"{GEO_BASE}/{bucket}/{gsm}/suppl/{fname}"


def download(url: str, dest: Path, expected_size: int, max_retries: int = 4) -> bool:
    """Resumable-ish: skip if dest exists at expected size; otherwise download."""
    if dest.exists() and dest.stat().st_size == expected_size:
        return True
    if dest.exists():
        print(f"  partial/wrong-size, removing: {dest.name} ({dest.stat().st_size} != {expected_size})")
        dest.unlink()
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            urllib.request.urlretrieve(url, tmp)
            sz = tmp.stat().st_size
            dt = time.time() - t0
            if sz != expected_size:
                print(f"  attempt {attempt}: size mismatch {sz} != {expected_size}")
                tmp.unlink()
                time.sleep(2 * attempt)
                continue
            tmp.rename(dest)
            mbps = sz / dt / 1e6 if dt > 0 else 0
            print(f"  OK {dest.name} ({sz/1e6:.1f} MB, {mbps:.1f} MB/s)")
            return True
        except (urllib.error.URLError, OSError, ConnectionError) as e:
            print(f"  attempt {attempt}: {e}")
            if tmp.exists():
                tmp.unlink()
            time.sleep(2 * attempt)
    return False


def main():
    samples = parse_filelist()
    print(f"Parsed {len(samples)} samples in range GSM{GSM_START}..GSM{GSM_END}")
    if len(samples) != 46:
        print(f"WARNING: expected 46 samples, got {len(samples)}", file=sys.stderr)

    # Total disk
    total_bytes = sum(sz for gsm in samples.values() for (_, sz) in gsm.values())
    print(f"Total expected download: {total_bytes/1e9:.2f} GB")

    manifest = {"gsms": {}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    n_ok = 0
    n_fail = 0
    for gsm in sorted(samples):
        print(f"[{gsm}]")
        manifest["gsms"][gsm] = {}
        files = samples[gsm]
        all_ok = True
        for sfx, (fname, sz) in files.items():
            dest = OUT_DIR / fname
            url = gsm_url(gsm, fname)
            ok = download(url, dest, sz)
            manifest["gsms"][gsm][sfx] = {
                "filename": fname, "size": sz, "ok": ok,
            }
            if not ok:
                all_ok = False
        if all_ok:
            n_ok += 1
        else:
            n_fail += 1
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["n_ok"] = n_ok
    manifest["n_fail"] = n_fail
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. ok={n_ok} fail={n_fail} manifest={manifest_path}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

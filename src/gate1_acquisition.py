"""
Gate 1: Data acquisition for cfDNA methylation cancer detection.

This script is resumable. Each cohort downloads to its own directory under
work/raw/. Existing non-empty files are skipped.

Downloads (in priority order):
  - Loyfer UXM atlas (block-level reference; hg19 + hg38)        DONE inline
  - GSE186458 block coords + series matrix (atlas source)        DONE inline
  - GSE122126 series matrices (Moss 2018 atlas+cfDNA, 450K/EPIC) DONE inline
  - GSE97923 processed COAD cfDNA (Zhang 2017, n=773)            DONE inline
  - GSE97932 processed LUNC cfDNA (Zhang 2017, n=654)            DONE inline
  - GSE79277 RRBS haplotype bed (Guo 2017, n=134 labeled)        DONE inline
  - GSE186007 per-sample .cov.gz (LABS 2024, n=100 cfDNA labeled)
  - TCGA methylation manifests (450K array, processed)

Run:
    source venvs/cfdna/bin/activate
    python work/src/gate1_acquisition.py
"""

import os
import sys
import json
import time
import gzip
import shutil
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


HDD = Path('work/raw')
SSD = Path('work')
LOG_DIR = SSD / 'logs'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'gate1.log'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger('gate1')


def download_file(url, dest, retries=3, chunk=1 << 20):
    """Stream-download a single file with retries. Skips if dest exists nonempty."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return ('skip', dest.stat().st_size)
    for attempt in range(retries):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + '.part')
            with open(tmp, 'wb') as f:
                for blk in r.iter_content(chunk_size=chunk):
                    if blk:
                        f.write(blk)
            tmp.rename(dest)
            return ('ok', dest.stat().st_size)
        except Exception as e:
            log.warning(f'download attempt {attempt+1}/{retries} failed for {url}: {e}')
            time.sleep(2 ** attempt)
    return ('fail', 0)


def gse186007_download_parallel(manifest_path, out_dir, max_workers=4):
    """Download per-sample .cov.gz for GSE186007 in parallel."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path) as f:
        manifest = json.load(f)
    log.info(f'GSE186007: downloading {len(manifest)} cfDNA samples')
    tasks = []
    for m in manifest:
        url = m['suppl'].replace('ftp://', 'https://')
        fname = url.rsplit('/', 1)[-1]
        dest = out_dir / fname
        tasks.append((url, dest, m))
    done = 0
    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(download_file, t[0], t[1]): t[2] for t in tasks}
        for fut in tqdm(as_completed(futs), total=len(futs), desc='GSE186007'):
            m = futs[fut]
            status, sz = fut.result()
            if status == 'fail':
                failed.append(m['gsm'])
            else:
                done += 1
    log.info(f'GSE186007 done: {done}/{len(manifest)} files, failures: {len(failed)}')
    if failed:
        log.warning(f'GSE186007 failed gsms: {failed[:10]}')
    return done, failed


def tcga_methylation_manifests(out_dir, cancer_types=None):
    """Pull TCGA 450K methylation file manifests from GDC API for priority cancers."""
    if cancer_types is None:
        cancer_types = ['TCGA-COAD', 'TCGA-LIHC', 'TCGA-BRCA', 'TCGA-LUAD',
                        'TCGA-LUSC', 'TCGA-PAAD']
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = 'https://api.gdc.cancer.gov'
    summaries = {}
    for proj in cancer_types:
        log.info(f'TCGA: querying {proj}')
        # Filters: methylation beta-value files, 450K array, project=proj
        filters = {
            "op": "and",
            "content": [
                {"op": "in", "content": {"field": "cases.project.project_id", "value": [proj]}},
                {"op": "in", "content": {"field": "data_category", "value": ["DNA Methylation"]}},
                {"op": "in", "content": {"field": "data_type", "value": ["Methylation Beta Value"]}},
                {"op": "in", "content": {"field": "platform", "value": ["Illumina Human Methylation 450"]}},
            ],
        }
        params = {
            'filters': json.dumps(filters),
            'fields': 'file_id,file_name,file_size,cases.submitter_id,cases.diagnoses.tumor_stage,cases.samples.sample_type',
            'format': 'JSON',
            'size': '2000',
        }
        try:
            r = requests.get(f'{base}/files', params=params, timeout=120)
            r.raise_for_status()
            data = r.json()
            hits = data.get('data', {}).get('hits', [])
            n = len(hits)
            total_size = sum(int(h.get('file_size', 0)) for h in hits)
            summaries[proj] = {'n_files': n, 'total_bytes': total_size}
            with open(out_dir / f'{proj}_manifest.json', 'w') as f:
                json.dump(hits, f, indent=2)
            log.info(f'  {proj}: {n} files, {total_size/1e9:.2f} GB total')
        except Exception as e:
            log.error(f'  {proj} ERROR: {e}')
            summaries[proj] = {'error': str(e)}
    with open(out_dir / 'tcga_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)
    return summaries


def tcga_methylation_download_sample(out_dir, n_per_project=5):
    """Download a small representative sample (n_per_project files) per TCGA project
    for format assessment in Gate 1. Full download is deferred to Gate 2/3."""
    base = 'https://api.gdc.cancer.gov/data'
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = out_dir / 'manifests'
    if not manifest_dir.exists():
        log.warning('no TCGA manifests, skip sample download')
        return {}
    summary = {}
    for mf in sorted(manifest_dir.glob('TCGA-*_manifest.json')):
        proj = mf.stem.replace('_manifest', '')
        with open(mf) as f:
            hits = json.load(f)
        if not hits:
            summary[proj] = 0
            continue
        sample_hits = hits[:n_per_project]
        proj_dir = out_dir / proj
        proj_dir.mkdir(exist_ok=True)
        n_ok = 0
        for h in sample_hits:
            fid = h['file_id']
            fname = h['file_name']
            url = f'{base}/{fid}'
            status, _ = download_file(url, proj_dir / fname)
            if status in ('ok', 'skip'):
                n_ok += 1
        summary[proj] = n_ok
        log.info(f'TCGA {proj} sample: {n_ok}/{len(sample_hits)} downloaded for format check')
    return summary


if __name__ == '__main__':
    log.info('Gate 1 acquisition starting')

    # Most cohorts already downloaded interactively. This script handles the
    # remaining bulk download (GSE186007 per-sample .cov.gz, TCGA manifests).

    # 1. GSE186007 per-sample download
    gse186007_dir = HDD / 'GSE186007'
    manifest_path = gse186007_dir / 'manifest.json'
    if manifest_path.exists():
        n_done, failed = gse186007_download_parallel(manifest_path, gse186007_dir / 'covs', max_workers=4)
    else:
        log.warning('GSE186007 manifest missing; skipping')

    # 2. TCGA methylation manifests + small sample downloads
    tcga_dir = HDD / 'TCGA'
    tcga_dir.mkdir(parents=True, exist_ok=True)
    tcga_manifest_dir = tcga_dir / 'manifests'
    summaries = tcga_methylation_manifests(tcga_manifest_dir)
    # Pull a few representative files for format assessment
    tcga_methylation_download_sample(tcga_dir, n_per_project=3)

    log.info('Gate 1 acquisition complete')

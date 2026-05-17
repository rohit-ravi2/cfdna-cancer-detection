#!/usr/bin/env python3
"""
REDO-1: Pre-flight probe of GSE149438 (Wu 2021) capture panel coordinates
vs Loyfer atlas blocks.

Strategy: 5 probe samples already downloaded. The targeted bisulfite panel
covers a SeqCap Epi panel of ~25.6 Mb / 67,832 DMRs (per paper). Each sample
beta file is filtered to >=4 reads, so the panel is approx the union of
positions across many samples. Five samples is enough to give a tight lower
bound on panel coverage (since deep panels saturate quickly).

Output: print intersection numbers + write coverage probe parquet.

Decision rule:
  >=10% atlas blocks covered  -> PASS
  >=5% atlas blocks covered   -> PASS_LOW (proceed with caveat)
  <5%                          -> HARD_STOP
"""

from __future__ import annotations
import os, sys, time, json, gzip, glob
import pandas as pd
import numpy as np
from pyliftover import LiftOver


PROBE_DIR = '/mnt/hdd/research/cfdna/raw/GSE149438/probe'
ATLAS_BLOCKS = '/mnt/ssd/research/cfdna/processed/_atlas_blocks.parquet'
OUT_DIR = '/mnt/ssd/research/cfdna/processed'
LOG = '/mnt/ssd/research/cfdna/logs/gate_redo1.log'


def log(msg):
    ts = time.strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(LOG, 'a') as f:
        f.write(line + '\n')


def load_sample(path):
    """Return DataFrame with chr, pos, ratio, depth."""
    df = pd.read_csv(path, sep='\t', usecols=['chr','pos','context','ratio','eff_CT_count'],
                     dtype={'chr':'string','pos':'int32','context':'category','ratio':'float32','eff_CT_count':'float32'})
    df = df[df['context'] == 'CG']
    df = df.drop(columns=['context'])
    df = df.rename(columns={'eff_CT_count':'depth'})
    return df


def main():
    open(LOG, 'w').close()
    log('REDO-1 starting')
    files = sorted(glob.glob(os.path.join(PROBE_DIR, 'GSM*.txt.gz')))
    log(f'probe samples: {len(files)}')

    atlas = pd.read_parquet(ATLAS_BLOCKS)
    log(f'atlas blocks loaded: shape={atlas.shape} columns={list(atlas.columns)}')

    # Build per-sample row counts & coordinate sets (hg19 native)
    per_sample = {}
    union_positions = set()
    for f in files:
        gsm = os.path.basename(f).split('_')[0]
        df = load_sample(f)
        per_sample[gsm] = {'n_cpgs': len(df),
                           'mean_depth': float(df['depth'].mean()),
                           'median_depth': float(df['depth'].median())}
        union_positions.update(zip(df['chr'].tolist(), df['pos'].tolist()))
        log(f'  {gsm}: n={len(df)} mean_depth={df["depth"].mean():.1f}')
    log(f'union of unique (chr,pos) across {len(files)} samples: {len(union_positions):,}')

    # Liftover union hg19 -> hg38
    lo = LiftOver('hg19', 'hg38')
    union_hg38 = {}
    miss = 0
    for chrom, pos in union_positions:
        # BSMAP pos is 1-based; pyliftover expects 0-based. Subtract 1.
        res = lo.convert_coordinate(chrom, pos - 1)
        if res:
            c2, p2, _, _ = res[0]
            union_hg38.setdefault(c2, []).append(p2)
        else:
            miss += 1
    n_lifted = sum(len(v) for v in union_hg38.values())
    log(f'liftover hg19->hg38: lifted={n_lifted:,} missed={miss:,} ({miss/(n_lifted+miss):.2%})')

    # Sort hg38 positions per chr
    union_hg38 = {c: np.array(sorted(set(p)), dtype=np.int64) for c, p in union_hg38.items()}
    log(f'unique hg38 positions: {sum(len(v) for v in union_hg38.values()):,}')

    # Intersect with atlas blocks. For each atlas block (chr, start, end),
    # any position in [start, end) counts as covered.
    atlas['n_panel_positions'] = 0
    covered_mask = np.zeros(len(atlas), dtype=bool)
    for chrom, sub in atlas.groupby('chr'):
        positions = union_hg38.get(chrom, None)
        if positions is None or len(positions) == 0:
            continue
        starts = sub['start'].to_numpy()
        ends = sub['end'].to_numpy()
        idx_arr = sub.index.to_numpy()
        # vectorized: for each block, count positions in [start, end)
        for i, (s, e) in enumerate(zip(starts, ends)):
            lo_i = np.searchsorted(positions, s, side='left')
            hi_i = np.searchsorted(positions, e, side='left')
            n = hi_i - lo_i
            if n > 0:
                covered_mask[idx_arr[i]] = True
                atlas.at[idx_arr[i], 'n_panel_positions'] = int(n)

    atlas['covered'] = covered_mask
    n_covered = int(covered_mask.sum())
    coverage = n_covered / len(atlas)
    log(f'atlas blocks covered by panel: {n_covered}/{len(atlas)} = {coverage:.2%}')

    # Save augmented atlas
    atlas.to_parquet(os.path.join(OUT_DIR, '_atlas_blocks_GSE149438_coverage.parquet'))

    # Per-target breakdown
    by_target = atlas.groupby('target').agg(
        n_blocks=('block_id','count'),
        n_covered=('covered','sum'),
    ).reset_index()
    by_target['frac_covered'] = by_target['n_covered'] / by_target['n_blocks']
    log('coverage by atlas target cell type:')
    for _, r in by_target.sort_values('frac_covered', ascending=False).iterrows():
        log(f'  {r["target"]:>18s}: {int(r["n_covered"]):>5d} / {int(r["n_blocks"]):>5d}  ({r["frac_covered"]:.1%})')

    # Decision
    if coverage >= 0.10:
        decision = 'PASS'
    elif coverage >= 0.05:
        decision = 'PASS_LOW'
    else:
        decision = 'HARD_STOP'
    log(f'REDO-1 decision: {decision} coverage={coverage:.2%}')

    # Write a summary json
    summary = {
        'n_probe_samples': len(files),
        'union_positions_hg19': len(union_positions),
        'union_positions_hg38': int(sum(len(v) for v in union_hg38.values())),
        'liftover_miss_rate': miss / (n_lifted + miss) if (n_lifted+miss) else 0.0,
        'atlas_blocks': int(len(atlas)),
        'atlas_blocks_covered': int(n_covered),
        'coverage_frac': float(coverage),
        'per_sample': per_sample,
        'decision': decision,
        'by_target': by_target.sort_values('frac_covered', ascending=False).to_dict('records'),
    }
    with open(os.path.join(OUT_DIR, '_GSE149438_panel_probe.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    log(f'summary written to {OUT_DIR}/_GSE149438_panel_probe.json')

    return decision, coverage


if __name__ == '__main__':
    decision, cov = main()
    print(f'\nFINAL REDO-1: {decision}  coverage={cov:.2%}')
    sys.exit(0 if decision != 'HARD_STOP' else 1)

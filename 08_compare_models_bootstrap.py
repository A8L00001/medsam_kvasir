#!/usr/bin/env python3
"""Image-level paired bootstrap for any two MedSAM 2080Ti experiments.

Reads existing outputs/evaluations/{EXPERIMENT}/{split}/noise_XXpct/per_image.csv.
No GPU needed; will not alter checkpoints or preexisting analysis files.

Usage:
 python 08_compare_models_bootstrap.py --work outputs --split val \
        --baseline E3 --candidate E3C --bootstrap 5000
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ('dice', 'iou', 'boundary_f1_2px', 'hd95_pixels')
LEVELS = (0, 5, 10, 20)


def load_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'找不到评估文件: {path}')
    with path.open('r', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'空 CSV：{path}')
    result = {}
    for r in rows:
        for name in ('stem', 'trial', *METRICS):
            if name not in r:
                raise ValueError(f'CSV 缺少 {name}: {path}')
        key = (r['stem'], int(r['trial']))
        if key in result:
            raise ValueError(f'重复 (stem, trial)={key}：{path}')
        vals = {name: float(r[name]) for name in METRICS}
        if not np.isfinite(list(vals.values())).all():
            raise ValueError(f'存在 NaN/Inf: {path}, {key}')
        result[key] = vals
    return result


def save_csv(path: Path, rows):
    if not rows:
        raise ValueError('No CSV rows')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def paired_analysis(work: Path, split: str, base: str, candidate: str,
                    n_boot: int, seed: int):
    outdir = work / 'reports' / split
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = f'paired_{candidate}_vs_{base}'
    detail = []
    summary = []
    n_ref = None
    trials_ref = None
    identities_ref = None
    rng = np.random.default_rng(seed)

    for level in LEVELS:
        directory = f'noise_{level:02d}pct'
        baseline = load_csv(work/'evaluations'/base/split/directory/'per_image.csv')
        improved = load_csv(work/'evaluations'/candidate/split/directory/'per_image.csv')
        if baseline.keys() != improved.keys():
            raise ValueError(f'noise={level}%：两组 (stem,trial) 不完全匹配，无法做配对统计')
        trials_by_image = defaultdict(list)
        for stem, trial in baseline:
            trials_by_image[stem].append(trial)
        image_names = sorted(trials_by_image)
        counts = {len(ids) for ids in trials_by_image.values()}
        if len(counts) != 1:
            raise ValueError(f'noise={level}%：部分图像有缺失的 trial')
        if n_ref is None:
            n_ref = len(image_names)
            trials_ref = next(iter(counts))
            identities_ref = image_names
        if image_names != identities_ref or next(iter(counts)) != trials_ref:
            raise ValueError('各个扰动档位的图像或重复次数不一致')

        metric_delta = {m: np.zeros(n_ref, dtype=np.float64) for m in METRICS}
        for i, stem in enumerate(image_names):
            tlist = sorted(trials_by_image[stem])
            row = {'noise_pct':level, 'stem':stem, 'trials':len(tlist)}
            for m in METRICS:
                m_base = float(np.mean([baseline[(stem, t)][m] for t in tlist]))
                m_cand = float(np.mean([improved[(stem, t)][m] for t in tlist]))
                delta = m_cand - m_base
                row[f'{base}_{m}'] = m_base
                row[f'{candidate}_{m}'] = m_cand
                row[f'delta_{m}'] = delta
                metric_delta[m][i] = delta
            detail.append(row)

        # Resample IMAGE identities jointly for all metrics to preserve image-level dependence.
        drawn = rng.integers(0, n_ref, size=(n_boot, n_ref))
        for m in METRICS:
            x = metric_delta[m]
            ci = np.percentile(x[drawn].mean(axis=1), [2.5,97.5])
            summary.append({
                'noise_pct':level,
                'metric':m,
                'baseline':base,
                'candidate':candidate,
                'n_images':n_ref,
                'trials_per_image':trials_ref,
                'mean_delta_candidate_minus_baseline':float(x.mean()),
                'ci95_low':float(ci[0]),
                'ci95_high':float(ci[1]),
                'positive_images':int(np.sum(x > 1e-10)),
                'negative_images':int(np.sum(x < -1e-10)),
                'unchanged_images':int(np.sum(np.abs(x) <= 1e-10)),
            })

    save_csv(outdir/f'{prefix}_bootstrap.csv', summary)
    save_csv(outdir/f'{prefix}_by_image.csv', detail)
    hvals = [r['delta_dice'] for r in detail if r['noise_pct'] == 20]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(hvals, bins=20)
    ax.axvline(0, color='black', linestyle='--')
    ax.set(xlabel=f'20% perturbation: Dice({candidate}) - Dice({base})',
           ylabel='Number of images', title='Per-image paired Dice difference')
    fig.tight_layout()
    fig.savefig(outdir/f'{prefix}_20pct_hist.png', dpi=150)
    plt.close(fig)
    print(f'PAIRED ANALYSIS PASS: {candidate} minus {base} | n_images={n_ref} | trials={trials_ref} | bootstrap={n_boot}')
    print('Box noise   Dice difference [95% image bootstrap CI]   improved/worsened')
    for r in summary:
        if r['metric']=='dice':
            print(f"{r['noise_pct']:>3}%      {r['mean_delta_candidate_minus_baseline']:+.5f} "
                  f"[{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}]   "
                  f"{r['positive_images']}/{r['negative_images']}")
    print('OTHER METRICS (candidate minus baseline; HD95 negative is better):')
    for level in LEVELS:
        for name in ('boundary_f1_2px','hd95_pixels'):
            rr = next(r for r in summary if r['noise_pct']==level and r['metric']==name)
            print(f"  {level:>2}% {name}: {rr['mean_delta_candidate_minus_baseline']:+.5f} "
                  f"[{rr['ci95_low']:+.5f}, {rr['ci95_high']:+.5f}]")
    print('FILES:')
    for suffix in ('_bootstrap.csv','_by_image.csv','_20pct_hist.png'):
        print(' ',outdir/f'{prefix}{suffix}')
    print('注意：图像级 Bootstrap 衡量固定训练模型和固定扰动条件下的图像抽样不确定性；不能替代跨训练种子验证。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', default='outputs')
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--baseline', default='E3')
    parser.add_argument('--candidate', default='E3C')
    parser.add_argument('--bootstrap', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.baseline == args.candidate:
        parser.error('baseline and candidate must differ')
    if args.bootstrap < 100:
        parser.error('bootstrap must be >= 100')
    paired_analysis(Path(args.work), args.split, args.baseline,
                    args.candidate, args.bootstrap, args.seed)


if __name__=='__main__':
    main()

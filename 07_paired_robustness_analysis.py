#!/usr/bin/env python3
"""Image-level paired bootstrap analysis for existing MedSAM E1/E3 noise evaluations.

No GPU required and never reads/writes any checkpoints. Retains within-image
correlation from repeated perturbation trials by reducing to one observation
per image before resampling images.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ('dice','iou','boundary_f1_2px','hd95_pixels')
LEVELS = (0, 5, 10, 20)


def read_file(path):
    with Path(path).open('r', encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'无数据：{path}')
    out = {}
    for row in rows:
        key = (row['stem'], int(row['trial']))
        if key in out:
            raise RuntimeError(f'{path} 出现重复 (stem,trial)：{key}')
        out[key] = {m: float(row[m]) for m in METRICS}
    return out


def write_rows(path, rows):
    if not rows: return
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(work, split, n_boot, seed):
    root = Path(work)
    outdir=root/'reports'/split
    outdir.mkdir(parents=True,exist_ok=True)
    summary=[]
    detail=[]
    rng = np.random.default_rng(seed)
    n_images_ref=None
    for pct in LEVELS:
        folder=f'noise_{pct:02d}pct'
        p1=root/'evaluations'/'E1'/split/folder/'per_image.csv'
        p3=root/'evaluations'/'E3'/split/folder/'per_image.csv'
        a,b=read_file(p1),read_file(p3)
        if set(a)!=set(b):
            raise RuntimeError(f'{pct}%：E1 与 E3 的图像/扰动重复编号不一致，不能配对')
        stem_map=defaultdict(list)
        for stem,trial in a:
            stem_map[stem].append(trial)
        stems=sorted(stem_map)
        n_images=len(stems)
        n_preds=len(a)
        if n_images_ref is None:n_images_ref=n_images
        if n_images!=n_images_ref:
            raise RuntimeError('各档扰动的图像数量不一致')
        trials={len(v) for v in stem_map.values()}
        if len(trials)!=1:
            raise RuntimeError(f'{pct}%：各图的重复次数不一致：{trials}')
        image_values={metric:[] for metric in METRICS}
        per_stem={}
        for stem in stems:
            ts=sorted(stem_map[stem])
            row={'noise_pct':pct, 'stem':stem, 'trials':len(ts)}
            for metric in METRICS:
                av=np.mean([a[(stem,t)][metric] for t in ts])
                bv=np.mean([b[(stem,t)][metric] for t in ts])
                delta=float(bv-av)
                image_values[metric].append(delta)
                row[f'E1_{metric}']=float(av)
                row[f'E3_{metric}']=float(bv)
                row[f'delta_{metric}']=delta
            per_stem[stem]=row
            detail.append(row)
        for metric in METRICS:
            vals=np.array(image_values[metric], dtype=np.float64)
            # Resample independent IMAGE identities, not 3*n correlated predictions.
            draws=rng.integers(0,n_images,size=(n_boot,n_images))
            boot=vals[draws].mean(axis=1)
            low,high=np.percentile(boot,[2.5,97.5])
            summary.append({'noise_pct':pct,'metric':metric,'n_images':n_images,
                'paired_predictions':n_preds,'repeats_per_image':len(next(iter(stem_map.values()))),
                'mean_delta_E3_minus_E1':float(vals.mean()),
                'ci95_low':float(low),'ci95_high':float(high),
                'images_positive_delta':int((vals>1e-10).sum()),
                'images_negative_delta':int((vals< -1e-10).sum()),
                'images_unchanged':int((np.abs(vals)<=1e-10).sum())})
    sum_path=outdir/'paired_E3_vs_E1_bootstrap.csv'
    detail_path=outdir/'paired_E3_vs_E1_by_image.csv'
    write_rows(sum_path,summary)
    write_rows(detail_path,detail)
    plot_vals=[r['delta_dice'] for r in detail if r['noise_pct']==20]
    fig,ax=plt.subplots(figsize=(8,5))
    ax.hist(plot_vals,bins=20)
    ax.axvline(0,linestyle='--',color='black')
    ax.set(xlabel='20% perturbation: per-image Dice(E3) - Dice(E1)',
           ylabel='Number of images',title='Paired robustness improvement (image-level)')
    fig.tight_layout()
    image_path=outdir/'paired_E3_vs_E1_20pct_hist.png'
    fig.savefig(image_path,dpi=160)
    plt.close(fig)
    print(f'Paired analysis DONE: n_images={n_images_ref}, bootstrap={n_boot}')
    print('Noise  |  Dice E3-E1 [95% image bootstrap CI] | # improved / # worsened')
    for r in summary:
        if r['metric']=='dice':
            print(f"{r['noise_pct']:>3}%   | {r['mean_delta_E3_minus_E1']:+.5f} [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] | {r['images_positive_delta']}/{r['images_negative_delta']}")
    print('FILES:',sum_path,detail_path,image_path,sep='\n  ')
    print('说明：置信区间反映在当前固定模型及固定扰动采样下，对图像集合重采样的不确定性；不等于跨训练随机种子的稳定性验证。')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--work',default='outputs')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--bootstrap',type=int,default=5000)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args()
    if a.bootstrap<100:raise ValueError('--bootstrap 最少 100')
    run(a.work,a.split,a.bootstrap,a.seed)

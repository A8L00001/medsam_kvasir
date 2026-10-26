#!/usr/bin/env python3
"""Image-level paired bootstrap for fixed YOLO-box E1 vs E3C segmentation.

Reads Step 11 outputs/auto_yolo/<split>/conf_<conf*1000>/per_image.csv.
Does not evaluate checkpoints, tune settings, or modify existing evaluation files.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

METRICS = ('dice', 'iou', 'boundary_f1_2px', 'hd95_pixels')


def load_pairs(filename: Path):
    paired = {}
    with filename.open('r', newline='', encoding='utf-8-sig') as fp:
        reader = csv.DictReader(fp)
        required = {'stem', 'experiment', 'n_boxes', *METRICS}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f'CSV missing fields: {sorted(missing)}; found {reader.fieldnames}')
        for raw in reader:
            stem, exp = raw['stem'], raw['experiment']
            if exp not in ('E1', 'E3C'):
                continue
            row = {'n_boxes': int(raw['n_boxes'])}
            row.update({m: float(raw[m]) for m in METRICS})
            if not all(np.isfinite(row[m]) for m in METRICS):
                raise ValueError(f'Nonfinite metric: {stem}/{exp}')
            if exp in paired.setdefault(stem, {}):
                raise ValueError(f'Duplicate record for {stem}/{exp}')
            paired[stem][exp] = row
    result = []
    for stem, experiments in sorted(paired.items()):
        if set(experiments) != {'E1', 'E3C'}:
            raise ValueError(f'Unpaired models at image {stem}: {list(experiments)}')
        a, b = experiments['E1'], experiments['E3C']
        if a['n_boxes'] != b['n_boxes']:
            raise ValueError(f'Mismatched YOLO detections at {stem}')
        item = {'stem': stem, 'n_boxes': a['n_boxes']}
        for m in METRICS:
            item[f'E1_{m}'] = a[m]
            item[f'E3C_{m}'] = b[m]
            item[f'delta_{m}'] = b[m] - a[m]
        result.append(item)
    if not result:
        raise ValueError('No paired E1 and E3C rows found')
    return result


def bootstrap(values, rng, samples):
    v = np.asarray(values, dtype=np.float64)
    idx = rng.integers(0, len(v), size=(samples, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(v.mean()), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', default='outputs')
    ap.add_argument('--split', choices=('val', 'test'), default='test')
    ap.add_argument('--conf', type=float, default=0.10)
    ap.add_argument('--bootstrap', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    if a.bootstrap < 100:
        ap.error('--bootstrap must be >= 100')
    base = Path(a.work) / 'auto_yolo' / a.split / f'conf_{round(a.conf*1000):03d}'
    source = base / 'per_image.csv'
    if not source.exists():
        raise FileNotFoundError(f'Expected Step 11 per-image file: {source}')
    items = load_pairs(source)
    rng = np.random.default_rng(a.seed)
    lines = [f'Automatic YOLO -> MedSAM paired analysis | split={a.split}, conf={a.conf}',
             f'Number of independent images: {len(items)} | bootstrap resamples: {a.bootstrap}',
             'Fixed YOLO detections are identical for E1 and E3C.',
             'Positive differences favor E3C except HD95, where negative is better.', '']
    for group, subset in [('ALL', items), ('SINGLE BOX', [x for x in items if x['n_boxes']==1]),
                          ('MULTIPLE BOXES', [x for x in items if x['n_boxes']>1])]:
        if not subset:
            continue
        lines.append(f'{group}: n={len(subset)}')
        for m in METRICS:
            d = np.asarray([x[f'delta_{m}'] for x in subset])
            v1 = np.mean([x[f'E1_{m}'] for x in subset])
            v3 = np.mean([x[f'E3C_{m}'] for x in subset])
            delta, lo, hi = bootstrap(d, rng, a.bootstrap)
            imp = int(np.sum(d < 0)) if m == 'hd95_pixels' else int(np.sum(d > 0))
            worse = int(np.sum(d > 0)) if m == 'hd95_pixels' else int(np.sum(d < 0))
            unit = 'px' if m == 'hd95_pixels' else 'absolute fraction'
            lines.append(f'  {m}: E1={v1:.6f}, E3C={v3:.6f}, '
                         f'change={delta:+.6f}, 95% image-bootstrap CI=[{lo:+.6f}, {hi:+.6f}] '
                         f'({unit}); improved/worsened/tied={imp}/{worse}/{len(d)-imp-worse}')
        lines.append('')
    out_csv = base / 'paired_auto_E3C_vs_E1_by_image.csv'
    with out_csv.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=items[0].keys())
        writer.writeheader()
        writer.writerows(items)
    lines.extend(['Five lowest E3C Dice examples (for qualitative failure analysis):'])
    for x in sorted(items, key=lambda r: r['E3C_dice'])[:5]:
        lines.append(f"  {x['stem']} | boxes={x['n_boxes']} | E1={x['E1_dice']:.4f} | "
                     f"E3C={x['E3C_dice']:.4f} | delta={x['delta_dice']:+.4f}")
    report = base / 'paired_auto_E3C_vs_E1.txt'
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))
    print(f'FILES:\n  {report}\n  {out_csv}')


if __name__ == '__main__':
    main()

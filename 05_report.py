"""Read saved summaries and training logs, plot validation Dice/robustness. No invented results."""
from pathlib import Path
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.utils import config,load_json,save_csv,work_dir


def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['val','test'],default='val')
    args=p.parse_args();cfg=config();w=work_dir(cfg)
    folder=w/'reports'/args.split;folder.mkdir(parents=True,exist_ok=True)
    rows=[];fig,ax=plt.subplots(figsize=(8,5));has=False
    for exp in ('E0','E1','E2','E3','E3C','E4'):
        xs=[];ys=[]
        for pct in (0,5,10,20):
            file=w/'evaluations'/exp/args.split/f'noise_{pct:02}pct'/'summary.json'
            if file.is_file():
                r=load_json(file);xs.append(pct);ys.append(r['mean_dice'])
                rows.append({'exp':exp,'noise_pct':pct,'mean_dice':r['mean_dice'],
                             'mean_iou':r['mean_iou'],'n_trials':r['trials']})
        if xs:ax.plot(xs,ys,marker='o',label=exp);has=True
    if has:
        ax.set(xlabel='Bounding-box side perturbation (%)',ylabel='Mean Dice',
               title=f'MedSAM box robustness / {args.split}')
        ax.legend();ax.grid(alpha=.3);fig.tight_layout()
        fig.savefig(folder/'robustness.png',dpi=150)
        save_csv(folder/'metrics.csv',rows)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5));has=False
    for exp in ('E1','E2','E3','E3C','E4'):
        file=w/'runs'/exp/'history.csv'
        if file.is_file():
            with file.open(encoding='utf-8') as f:dat=list(csv.DictReader(f))
            ax.plot([int(r['epoch']) for r in dat],[float(r['val_dice']) for r in dat],label=exp)
            has=True
    if has:
        ax.set(xlabel='Epoch',ylabel='Validation Dice',title='MedSAM training curves')
        ax.legend();fig.tight_layout();fig.savefig(folder/'val_curves.png',dpi=150)
    plt.close(fig)
    print('REPORT DIRECTORY:',folder)
    if not rows:print('没有评估数据；请先运行 03_evaluate.py 或 06_robustness.py')

if __name__=='__main__':main()

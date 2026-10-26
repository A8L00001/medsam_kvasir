"""Step 03: evaluate pretrained E0 or fine-tuned E1/E2/E3/E3C/E4 with one Oracle Box protocol."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from core.seg import check_features,model,read_rgb,read_gt
from core.utils import config,require_gpu,work_dir,save_json,save_csv
from core.evaluate import evaluate


def pictures(stems,img_map,gt_map,preds,path):
    if not preds:return
    chosen=[s for s in stems if s in preds]
    fig,axes=plt.subplots(len(chosen),3,figsize=(12,3*len(chosen)),squeeze=False)
    for i,s in enumerate(chosen):
        rgb=read_rgb(img_map[s]);gt=read_gt(gt_map[s]);pred=preds[s]
        axes[i,0].imshow(rgb);axes[i,0].set_title(s+' / RGB')
        axes[i,1].imshow(gt,cmap='gray',vmin=0,vmax=1);axes[i,1].set_title('Ground truth')
        axes[i,2].imshow(rgb)
        axes[i,2].imshow(np.ma.masked_where(pred==0,pred),cmap='spring',alpha=.5,vmin=0,vmax=1)
        axes[i,2].set_title('Prediction overlay')
        for ax in axes[i]:ax.axis('off')
    fig.tight_layout();fig.savefig(path,dpi=125);plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--exp',choices=['E0','E1','E2','E3','E3C','E4'],default='E0')
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--final-test',action='store_true',help='测试集只能在模型和参数全部选定后使用')
    p.add_argument('--noise',type=float,default=0.,help='Box 四条边相对于宽高的扰动比例')
    p.add_argument('--trials',type=int,default=1)
    a=p.parse_args()
    if a.split=='test' and not a.final_test:
        raise RuntimeError('测试集已锁定，必须在验证集选定方案后显式增加 --final-test')
    if not 0<=a.noise<=.3 or a.trials<1:
        raise ValueError('noise 必须在 0..0.3，trials 至少为 1')
    cfg=config();device=require_gpu();w=work_dir(cfg)
    m,images,gts=check_features(cfg,(a.split,))
    sam=model(cfg['model']['medsam_repo'],cfg['model']['checkpoint'])
    sam.image_encoder=nn.Identity()
    sam.prompt_encoder.to(device).eval();sam.mask_decoder.to(device).eval()
    if a.exp!='E0':
        checkpoint=w/'runs'/a.exp/'best.pth'
        if not checkpoint.is_file():raise FileNotFoundError(f'{checkpoint} 不存在，请先训练 {a.exp}')
        state=torch.load(checkpoint,map_location='cpu')
        sam.mask_decoder.load_state_dict(state['decoder'],strict=True)
    noise_key=f'noise_{int(round(a.noise*100)):02}pct'
    folder=w/'evaluations'/a.exp/a.split/noise_key
    folder.mkdir(parents=True,exist_ok=True)
    report,rows,preds=evaluate(sam,cfg,m[a.split],gts,split=a.split,
                        noise=a.noise,trials=a.trials,details=True,keep_preds=6,
                        amp=cfg['training']['amp'])
    report['experiment']=a.exp;report['split']=a.split
    save_json(folder/'summary.json',report);save_csv(folder/'per_image.csv',rows)
    pictures(m[a.split],images,gts,preds,folder/'examples.png')
    print('\n===== EVALUATION RESULT =====')
    for k,v in report.items():print(f'{k}: {v}')
    print('FILES:',folder)

if __name__=='__main__':main()

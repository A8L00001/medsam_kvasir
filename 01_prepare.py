"""Step 01: exhaustive paired-image verification and immutable 80/10/10 split."""
from pathlib import Path
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tqdm import tqdm

from core.seg import SIDE,folder_map,read_rgb,read_gt,gt_box
from core.utils import config,save_json,load_json,save_csv,sha,work_dir


def run(cfg):
    w=work_dir(cfg)
    im_dir=cfg['data']['images_dir'];gt_dir=cfg['data']['masks_dir']
    if not im_dir or not gt_dir:
        raise RuntimeError('config.yaml 的 images_dir/masks_dir 未设置：先运行 00_preflight.py')
    images=folder_map(im_dir);gts=folder_map(gt_dir)
    if images.keys()!=gts.keys():
        raise RuntimeError(f'Images 与 Masks 不配对：仅图像={list(images.keys()-gts.keys())[:5]}；仅标注={list(gts.keys()-images.keys())[:5]}')
    names=sorted(images)
    if len(names)<10:raise RuntimeError(f'配对样本仅 {len(names)} 张；检查是否使用了完整 Kvasir-SEG')
    seed=cfg['project']['seed']
    rng=random.Random(seed);rng.shuffle(names)
    ntrain=int(len(names)*.8);nval=int(len(names)*.1)
    manifest={'images_dir':str(Path(im_dir).resolve()),'masks_dir':str(Path(gt_dir).resolve()),
              'seed':seed,'protocol':'image-level random 80/10/10 (not patient-level)',
              'train':names[:ntrain],'val':names[ntrain:ntrain+nval],
              'test':names[ntrain+nval:]}
    manifest['split_hash']=sha('\n'.join(f'{k}:{s}' for k in ('train','val','test') for s in manifest[k]))
    saved=w/'splits.json'
    if saved.exists():
        old=load_json(saved)
        if old!=manifest:
            raise RuntimeError('已有 splits.json 与当前路径、随机种子或文件列表不符！不允许覆盖。')
        print('已存在相同 splits.json：沿用旧划分，不覆盖。')
    rows=[]; mask_areas=[]
    for spl in ('train','val','test'):
        for name in tqdm(manifest[spl],desc=f'Validating {spl}'):
            rgb=read_rgb(images[name]);gt=read_gt(gts[name])
            if gt.shape!=rgb.shape[:2]:raise RuntimeError(f'{name} 图像/Mask 形状不同：{rgb.shape} vs {gt.shape}')
            if gt.sum()==0:raise RuntimeError(f'{name} Mask 完全空，需要检查数据源')
            b=gt_box(gt)
            mask_areas.append(float(gt.mean()))
            rows.append({'id':name,'split':spl,'height':int(gt.shape[0]),'width':int(gt.shape[1]),
                         'foreground_ratio':round(float(gt.mean()),6),
                         'box_x0':int(b[0]),'box_y0':int(b[1]),'box_x1':int(b[2]),'box_y1':int(b[3])})
    save_json(saved,manifest)
    save_csv(w/'01_data'/'statistics.csv',rows)
    picks=random.Random(seed+23).sample(sorted(images), min(8,len(images)))
    fig,ax=plt.subplots(len(picks),2,figsize=(11,len(picks)*3.1),squeeze=False)
    for i,name in enumerate(picks):
        rgb=read_rgb(images[name]);gt=read_gt(gts[name]);h,ww=gt.shape;b=gt_box(gt)
        x0,y0,x1,y1=b*[ww/SIDE,h/SIDE,ww/SIDE,h/SIDE]
        ax[i,0].imshow(rgb);ax[i,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='lime',linewidth=2))
        ax[i,0].set_title(name+' / Oracle Box')
        ax[i,1].imshow(gt,cmap='gray',vmin=0,vmax=1);ax[i,1].set_title('GT mask')
        for a in ax[i]:a.axis('off')
    fig.tight_layout();fig.savefig(w/'01_data'/'samples.png',dpi=135);plt.close(fig)
    summary={'status':'PASS','total_pairs':len(names),
             'split_counts':{k:len(manifest[k]) for k in ('train','val','test')},
             'split_hash':manifest['split_hash'],'mask_area_mean':float(np.mean(mask_areas)),
             'mask_area_min':float(np.min(mask_areas)),'mask_area_max':float(np.max(mask_areas)),
             'oracle_box_warning':'基于真实掩码生成提示框。最终自动推理另做 YOLO 检测框。',
             'samples_figure':str(w/'01_data'/'samples.png')}
    save_json(w/'01_data'/'report.json',summary)
    print('\n===== STEP 01 RESULT =====')
    print('STATUS:',summary['status'])
    print('Pairs:',summary['total_pairs'])
    print('Split:',summary['split_counts'])
    print('Mean foreground fraction:',round(summary['mask_area_mean'],4))
    print('REPORT:',w/'01_data'/'report.json')
    print('VISUAL CHECK:',w/'01_data'/'samples.png')
    print('NEXT: 手工确认绿色 Box 是否框住息肉，再进入 Step 02')

if __name__=='__main__':run(config())

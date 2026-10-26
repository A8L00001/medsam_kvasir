#!/usr/bin/env python3
"""Step 11: fixed YOLOv8 detector -> MedSAM E1/E3C automatic box evaluation.

Do not use ground-truth boxes at inference time. GT masks are for metric computation only.
Select confidence threshold on VAL only. Test is protected by --final-test.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn as nn

from core.seg import (SIDE, check_features, decoder_forward, feature_file,
                      metrics, boundary_f1, hd95, model as make_medsam,
                      output_mask, read_gt, read_rgb)
from core.utils import config, require_gpu, save_csv, save_json, work_dir


def original_to_medsam_box(xyxy, width, height):
    """YOLO returns original-image xyxy. MedSAM preprocess does direct square resize."""
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    v = np.asarray(xyxy, dtype=np.float32)
    if v.shape != (4,) or not np.isfinite(v).all():
        raise ValueError('Invalid xyxy')
    x1, y1, x2, y2 = v.tolist()
    x1, x2 = sorted([max(0., min(float(width), x1)), max(0., min(float(width), x2))])
    y1, y2 = sorted([max(0., min(float(height), y1)), max(0., min(float(height), y2))])
    if x2-x1 < 1.0 or y2-y1 < 1.0:
        return None
    return np.asarray([x1*SIDE/width,y1*SIDE/height,
                       x2*SIDE/width,y2*SIDE/height], dtype=np.float32)


def build_decoders(sam, work, device):
    '''Load each decoder ONCE, avoiding repeated checkpoint I/O across images.'''
    decoders={}
    for exp in ('E1','E3C'):
        filename = work/'runs'/exp/'best.pth'
        if not filename.is_file():
            raise FileNotFoundError(f'Missing trained MedSAM decoder: {filename}')
        state = torch.load(filename,map_location='cpu')
        decoder = copy.deepcopy(sam.mask_decoder)
        decoder.load_state_dict(state['decoder'],strict=True)
        decoder.to(device).eval()
        decoders[exp]=decoder
    return decoders


@torch.inference_mode()
def segment_for_boxes(sam, feature, boxes_xyxy, shape, use_amp=True):
    """Run shared MedSAM decoder once per box; union binary masks. No detection => empty mask."""
    height, width = shape
    merged = np.zeros((height,width),dtype=np.uint8)
    if not boxes_xyxy:
        return merged
    device = feature.device
    for xyxy in boxes_xyxy:
        medbox = original_to_medsam_box(xyxy,width,height)
        if medbox is None:
            continue
        prompt = torch.from_numpy(medbox[None]).to(device)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=='cuda'):
            logits=decoder_forward(sam,feature,prompt)
        mask=output_mask(logits,height,width)
        np.maximum(merged,mask,out=merged)
    return merged


def plot_cases(examples, out):
    if not examples:
        return
    fig,axs=plt.subplots(len(examples),5,figsize=(18,3.2*len(examples)),squeeze=False)
    for idx,e in enumerate(examples):
        rgb=e['rgb']; gt=e['gt']; boxes=e['boxes']
        axs[idx,0].imshow(rgb);axs[idx,0].set_title(e['stem']+' / RGB',fontsize=9)
        axs[idx,1].imshow(rgb)
        for j,b in enumerate(boxes):
            x0,y0,x1,y1=b['xyxy']
            axs[idx,1].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,
                                          fill=False,edgecolor='lime',linewidth=1.5))
            axs[idx,1].text(x0,y0, f"{b['conf']:.2f}",fontsize=7,color='white',
                             bbox={'facecolor':'green','alpha':.7,'pad':1})
        axs[idx,1].set_title('YOLO boxes ('+str(len(boxes))+')',fontsize=9)
        axs[idx,2].imshow(gt,cmap='gray',vmin=0,vmax=1);axs[idx,2].set_title('GT mask (metric only)',fontsize=9)
        for col,exp in [(3,'E1'),(4,'E3C')]:
            axs[idx,col].imshow(rgb)
            pred=e['preds'][exp]
            axs[idx,col].imshow(np.ma.masked_where(pred==0,pred),cmap='spring',alpha=.50,vmin=0,vmax=1)
            axs[idx,col].set_title(f"{exp} Dice={e['scores'][exp]:.3f}",fontsize=9)
        for a in axs[idx]:a.axis('off')
    fig.tight_layout();fig.savefig(out,dpi=115);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--final-test',action='store_true',help='Required to evaluate the held-out test split')
    p.add_argument('--conf',type=float,default=.25,help='YOLO prediction confidence threshold; tune on VAL only')
    p.add_argument('--iou',type=float,default=.50,help='YOLO NMS IoU threshold')
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--max-det',type=int,default=10)
    p.add_argument('--examples',type=int,default=8)
    p.add_argument('--weights',type=Path,default=None,help='Optional YOLO best.pt override')
    p.add_argument('--device',default='0',help='CUDA device index for Ultralytics inference')
    a=p.parse_args()
    if a.split=='test' and not a.final_test:
        p.error('Test split is locked; use --final-test only after validation-based choices are frozen')
    if not 0<a.conf<1 or not 0<a.iou<1 or a.max_det<1 or a.imgsz<32 or a.examples<0:
        p.error('Bad confidence/NMS/image-size/max-det/examples argument')

    cfg=config(); work=work_dir(cfg); device=require_gpu()
    manifest,imgs,gts=check_features(cfg,(a.split,))
    stems=manifest[a.split]
    detector_weights=a.weights or work/'yolo_runs'/'polyp_v8n'/'weights'/'best.pt'
    detector_weights=Path(detector_weights).resolve()
    if not detector_weights.is_file():
        raise FileNotFoundError(f'YOLO checkpoint not found: {detector_weights}')

    from ultralytics import YOLO
    detector=YOLO(str(detector_weights))
    sam=make_medsam(cfg['model']['medsam_repo'],cfg['model']['checkpoint'])
    sam.image_encoder=nn.Identity()  # embeddings were already cached with same official MedSAM
    sam.prompt_encoder.to(device).eval()
    for param in sam.prompt_encoder.parameters():
        param.requires_grad_(False)
    decoders=build_decoders(sam,work,device)

    output=work/'auto_yolo'/a.split/f'conf_{int(round(a.conf*1000)):03d}'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'{output} is not empty; will not overwrite a completed report')
    output.mkdir(parents=True,exist_ok=True)
    predictions={}
    rows=[]; examples=[]
    count_no_boxes=0; count_many_boxes=0; all_boxes=0
    for i,stem in enumerate(stems,1):
        image_path=imgs[stem]
        rgb=read_rgb(image_path)
        gt=read_gt(gts[stem]); h,w=gt.shape
        if rgb.shape[:2]!=(h,w):
            raise RuntimeError(f'RGB/GT shape mismatch for {stem}')
        pred_results=detector.predict(source=str(image_path),imgsz=a.imgsz,
                                      conf=a.conf,iou=a.iou,max_det=a.max_det,
                                      device=a.device,verbose=False)
        r=pred_results[0]
        dets=[]
        if r.boxes is not None and len(r.boxes)>0:
            coords=r.boxes.xyxy.detach().cpu().numpy()
            scores=r.boxes.conf.detach().cpu().numpy()
            classes=r.boxes.cls.detach().cpu().numpy().astype(int)
            for coords_i,score,cls in zip(coords,scores,classes):
                if cls!=0:
                    continue
                if original_to_medsam_box(coords_i,w,h) is None:
                    continue
                dets.append({'xyxy':[float(z) for z in coords_i],
                             'conf':float(score),'class_id':int(cls)})
        predictions[stem]=dets
        all_boxes+=len(dets)
        count_no_boxes+=len(dets)==0
        count_many_boxes+=len(dets)>1
        emb=np.load(feature_file(cfg,a.split,stem),allow_pickle=False).astype(np.float32)
        feat=torch.from_numpy(emb[None]).to(device)
        item={'stem':stem,'rgb':rgb,'gt':gt,'boxes':dets,'preds':{},'scores':{}}
        for exp in ('E1','E3C'):
            sam.mask_decoder=decoders[exp]
            seg=segment_for_boxes(sam,feat,[d['xyxy'] for d in dets],(h,w),cfg['training']['amp'])
            row={'stem':stem,'experiment':exp,'n_boxes':len(dets),
                 'max_conf':max((d['conf'] for d in dets),default=0.),
                 **metrics(seg,gt),'boundary_f1_2px':boundary_f1(seg,gt),
                 'hd95_pixels':hd95(seg,gt)}
            rows.append(row)
            if len(examples)<a.examples:
                item['preds'][exp]=seg
                item['scores'][exp]=row['dice']
        if len(examples)<a.examples:
            examples.append(item)
        if i%20==0 or i==len(stems):
            print(f'Processed {i}/{len(stems)} images',flush=True)

    summary={'split':a.split,'n_images':len(stems),
             'box_source':'YOLOv8n PREDICTED boxes only (no GT boxes used for prompts)',
             'yolo_weights':str(detector_weights),'confidence_threshold':a.conf,
             'nms_iou':a.iou,'imgsz':a.imgsz,'max_det':a.max_det,
             'zero_detection_images':count_no_boxes,'multiple_detection_images':count_many_boxes,
             'total_predicted_boxes':all_boxes,
             'warning':'All images included in segmentation means, even those with zero YOLO detections; empty prediction scored vs GT.',
             'models':{}}
    for exp in ('E1','E3C'):
        es=[r for r in rows if r['experiment']==exp]
        detected=[r for r in es if r['n_boxes']>0]
        summary['models'][exp]={
            'mean_dice_detected_only':float(np.mean([r['dice'] for r in detected])) if detected else None,
            'mean_dice':float(np.mean([r['dice'] for r in es])),
            'mean_iou':float(np.mean([r['iou'] for r in es])),
            'mean_boundary_f1_2px':float(np.mean([r['boundary_f1_2px'] for r in es])),
            'mean_hd95_pixels':float(np.mean([r['hd95_pixels'] for r in es]))}
    save_json(output/'summary.json',summary)
    save_json(output/'yolo_predictions.json',predictions)
    save_csv(output/'per_image.csv',rows)
    if examples:plot_cases(examples,output/'examples.png')
    print('\n===== STEP 11: YOLO -> MedSAM RESULT =====')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('FILES:',output)

if __name__=='__main__':
    main()

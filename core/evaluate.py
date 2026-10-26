"""One official-size mask inference protocol for E0, E1-E4 and robustness."""
import numpy as np
import torch
from .seg import (gt_box,read_gt,feature_file,decoder_forward,output_mask,
                  metrics,boundary_f1,hd95,perturb_box,stable_rng)

@torch.inference_mode()
def evaluate(sam,cfg,stems,mask_map,split='val',noise=0.,trials=1,
             details=False,keep_preds=0,amp=True):
    sam.mask_decoder.eval();sam.prompt_encoder.eval()
    device=next(sam.mask_decoder.parameters()).device
    rows=[];preds={}
    for idx,stem in enumerate(stems):
        gt=read_gt(mask_map[stem]);h,w=gt.shape
        emb=np.load(feature_file(cfg,split,stem),allow_pickle=False).astype(np.float32)
        feat=torch.from_numpy(emb[None]).to(device)
        clean=gt_box(gt)
        for j in range(trials):
            box=perturb_box(clean,noise,stable_rng(cfg['project']['seed'],stem,j,noise))
            bt=torch.as_tensor(box,device=device,dtype=torch.float32)[None]
            with torch.cuda.amp.autocast(enabled=amp):
                logits=decoder_forward(sam,feat,bt)
            pred=output_mask(logits,h,w)
            row={'stem':stem,'trial':j,'noise':noise,**metrics(pred,gt)}
            if details:
                row['boundary_f1_2px']=boundary_f1(pred,gt)
                row['hd95_pixels']=hd95(pred,gt)
            rows.append(row)
            if j==0 and idx<keep_preds:preds[stem]=pred
    result={'n_images':len(stems),'n_predictions':len(rows),
            'mean_dice':float(np.mean([r['dice'] for r in rows])),
            'mean_iou':float(np.mean([r['iou'] for r in rows])),
            'noise':noise,'trials':trials,
            'box_source':'GT-derived Oracle Box (NOT fully automatic)'}
    if details:
        result.update({'mean_boundary_f1_2px':float(np.mean([r['boundary_f1_2px'] for r in rows])),
                       'mean_hd95_pixels':float(np.mean([r['hd95_pixels'] for r in rows]))})
    return result,rows,preds

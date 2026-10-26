"""Step 04: decoder-only fine-tuning on a single RTX 2080 Ti 22 GB.

E1 : Dice+BCE
E2 : E1 + edge-weighted BCE
E3 : two prompt views, shared decoder + mask consistency loss
E3C: same two prompt views as E3 but consistency_lambda=0 (control for the loss)
E4 : E2 and E3 combined
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from core.seg import model,decoder_forward,base_loss,boundary_loss,check_features
from core.data import DecoderData
from core.evaluate import evaluate
from core.utils import config,require_gpu,work_dir,save_json,save_csv,seed_all


def loss_one(logits,gt,exp,train_cfg):
    loss=base_loss(logits,gt)
    if exp in ('E2','E4'):
        loss=loss+train_cfg['boundary_lambda']*boundary_loss(logits,gt)
    return loss


def make_checkpoint(sam,optimizer,scheduler,scaler,epoch,best,wait,history,
                    exp,settings):
    return {'decoder':sam.mask_decoder.state_dict(), 'optimizer':optimizer.state_dict(),
            'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
            'epoch':epoch,'best':best,'wait':wait,'history':history,'exp':exp,
            'settings':settings,'rng_python':random.getstate(),
            'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all()}


def restore_rng(checkpoint):
    random.setstate(checkpoint['rng_python'])
    np.random.set_state(checkpoint['rng_numpy'])
    torch.set_rng_state(checkpoint['rng_torch'])
    torch.cuda.set_rng_state_all(checkpoint['rng_cuda'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exp',choices=['E1','E2','E3','E3C','E4'],required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--epochs',type=int,default=None,help='不填写则读取 config.yaml')
    parser.add_argument('--batch',type=int,default=None,help='不填写则读取 config.yaml')
    args=parser.parse_args()
    cfg=config();device=require_gpu();seed_all(cfg['project']['seed']);w=work_dir(cfg)
    train_cfg=dict(cfg['training'])
    if args.epochs is not None:train_cfg['epochs']=args.epochs
    if args.batch is not None:train_cfg['batch_size']=args.batch
    # E3C is EXACTLY the E3 dual-prompt setup, except no consistency loss.
    # Set only the experiment-local copy; never mutate global config.yaml.
    if args.exp == 'E3C':
        train_cfg['consistency_lambda'] = 0.0
    if train_cfg['batch_size']<1 or train_cfg['epochs']<1:raise ValueError('batch/epochs must be >=1')
    m,_,gt_files=check_features(cfg,('train','val'))
    out=w/'runs'/args.exp;out.mkdir(parents=True,exist_ok=True)
    last=out/'last.pth';best_file=out/'best.pth'
    if last.is_file() and not args.resume:
        raise FileExistsError(f'{last} 已存在，为防止覆盖请增加 --resume 或先备份该实验目录')
    if args.resume and not last.is_file():raise FileNotFoundError(f'{last} 不存在，不能续训')
    if not args.resume:
        save_json(out/'config.json',{'exp':args.exp,'train':train_cfg,
                     'seed':cfg['project']['seed'],'split_hash':m['split_hash'],
                     'trainable_modules':['mask_decoder']})
    sam=model(cfg['model']['medsam_repo'],cfg['model']['checkpoint'])
    sam.image_encoder=nn.Identity() # frozen ViT-B features were cached in Step 02
    sam.prompt_encoder.to(device).eval();sam.mask_decoder.to(device).train()
    for p in sam.prompt_encoder.parameters():p.requires_grad_(False)
    for p in sam.mask_decoder.parameters():p.requires_grad_(True)
    print('Trainable decoder params:',sum(p.numel() for p in sam.mask_decoder.parameters()))
    dataset=DecoderData(cfg,m['train'],gt_files,'E3' if args.exp == 'E3C' else args.exp)
    loader=DataLoader(dataset,batch_size=train_cfg['batch_size'],shuffle=True,
                      num_workers=train_cfg['num_workers'],pin_memory=True)
    opt=torch.optim.AdamW(sam.mask_decoder.parameters(),lr=train_cfg['lr'],
                          weight_decay=train_cfg['weight_decay'])
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=train_cfg['epochs'])
    scaler=torch.cuda.amp.GradScaler(enabled=train_cfg['amp'])  # torch 2.1 API
    first=1;best=-1.;wait=0;history=[]
    if args.resume:
        cp=torch.load(last,map_location='cpu')
        if cp['exp']!=args.exp or cp['settings']!=train_cfg:
            raise RuntimeError('续训配置与上次不一致。使用原先 epochs/batch/学习率等设置。')
        sam.mask_decoder.load_state_dict(cp['decoder'])
        opt.load_state_dict(cp['optimizer']);sched.load_state_dict(cp['scheduler'])
        scaler.load_state_dict(cp['scaler'])
        first=cp['epoch']+1;best=cp['best'];wait=cp['wait'];history=cp['history']
        restore_rng(cp)
        print(f'Resuming from epoch={first} best={best:.5f}')
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(first,train_cfg['epochs']+1):
        sam.mask_decoder.train()
        seen=0;epoch_sum=0.
        progress=tqdm(loader,desc=f'{args.exp} epoch {epoch}/{train_cfg["epochs"]}',leave=False)
        for feat,box_a,box_b,gt in progress:
            feat=feat.to(device,non_blocking=True)
            box_a=box_a.to(device,non_blocking=True)
            box_b=box_b.to(device,non_blocking=True)
            gt=gt.to(device,non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=train_cfg['amp']):
                pred_a=decoder_forward(sam,feat,box_a)
                l_a=loss_one(pred_a,gt,args.exp,train_cfg)
                if args.exp in ('E3','E3C','E4'):
                    pred_b=decoder_forward(sam,feat,box_b)
                    l_b=loss_one(pred_b,gt,args.exp,train_cfg)
                    consistency=F.l1_loss(torch.sigmoid(pred_a.float()),torch.sigmoid(pred_b.float()))
                    loss=.5*(l_a+l_b)+train_cfg['consistency_lambda']*consistency
                else:loss=l_a
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError('检测到 NaN/Inf loss；检查数据、AMP、学习率和模型输出')
            scaler.scale(loss).backward()
            scaler.step(opt);scaler.update()
            n=feat.shape[0];seen+=n;epoch_sum+=float(loss.detach())*n
            progress.set_postfix(loss=f'{epoch_sum/seen:.4f}')
        sched.step()
        with torch.inference_mode():
            val,_,_=evaluate(sam,cfg,m['val'],gt_files,split='val',details=False,
                              amp=train_cfg['amp'])
        mean_loss=epoch_sum/seen;val_dice=val['mean_dice']
        row={'epoch':epoch,'train_loss':mean_loss,'val_dice':val_dice,
             'val_iou':val['mean_iou'],'lr':opt.param_groups[0]['lr']}
        history.append(row);save_csv(out/'history.csv',history)
        improved=val_dice>best+1e-5
        if improved:best=val_dice;wait=0
        else:wait+=1
        cp=make_checkpoint(sam,opt,sched,scaler,epoch,best,wait,history,args.exp,train_cfg)
        torch.save(cp,last)
        if improved:torch.save(cp,best_file)
        print(f'[{args.exp}] epoch={epoch:03d} train={mean_loss:.5f} val_Dice={val_dice:.5f} '
              f'val_IoU={val["mean_iou"]:.5f} best={best:.5f} '
              f'peakVRAM={torch.cuda.max_memory_allocated()/2**30:.2f}GiB',flush=True)
        if wait>=train_cfg['patience']:
            print(f'Early stop: patience={train_cfg["patience"]}')
            break
    print('\n===== TRAIN RESULT =====')
    print('EXP:',args.exp,'BEST VAL DICE:',round(best,5),'WEIGHTS:',best_file)
    print('HISTORY:',out/'history.csv')

if __name__=='__main__':main()

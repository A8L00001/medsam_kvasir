"""Step 02: one-time FP16 ViT-B embedding extraction; safely resumable."""
from pathlib import Path
import argparse
import shutil

import numpy as np
import torch
from tqdm import tqdm

from core.seg import model, preprocess, read_rgb, feature_file, load_manifest
from core.utils import config, require_gpu, seed_all, work_dir, weight_id, load_json, save_json


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reset',action='store_true',help='危险：删除已有全部特征并重新生成')
    args=parser.parse_args()
    cfg=config();device=require_gpu();seed_all(cfg['project']['seed']);w=work_dir(cfg)
    m,images,_=load_manifest(cfg)
    if not cfg['model']['medsam_repo'] or not cfg['model']['checkpoint']:
        raise RuntimeError('config.yaml 还没填 MedSAM 仓库/官方权重路径，先执行 00_preflight.py')
    meta={'split_hash':m['split_hash'],'weights':weight_id(cfg['model']['checkpoint']),
          'preprocess':'official skimage cubic resize 1024 anti_alias -> uint8 minmax; ViT-B float16 feature'}
    cdir=w/'features';metafile=cdir/'metadata.json'
    if args.reset and cdir.exists():shutil.rmtree(cdir)
    if cdir.exists() and not metafile.exists() and list(cdir.rglob('*.npy')):
        raise RuntimeError('cache 缺少元数据，防止混用。确认后 --reset 重建')
    if metafile.exists() and load_json(metafile)!=meta:
        raise RuntimeError('数据划分、官方权重或预处理配置与已有缓存不一致；请换输出目录或 --reset')
    cdir.mkdir(parents=True,exist_ok=True);save_json(metafile,meta)
    sam=model(cfg['model']['medsam_repo'],cfg['model']['checkpoint'])
    encoder=sam.image_encoder.to(device).eval()
    del sam
    total_new=0
    with torch.inference_mode():
        for split in ('train','val','test'):
            for stem in tqdm(m[split],desc=f'Encoding {split}'):
                path=feature_file(cfg,split,stem)
                if path.exists():continue
                img=preprocess(read_rgb(images[stem]))[None].to(device)
                with torch.cuda.amp.autocast(enabled=True):
                    enc=encoder(img)
                array=enc[0].float().cpu().numpy().astype(np.float16)
                if array.shape!=(256,64,64):raise RuntimeError(f'预期 (256,64,64)；实际 {array.shape}')
                path.parent.mkdir(parents=True,exist_ok=True)
                tmp=path.with_suffix('.partial')
                with tmp.open('wb') as f:np.save(f,array)
                tmp.replace(path)
                total_new+=1
    report={'status':'PASS','new_embeddings':total_new,
            'total_embeddings':sum(len(m[s]) for s in ('train','val','test')),
            'embedding_shape':[256,64,64],
            'peak_gpu_allocated_gib':round(torch.cuda.max_memory_allocated()/2**30,3),
            'cache_dir':str(cdir)}
    save_json(w/'02_cache'/'report.json',report)
    print('\n===== STEP 02 RESULT =====')
    for key,value in report.items():print(f'{key}: {value}')
    print('REPORT:',w/'02_cache'/'report.json')

if __name__=='__main__':main()

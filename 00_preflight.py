"""Step 00: verify already-working server, locate assets and print actionable report."""
import argparse
import os
import platform
import shutil
import sys
from pathlib import Path

import torch
import yaml
from core.utils import HERE, config, save_json, work_dir

SKIP={'__pycache__','.git','.cache','site-packages','.venv','venv','outputs','node_modules'}


def discover(root, max_depth=6):
    root=Path(root).resolve()
    pairs=[];repos=[];weights=[]
    if not root.exists(): return pairs,repos,weights
    for dirpath, dirs, files in os.walk(root):
        p=Path(dirpath)
        depth=len(p.relative_to(root).parts)
        dirs[:]=[d for d in dirs if d not in SKIP and not d.startswith('.')]
        if depth>=max_depth:dirs[:]=[]
        low={d.lower():d for d in dirs}
        if 'segment_anything' in low:
            repos.append(str(p))
        if 'images' in low and 'masks' in low:
            image_dir=p/low['images']; mask_dir=p/low['masks']
            n1=len([x for x in image_dir.iterdir() if x.suffix.lower() in {'.png','.jpeg','.jpg'}])
            n2=len([x for x in mask_dir.iterdir() if x.suffix.lower() in {'.png','.jpeg','.jpg'}])
            if n1>0 and n2>0:pairs.append((n1,n2,str(image_dir),str(mask_dir)))
        for f in files:
            if f=='medsam_vit_b.pth':weights.append(str(p/f))
    return pairs,repos,weights


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scan-root',default=None,help='递归搜索数据与 MedSAM 的根目录，默认当前目录')
    parser.add_argument('--config',default=str(HERE/'config.yaml'))
    args=parser.parse_args()
    cfg_file=Path(args.config).resolve()
    if not cfg_file.is_file():
        example=cfg_file.with_name('config.example.yaml')
        if not example.is_file():
            raise FileNotFoundError(f'缺少 {cfg_file}，也没有模板 {example}')
        cfg_file.write_text(example.read_text(encoding='utf-8'),encoding='utf-8')
        print(f'{cfg_file.name} 不存在，已按 {example.name} 生成。')
    scan=args.scan_root or str(Path.cwd())
    print('\n========== STEP 00 | RTX 2080 Ti 环境预检 ==========')
    print('Current directory:',Path.cwd())
    print('Scan root:',Path(scan).resolve())
    report={'python':sys.version.split()[0], 'torch':torch.__version__,
            'torch_cuda':torch.version.cuda, 'os':platform.platform(),
            'cuda_available':torch.cuda.is_available(),
            'scan_root':str(Path(scan).resolve())}
    if torch.cuda.is_available():
        p=torch.cuda.get_device_properties(0)
        report.update({'gpu':p.name,'gpu_memory_gib':round(p.total_memory/2**30,3),
                       'compute_capability':f'{p.major}.{p.minor}'})
        x=torch.randn((64,64),device='cuda',dtype=torch.float16)
        with torch.no_grad():y=x@x
        report['fp16_ok']=bool(torch.isfinite(y).all().item())
    else:
        report['fp16_ok']=False
    du=shutil.disk_usage(HERE)
    report['free_disk_gib']=round(du.free/2**30,2)
    pairs,repos,weights=discover(scan)
    report['data_candidates']=pairs
    report['medsam_repos']=repos
    report['checkpoint_candidates']=weights
    cfg=yaml.safe_load(cfg_file.read_text(encoding='utf8'))
    changed=False
    if len(pairs)==1:
        _,_,im,gt=pairs[0]
        if not cfg['data']['images_dir'] and not cfg['data']['masks_dir']:
            cfg['data']['images_dir']=im;cfg['data']['masks_dir']=gt;changed=True
    if len(repos)==1 and not cfg['model']['medsam_repo']:
        cfg['model']['medsam_repo']=repos[0];changed=True
    if len(weights)==1 and not cfg['model']['checkpoint']:
        cfg['model']['checkpoint']=weights[0];changed=True
    if changed:
        cfg_file.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf8')
        print('自动填充 config.yaml 中唯一识别的数据/模型路径。')
    d=config(cfg_file)
    report['configured_paths']={key: {k:(Path(v).exists() if v else False) for k,v in d[key].items()
                                     if k in ('images_dir','masks_dir','medsam_repo','checkpoint')}
                                for key in ('data','model')}
    paths_ok=(all(report['configured_paths']['data'].values()) and
              all(report['configured_paths']['model'].values()) and
              bool(report['configured_paths']['data']) and bool(report['configured_paths']['model']))
    report['status']='PASS' if report['cuda_available'] and report['fp16_ok'] else 'FAIL'
    report['assets_status']='READY' if paths_ok else 'MISSING_PATHS'
    output=work_dir(d)/'00_preflight'/'report.json'
    save_json(output,report)
    print('GPU:',report.get('gpu'), '| VRAM GiB:',report.get('gpu_memory_gib'))
    print('Torch:',report['torch'], '| CUDA:',report['torch_cuda'], '| FP16:',report['fp16_ok'])
    print('Disk free GiB:',report['free_disk_gib'])
    print('Data candidate folders:',pairs[:8])
    print('MedSAM repos:',repos[:8], 'Official weights:',weights[:8])
    print('Config paths:',d['data'],d['model'])
    print('STATUS:',report['status'],'| DATA/MODEL:',report['assets_status'])
    print('REPORT:',output)
    if not paths_ok:
        print('NEXT: 检查 config.yaml 中未配置的 4 个路径；提供 --scan-root <更大的目录> 可以扩大搜索范围。')
    else:
        print('NEXT: python 01_prepare.py')
    if report['status']!='PASS':sys.exit(2)

if __name__=='__main__':main()

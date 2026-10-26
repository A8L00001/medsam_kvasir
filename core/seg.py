"""MedSAM ViT-B preprocessing, prompt / mask resolution, and losses."""
from pathlib import Path
import importlib
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as skresize

from .utils import load_json, weight_id

SIDE=1024
LOW=256
EXT={'.jpg','.jpeg','.png','.bmp','.tif','.tiff'}


def folder_map(directory):
    p=Path(directory)
    if not p.is_dir():raise FileNotFoundError(f'目录不存在：{p}')
    files={}
    for path in sorted(p.iterdir()):
        if path.is_file() and path.suffix.lower() in EXT:
            if path.stem in files:raise ValueError(f'重复文件名（不同扩展名）：{path.stem}')
            files[path.stem]=path.resolve()
    if not files:raise RuntimeError(f'目录中没有图像文件：{p}')
    return files


def read_rgb(path):
    a=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if a is None:raise IOError(f'无法读取图像：{path}')
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def read_gt(path):
    a=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
    if a is None:raise IOError(f'无法读取标注：{path}')
    return (a>=128).astype(np.uint8)


def mask_resize(mask, side):
    return cv2.resize(mask,(side,side),interpolation=cv2.INTER_NEAREST)


def gt_box(mask):
    ys,xs=np.where(mask_resize(mask,SIDE)>0)
    if len(xs)==0:raise ValueError('发现空 Mask，无法提取 Bounding Box')
    return np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1], dtype=np.float32)


def preprocess(rgb):
    # Official MedSAM_Inference.py: cubic anti-aliased 1024 resize -> uint8 -> min/max.
    resized=skresize(rgb,(SIDE,SIDE),order=3,preserve_range=True,
                     anti_aliasing=True).astype(np.uint8)
    image=resized.astype(np.float32)
    image=(image-image.min())/max(1e-8,float(image.max()-image.min()))
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2,0,1)))


def outward_box(box, max_px, rng=None):
    import random
    rng = rng if rng is not None else random
    a,b,c,d=box
    shifts=[rng.randint(0,max_px) for _ in range(4)]
    return np.array([max(0,a-shifts[0]),max(0,b-shifts[1]),
                     min(SIDE,c+shifts[2]),min(SIDE,d+shifts[3])],dtype=np.float32)


def perturb_box(box, frac, rng):
    if frac==0: return box.copy().astype(np.float32)
    a,b,c,d=map(float,box);w=c-a;h=d-b
    shifts=[rng.uniform(-frac,frac)*v for v in (w,h,w,h)]
    vals=np.clip(np.array([a,b,c,d])+shifts,0,SIDE)
    if vals[2]-vals[0]<2 or vals[3]-vals[1]<2:return box.copy().astype(np.float32)
    return vals.astype(np.float32)


def stable_rng(seed,stem,trial,noise):
    import hashlib,random
    digest=hashlib.sha256(f'{seed}-{stem}-{trial}-{noise}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8],'big'))


def model(repo,checkpoint):
    repo=Path(repo)
    if not (repo/'segment_anything').is_dir():
        raise FileNotFoundError(f'MedSAM 仓库中找不到 segment_anything/: {repo}')
    if not Path(checkpoint).is_file():
        raise FileNotFoundError(f'找不到 medsam_vit_b.pth: {checkpoint}')
    sys.path.insert(0,str(repo))
    # Avoid accidental mismatch when a global 'segment_anything' is already imported.
    if 'segment_anything' in sys.modules:
        old=Path(sys.modules['segment_anything'].__file__).resolve()
        if repo.resolve() not in old.parents:
            raise RuntimeError('导入了非指定路径的 segment_anything，请用新 Python 进程运行')
    from segment_anything import sam_model_registry
    return sam_model_registry['vit_b'](checkpoint=str(checkpoint))


def decoder_forward(sam, feature, boxes):
    with torch.no_grad():
        sparse,dense=sam.prompt_encoder(points=None,boxes=boxes.float()[:,None,:],masks=None)
    low_logits, _=sam.mask_decoder(image_embeddings=feature,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=False)
    return low_logits


def base_loss(logits,gt):
    logits=logits.float();gt=gt.float()
    p=torch.sigmoid(logits);dims=(1,2,3)
    dice=1-((2*(p*gt).sum(dims)+1)/(p.sum(dims)+gt.sum(dims)+1)).mean()
    bce=F.binary_cross_entropy_with_logits(logits,gt)
    return dice+bce


def boundary_loss(logits,gt,band_px=5,weight=4.0):
    gt=gt.float()
    k=2*band_px+1
    expand=F.max_pool2d(gt,k,stride=1,padding=band_px)
    shrink=1-F.max_pool2d(1-gt,k,stride=1,padding=band_px)
    band=(expand-shrink).clamp(0,1)
    weights=1+weight*band
    loss=F.binary_cross_entropy_with_logits(logits.float(),gt,reduction='none')
    return (loss*weights).sum()/weights.sum()


def output_mask(low_logits,h,w):
    # Official inference interpolates probabilities (NOT logits) then thresholds.
    prob=torch.sigmoid(low_logits.float())
    prob=F.interpolate(prob,size=(h,w),mode='bilinear',align_corners=False)
    return (prob[0,0]>0.5).cpu().numpy().astype(np.uint8)


def metrics(pred,gt):
    a=pred.astype(bool);b=gt.astype(bool)
    tp=np.logical_and(a,b).sum();sa=a.sum()+b.sum()
    union=np.logical_or(a,b).sum()
    return {'dice':float(2*tp/sa if sa else 1.),
            'iou':float(tp/union if union else 1.)}


def boundary_f1(pred,gt,r=2):
    # Fixed 2-native-pixel tolerance. Same metric protocol for every experiment.
    k=np.ones((3,3),np.uint8)
    a=cv2.morphologyEx(pred.astype(np.uint8),cv2.MORPH_GRADIENT,k)>0
    b=cv2.morphologyEx(gt.astype(np.uint8),cv2.MORPH_GRADIENT,k)>0
    if not a.any() and not b.any():return 1.
    if not a.any() or not b.any():return 0.
    dil=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    ad=cv2.dilate(a.astype(np.uint8),dil)>0
    bd=cv2.dilate(b.astype(np.uint8),dil)>0
    precision=np.logical_and(a,bd).sum()/a.sum()
    recall=np.logical_and(b,ad).sum()/b.sum()
    return float(2*precision*recall/(precision+recall+1e-8))


def hd95(pred,gt):
    # Pixel units, HD95 of symmetric boundary distances. Empty masks get diagonal penalty.
    a=pred.astype(np.uint8);b=gt.astype(np.uint8)
    if not a.any() and not b.any():return 0.
    if not a.any() or not b.any():return float(np.hypot(*gt.shape))
    k=np.ones((3,3),np.uint8)
    ea=cv2.morphologyEx(a,cv2.MORPH_GRADIENT,k)>0
    eb=cv2.morphologyEx(b,cv2.MORPH_GRADIENT,k)>0
    if not ea.any() or not eb.any():return float(np.hypot(*gt.shape))
    db=cv2.distanceTransform((~eb).astype(np.uint8),cv2.DIST_L2,5)
    da=cv2.distanceTransform((~ea).astype(np.uint8),cv2.DIST_L2,5)
    return float(np.percentile(np.concatenate([db[ea],da[eb]]),95))


def load_manifest(cfg):
    from .utils import work_dir
    path=work_dir(cfg)/'splits.json'
    if not path.is_file():raise FileNotFoundError(f'先运行 01_prepare.py：{path}')
    m=load_json(path)
    im=folder_map(m['images_dir']);gt=folder_map(m['masks_dir'])
    names=[v for k in ('train','val','test') for v in m[k]]
    if len(names)!=len(set(names)):raise RuntimeError('检测到跨 split 重复样本')
    if any(name not in im or name not in gt for name in names):
        raise RuntimeError('数据文件缺失或被更改')
    return m,im,gt


def feature_file(cfg,split,stem):
    from .utils import work_dir
    return work_dir(cfg)/'features'/split/f'{stem}.npy'


def check_features(cfg,splits=('train','val')):
    from .utils import work_dir
    m,im,gt=load_manifest(cfg)
    meta=load_json(work_dir(cfg)/'features'/'metadata.json')
    if meta['split_hash']!=m['split_hash'] or meta['weights']!=weight_id(cfg['model']['checkpoint']):
        raise RuntimeError('当前图像缓存与数据划分/官方权重不一致，不能混用')
    for split in splits:
        missing=[s for s in m[split] if not feature_file(cfg,split,s).is_file()]
        if missing:raise FileNotFoundError(f'{split} 缺少 {len(missing)} 份特征；例如 {missing[0]}')
    return m,im,gt

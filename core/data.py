"""Feature-cache dataset for decoder-only fine tuning (no full 1024 image batches)."""
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from .seg import feature_file,read_gt,mask_resize,gt_box,outward_box,perturb_box,LOW

class DecoderData(Dataset):
    def __init__(self,cfg, names, mask_files, mode):
        self.cfg=cfg; self.names=names;self.mask_files=mask_files;self.mode=mode
        self.jitter=cfg['training']['box_jitter_px']
        self.frac=cfg['training']['consistency_box_ratio']

    def __len__(self):return len(self.names)

    def __getitem__(self,idx):
        name=self.names[idx]
        feat=np.load(feature_file(self.cfg,'train',name),allow_pickle=False)
        if feat.shape!=(256,64,64):raise ValueError(f'特征异常：{name}: {feat.shape}')
        gt=read_gt(self.mask_files[name])
        box=gt_box(gt)
        a=outward_box(box,self.jitter)
        if self.mode in ('E3','E4'):
            # Secondary prompt includes ±jitter and can shift slightly inward.
            # Reject proposals excluding excessive GT bounding area.
            x0,y0,x1,y1=box;area=(x1-x0)*(y1-y0)
            b=None
            for _ in range(15):
                trial=perturb_box(box,self.frac,random)
                inter=max(0.,min(x1,trial[2])-max(x0,trial[0]))*max(0.,min(y1,trial[3])-max(y0,trial[1]))
                if inter/max(area,1)>0.85:
                    b=trial;break
            if b is None:b=outward_box(box,self.jitter)
        else:b=a.copy()
        target=mask_resize(gt,LOW).astype(np.float32)[None]
        return (torch.from_numpy(feat.astype(np.float32)),
                torch.from_numpy(a),torch.from_numpy(b),torch.from_numpy(target))

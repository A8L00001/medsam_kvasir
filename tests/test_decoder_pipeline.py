"""Synthetic batch testing without downloading official MedSAM checkpoint."""
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from torch import nn

from core.seg import decoder_forward, base_loss
from core.data import DecoderData

class FakePrompt(nn.Module):
    def forward(self,points,boxes,masks):
        b=boxes.size(0)
        return torch.zeros(b,2,256,device=boxes.device), torch.zeros(b,256,8,8,device=boxes.device)
    def get_dense_pe(self): return torch.zeros(1,256,8,8)

class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Conv2d(256,1,kernel_size=1)
    def forward(self,image_embeddings,image_pe,sparse_prompt_embeddings,dense_prompt_embeddings,multimask_output):
        pred=self.head(image_embeddings+dense_prompt_embeddings)
        return nn.functional.interpolate(pred,size=(256,256),mode='bilinear'),None

class TestPipeline(unittest.TestCase):
    def test_prompt_decoder_and_gradient(self):
        torch.set_num_threads(2)
        sam=SimpleNamespace(prompt_encoder=FakePrompt(),mask_decoder=FakeDecoder())
        imgs=torch.randn(2,256,8,8)
        boxes=torch.tensor([[10,20,200,300],[11,22,245,345]],dtype=torch.float32)
        pred=decoder_forward(sam,imgs,boxes)
        self.assertEqual(pred.shape,(2,1,256,256))
        loss=base_loss(pred,torch.ones_like(pred));loss.backward()
        self.assertIsNotNone(sam.mask_decoder.head.weight.grad)

    def test_feature_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'features'/'train').mkdir(parents=True)
            np.save(root/'features'/'train'/'case01.npy',np.ones((256,64,64),dtype=np.float16))
            gt=np.zeros((150,250),dtype=np.uint8);gt[25:90,35:120]=255
            mask=root/'case01.jpg';cv2.imwrite(str(mask),gt)
            cfg={'project':{'work_dir':str(root)},
                 'training':{'box_jitter_px':20,'consistency_box_ratio':.1}}
            ds=DecoderData(cfg,['case01'],{'case01':mask},'E4')
            x,a,b,target=ds[0]
            self.assertEqual(x.shape,(256,64,64))
            self.assertEqual(target.shape,(1,256,256))
            self.assertEqual(a.shape,(4,))
            self.assertEqual(b.shape,(4,))

if __name__=='__main__':unittest.main()

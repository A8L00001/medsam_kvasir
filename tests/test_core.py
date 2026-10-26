import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from core.seg import (SIDE,LOW,gt_box,outward_box,perturb_box,stable_rng,
                      mask_resize,metrics,boundary_f1,hd95,base_loss,boundary_loss,
                      output_mask)

class TestCore(unittest.TestCase):
    def setUp(self):
        self.gt=np.zeros((131,227),dtype=np.uint8)
        self.gt[15:110,24:155]=1

    def test_box_jitter_and_repeatability(self):
        box=gt_box(self.gt)
        self.assertTrue(np.all(box>=0) and np.all(box<=1024))
        self.assertEqual(mask_resize(self.gt,LOW).shape,(256,256))
        jitter=outward_box(box,20)
        self.assertLessEqual(jitter[0],box[0]);self.assertGreaterEqual(jitter[2],box[2])
        a=perturb_box(box,.1,stable_rng(42,'a',1,.1))
        b=perturb_box(box,.1,stable_rng(42,'a',1,.1))
        np.testing.assert_allclose(a,b)

    def test_metrics(self):
        self.assertEqual(metrics(self.gt,self.gt),{'dice':1.,'iou':1.})
        self.assertEqual(hd95(self.gt,self.gt),0.)
        self.assertGreater(boundary_f1(self.gt,self.gt),.999)
        blank=np.zeros_like(self.gt)
        self.assertEqual(metrics(blank,self.gt),{'dice':0.,'iou':0.})

    def test_losses_and_backward(self):
        logits=torch.randn(2,1,256,256,requires_grad=True)
        labels=torch.randint(0,2,(2,1,256,256)).float()
        loss=base_loss(logits,labels)+.2*boundary_loss(logits,labels)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_output_geometry(self):
        z=torch.ones(1,1,256,256)
        self.assertEqual(output_mask(z,131,227).shape,self.gt.shape)

if __name__=='__main__':unittest.main()

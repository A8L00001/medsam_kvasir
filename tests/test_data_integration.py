import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.utils import config,load_json

class TestIntegration(unittest.TestCase):
    def test_prepare(self):
        import importlib.util
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('stage_one_prepare',root/'01_prepare.py')
        prepare=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prepare)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);imgs=base/'images';gts=base/'masks'
            imgs.mkdir();gts.mkdir()
            for i in range(12):
                h,w=120+i,180+i
                rgb=np.full((h,w,3),35+i,dtype=np.uint8)
                gt=np.zeros((h,w),dtype=np.uint8)
                cv2.rectangle(gt,(30,20),(70+i,70+i),255,-1)
                cv2.imwrite(str(imgs/f'{i:03d}.jpg'),rgb)
                cv2.imwrite(str(gts/f'{i:03d}.png'),gt)
            cfg={'project':{'work_dir':str(base/'outputs'),'seed':42},
                 'data':{'images_dir':str(imgs),'masks_dir':str(gts)}}
            prepare.run(cfg)
            s=load_json(base/'outputs'/'01_data'/'report.json')
            self.assertEqual(s['status'],'PASS')
            self.assertEqual(s['split_counts'],{'train':9,'val':1,'test':2})
            self.assertTrue((base/'outputs'/'01_data'/'samples.png').is_file())
            prepare.run(cfg)  # recheck is idempotent and preserves splits

if __name__=='__main__':unittest.main()

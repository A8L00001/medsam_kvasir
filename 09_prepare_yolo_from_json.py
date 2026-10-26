#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 09: create YOLOv8 train/val detection labels from the supplied Kvasir-SEG bboxes JSON.

Run from the existing medsam_2080ti_stepwise project root. Uses the immutable
outputs/splits.json and NEVER writes test images/labels. Retains multiple boxes
per image and preserves the original bbox annotations without filtering.

The existing 10_train_yolo.py can be used unchanged because the output directory
is outputs/yolo_dataset. Refuses to replace an existing directory.
"""
import argparse
import hashlib
import json
import random
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

from core.seg import folder_map, read_rgb
from core.utils import config, work_dir, save_json


def convert_box(item, width, height, stem, index):
    """Raw pixel xyxy to YOLO normalized cx cy w h; allow xmax==image width."""
    if item.get('label') != 'polyp':
        raise ValueError(f'{stem} bbox[{index}]: unexpected class {item.get("label")!r}')
    try:
        x0, y0, x1, y1 = (float(item[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f'{stem} bbox[{index}]: invalid xyxy fields') from exc
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f'{stem} bbox[{index}]: out of range {(x0,y0,x1,y1)} on {(width,height)}')
    cx = (x0+x1)/(2*width)
    cy = (y0+y1)/(2*height)
    bw = (x1-x0)/width
    bh = (y1-y0)/height
    return (cx,cy,bw,bh), (x0,y0,x1,y1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bbox-json', type=Path, default=Path('kavsir_bboxes.json'),
                   help='Official Kvasir-SEG bounding-box file; the "kavsir" spelling is upstream')
    p.add_argument('--copy', action='store_true', help='Copy instead of symlink')
    p.add_argument('--samples', type=int, default=8)
    args=p.parse_args()
    if args.samples < 1:
        p.error('--samples must be >= 1')
    bbox_path=args.bbox_json.expanduser().resolve()
    if not bbox_path.is_file():
        raise FileNotFoundError(f'No bbox JSON at {bbox_path}. Pass --bbox-json /absolute/path/file.json')
    cfg=config()
    work=work_dir(cfg)
    split_file=work/'splits.json'
    if not split_file.is_file():
        raise FileNotFoundError(f'Missing immutable train/val/test split: {split_file}')
    manifest=json.loads(split_file.read_text(encoding='utf-8'))
    train=list(manifest['train']); val=list(manifest['val']); test=list(manifest['test'])
    ids=train+val+test
    if len(ids)!=len(set(ids)):
        raise RuntimeError('Train/val/test overlap. Refuse to proceed.')
    images=folder_map(manifest['images_dir'])
    for stem in train+val:
        if stem not in images:
            raise FileNotFoundError(f'Image missing from original dataset: {stem}')
    annotation=json.loads(bbox_path.read_text(encoding='utf-8'))
    if not isinstance(annotation,dict):
        raise ValueError('Expected JSON mapping image stem -> {width,height,bbox:[...]}')
    missing=set(train+val)-annotation.keys()
    if missing:
        raise ValueError(f'Missing bbox annotations for {len(missing)} train/val files: {sorted(missing)[:8]}')
    root=work/'yolo_dataset'
    if root.exists():
        raise FileExistsError(
            f'{root} already exists. Back it up or move it away, then retry: '
            f'mv {root} {work / "yolo_dataset_prev"}. '
            'Do not overwrite old results.')
    work.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.yolo_json_stage_',dir=work))
    rows=[]; tiny=[]
    counts=Counter(); many=Counter()
    boxes_for_plot={}
    try:
        for split, names in [('train',train),('val',val)]:
            for stem in names:
                src=images[stem]
                im=read_rgb(src)
                h,w=im.shape[:2]
                obj=annotation[stem]
                if (int(obj['width']),int(obj['height'])) != (w,h):
                    raise ValueError(f'{stem}: bbox JSON width/height={(obj["width"],obj["height"])} != image {(w,h)}')
                bb=obj.get('bbox')
                if not isinstance(bb,list) or not bb:
                    raise ValueError(f'{stem}: missing or empty bbox list')
                labels=[]; rawboxes=[]
                for i, box in enumerate(bb):
                    norm, raw=convert_box(box,w,h,stem,i)
                    labels.append('0 '+' '.join(f'{v:.8f}' for v in norm))
                    rawboxes.append(raw)
                    if (raw[2]-raw[0])*(raw[3]-raw[1])/(w*h)<0.0001:
                        tiny.append({'stem':stem,'split':split,'box_index':i,'xyxy':list(raw)})
                out_img=stage/'images'/split/src.name
                out_lbl=stage/'labels'/split/(stem+'.txt')
                out_img.parent.mkdir(parents=True,exist_ok=True)
                out_lbl.parent.mkdir(parents=True,exist_ok=True)
                if args.copy:
                    shutil.copy2(src,out_img)
                else:
                    out_img.symlink_to(src)
                out_lbl.write_text('\n'.join(labels)+'\n',encoding='utf8')
                counts[split]+=len(bb)
                if len(bb)>1:
                    many[split]+=1
                rows.append({'stem':stem,'split':split,'box_count':len(bb)})
                boxes_for_plot[stem]=rawboxes
        cfg_yaml={'path':str(root.resolve()),'train':'images/train','val':'images/val', 'names':{0:'polyp'}}
        with (stage/'data.yaml').open('w',encoding='utf8') as f:
            yaml.safe_dump(cfg_yaml,f,allow_unicode=True,sort_keys=False)
        chosen=random.Random(2026).sample(rows,min(args.samples,len(rows)))
        fig,ax=plt.subplots(len(chosen),2,figsize=(11,len(chosen)*3),squeeze=False)
        for j,row in enumerate(chosen):
            stem=row['stem']; pic=read_rgb(images[stem])
            ax[j,0].imshow(pic); ax[j,1].imshow(pic)
            for box in boxes_for_plot[stem]:
                x0,y0,x1,y1=box
                ax[j,0].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,
                                            edgecolor='lime',facecolor='none',linewidth=2))
            ax[j,0].set_title(f'{row["split"]}: {stem} | original JSON boxes ({len(boxes_for_plot[stem])})',fontsize=8)
            ax[j,1].set_title('Original image (no labels overlaid)',fontsize=8)
            for a in ax[j]:a.axis('off')
        fig.tight_layout()
        fig.savefig(stage/'sample_detection_labels.png',dpi=130)
        plt.close(fig)
        sha256=hashlib.sha256(bbox_path.read_bytes()).hexdigest()
        report={
            'status':'PASS', 'dataset':'Kvasir-SEG',
            'source':'provided bounding-box JSON; no boxes derived from segmentation masks',
            'source_bbox_json':str(bbox_path), 'source_bbox_sha256':sha256,
            'source_split_hash':manifest.get('split_hash'),
            'train_images':len(train), 'val_images':len(val),
            'train_boxes':counts['train'],'val_boxes':counts['val'],
            'train_images_multiple_boxes':many['train'],
            'val_images_multiple_boxes':many['val'],
            'test_labels_written':0, 'train_val_test_disjoint':True,
            'tiny_bbox_count_train_val':len(tiny),
            'tiny_boxes_train_val':tiny[:25],
            'data_yaml':str(root/'data.yaml'),
            'sample_plot':str(root/'sample_detection_labels.png'),
            'note':'YOLO boxes from dataset JSON are GT labels for training only; never feed GT JSON boxes to test-time MedSAM.'
        }
        save_json(stage/'report.json',report)
        stage.rename(root)   # atomic commit after all items pass validation
    except Exception:
        shutil.rmtree(stage,ignore_errors=True)
        raise
    print('\n===== YOLO DATA FROM Kvasir-SEG BBOX JSON =====')
    for k in ('status','train_images','val_images','train_boxes','val_boxes',
              'train_images_multiple_boxes','val_images_multiple_boxes',
              'test_labels_written','tiny_bbox_count_train_val'):
        print(f'{k}: {report[k]}')
    print('REPORT:',root/'report.json')
    print('VISUAL:',root/'sample_detection_labels.png')
    print('TRAINER: existing 10_train_yolo.py (unchanged)')

if __name__=='__main__':
    main()

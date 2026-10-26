#!/usr/bin/env python3
"""Step 10: train YOLOv8n on the Kvasir-SEG train split and select best on val.

Use only after 09_prepare_yolo_from_json.py passed. No test data is accessed.
"""
import argparse
from pathlib import Path
import yaml
from core.utils import config, work_dir, save_json, require_gpu


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', default='yolov8n.pt', help='pretrained YOLOv8n model path or name')
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--device', default='0', help='CUDA device index for Ultralytics')
    a = p.parse_args()
    if a.batch < 1 or a.epochs < 1 or a.imgsz < 32:
        raise ValueError('invalid training parameters')
    require_gpu()
    cfg = config(); work = work_dir(cfg)
    data_yaml = work / 'yolo_dataset' / 'data.yaml'
    if not data_yaml.is_file():
        raise FileNotFoundError('Run 09_prepare_yolo_from_json.py first')
    with data_yaml.open(encoding='utf8') as f:
        data = yaml.safe_load(f)
    if 'test' in data:
        raise RuntimeError('Training YAML must not contain test path')
    from ultralytics import YOLO
    # Separate output directory and run name, do not overwrite prior training.
    run_dir = work / 'yolo_runs' / 'polyp_v8n'
    if run_dir.exists():
        raise FileExistsError(f'{run_dir} exists. Review previous results before re-running.')
    settings = {'weights':a.weights,'epochs':a.epochs,'batch':a.batch,
                'imgsz':a.imgsz,'patience':a.patience,'seed':a.seed,
                'data_yaml':str(data_yaml),'split':'train/val only'}
    save_json(work/'yolo_dataset'/'train_config.json', settings)
    model = YOLO(a.weights)
    print('START DETECTOR TRAINING: only train/val, no test.')
    model.train(data=str(data_yaml), epochs=a.epochs, batch=a.batch,
                imgsz=a.imgsz, patience=a.patience, seed=a.seed,
                workers=a.workers, device=a.device, amp=True,
                project=str(work/'yolo_runs'), name='polyp_v8n', exist_ok=False)
    best = run_dir / 'weights' / 'best.pt'
    if not best.is_file():
        raise RuntimeError(f'Training finished but best model not found: {best}')
    print('\n===== STEP 10 | YOLO TRAIN RESULT =====')
    print('BEST WEIGHT:', best)
    print('VAL RESULTS:', run_dir/'results.csv')
    print('NEXT: check detector validation metrics and sample predictions before end-to-end MedSAM inference.')


if __name__=='__main__': main()

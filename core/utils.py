"""Reproducibility and fail-fast helpers (PyTorch 2.1 compatible)."""
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parents[1]


def config(path=None):
    path = Path(path) if path else HERE / 'config.yaml'
    with path.open(encoding='utf-8') as f:
        d = yaml.safe_load(f)
    # Paths in config are relative to the project directory, not shell pwd.
    for group, keys in [('project', ['work_dir']), ('data', ['images_dir', 'masks_dir']),
                        ('model', ['medsam_repo', 'checkpoint'])]:
        for key in keys:
            v = d[group].get(key)
            if v is not None:
                p = Path(v).expanduser()
                d[group][key] = str((HERE/p).resolve() if not p.is_absolute() else p.resolve())
    return d


def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def require_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError('GPU 不可用：请确认 torch 是 CUDA 版本，并运行 00_preflight.py')
    return torch.device('cuda:0')


def save_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open(encoding='utf-8') as f: return json.load(f)


def save_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def sha(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()


def weight_id(p):
    p=Path(p).resolve(); s=p.stat()
    return {'path':str(p),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def work_dir(cfg):
    w=Path(cfg['project']['work_dir']);w.mkdir(parents=True, exist_ok=True)
    return w

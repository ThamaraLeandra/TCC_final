
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_with_tta_min.py
Avalia um modelo salvo N vezes com TTA, SEM usar DataLoader de PIL (evita collate error).
Uso:
python evaluate_with_tta_min.py --weights output/resnet18_global.pth --data_root client2_original --runs 10 --tta 8 --img_size 224 --csv_out resultados_eval_tta.csv
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import pandas as pd

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def find_classes_and_samples(root: Path):
    classes = [d.name for d in root.iterdir() if d.is_dir()]
    classes.sort()
    class_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for cls in classes:
        for p in (root / cls).rglob("*"):
            if p.is_file() and is_img(p):
                samples.append((p, class_to_idx[cls]))
    return classes, samples

def confusion_matrix(y_true, y_pred, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def precision_recall_f1_macro_from_cm(cm: np.ndarray):
    num_classes = cm.shape[0]
    precisions, recalls, f1s = [], [], []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))

def get_resnet18_model(num_classes: int):
    model = models.resnet18(weights=None)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def build_tta_transforms(img_size: int, tta: int):
    if tta <= 1:
        return [T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])]
    tfms = []
    for _ in range(tta):
        tfms.append(T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.RandomRotation(degrees=10)], p=0.5),
            T.RandomApply([T.RandomAffine(degrees=0, translate=(0.05,0.05), scale=(0.95,1.05))], p=0.5),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ]))
    return tfms

@torch.inference_mode()
def eval_once(weights_path: str, data_root: str, img_size: int, tta: int, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(data_root)
    classes, samples = find_classes_and_samples(data_root)
    num_classes = len(classes)

    model = get_resnet18_model(num_classes).to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()

    tta_transforms = build_tta_transforms(img_size, tta)
    y_true_all, y_pred_all = [], []

    for p, y in samples:
        y_true_all.append(y)
        logits_acc = None
        for t in tta_transforms:
            with Image.open(p) as im:
                im = im.convert("RGB")
                x = t(im).unsqueeze(0).to(device)
                logits = model(x)
                logits_acc = logits if logits_acc is None else (logits_acc + logits)
        pred = torch.argmax(logits_acc, dim=1).item()
        y_pred_all.append(pred)

    cm = confusion_matrix(y_true_all, y_pred_all, num_classes)
    acc = float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0
    prec_macro, rec_macro, f1_macro = precision_recall_f1_macro_from_cm(cm)
    return {"accuracy": acc, "precision_macro": prec_macro, "recall_macro": rec_macro, "f1_macro": f1_macro}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--tta", type=int, default=8)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--csv_out", type=str, default="")
    args = ap.parse_args()

    rows = []
    for run in range(1, args.runs+1):
        print(f"\n===== EVAL RUN {run}/{args.runs} | TTA={args.tta} =====")
        m = eval_once(args.weights, args.data_root, args.img_size, args.tta)
        print(f"accuracy: {m['accuracy']:.6f}  f1_macro: {m['f1_macro']:.6f}  precision_macro: {m['precision_macro']:.6f}  recall_macro: {m['recall_macro']:.6f}")
        rows.append({"run": run, **m})

    if args.csv_out:
        pd.DataFrame(rows).to_csv(args.csv_out, index=False)
        print(f"[OK] CSV salvo em {args.csv_out}")

if __name__ == "__main__":
    main()

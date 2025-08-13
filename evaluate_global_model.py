import argparse
import os
import torch
from torch.utils.data import DataLoader
from dataset.kaggle_dataset import TumorKaggleDataset, get_transforms
from model.resnet_model import get_resnet18_model
from utils.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_eval(weights_path: str, data_root: str, batch_size=64):
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Pesos não encontrados: {weights_path}")
    print(f"Carregando modelo de {weights_path}")
    model = get_resnet18_model(pretrained=False, num_classes=4).to(DEVICE)
    sd = torch.load(weights_path, map_location=DEVICE)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tfm = get_transforms(img_size=224, augment=False)
    ds = TumorKaggleDataset(data_root, transform=tfm, verbose=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                    pin_memory=torch.cuda.is_available())

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().tolist())
            y_true.extend(yb.tolist())
    m = compute_metrics(y_true, y_pred)
    print(f"Global | acc={m['accuracy']:.4f} | f1={m['f1_score']:.4f} "
          f"| prec={m['precision']:.4f} | rec={m['recall']:.4f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data_root", required=True)
    args = ap.parse_args()
    run_eval(args.weights, args.data_root)

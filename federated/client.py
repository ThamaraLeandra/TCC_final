import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from dataset.kaggle_dataset import TumorKaggleDataset, CLASS_MAP, get_transforms
from model.resnet_model import get_resnet18_model
from utils.metrics import compute_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLIENT_VERSION = "client_v3_rgb_imagenet_macro_val"
print(f"[CLIENT] Versão ativa: {CLIENT_VERSION}")

USE_AUG      = os.getenv("AUG", "1") == "1"       
USE_MIXUP    = os.getenv("MIXUP", "0") == "1"
USE_FOCAL    = os.getenv("FOCAL", "0") == "1"
LOCAL_EPOCHS = int(os.getenv("LOCAL_EPOCHS", "1"))
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "32"))
IMG_SIZE     = int(os.getenv("IMG_SIZE", "224"))

def _mixup(x, y, alpha=0.2):
    if alpha <= 0: 
        return x, y, None
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx, :]
    return mixed_x, (y, y[idx], lam), True

def _build_loaders(root: str, batch_size=BATCH_SIZE, num_workers=0):
    tfm = get_transforms(img_size=IMG_SIZE, augment=USE_AUG)
    ds = TumorKaggleDataset(root, transform=tfm, verbose=True)

    labels = [y for _, y in ds.samples]
    idx = np.arange(len(ds))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_idx, val_idx = next(sss.split(idx, labels))
    train_ds = Subset(ds, train_idx)
    val_ds   = Subset(ds, val_idx)

    train_labels = np.array([labels[i] for i in train_idx])
    class_sample_count = np.bincount(train_labels, minlength=len(CLASS_MAP))
    class_weights = class_sample_count.sum() / (class_sample_count + 1e-6)
    class_weights = class_weights / class_weights.sum()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    return train_loader, val_loader, len(train_idx), len(val_idx), class_weights_t

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction="none")
        self.reduction = reduction
    def forward(self, logits, target):
        ce = self.ce(logits, target)  # [N]
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss

def _make_model_and_optim(class_weights, lr_head=1e-3):
    model = get_resnet18_model(pretrained=True, num_classes=len(CLASS_MAP)).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights) if USE_FOCAL else nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr_head, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    return model, optimizer, scheduler, criterion

class TumorClient(fl.client.NumPyClient):
    def __init__(self, data_path: str):
        if not os.path.isdir(data_path):
            print(f"[Dataset] Caminho inexistente: {data_path}")
            raise SystemExit(1)

        self.trainloader, self.valloader, self.n_train, self.n_val, self.class_weights = _build_loaders(
            data_path, batch_size=BATCH_SIZE, num_workers=0
        )
        self.model, self.optimizer, self.scheduler, self.criterion = _make_model_and_optim(self.class_weights)

        print(f"  - train: {self.n_train} | val: {self.n_val} | aug={USE_AUG} mixup={USE_MIXUP} focal={USE_FOCAL}")

        xb_chk, yb_chk = next(iter(self.valloader))
        print(f"[SANITY] batch_val shape = {xb_chk.shape} (esperado [N,3,224,224])")
        print(f"[SANITY] y unique (val) = {sorted(set(yb_chk.tolist()))}")
        if xb_chk.dim() != 4 or xb_chk.shape[1] != 3:
            raise RuntimeError(
                f"Transforms incorretos! Esperado 3 canais, veio {xb_chk.shape}. "
                "Confira dataset.kaggle_dataset.get_transforms() e __getitem__ (convert('RGB'))."
            )

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        for (k, _), p in zip(state_dict.items(), parameters):
            state_dict[k] = torch.as_tensor(p)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()

        local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
        for _ in range(local_epochs):
            for xb, yb in self.trainloader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.to(DEVICE, non_blocking=True)

                self.optimizer.zero_grad()
                if USE_MIXUP:
                    xb, mix, ok = _mixup(xb, yb, alpha=0.2)
                    logits = self.model(xb)
                    if ok:
                        y1, y2, lam = mix
                        loss = lam * self.criterion(logits, y1) + (1 - lam) * self.criterion(logits, y2)
                    else:
                        loss = self.criterion(logits, yb)
                else:
                    logits = self.model(xb)
                    loss = self.criterion(logits, yb)

                loss.backward()
                self.optimizer.step()
            self.scheduler.step()

        metrics = self._evaluate_loader(self.valloader, label="val")
        return self.get_parameters({}), self.n_train, metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        metrics = self._evaluate_loader(self.valloader, label="val")
        return float(1.0 - metrics["accuracy"]), self.n_val, metrics

    @torch.no_grad()
    def _evaluate_loader(self, loader, label="val"):
        self.model.eval()
        all_preds, all_true = [], []
        for xb, yb in loader:
            assert xb.dim() == 4 and xb.shape[1] == 3, f"Esperado [N,3,H,W], veio {xb.shape}"
            xb = xb.to(DEVICE, non_blocking=True)
            logits = self.model(xb)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds); all_true.extend(yb.tolist())

        # métricas macro
        prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_preds, average="macro", zero_division=0)
        acc = float(np.mean(np.array(all_true) == np.array(all_preds)))

        # por-classe + matriz de confusão + dist de previsão (diagnóstico)
        prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(all_true, all_preds, average=None, zero_division=0)
        cm = confusion_matrix(all_true, all_preds, labels=list(range(len(CLASS_MAP))))
        unique, counts = np.unique(np.array(all_preds), return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique, counts)}

        print(
            f"📊 Cliente ({label}): acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} | "
            f"prec_por_classe={np.round(prec_c,3).tolist()} | preds_dist={dist}"
        )
        # descomente se quiser ver a matriz sempre:
        # print(f"CM:\n{cm}")

        return {"accuracy": acc, "f1_score": f1, "precision": prec, "recall": rec}

def start_client(data_path: str):
    client = TumorClient(data_path)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

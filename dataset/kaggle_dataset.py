import os
from glob import glob
from typing import Callable, List, Tuple, Dict
from PIL import Image
from torch.utils.data import Dataset

CLASS_MAP: Dict[str, int] = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _is_img(p: str) -> bool:
    return os.path.splitext(p)[1].lower() in IMG_EXT

def _iter_class_dirs(root: str) -> List[Tuple[str, int]]:
    items = []
    for cname, cid in CLASS_MAP.items():
        cdir = os.path.join(root, cname)
        if os.path.isdir(cdir):
            items.append((cdir, cid))
    return items

class TumorKaggleDataset(Dataset):
    def __init__(self, root: str, transform: Callable = None, verbose: bool = False):
        self.root = root
        self.transform = transform
        self.samples = []  
        per_class_count = {k: 0 for k in CLASS_MAP.keys()}

        for cdir, cid in _iter_class_dirs(root):
            files = [p for p in glob(os.path.join(cdir, "**", "*"), recursive=True) if _is_img(p)]
            self.samples.extend([(p, cid) for p in files])
            cname = os.path.basename(cdir.rstrip("/\\"))
            per_class_count[cname] = len(files)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"Nenhuma imagem encontrada em {root}. "
                f"Esperado pastas: {', '.join(CLASS_MAP.keys())}"
            )

        if verbose:
            total = len(self.samples)
            print(f"[Dataset] {root} -> {total} imagens")
            for cname in CLASS_MAP.keys():
                print(f"  - {cname}: {per_class_count.get(cname, 0)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Carrega em RGB direto (3 canais) para bater com ResNet-18 pretrained
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

KaggleTumorDataset = TumorKaggleDataset

# Transforms padronizados para TODOS os clientes
def get_transforms(img_size=224, augment: bool = False):
    from torchvision import transforms
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    aug_list = []
    if augment:
        aug_list = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        ]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        *aug_list,
        transforms.ToTensor(),                       
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]),
    ])

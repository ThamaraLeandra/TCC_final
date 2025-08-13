import argparse, os
from pathlib import Path
from PIL import Image, ImageFile
from concurrent.futures import ThreadPoolExecutor, as_completed

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]

def is_img(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT

def process_one(src: Path, dst: Path, img_size: int) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            im = im.resize((img_size, img_size), Image.BILINEAR)
            dst = dst.with_suffix(".jpg")
            im.save(dst, format="JPEG", quality=95, optimize=True)
        return True
    except Exception as e:
        print(f"[ERRO] {src} -> {e}")
        return False

def normalize_split_name(name: str) -> str:
    n = name.lower()
    if n in {"train", "training"}:  return "training"
    if n in {"test", "testing"}:    return "Testing" 
    return name

def run_split(src_root: Path, split: str, dst_root: Path, out_train_name: str, img_size: int):
    norm = normalize_split_name(split)
    if norm == "training":
        src_split = src_root / "training" 
        dst_split = dst_root / out_train_name
    elif norm == "Testing":
        src_split = src_root / "Testing"
        dst_split = dst_root / "Testing"
    else:
        src_split = src_root / split
        dst_split = dst_root / split

    if not src_split.exists():
        print(f"[AVISO] Split não encontrado: {src_split}")
        return

    print(f"[RUN] {src_split}  ->  {dst_split}  (size={img_size})")

    tasks = []
    total = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        for cls in CLASSES:
            src_cls = src_split / cls
            if not src_cls.exists():
                print(f"[AVISO] Classe ausente no split {split}: {cls}")
                continue
            for p in src_cls.rglob("*"):
                if p.is_file() and is_img(p):
                    rel = p.relative_to(src_split) 
                    dst = dst_split / rel
                    tasks.append(ex.submit(process_one, p, dst, img_size))
                    total += 1
        ok = 0
        for fut in as_completed(tasks):
            ok += 1 if fut.result() else 0
    print(f"[OK] {split}: {ok}/{total} imagens processadas -> {dst_split}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", default="dataset_kaggle", help="pasta de origem (onde estão training/ e Testing/)")
    ap.add_argument("--dst_root", default="dataset_kaggle_preprocessed", help="pasta de destino")
    ap.add_argument("--splits", nargs="+", default=["training", "Testing"], help="quais splits processar")
    ap.add_argument("--out_train_name", default="Train", help="nome do split de treino na saída (ex.: Train)")
    ap.add_argument("--img_size", type=int, default=224)
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[START] src={src_root}  dst={dst_root}  splits={args.splits}  size={args.img_size}")
    for sp in args.splits:
        run_split(src_root, sp, dst_root, args.out_train_name, args.img_size)
    print("[DONE]")

if __name__ == "__main__":
    main()

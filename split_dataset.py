# split_mass_shape_dataset.py
# מפצל את mass_shape_data ל- train/val/test ביחסים 70/10/20
# העתקה בלבד (לא מוחק/מזיז את המקור)

from pathlib import Path
import random
import shutil

# ====== עדכון נתיבים במידת הצורך ======
SOURCE_ROOT = Path("dataset_mask_testing")
DEST_ROOT   = Path("TESTING_DATASET_SPLIT")
# ======================================

SPLIT = {"train": 0.70, "val": 0.10, "test": 0.20}
SEED = 42  # לשחזור

# סיומות מותרות (אפשר להרחיב)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(dir_path: Path):
    return [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def split_counts(n: int, p_train: float, p_val: float, p_test: float):
    # חישוב אינטגרלי שמכסה את כל התמונות
    n_train = int(n * p_train)
    n_val   = int(n * p_val)
    n_test  = n - n_train - n_val
    return n_train, n_val, n_test

def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"לא נמצאה תיקיית מקור: {SOURCE_ROOT}")

    random.seed(SEED)
    ensure_dir(DEST_ROOT)

    # איסוף קטגוריות (תיקיות משנה)
    classes = sorted([d for d in SOURCE_ROOT.iterdir() if d.is_dir()])
    if not classes:
        print("לא נמצאו תיקיות קטגוריה בתוך mass_shape_data.")
        return

    # יצירת תיקיות היעד
    for split in SPLIT.keys():
        for c in classes:
            ensure_dir(DEST_ROOT / split / c.name)

    summary = []

    # לכל קטגוריה – פיצול העתקתי
    for c in classes:
        imgs = list_images(c)
        random.shuffle(imgs)
        n = len(imgs)
        n_train, n_val, n_test = split_counts(n, SPLIT["train"], SPLIT["val"], SPLIT["test"])

        # חלוקה
        parts = {
            "train": imgs[:n_train],
            "val":   imgs[n_train:n_train + n_val],
            "test":  imgs[n_train + n_val:],
        }

        # העתקה
        for split, files in parts.items():
            dst_dir = DEST_ROOT / split / c.name
            for src in files:
                shutil.copy2(src, dst_dir / src.name)

        summary.append((c.name, n, n_train, n_val, n_test))

    # הדפסה מסכמת
    total = sum(x[1] for x in summary)
    total_train = sum(x[2] for x in summary)
    total_val   = sum(x[3] for x in summary)
    total_test  = sum(x[4] for x in summary)

    print(f"[DONE] Dataset created at: {DEST_ROOT}")
    print("Per-class counts (total/train/val/test):")
    for name, n, tr, va, te in summary:
        print(f"  {name:15s}  {n:5d}  {tr:5d}  {va:5d}  {te:5d}")
    print(f"TOTAL: {total}  train={total_train}  val={total_val}  test={total_test}")

if __name__ == "__main__":
    main()

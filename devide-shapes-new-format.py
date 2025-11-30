from pathlib import Path
import shutil
import re
import pandas as pd

# ========= שנה כאן את הנתיבים =========
IMAGES_DIR = Path("binary_png_sharon_rod")  # היכן שהתמונות .png
MAPPING_XLSX_PATH = Path("first_try.xlsx")  # הקובץ עם filename ו-shape
OUT_BASE = Path("first_try_out")           # תיקיית יעד (אפשר לשנות)
# =====================================


def norm_shape_name_for_dir(s: str) -> str:
    """שם תיקייה בטוח: רווחים -> '_', והסרת מפרידים בעייתיים."""
    safe = s.strip().replace("/", "_").replace("\\", "_")
    safe = re.sub(r"\s+", "_", safe)
    return safe


def load_fname2shape(mapping_xlsx: Path) -> dict[str, str]:
    """
    קורא את קובץ האקסל ומחזיר מילון:
        { 'D1-0001_MLO_R_patch.png': 'ROUND', ... }
    מניח שיש עמודות בשם: 'filename' ו-'shape'.
    """
    # אם יש שם גיליון ספציפי – אפשר להוסיף sheet_name="ID_shape" או אחר
    df = pd.read_excel(mapping_xlsx, engine="openpyxl")

    # ניקוי רווחים וערכים חסרים
    df["filename"] = df["filename"].astype(str).str.strip()
    df["shape"] = df["shape"].astype(str).str.strip()

    df = df[
        (df["filename"] != "") &
        (df["shape"] != "") &
        (df["shape"].str.lower() != "nan")
    ]

    # מילון: שם קובץ -> צורה
    return dict(zip(df["filename"], df["shape"]))


def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"IMAGES_DIR לא קיים: {IMAGES_DIR}")
    if not MAPPING_XLSX_PATH.exists():
        raise FileNotFoundError(f"קובץ המיפוי לא נמצא: {MAPPING_XLSX_PATH}")

    fname2shape = load_fname2shape(MAPPING_XLSX_PATH)
    print(f"[INFO] נטענו {len(fname2shape)} filename עם shape מתוך: {MAPPING_XLSX_PATH}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    total_files = 0
    matched = 0
    skipped_no_shape = 0

    # אם יש רק PNG:
    for img in IMAGES_DIR.glob("*.png"):
        total_files += 1
        fname = img.name  # שם הקובץ בדיוק כמו בתיקייה

        shape = fname2shape.get(fname)
        if not shape:
            skipped_no_shape += 1
            continue

        dst_dir = OUT_BASE / norm_shape_name_for_dir(shape)
        dst_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, dst_dir / img.name)
        matched += 1

    print(f"סיימתי. נסרקו {total_files} קבצים.")
    print(f"  הועתקו (עם התאמת filename->shape): {matched}")
    print(f"  דילוג – אין shape עבור filename במפה: {skipped_no_shape}")
    print(f"הפלט נמצא תחת: {OUT_BASE}")


if __name__ == "__main__":
    main()

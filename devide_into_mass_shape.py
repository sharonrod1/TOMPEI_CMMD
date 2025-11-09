# dispatch_images_by_shape.py
# דרישות: pip install pandas openpyxl

from pathlib import Path
import shutil
import re
import pandas as pd

# ========= שנה כאן את הנתיבים =========
IMAGES_DIR = Path(r"C:\Users\User\PycharmProjects\TOMPEI-CMMD-main\binary_mask_png")  # היכן שהתמונות .png
MAPPING_XLSX_PATH = Path(r"C:\Users\User\DATA_CMMD\ID_SHAPE_from_mass.xlsx")      # הקובץ שיצרנו (עם ID, SHAPE)
OUT_BASE = Path(r"C:\Users\User\PycharmProjects\TOMPEI-CMMD-main\mass_shape_data_masking") # תיקיית יעד
# =====================================

def _cell_str(x):
    return str(x).strip() if x is not None else ""

def _norm_pid(s: str) -> str | None:
    """חלץ בדיוק D?-dddd מכל מחרוזת ונרמל ל-UPPER."""
    s = _cell_str(s).upper().replace("\u00A0", " ").strip()
    m = re.search(r"\bD\d-\d{4}\b", s)
    return m.group(0) if m else None

def get_pid_from_filename(name: str) -> str | None:
    """
    'D1-0001_MLO_R_patch.png' -> 'D1-0001'
    לוקח את כל מה שלפני ה-underscore הראשון ומוודא פורמט PID חוקי.
    """
    head = name.split("_", 1)[0]
    return _norm_pid(head)

def norm_shape_name_for_dir(s: str) -> str:
    """שם תיקייה בטוח: רווחים -> '_', והסרת מפרידים בעייתיים."""
    safe = s.strip().replace("/", "_").replace("\\", "_")
    safe = re.sub(r"\s+", "_", safe)
    return safe

def load_id2shape(mapping_xlsx: Path) -> dict[str, str]:
    """קורא את גיליון ID_SHAPE ומחזיר מילון {ID: SHAPE} אחרי סינון NaN/ריקים."""
    df = pd.read_excel(mapping_xlsx, sheet_name="ID_SHAPE", engine="openpyxl")
    # ניקוי ערכים חסרים/ריקים
    df["ID"] = df["ID"].astype(str).str.strip().str.upper()
    df["SHAPE"] = df["SHAPE"].astype(str).str.strip()
    df = df[(df["ID"] != "") & (df["SHAPE"] != "") & (df["SHAPE"].str.lower() != "nan")]
    return dict(zip(df["ID"], df["SHAPE"]))

def main():
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"IMAGES_DIR לא קיים: {IMAGES_DIR}")
    if not MAPPING_XLSX_PATH.exists():
        raise FileNotFoundError(f"מפת ID->SHAPE לא נמצאה: {MAPPING_XLSX_PATH}")

    id2shape = load_id2shape(MAPPING_XLSX_PATH)
    print(f"[INFO] נטענו {len(id2shape)} IDs עם SHAPE מתוך: {MAPPING_XLSX_PATH}")

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    total_files = 0
    matched = 0
    skipped_no_id = 0
    skipped_no_shape = 0

    # אם יש רק PNG:
    for img in IMAGES_DIR.glob("*.png"):
        total_files += 1
        pid = get_pid_from_filename(img.name)
        if not pid:
            skipped_no_id += 1
            continue

        shape = id2shape.get(pid)
        if not shape:
            skipped_no_shape += 1
            continue

        dst_dir = OUT_BASE / norm_shape_name_for_dir(shape)
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, dst_dir / img.name)  # שומר מטה-דאטה בסיסי
        matched += 1

    print(f"סיימתי. נסרקו {total_files} קבצים.")
    print(f"  הועתקו (עם התאמת ID->SHAPE): {matched}")
    print(f"  דילוג – בלי PID חוקי בשם: {skipped_no_id}")
    print(f"  דילוג – אין SHAPE למזהה במפה: {skipped_no_shape}")
    print(f"הפלט נמצא תחת: {OUT_BASE}")

if __name__ == "__main__":
    main()

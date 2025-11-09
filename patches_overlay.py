import os, json, numpy as np, pydicom
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from glob import glob
from collections import defaultdict
import pandas as pd
from tqdm import tqdm


#this code takes the json and the dcm and creates a dcm that more focused around the lession " the patch " made bv me
MIN_POLY_PIX = 10  # דלג על מסכות עם פחות מ-10 פיקסלים "דולקים"



# ---------- הגדרות נתיבים ----------
CSV_PATH   = "TOMPEI-CMMD.csv"
JSON_DIR   = r"TOMPEI-CMMD_v01_20250123"
ROOT_DICOM = r"C:\Users\User\DATA_CMMD\manifest-1616439774456\CMMD"
OUT_DIR    = "binary_mask_dcm_new"
MARGIN_FRAC = 0.10        # שוליים סביב ה-BBox
UPDATE_IPP  = True        # לעדכן ImagePositionPatient בהתאם לחיתוך

# ---------- מסכה מפוליגון(ים) ----------
def polygons_to_mask(polygons, img_shape):
    """
    קולט רשימת פוליגונים (כל פוליגון: רשימת נקודות {"x","y"}),
    ומחזיר מסכה בוליאנית בגודל התמונה שבה 1 בתוך הצורה ו-0 מחוץ לה.
    מנסה להשתמש ב-OpenCV, ואם לא קיים — נופל ל-matplotlib.path.
    """
    H, W = img_shape
    try:
        import cv2
        mask = np.zeros((H, W), dtype=np.uint8)
        pts_list = []
        for poly in polygons:
            if not poly:
                continue
            pts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.int32)
            # CV2 מצפה לצורת Nx1x2 או רשימה של כאלה
            pts_list.append(pts.reshape((-1, 1, 2)))
        if pts_list:
            cv2.fillPoly(mask, pts_list, 1)
        return mask.astype(bool)
    except Exception:
        # Fallback: matplotlib.path
        from matplotlib.path import Path
        yy, xx = np.mgrid[0:H, 0:W]
        coords = np.vstack((xx.ravel(), yy.ravel())).T
        mask = np.zeros(H*W, dtype=bool)
        for poly in polygons:
            if not poly:
                continue
            verts = np.array([[p["x"], p["y"]] for p in poly], dtype=np.float32)
            path = Path(verts)
            mask |= path.contains_points(coords)
        return mask.reshape(H, W)


def largest_polygon(polygons):
    def area(poly):
        x = np.array([p["x"] for p in poly], dtype=np.float32)
        y = np.array([p["y"] for p in poly], dtype=np.float32)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return max(polygons, key=area) if polygons else None
def largest_polygon_in_range(polygons, img_shape, min_frac=1e-4, max_frac=0.8):
    H, W = img_shape
    img_area = float(H * W)
    def area(poly):
        x = np.array([p["x"] for p in poly], dtype=np.float32)
        y = np.array([p["y"] for p in poly], dtype=np.float32)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    cands = [(area(p), p) for p in polygons]
    cands = [(a, p) for (a, p) in cands if min_frac*img_area <= a <= max_frac*img_area]
    if cands:
        return max(cands, key=lambda t: t[0])[1]
    # אם אין מתאים בתחום – נבחר את הגדול ביותר בכלל
    return max(polygons, key=lambda p: area(p)) if polygons else None

# ---------- עזרים ----------
def choose_lesion_polygon(polygons, img_shape, min_frac=1e-4, max_frac=5e-2):
    H, W = img_shape
    img_area = float(H * W)
    def area(poly):
        x = np.array([p["x"] for p in poly], dtype=np.float32)
        y = np.array([p["y"] for p in poly], dtype=np.float32)
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    cands = []
    for poly in polygons:
        a = area(poly); frac = a / img_area
        if min_frac <= frac <= max_frac:
            cands.append((a, poly))
    if cands:
        cands.sort(key=lambda t: t[0])  # הכי קטן בין הסבירים
        return cands[0][1]
    return min(polygons, key=lambda p: 0.5*abs(
        np.dot(np.array([q["x"] for q in p]), np.roll(np.array([q["y"] for q in p]), -1)) -
        np.dot(np.array([q["y"] for q in p]), np.roll(np.array([q["x"] for q in p]), -1))
    )) if polygons else None

def bbox_from_polygon(points, margin_frac=0.10, img_shape=None):
    xs = np.array([p["x"] for p in points], dtype=np.float32)
    ys = np.array([p["y"] for p in points], dtype=np.float32)
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    w, h = (maxx - minx), (maxy - miny)
    minx -= w * margin_frac; maxx += w * margin_frac
    miny -= h * margin_frac; maxy += h * margin_frac
    if img_shape is not None:
        H, W = img_shape
        minx = max(0.0, minx); miny = max(0.0, miny)
        maxx = min(float(W - 1), maxx); maxy = min(float(H - 1), maxy)
    return tuple(int(round(v)) for v in [minx, miny, maxx, maxy])

def crop_to_bbox(arr, bbox):
    minx, miny, maxx, maxy = bbox
    return arr[miny:maxy+1, minx:maxx+1]

def save_patch_as_dicom(ds, patch_arr, bbox, out_path, update_position=True):
    new = ds.copy()
    new.Rows, new.Columns = patch_arr.shape
    new.PixelData = patch_arr.tobytes()

    # הבטחת מטא של קובץ ו-TS לא דחוס
    if not hasattr(new, "file_meta"):
        from pydicom.dataset import Dataset
        new.file_meta = Dataset()
    new.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    new.is_little_endian = True
    new.is_implicit_VR = False

    # SOP Instance UID חדש
    new.SOPInstanceUID = generate_uid()

    # עדכון ImagePositionPatient לפי היסט (אם יש מידע רלוונטי)
    if update_position and ("PixelSpacing" in ds) and ("ImagePositionPatient" in ds):
        ps = [float(x) for x in ds.PixelSpacing]          # [row_spacing, col_spacing]
        ipp = [float(x) for x in ds.ImagePositionPatient] # [X, Y, Z]
        minx, miny, _, _ = bbox
        new.ImagePositionPatient = [ipp[0] + minx*ps[1], ipp[1] + miny*ps[0], ipp[2]]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    new.save_as(out_path)

def get_mammo_laterality_and_view(path_dicom: str):
    ds = pydicom.dcmread(path_dicom)
    laterality = getattr(ds.get((0x0020, 0x0062)), "value", "Unknown")
    view = getattr(ds.get((0x0018, 0x5101)), "value", None)
    if not view:
        try:
            seq = ds[(0x0054, 0x0220)].value  # View Code Sequence
            if seq and seq[0].get((0x0008, 0x0104)):
                meaning = seq[0][(0x0008, 0x0104)].value
                view = "CC" if "cranio" in str(meaning).lower() else "MLO"
        except Exception:
            view = "Unknown"
    view_map = {"CC":"CC","MLO":"MLO","CRANIO-CAUDAL":"CC","CRANIO CAUDAL":"CC","MEDIO-LATERAL OBLIQUE":"MLO"}
    view_norm = view_map.get(str(view).strip().upper(), str(view).strip().upper())
    return (str(laterality).upper() or "UNKNOWN"), (view_norm or "UNKNOWN")

# ---------- ריצה ראשית ----------
def main():
    df = pd.read_csv(CSV_PATH).sort_values(by="Subject ID").reset_index(drop=True)
    uid_list = df["Series UID"].astype(str).tolist()
    id_list  = df["Subject ID"].astype(str).tolist()

    paths_json = sorted(glob(os.path.join(JSON_DIR, "*.json")))
    json_index = defaultdict(list)
    for p in paths_json:
        sid = os.path.basename(p).split("_")[0].strip().upper()
        json_index[sid].append(p)

    os.makedirs(OUT_DIR, exist_ok=True)

    stats = dict(total=len(id_list), patches=0, json_miss=0, dcm_miss=0, no_match=0, errors=0)

    for uid, subj_id in tqdm(list(zip(uid_list, id_list)), total=len(id_list)):
        sid = subj_id.strip().upper()

        candidates = json_index.get(sid, [])
        if not candidates:
            stats["json_miss"] += 1
            continue

        dicoms = sorted(glob(fr"{ROOT_DICOM}\**\{uid}\*.dcm", recursive=True))
        if not dicoms:
            stats["dcm_miss"] += 1
            continue

        # מפה lat+view -> נתיב
        lat_view = {}
        for pdcm in dicoms:
            try:
                lat, view = get_mammo_laterality_and_view(pdcm)
                lat_view[(lat, view)] = pdcm
            except Exception:
                continue

        for path_json in candidates:
            parts = os.path.basename(path_json).split("_")
            if len(parts) < 3:
                continue
            view_json = parts[1].upper()
            lat_json  = parts[2].upper()

            pdcm = lat_view.get((lat_json, view_json))
            if not pdcm:
                stats["no_match"] += 1
                continue

            try:
                ds = pydicom.dcmread(pdcm)
                img = ds.pixel_array  # שומר מקור

                with open(path_json, "r") as f:
                    data = json.load(f)
                polys = [d.get("cgPoints", []) for d in data]


                lesion_count = 0
                for idx, poly in enumerate(polys, start=1):
                    # דילוג על פוליגון לא תקין/קצר
                    if not poly or len(poly) < 3:
                        continue

                    # מסכה בדיוק לצורת הפוליגון הנוכחי
                    mask = polygons_to_mask([poly], img.shape)
                    if not mask.any():
                        continue

                    # (אופציונלי) דלג על פוליגון זעיר
                    if 'MIN_POLY_PIX' in globals():
                        if mask.sum() < max(1, int(MIN_POLY_PIX)):
                            continue

                    # מעטפת הדוקה למסכה
                    ys, xs = np.where(mask)
                    minx, maxx = int(xs.min()), int(xs.max())
                    miny, maxy = int(ys.min()), int(ys.max())
                    bbox = (minx, miny, maxx, maxy)

                    # חיתוך תמונה+מסכה
                    patch_img = crop_to_bbox(img, bbox)
                    patch_mask = crop_to_bbox(mask.astype(np.uint8), bbox).astype(bool)

                    # השחרת הרקע, שמירת עוצמת אות מקורית בתוך הפוליגון
                    patch_masked = np.where(patch_mask, patch_img, 0).astype(img.dtype)

                    # שם קובץ לפי האינדקס (poly1, poly2, ...) — סדר זהה לסדר ב-JSON
                    out_name = f"{sid}_{view_json}_{lat_json}_{idx}.dcm"
                    out_path = os.path.join(OUT_DIR, out_name)

                    # שמירה כ-DICOM חדש עם עדכון IPP לפי ההיסט של ה-bbox
                    save_patch_as_dicom(ds, patch_masked, bbox, out_path, update_position=UPDATE_IPP)

                    lesion_count += 1
                    stats["patches"] += 1

                # אם לא נשמר אף פוליגון עבור ה-JSON הזה
                if lesion_count == 0:
                    stats["no_match"] += 1


            except Exception as e:
                print(f"[ERROR] {sid} {view_json} {lat_json}: {e}")
                stats["errors"] += 1

    print("\n=== PATCH DCM STATS ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
import numpy as np
import math

def square_bbox_from_polygon(points, margin_frac=0.10, img_shape=None):
    """
    יוצר ריבוע שמכסה את הפוליגון + שוליים יחסיים.
    משמר את המרכז, ומזיז את הריבוע לגמרי לתוך גבולות התמונה (ללא חיתוך).
    """
    xs = np.array([p["x"] for p in points], dtype=np.float32)
    ys = np.array([p["y"] for p in points], dtype=np.float32)

    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    w, h = (maxx - minx), (maxy - miny)

    # מרכז המלבן המקורי
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    # אורך צלע ריבוע: הצד הארוך * (1 + 2*margin)
    base = max(w, h)
    side = base * (1.0 + 2.0 * margin_frac)

    if img_shape is not None:
        H, W = img_shape
        # אם הצד גדול מדי לתמונה – נגביל למינימום האפשרי
        side = min(side, float(min(W, H) - 1))

    # מלבן ריבועי ראשוני סביב המרכז
    x0 = cx - side / 2.0
    y0 = cy - side / 2.0
    x1 = cx + side / 2.0
    y1 = cy + side / 2.0

    if img_shape is not None:
        H, W = img_shape
        # הזזה פנימה אם חורג, בלי לשנות את ה-side
        if x0 < 0:
            shift = -x0
            x0 += shift; x1 += shift
        if x1 > W - 1:
            shift = x1 - (W - 1)
            x0 -= shift; x1 -= shift
        if y0 < 0:
            shift = -y0
            y0 += shift; y1 += shift
        if y1 > H - 1:
            shift = y1 - (H - 1)
            y0 -= shift; y1 -= shift

        # אם גם אחרי הזזה יש מגבלה (למשל side גדול מגודל התמונה), נכווץ מעט
        x0 = max(0.0, x0); y0 = max(0.0, y0)
        x1 = min(float(W - 1), x1); y1 = min(float(H - 1), y1)
        # ודא שזה ריבוע אחרי עיגול: נתקן לפי המינימום מהצירים
        side_int = int(round(min(x1 - x0, y1 - y0)))
        # נבנה מחדש על פי פינה שמאלית-עליונה
        x1 = x0 + side_int
        y1 = y0 + side_int

    # המרה ל-int עם הכלה מלאה של הפיקסלים
    return tuple(int(round(v)) for v in [x0, y0, x1, y1])

if __name__ == "__main__":
    main()

import os
import pydicom
import numpy as np
import imageio.v2 as imageio

# נתיב התיקייה המקורית עם קבצי DCM
source_dir = "binary_mask_dcm_new_sharon_rod"
# נתיב לתיקייה החדשה
target_dir = "binary_png_sharon_rod"

# יצירת תיקייה חדשה אם אינה קיימת
os.makedirs(target_dir, exist_ok=True)

# קובץ טקסט לרישום הנתיבים
log_path = os.path.join(target_dir, "paths_log.txt")

with open(log_path, "w", encoding="utf-8") as log_file:
    log_file.write(f"Source Directory: {source_dir}\n")
    log_file.write(f"Target Directory: {target_dir}\n\n")
    log_file.write("Converted files:\n")

    for filename in os.listdir(source_dir):
        if filename.lower().endswith(".dcm"):
            dcm_path = os.path.join(source_dir, filename)
            ds = pydicom.dcmread(dcm_path)
            img = ds.pixel_array.astype(np.float32)
            
            # נרמול טווח הערכים (לא חובה אם רוצים "כמה שפחות עיבוד")
            img -= img.min()
            img /= img.max()
            img = (img * 255).astype(np.uint8)

            png_filename = os.path.splitext(filename)[0] + ".png"
            png_path = os.path.join(target_dir, png_filename)
            imageio.imwrite(png_path, img)
            
            log_file.write(f"{dcm_path} -> {png_path}\n")

print(f"ההמרה הושלמה. הקבצים נמצאים ב: {target_dir}")
print(f"נוצר גם קובץ: {log_path}")

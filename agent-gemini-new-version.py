import os
import pandas as pd
from google.api_core import exceptions
from google import genai
from google.genai import types
import time

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

IMAGES_DIR = "binary_png_sharon_rod"

# Main results CSV (append)
OUTPUT_CSV = "latest_gemini_mass_results-agata.csv"

# Unsafe/default list CSV (append)
UNSAFE_OUTPUT_CSV = "unsafe_default_files.csv"

# Process N images per run
MAX_IMAGES = 250

# Start offset (0=first 250, 250=next 250, ...)
START_INDEX = 250

GEMINI_MODEL = "gemini-3-pro-preview"

# ---------------------------------------------------------------------
# GEMINI CLIENT
# ---------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Please export your Gemini API key, e.g.\n"
        '  export GEMINI_API_KEY="YOUR_GEMINI_API_KEY_HERE"'
    )

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------

SCHEMA = [
    "file_id",
    "mass_shape",
    "mass_margins",
    "mass_density",
    "architectural_distortion",
    "associated_features",
    "bi_rads_assessment",
    "pathology",
    "confidence",
]

DEFAULT_ROW = {
    "file_id": "missing_file_id",
    "mass_shape": "irregular",
    "mass_margins": "indistinct",
    "mass_density": "high",
    "architectural_distortion": "absent",
    "associated_features": "none",
    "bi_rads_assessment": "4B",
    "pathology": "malignant",
    "confidence": "0.5",
}


def append_df_to_csv(df: pd.DataFrame, csv_path: str, columns: list[str]):
    """
    Append df to csv_path. Writes header only if file does not exist or is empty.
    Ensures column order.
    """
    df = df.reindex(columns=columns)
    file_exists = os.path.exists(csv_path)
    write_header = True
    if file_exists:
        try:
            write_header = (os.path.getsize(csv_path) == 0)
        except OSError:
            write_header = True

    df.to_csv(
        csv_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )


def parse_model_csv_row(raw_answer: str):
    """
    Returns: (parsed_dict, used_default: bool)
    used_default=True only when parsing failed and we fell back to DEFAULT_ROW.
    """
    line = (raw_answer or "").strip()

    # If the model returned multiple lines, take the last non-empty line
    if "\n" in line:
        lines = [l.strip() for l in line.splitlines() if l.strip()]
        if lines:
            line = lines[-1]

    parts = line.split(",")

    if len(parts) == len(SCHEMA):
        out = dict(zip(SCHEMA, parts))
        try:
            conf = float(out["confidence"])
            conf = max(0.0, min(1.0, conf))
            out["confidence"] = f"{conf:.4f}".rstrip("0").rstrip(".")
        except Exception:
            pass
        return out, False

    fallback = dict(DEFAULT_ROW)
    fallback["file_id"] = parts[0] if len(parts) > 0 and parts[0] else "missing_file_id"
    return fallback, True


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------

def classify_image_with_gemini(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    system_prompt = (
        "You are an assistant for medical image analysis, specializing in tumor morphology. "
        "You receive breast mass image patches. "
        "Your job is to assign structured labels using the requested schema and allowed values only."
    )

    user_prompt = """
You are given a single breast MASS tumor image patch. The case always contains a mass.

The image itself contains:
1) The tumor patch


Your task is to output EXACTLY ONE single-line CSV row with a fixed schema.

Return exactly one line (no extra text, no header, no markdown), in this exact order:
file_id,mass_shape,mass_margins,mass_density,architectural_distortion,associated_features,bi_rads_assessment,pathology,confidence

Rules:
- file_id must be this file id for every file "0001"
- You MUST choose exactly one value for each field from the allowed values below.
- Do NOT output "unknown", "N/A", or leave fields empty.
- If uncertain, choose the most likely value based on visual evidence.
- Do NOT add spaces after commas.
- confidence must be a number between 0 and 1 (e.g., 0.81).

Allowed values:
mass_shape: round|oval|lobulated|irregular
mass_margins: circumscribed|obscured|microlobulated|indistinct|spiculated
mass_density: fat-containing|low|equal|high
architectural_distortion: present|absent
associated_features: none|skin_thickening|nipple_retraction|edema|axillary_adenopathy|other
bi_rads_assessment: 1|2|3|4A|4B|4C|5|6
pathology: benign|malignant

Return only the single CSV row.
"""

    safety_settings = [
        types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
        types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
    ]

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=20000,
            temperature=0.0,
            safety_settings=safety_settings,
        ),
    )

    raw_answer = (response.text or "").strip()
    print("RAW ANSWER:", raw_answer)
    return raw_answer


# ---------------------------------------------------------------------
# FOLDER PROCESSING (APPEND CSV)
# ---------------------------------------------------------------------

def process_folder(
    images_dir: str,
    output_csv: str,
    unsafe_output_csv: str,
    batch_size: int = 30,
    max_images: int = 250,
    start_index: int = 0,
):
    rows_buffer = []
    unsafe_buffer = []
    processed = 0

    all_pngs = [fn for fn in sorted(os.listdir(images_dir)) if fn.lower().endswith(".png")]
    target_pngs = all_pngs[start_index : start_index + max_images]

    if not target_pngs:
        print(
            f"No PNG files found for slice. start_index={start_index}, "
            f"max_images={max_images}, total_pngs={len(all_pngs)}"
        )
        return

    print(f"Total PNGs in folder: {len(all_pngs)}")
    print(f"Processing slice: [{start_index} : {start_index + max_images}] -> {len(target_pngs)} files")
    print(f"Appending to: {output_csv}")
    print(f"Appending unsafe list to: {unsafe_output_csv}")

    for filename in target_pngs:
        full_path = os.path.join(images_dir, filename)
        print(f"Classifying {filename} ...")

        success = False
        used_default = False
        parsed = None

        while not success:
            try:
                raw_row = classify_image_with_gemini(full_path)
                parsed, used_default = parse_model_csv_row(raw_row)
                success = True

                # Force file_id to actual filename (without extension)
                parsed["file_id"] = os.path.splitext(filename)[0]
                print(parsed)

            except exceptions.ResourceExhausted:
                print("  Rate limit exceeded. Waiting for 30 seconds before retrying...")
                time.sleep(30)

            except Exception as e:
                print(f"  ERROR for {filename}: {e}")
                parsed = dict(DEFAULT_ROW)
                parsed["file_id"] = os.path.splitext(filename)[0]
                used_default = True
                success = True  # stop retrying on generic errors

        if used_default:
            unsafe_buffer.append({"FILE_NAME": filename})

        rows_buffer.append(parsed)
        processed += 1

        # Flush buffers every batch_size
        if processed % batch_size == 0:
            append_df_to_csv(pd.DataFrame(rows_buffer), output_csv, SCHEMA)
            append_df_to_csv(pd.DataFrame(unsafe_buffer), unsafe_output_csv, ["FILE_NAME"])
            print(f"Appended progress for {processed} images.")
            rows_buffer.clear()
            unsafe_buffer.clear()

    # Final flush (remaining)
    if rows_buffer:
        append_df_to_csv(pd.DataFrame(rows_buffer), output_csv, SCHEMA)
    if unsafe_buffer:
        append_df_to_csv(pd.DataFrame(unsafe_buffer), unsafe_output_csv, ["FILE_NAME"])

    print(f"\nDone. Appended {processed} images to {output_csv}.")


if __name__ == "__main__":
    process_folder(
        IMAGES_DIR,
        OUTPUT_CSV,
        UNSAFE_OUTPUT_CSV,
        batch_size=30,
        max_images=100,
        start_index=500,
    )

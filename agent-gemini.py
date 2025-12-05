import os
import pandas as pd

from google import genai
from google.genai import types

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Folder with PNG images
IMAGES_DIR = "binary_png_sharon_rod"  # change if needed

# Excel output file
OUTPUT_EXCEL = "gemini_results-many-choise.xlsx"

# (יכול להישאר, אבל כבר לא באמת בשימוש – Gemini לא מוגבל לרשימה הזו)
SHAPES = ["round/oval", "polygonal", "lobulated", "irregular"]

# Gemini model name (can be adjusted if you use a different model)
GEMINI_MODEL = "gemini-2.5-pro"  # see Gemini docs for other options


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
# CLASSIFICATION – GROK-STYLE BEHAVIOR
# ---------------------------------------------------------------------

def classify_image_with_gemini(image_path: str) -> str:
    """
    Classify a single PNG image using Gemini into a FREE-TEXT
    medical / radiology morphology label (like the GROK version).

    Output format expected from Gemini:
      - יכול להיות הסבר קצר
      - בשורה האחרונה בדיוק:
            label: <some term>

    הפונקציה מחזירה רק את התווית (החלק אחרי 'label:'),
    ואם לא מוצאת – מחזירה את כל הטקסט הגולמי.
    """
    import base64

    # Read image bytes
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    system_prompt = (
        "You are an assistant for medical image analysis, specializing in tumor morphology. "
        "You receive binary lesion mask images (white lesion on black background). "
        "Your job is to assign a single global SHAPE LABEL to the lesion, using standard "
        "and well-known medical/radiology terminology (e.g., spiculated, round, oval, lobulated, irregular, stellate, etc.). "
        "Always think as a careful radiology assistant: pick the term that best describes the GLOBAL outline of the lesion."
    )

    user_prompt = """
You are given a binary mask image of a single lesion (white foreground on black background).

Your task:
- Look at the GLOBAL outline shape of the lesion.
- Assign ONE main shape label using a short, standard term from medical / radiology morphology.
- Use commonly used terminology in breast imaging / oncology when possible, such as:
    - round
    - oval
    - lobulated
    - irregular
    - spiculated
    - stellate
    - linear
    - branching
    - tubular
    - nodular
    - etc.
- You are NOT limited to this list, but the label MUST be a real, recognizable medical/radiology term.

Guidelines:
- Focus on the global contour of the lesion in the mask.
- Prefer concise labels (usually a single word or at most two words).
- If the shape has radiating lines or spikes from the center → terms like "spiculated" / "stellate" may be appropriate.
- If it is smooth and approximately circular → "round".
- If it is smoothly elongated → "oval".
- If it has several smooth lobes → "lobulated".
- If it is chaotic, very jagged, or clearly atypical → "irregular".
- If it is clearly long and tube-like or branching → terms like "linear", "tubular", or "branching".

Reasoning:
- Briefly explain in 1–3 short sentences why you chose this label.
- You may mention key visual characteristics (compact vs elongated, smooth vs jagged, presence of spikes, etc.).

Output format:
- You may write a short explanation.
- On the LAST line, write exactly:
  label: <a single main shape term, e.g., spiculated, round, oval, lobulated, irregular, stellate, linear, branching, tubular, etc.>
- Do not output anything else after that line.
"""

    # Call Gemini with image + text prompt
    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=img_bytes,
                mime_type="image/png",
            ),
            user_prompt,
        ],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=200,
            temperature=0.0,
        ),
    )

    raw_answer = (response.text or "").strip()
    print("RAW ANSWER:", raw_answer)

    # ננסה לחלץ את השורה שמתחילה ב-"label:"
    label = None
    for line in reversed(raw_answer.splitlines()):
        line_stripped = line.strip()
        if line_stripped.lower().startswith("label:"):
            candidate = line_stripped.split("label:", 1)[1].strip()
            label = candidate
            break

    # אם מצאנו label – נחזיר אותו כמו שהוא (יכול להיות "spiculated", "oval", "stellate", וכו')
    if label:
        return label

    # אם לא מצאנו – נחזיר את כל התשובה הגולמית כדי שלא נאבד מידע
    return raw_answer


# ---------------------------------------------------------------------
# FOLDER PROCESSING
# ---------------------------------------------------------------------

def process_folder(images_dir: str, output_excel: str, batch_size: int = 30):
    """
    Loops over all PNG files in a folder, classifies them with Gemini,
    and saves the results into an Excel file.
    Progress is saved every `batch_size` images.
    """
    rows = []
    processed = 0

    for filename in sorted(os.listdir(images_dir)):
        if not filename.lower().endswith(".png"):
            continue

        full_path = os.path.join(images_dir, filename)
        print(f"Classifying {filename} ...")

        try:
            shape = classify_image_with_gemini(full_path)
        except Exception as e:
            print(f"  ERROR for {filename}: {e}")
            shape = "error"

        print(f"  → {shape}")

        rows.append({"filename": filename, "shape": shape})
        processed += 1

        # Save every `batch_size` images
        if processed % batch_size == 0:
            df = pd.DataFrame(rows)
            df.to_excel(output_excel, index=False)
            print(f"Saved progress for {processed} images to {output_excel}")

    if not rows:
        print("No PNG files were found in the folder.")
        return

    # Final save
    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"\nFinal Excel file saved with {processed} images: {output_excel}")


if __name__ == "__main__":
    process_folder(IMAGES_DIR, OUTPUT_EXCEL)

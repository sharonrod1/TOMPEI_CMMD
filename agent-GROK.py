import os
import base64
import pandas as pd
from openai import OpenAI

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

# Folder with PNG images
IMAGES_DIR = "binary_png_sharon_rod"  # change if needed

# Excel output file
OUTPUT_EXCEL = "grok_results-final.xlsx"

# Shape categories
SHAPES = ["round/oval", "polygonal", "lobulated", "irregular"]

# Grok model name (check xAI docs for the exact model id and adjust if needed)
GROK_MODEL = "grok-2-vision-1212"  # placeholder; change to the actual Grok model name


# ---------------------------------------------------------------------
# GROK CLIENT
# ---------------------------------------------------------------------

# Grok/xAI client – uses XAI_API_KEY from environment and xAI base URL
grok_client = OpenAI(
    base_url="https://api.x.ai/v1",
    api_key=os.getenv("XAI_API_KEY"),
)

if grok_client.api_key is None:
    raise RuntimeError(
        "XAI_API_KEY is not set. Please export your Grok API key, e.g.\n"
        '  export XAI_API_KEY="YOUR_GROK_API_KEY_HERE"'
    )


# ---------------------------------------------------------------------
# CLASSIFICATION
# ---------------------------------------------------------------------

def classify_image_with_grok(image_path: str) -> str:
    """
    Classify a single PNG image using Grok into one of SHAPES.
    Returns ONLY: 'round/oval', 'lobulated', 'polygonal', or 'irregular'.
    """
    import json

    # Read and encode image
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    system_prompt = (
        "You are an assistant that classifies the global shape in a binary lesion mask "
        "image into one of four predefined shapes."
    )

    # Same definitions as before, but with a VERY strict output rule (JSON only)
    user_prompt = """
You are given a binary lesion mask image (white lesion on black background).

Your task: classify the GLOBAL OUTLINE SHAPE of the lesion into EXACTLY ONE of the following four categories:

1. round/oval
   - One compact blob.
   - Mostly smooth border with no deep concavities.
   - Circular or elliptical.

2. lobulated
   - Still a compact blob.
   - Border has 2 or more rounded lobes (“bumps”).
   - No long thin extensions and no strong concavities.

3. polygonal
   - Outline shows straight-ish segments and visible corners.
   - Angular appearance rather than smooth.

4. irregular
   - Clearly non-compact or distorted.
   - Examples: long/snaking shapes, branching, star-like, strong concavities, long thin projections.
   - A shape that looks like a tube/worm, even with smooth edges, MUST be “irregular”.

Decision rules:

Step 1 — Compact vs non-compact:
- If the shape is long, elongated, curved like a snake, branching, or has long thin arms → classify as “irregular”.

Step 2 — If compact:
- Smooth border → “round/oval”.
- Smooth border with multiple rounded lobes → “lobulated”.
- Angular border with corners → “polygonal”.

Tie-breaking rules:
- Borderline round vs oval → “round/oval”.
- Borderline round/oval vs lobulated → prefer “round/oval” unless lobes are obvious.
- Borderline lobulated vs polygonal → choose the category that best matches the border (smooth lobes → lobulated; straight segments/corners → polygonal).
- Only choose “irregular” for clearly non-compact shapes.

STRICT OUTPUT FORMAT:
Return ONLY a single JSON object on one line:

{"label": "round/oval"}
{"label": "lobulated"}
{"label": "polygonal"}
{"label": "irregular"}

No explanations. No text before or after. Only the JSON.
"""

    response = grok_client.chat.completions.create(
        model=GROK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        },
                    },
                ],
            },
        ],
        max_completion_tokens=50,
        temperature=0.0,
    )

    raw_answer = response.choices[0].message.content.strip()
    print("RAW ANSWER:", raw_answer)

    # Try JSON parse first
    label = None
    try:
        data = json.loads(raw_answer)
        label = str(data.get("label", "")).lower().strip()
    except Exception:
        # If it's not valid JSON, fall back to text parsing
        raw_lower = raw_answer.lower()
        for shape in SHAPES:
            if shape in raw_lower:
                return shape
        return raw_lower  # last resort: return whatever the model said

    # Normalize label to one of SHAPES
    for shape in SHAPES:
        if shape in label:
            return shape

    # If we reach here, JSON existed but label was weird
    return label or raw_answer


# ---------------------------------------------------------------------
# FOLDER PROCESSING
# ---------------------------------------------------------------------

def process_folder(images_dir: str, output_excel: str, batch_size: int = 30):
    """
    Loops over all PNG files in a folder, classifies them with Grok,
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
            shape = classify_image_with_grok(full_path)
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

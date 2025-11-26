import os
import base64
import pandas as pd
from openai import OpenAI

# 1. Change this to your images folder path
IMAGES_DIR = "binary_png_sharon_rod"  # e.g.: r"D:\images\masks"

# 2. The Excel file that will be saved
OUTPUT_EXCEL = "first_try.xlsx"

# 3. The predefined list of shapes
SHAPES = ["round/oval", "polygonal", "lobulated", "irregular"]

# Create OpenAI client (API key is taken from the environment variable)
client = OpenAI()


def classify_image_with_llm(image_path: str) -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    system_prompt = (
        "You are an assistant that classifies the global shape in a binary lesion mask image "
        "into one of four predefined shapes."
    )

    user_prompt = """
You are given a binary mask image of a single lesion (white foreground on black background).

Your task is to classify the GLOBAL outline shape of the lesion into exactly ONE of these four categories:

1. round/oval
   - Approximately circular OR elliptical.
   - One compact blob.
   - Width and height may be similar (round) or one axis clearly longer (oval).
   - Border mostly smooth, without deep indentations or long projections.
   - If the shape is a single, reasonably compact blob, it is usually "round/oval".

2. lobulated
   - The lesion is still relatively compact (not long and snakelike).
   - Border is globally smooth but composed of MULTIPLE rounded lobes.
   - You can clearly see 2 or more rounded bumps along the border.
   - No long thin arms, no very deep concavities.

3. polygonal
   - The outline is mostly made of straight-ish segments with visible corners.
   - The shape has a faceted, polygon-like appearance.
   - Corners/vertices are apparent, but there are no long thin projections.
   - The shape may be compact or slightly elongated, but looks angular rather than smooth.

4. irregular
   - Use this for clearly non-compact or highly distorted shapes.
   - Examples:
       - Long, snakelike, S-shaped, or branching forms.
       - Shapes with long thin arms or extensions away from the main body.
       - Very jagged, chaotic, or star-like outlines.
       - Strong concavities and indentations that break the idea of a single compact blob.
   - IMPORTANT: A shape that looks like a long bent tube or snake, even with smooth edges,
     should be classified as "irregular", NOT lobulated or round/oval.

Decision strategy:

Step 1 – Compact vs non-compact
- If the lesion is long, strongly elongated, S-shaped, branching, or looks like a worm/snake or path:
    → classify as "irregular" (even if the edges are smooth).
- Only consider "round/oval" or "lobulated" when the lesion is relatively compact.

Step 2 – If compact:
- If the lesion is essentially a single blob with a mostly smooth border:
    → "round/oval".
- If you clearly see multiple rounded lobes on a compact blob:
    → "lobulated".
- If the contour is mostly straight segments with corners:
    → "polygonal".

Bias rules (to reduce overuse of irregular but still detect obvious irregular cases):
- Borderline between round and oval → "round/oval".
- Borderline between round/oval and lobulated on a compact blob → choose "round/oval" unless lobes are very clear.
- Borderline between lobulated and polygonal → choose the one that best matches (smooth lobes → "lobulated"; straight edges and corners → "polygonal").
- Borderline between lobulated/polygonal and irregular:
    - If the shape is fairly compact → prefer "lobulated" or "polygonal".
    - If the shape is clearly long, snakelike, or branching → MUST choose "irregular".

Output format:
- On the LAST line, write exactly:
  label: <one of round/oval | lobulated | polygonal | irregular>
- Do not output anything else after that line.
"""



    response = client.chat.completions.create(
        model="gpt-5.1",
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

    raw_answer = response.choices[0].message.content.strip().lower()
    print("RAW ANSWER:", raw_answer)  # <-- TEMPORARY DEBUG

    # Try to parse the "label: xxx" line from the bottom
    label = None
    for line in reversed(raw_answer.splitlines()):
        line = line.strip()
        if line.startswith("label:"):
            candidate = line.split("label:", 1)[1].strip()
            label = candidate
            break

    if label:
        # normalize and match to allowed shapes
        for shape in SHAPES:
            if shape in label:
                return shape

    # Fallback: old behavior
    for shape in SHAPES:
        if shape in raw_answer:
            return shape

    return raw_answer


def process_folder(images_dir: str, output_excel: str, batch_size: int = 30):
    """
    Loops over all PNG files in a folder, classifies them with the LLM,
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

        shape = classify_image_with_llm(full_path)
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

    # Final save for any remaining images not exactly on a batch boundary
    df = pd.DataFrame(rows)
    df.to_excel(output_excel, index=False)
    print(f"\nFinal Excel file saved with {processed} images: {output_excel}")


if __name__ == "__main__":
    process_folder(IMAGES_DIR, OUTPUT_EXCEL)

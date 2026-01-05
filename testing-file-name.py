# gemini_print_file_id_debug.py
import os
from google import genai
from google.genai import types

# ---------------------------
# CONFIG
# ---------------------------
IMAGES_DIR = "binary_png_sharon_rod"
GEMINI_MODEL = "gemini-3-pro-preview"  # keep your model

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        'PowerShell example:\n  $env:GEMINI_API_KEY="YOUR_KEY_HERE"\n'
    )

client = genai.Client(api_key=GEMINI_API_KEY)

SYSTEM_PROMPT = (
    "You are a strict OCR extractor. Return only the requested value, nothing else."
)

USER_PROMPT = (
    "Extract the file name that is the name of the file "
    "Output ONLY the file name WITHOUT the .png extension.\n"
    "Return exactly one line, no extra text.\n"
    "If you truly cannot find any file name in the image text, output NOT_FOUND."
)

def get_response_text(resp) -> str:
    """Robustly extract text from different response shapes."""
    if getattr(resp, "text", None):
        return (resp.text or "").strip()

    # Fallback: candidates[0].content.parts[*].text
    cands = getattr(resp, "candidates", None) or []
    if not cands:
        return ""

    parts = getattr(cands[0], "content", None)
    if not parts:
        return ""

    parts_list = getattr(parts, "parts", None) or []
    texts = []
    for p in parts_list:
        t = getattr(p, "text", None)
        if t:
            texts.append(t)
    return "\n".join(texts).strip()


def extract_file_id(image_path: str) -> tuple[str, object]:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    image_part = types.Part.from_bytes(data=img_bytes, mime_type="image/png")
    text_part = types.Part(text=USER_PROMPT)

    # Put BOTH image + prompt inside a single user content (more reliable)
    contents = [
        types.Content(role="user", parts=[image_part, text_part])
    ]

    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=1000,
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
            ],
        ),
    )

    return get_response_text(resp), resp


def main():
    if not os.path.isdir(IMAGES_DIR):
        raise RuntimeError(f"Folder not found: {IMAGES_DIR}")

    pngs = [fn for fn in sorted(os.listdir(IMAGES_DIR)) if fn.lower().endswith(".png")]
    if not pngs:
        print(f"No PNG files found in: {IMAGES_DIR}")
        return

    for fn in pngs:
        path = os.path.join(IMAGES_DIR, fn)
        try:
            out, resp = extract_file_id(path)

            # If still empty -> print diagnostics
            if not out:
                cands = getattr(resp, "candidates", None) or []
                if not cands:
                    print(f"{fn} -> EMPTY (no candidates returned)")
                    continue

                fr = getattr(cands[0], "finish_reason", None)
                safety = getattr(cands[0], "safety_ratings", None)
                print(f"{fn} -> EMPTY | finish_reason={fr} | safety_ratings={safety}")
                continue

            print(f"{fn} -> {out}")

        except Exception as e:
            print(f"{fn} -> ERROR: {e}")


if __name__ == "__main__":
    main()


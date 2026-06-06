import os, json, glob, tempfile
import gradio as gr
from PIL import Image
from receipt_extractor import ReceiptExtractor

MODEL_ID = os.environ.get("MODEL_ID", "Tanishq71/sroie-layoutlmv3")
MAX_SIDE = 1600  # downscale cap so OCR stays fast on free CPU

print("Loading pipeline from", MODEL_ID)
extractor = ReceiptExtractor(model_dir=MODEL_ID)
print("Pipeline ready.")

EMPTY = [["Company", ""], ["Date", ""], ["Address", ""], ["Total", ""]]

# Pick up any receipt images placed in the examples/ folder (no hardcoded names)
_example_files = sorted(glob.glob("examples/*.jpg") + glob.glob("examples/*.png"))
EXAMPLES = [[f] for f in _example_files]


def _downscale(img):
    w, h = img.size
    m = max(w, h)
    if m > MAX_SIDE:
        s = MAX_SIDE / m
        img = img.resize((int(w * s), int(h * s)))
    return img


def predict(image):
    if image is None:
        return EMPTY, "Please upload a receipt image first.", None
    image = _downscale(image.convert("RGB"))
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        image.save(tmp.name, format="JPEG")
        tmp_path = tmp.name
    try:
        entities, timings = extractor.extract_with_timing(tmp_path)
    finally:
        os.unlink(tmp_path)
    table = [
        ["Company", entities["company"] or "(not detected)"],
        ["Date",    entities["date"]    or "(not detected)"],
        ["Address", entities["address"] or "(not detected)"],
        ["Total",   entities["total"]   or "(not detected)"],
    ]
    timing = "OCR %.1fs   .   Model %.2fs   .   Total %.1fs" % (
        timings["ocr_s"], timings["model_s"], timings["total_s"])
    json_path = os.path.join(tempfile.gettempdir(), "receipt_result.json")
    with open(json_path, "w") as f:
        json.dump(entities, f, indent=2)
    return table, timing, json_path


def clear_all():
    return None, EMPTY, "", None


DESCRIPTION = """
# Receipt Entity Extractor
Extracts **Company**, **Date**, **Address**, and **Total** from receipts using
**PaddleOCR + LayoutLMv3** fine-tuned on the SROIE dataset (Malaysian receipts).

**Performance (SROIE test, 347 receipts):** macro F1 **0.81 fuzzy** / **0.42 exact**.
Date recall is boosted by a regex fallback when the model abstains; address fuzzy-F1 is 0.91.

**Latency:** runs on free CPU. Large images are downscaled to 1600px; expect ~5-20s.
First load after inactivity takes 1-2 min (cold start).
"""

with gr.Blocks(theme=gr.themes.Soft(), title="Receipt Entity Extractor") as demo:
    gr.Markdown(DESCRIPTION)
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Receipt", height=420,
                                   sources=["upload", "clipboard", "webcam"])
            with gr.Row():
                submit_btn = gr.Button("Extract Entities", variant="primary")
                clear_btn  = gr.Button("Clear", variant="secondary")
        with gr.Column(scale=1):
            output_table  = gr.Dataframe(headers=["Field", "Extracted Value"],
                                         datatype=["str", "str"], value=EMPTY,
                                         interactive=False, label="Extracted Entities")
            timing_text   = gr.Textbox(label="Processing time", interactive=False)
            download_file = gr.File(label="Download result (JSON)")

    if EXAMPLES:
        gr.Examples(examples=EXAMPLES, inputs=image_input,
                    label="Example receipts — click to try")

    submit_btn.click(predict, inputs=image_input,
                     outputs=[output_table, timing_text, download_file])
    clear_btn.click(clear_all, inputs=None,
                    outputs=[image_input, output_table, timing_text, download_file])

if __name__ == "__main__":
    demo.launch()

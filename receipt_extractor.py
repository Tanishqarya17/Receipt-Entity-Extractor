"""
End-to-end receipt entity extraction pipeline.
Combines PaddleOCR + LayoutLMv3 + post-processing behind a single API.
Used by Day 21 (Gradio demo) and Day 22 (Hugging Face Spaces).
"""

import os, sys, time
import torch
from PIL import Image
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from paddleocr import PaddleOCR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from postprocessing import postprocess, recover_date

ENTITY_FIELDS = ["company", "date", "address", "total"]
SROIE_LABELS  = ["O", "B-COMPANY", "I-COMPANY", "B-DATE", "I-DATE",
                 "B-ADDRESS", "I-ADDRESS", "B-TOTAL", "I-TOTAL"]


class ReceiptExtractor:
    """End-to-end receipt entity extraction pipeline."""

    def __init__(self, model_dir, device=None, ocr_max_side=1500, apply_postprocess=True):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.apply_postprocess = apply_postprocess
        self.id2label = {i: l for i, l in enumerate(SROIE_LABELS)}

        self.processor = LayoutLMv3Processor.from_pretrained(model_dir, apply_ocr=False)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        self.ocr = PaddleOCR(
            use_textline_orientation=False, lang="en",
            text_det_limit_side_len=ocr_max_side,
        )

    @staticmethod
    def _normalize_box(polygon, W, H):
        xs = [p[0] for p in polygon]; ys = [p[1] for p in polygon]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        return [max(0,min(1000,int(1000*x0/W))), max(0,min(1000,int(1000*y0/H))),
                max(0,min(1000,int(1000*x1/W))), max(0,min(1000,int(1000*y1/H)))]

    def _run_ocr(self, image_path):
        raw = self.ocr.predict(image_path)
        if not raw or not isinstance(raw, list): return [], []
        page = raw[0]
        data = page.get("res", page) if isinstance(page, dict) else {}
        polys = data.get("dt_polys", []); texts = data.get("rec_texts", [])
        words, polygons = [], []
        for poly, text in zip(polys, texts):
            words.append(text)
            polygons.append([[float(p[0]), float(p[1])] for p in poly])
        return words, polygons

    def _run_model(self, image, words, polygons):
        W, H = image.size
        boxes = [self._normalize_box(p, W, H) for p in polygons]
        enc = self.processor(image, words, boxes=boxes, return_tensors="pt",
                             truncation=True, padding="max_length", max_length=512)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = self.model(**enc).logits
        pred_ids = logits.argmax(-1).squeeze(0).tolist()
        word_ids = self.processor.tokenizer(
            words, boxes=boxes, truncation=True, max_length=512,
            return_offsets_mapping=False
        ).word_ids()
        word_labels = {}
        for ti, wi in enumerate(word_ids):
            if wi is not None and wi not in word_labels:
                word_labels[wi] = self.id2label.get(pred_ids[ti], "O")
        return [word_labels.get(i, "O") for i in range(len(words))]

    @staticmethod
    def _walk_bio(words, labels):
        entities = {f: "" for f in ENTITY_FIELDS}
        fm = {"COMPANY":"company","DATE":"date","ADDRESS":"address","TOTAL":"total"}
        cf, ct = None, []
        for word, label in zip(words, labels):
            if label.startswith("B-"):
                if cf and ct: entities[cf] = " ".join(ct)
                cf = fm.get(label[2:]); ct = [word] if cf else []
            elif label.startswith("I-") and cf == fm.get(label[2:]):
                ct.append(word)
            else:
                if cf and ct: entities[cf] = " ".join(ct)
                cf, ct = None, []
        if cf and ct: 
          entities[cf] = " ".join(ct)
        if not entities['date']:                     
            entities['date'] = recover_date(words)    
        return entities

    def extract(self, image_path):
        entities, _ = self.extract_with_timing(image_path)
        return entities

    def extract_with_timing(self, image_path):
        timings = {}
        image = Image.open(image_path).convert("RGB")
        t0 = time.perf_counter()
        words, polygons = self._run_ocr(image_path)
        timings["ocr_s"] = time.perf_counter() - t0
        if not words:
            timings.update({"model_s": 0.0, "postprocess_s": 0.0})
            timings["total_s"] = timings["ocr_s"]
            return {f: "" for f in ENTITY_FIELDS}, timings
        t0 = time.perf_counter()
        word_labels = self._run_model(image, words, polygons)
        timings["model_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        raw = self._walk_bio(words, word_labels)
        entities = postprocess(raw) if self.apply_postprocess else raw
        timings["postprocess_s"] = time.perf_counter() - t0
        timings["total_s"] = sum(v for k,v in timings.items() if k.endswith("_s"))
        return entities, timings

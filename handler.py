"""
RunPod Serverless Handler — SAM3 Floor Plan Segmentation
arXiv:2511.16719 | transformers >= 4.45 | facebook/sam3

Stratégie hybride basée sur les capacités réelles de SAM3 sur plans d'architecte :

SAM3 PCS (text prompt) fonctionne sur les pièces avec équipements dessinés :
  - wc, toilet, bathroom, laundry room → très bons scores (0.7–0.93)
  - bedroom, living room, kitchen → scores nuls (rectangles vides sur le plan)

Pour les pièces sans fixtures, on utilise SAM3 en mode point prompt :
  - GPT-4o fournit le centre de la pièce
  - SAM3 segmente "ce qui se trouve à ce point"

Payload:
{
    "image_b64": "<base64>",
    "rooms": [
        {
            "id": 1,
            "type_en": "bathroom",        # label anglais pour SAM3 text
            "type_fr": "salle de bain",   # label français pour affichage
            "point": [cx, cy],            # centre en pixels (de GPT-4o)
            "box":   [x1, y1, x2, y2]    # bbox en pixels (de GPT-4o, optionnel)
        }
    ]
}

Response:
{
    "masks": [{"id": 1, "mask_rle": {...}, "score": 0.91, "method": "text_pcs"}],
    "model_used": "sam3"
}
"""
import base64
import io
import logging
import os
import time

import numpy as np
import runpod
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("RUNPOD_SECRET_HF_TOKEN")

_model = None
_processor = None


def load_sam3():
    global _model, _processor
    if _model is not None:
        return _model, _processor

    logger.info(f"Loading {SAM3_MODEL_ID}...")
    t0 = time.time()
    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    from transformers import Sam3Processor, Sam3Model
    _processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model = Sam3Model.from_pretrained(SAM3_MODEL_ID, **kwargs).to(DEVICE).eval()
    logger.info(f"SAM3 ready in {time.time()-t0:.1f}s")
    return _model, _processor


# ─── Concepts with distinctive visual features on floor plans ─────────────────
# These reliably work with SAM3 text prompts (confirmed on architectural drawings)
TEXT_PROMPT_TYPES = {
    "wc", "toilet", "bathroom", "bath", "shower room",
    "laundry room", "laundry", "utility room",
    "garage", "carport",
}


def segment_room(model, processor, image: Image.Image, room: dict) -> dict:
    """
    Segment one room using the best available strategy:
    1. SAM3 text prompt  — for rooms with distinctive fixtures (wc, bathroom…)
    2. SAM3 point prompt — for plain rooms (bedroom, living room, kitchen…)
    """
    h, w = image.size[1], image.size[0]
    rid = room["id"]
    type_en = room.get("type_en", room.get("text", "room")).lower().strip()
    point = room.get("point")   # [cx, cy] in pixels
    box = room.get("box")       # [x1, y1, x2, y2] in pixels

    # ── Strategy 1: SAM3 text PCS for fixture-bearing rooms ───────────────────
    if type_en in TEXT_PROMPT_TYPES:
        mask, score = _text_prompt(model, processor, image, type_en, h, w)
        if mask is not None and mask.sum() > 0:
            logger.info(f"[text] id={rid} '{type_en}' score={score:.3f} px={mask.sum()}")
            return {"id": rid, "mask": mask, "score": score, "method": "text_pcs"}

    # ── Strategy 2: SAM3 point prompt (GPT-4o center) ─────────────────────────
    if point is not None:
        mask, score = _point_prompt(model, processor, image, point, box, h, w)
        if mask is not None and mask.sum() > 0:
            logger.info(f"[point] id={rid} '{type_en}' score={score:.3f} px={mask.sum()}")
            return {"id": rid, "mask": mask, "score": score, "method": "point"}

    # ── Strategy 3: text prompt regardless of type (last attempt) ─────────────
    if type_en not in TEXT_PROMPT_TYPES:
        mask, score = _text_prompt(model, processor, image, type_en, h, w)
        if mask is not None and mask.sum() > 0:
            logger.info(f"[text_fallback] id={rid} '{type_en}' score={score:.3f}")
            return {"id": rid, "mask": mask, "score": score, "method": "text_fallback"}

    logger.warning(f"All SAM3 strategies failed for id={rid} '{type_en}'")
    return {"id": rid, "mask": np.zeros((h, w), dtype=np.uint8), "score": 0.0, "method": "failed"}


def _text_prompt(model, processor, image, text, h, w):
    """SAM3 text PCS — official API."""
    try:
        inputs = processor(images=image, text=text, return_tensors="pt").to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)

        orig_sizes = inputs.get("original_sizes")
        target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

        result = processor.post_process_instance_segmentation(
            outputs, threshold=0.35, mask_threshold=0.5, target_sizes=target_sizes
        )[0]

        masks = result.get("masks", [])
        scores = result.get("scores", [])

        if len(masks) == 0:
            return None, 0.0

        # Take highest-scoring mask
        if len(scores) > 0:
            best = int(torch.argmax(torch.tensor(scores)).item()) if isinstance(scores, list) else int(torch.argmax(scores).item())
            score = float(scores[best]) if isinstance(scores, list) else float(scores[best])
        else:
            best, score = 0, 0.7

        mask = masks[best].cpu().numpy().astype(np.uint8) * 255
        return (mask, score) if mask.sum() > 0 else (None, 0.0)

    except Exception as e:
        logger.debug(f"_text_prompt '{text}': {e}")
        return None, 0.0


def _point_prompt(model, processor, image, point, box, h, w):
    """SAM3 point prompt with optional box hint."""
    try:
        kwargs = {
            "images": image,
            "input_points": [[[[point[0], point[1]]]]],
            "input_labels": [[[1]]],
            "return_tensors": "pt",
        }
        if box:
            kwargs["input_boxes"] = [[[box]]]

        inputs = processor(**kwargs).to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)

        orig_sizes = inputs.get("original_sizes")
        target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

        result = processor.post_process_instance_segmentation(
            outputs, threshold=0.3, mask_threshold=0.5, target_sizes=target_sizes
        )[0]

        masks = result.get("masks", [])
        scores = result.get("scores", [])

        if len(masks) == 0:
            # Fallback: raw pred_masks
            return _raw_pred_masks(outputs, h, w)

        best = 0
        score = 0.7
        if len(scores) > 0:
            s = scores if isinstance(scores, torch.Tensor) else torch.tensor(scores)
            best = int(torch.argmax(s).item())
            score = float(s[best])

        mask = masks[best].cpu().numpy().astype(np.uint8) * 255
        return (mask, score) if mask.sum() > 0 else _raw_pred_masks(outputs, h, w)

    except Exception as e:
        logger.debug(f"_point_prompt: {e}")
        return None, 0.0


def _raw_pred_masks(outputs, h, w):
    """Last resort: upsample raw pred_masks tensor."""
    try:
        if not hasattr(outputs, "pred_masks"):
            return None, 0.0
        pm = outputs.pred_masks[0]  # (N, H_low, W_low)
        if pm.dim() != 3:
            return None, 0.0
        up = torch.nn.functional.interpolate(
            pm.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
        ).squeeze(0)
        binary = (up > 0).cpu().numpy().astype(np.uint8) * 255
        iou = getattr(outputs, "iou_scores", None)
        if iou is not None:
            sc = np.atleast_1d(iou[0].squeeze().cpu().numpy())
            best = int(np.argmax(sc))
            score = float(sc[best])
        else:
            best, score = 0, 0.6
        return (binary[best], score) if binary[best].sum() > 0 else (None, 0.0)
    except Exception as e:
        logger.debug(f"_raw_pred_masks: {e}")
        return None, 0.0


def encode_rle(mask: np.ndarray) -> dict:
    flat = mask.flatten()
    if len(flat) == 0:
        return {"rle": [], "shape": list(mask.shape), "starts_with": 0}
    rle, count = [], 1
    starts_with = current = int(flat[0] > 0)
    for v in flat[1:]:
        val = int(v > 0)
        if val == current:
            count += 1
        else:
            rle.append(count)
            count = 1
            current = val
    rle.append(count)
    return {"rle": rle, "shape": list(mask.shape), "starts_with": starts_with}


def handler(job: dict) -> dict:
    t0 = time.time()
    job_input = job.get("input", {})

    image_b64 = job_input.get("image_b64")
    if not image_b64:
        return {"error": "Missing image_b64"}

    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
        logger.info(f"Image: {image.size[0]}x{image.size[1]}")
    except Exception as e:
        return {"error": f"Image decode failed: {e}"}

    try:
        model, processor = load_sam3()
    except Exception as e:
        return {"error": f"Model load failed: {e}"}

    # Support "rooms" (new), "concepts" (v2), "prompts" (legacy)
    rooms = job_input.get("rooms")
    if not rooms:
        # Convert concepts or prompts to rooms format
        concepts = job_input.get("concepts") or job_input.get("prompts", [])
        if not concepts:
            return {"error": "No rooms, concepts, or prompts provided"}
        rooms = []
        for i, c in enumerate(concepts):
            rooms.append({
                "id": c.get("id", i+1),
                "type_en": c.get("type_en") or c.get("text") or c.get("type") or "room",
                "type_fr": c.get("type_fr", ""),
                "point": c.get("point"),
                "box": c.get("box"),
            })

    results = []
    for room in rooms:
        res = segment_room(model, processor, image, room)
        results.append({
            "id": res["id"],
            "mask_rle": encode_rle(res["mask"]),
            "score": round(res["score"], 4),
            "method": res["method"],
        })

    elapsed = round(time.time() - t0, 2)
    good = sum(1 for r in results if r["score"] > 0)
    logger.info(f"Done: {len(rooms)} rooms, {good} segmented in {elapsed}s")

    return {"masks": results, "model_used": "sam3", "processing_time_s": elapsed}


if __name__ == "__main__":
    load_sam3()
    logger.info("SAM3 ready.")
    runpod.serverless.start({"handler": handler})

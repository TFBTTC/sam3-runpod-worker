"""
RunPod Serverless Handler — SAM3 Floor Plan Segmentation

SAM3 key advantages used here:
- Text prompt: prompt with room label ("séjour cuisine") → SAM3 understands the concept
- Box hint: GPT-4o bounding box refines spatial location
- Point fallback: center point when text+box returns empty mask
- 4M+ concepts: understands French architectural terminology

Payload:
{
    "image_b64": "<base64 PNG/JPEG>",
    "prompts": [
        {"id": 1, "text": "séjour cuisine", "box": [x1,y1,x2,y2], "point": [cx,cy]}
    ]
}

Response:
{
    "masks": [{"id": 1, "mask_rle": {...}, "score": 0.95}],
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {DEVICE} | CUDA: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")

SAM3_MODEL_ID = os.getenv("SAM3_MODEL_ID", "facebook/sam3")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("RUNPOD_SECRET_HF_TOKEN")

_model = None
_processor = None


def load_sam3():
    """Load SAM3 model and processor once at container startup."""
    global _model, _processor
    if _model is not None:
        return _model, _processor

    logger.info(f"Loading SAM3 from {SAM3_MODEL_ID}...")
    t0 = time.time()

    from transformers import Sam3Processor, Sam3Model

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    _processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model = Sam3Model.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model = _model.to(DEVICE).eval()

    logger.info(f"SAM3 loaded in {time.time() - t0:.1f}s")
    return _model, _processor


def segment_rooms(model, processor, image: Image.Image, prompts: list) -> list:
    """
    Run SAM3 inference for each room prompt.

    SAM3 prompt strategy (best quality):
      text + box  →  SAM3 understands the room concept AND its location
      If empty → fallback to point prompt
    """
    h, w = image.size[1], image.size[0]
    results = []

    for prompt in prompts:
        text = prompt.get("text", "room")
        box = prompt.get("box")    # [x1, y1, x2, y2]
        point = prompt.get("point") # [cx, cy]

        mask, score = _segment_one(model, processor, image, text, box, point, h, w)
        results.append({"id": prompt["id"], "mask": mask, "score": score})

    return results


def _segment_one(model, processor, image, text, box, point, h, w):
    """Segment one room with text+box primary, point fallback."""
    # ── Primary: text + box ──────────────────────────────────────────────────
    try:
        kwargs = {"images": image, "return_tensors": "pt"}

        # SAM3's core feature: text concept prompting
        if text:
            kwargs["text"] = text

        # Box hint for spatial grounding (SAM3 format: [[[x1,y1,x2,y2]]])
        if box:
            kwargs["input_boxes"] = [[[box]]]
            kwargs["input_boxes_labels"] = [[[1]]]  # 1 = positive

        inputs = processor(**kwargs)
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model(**inputs)

        # target_sizes: original image dimensions for mask upscaling
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,
            mask_threshold=0.5,
            target_sizes=[(h, w)],
        )

        if post and post[0].get("masks") is not None and len(post[0]["masks"]) > 0:
            scores_t = post[0].get("scores")
            best_idx = int(torch.argmax(scores_t).item()) if scores_t is not None and len(scores_t) > 0 else 0
            score = float(scores_t[best_idx]) if scores_t is not None else 0.8
            mask = post[0]["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255

            if mask.sum() > 0:
                return mask, score

    except Exception as e:
        logger.warning(f"Text+box prompt failed: {e}")

    # ── Fallback: point prompt ───────────────────────────────────────────────
    if point is not None:
        try:
            # SAM3 point format: input_points=[[[point]]], input_labels=[[[1]]]
            inputs = processor(
                images=image,
                input_points=[[[[point[0], point[1]]]]],
                input_labels=[[[1]]],
                input_boxes=([[[box]]] if box else None),
                input_boxes_labels=([[[1]]] if box else None),
                return_tensors="pt",
            )
            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            post = processor.post_process_instance_segmentation(
                outputs, threshold=0.3, mask_threshold=0.5, target_sizes=[(h, w)]
            )

            if post and post[0].get("masks") is not None and len(post[0]["masks"]) > 0:
                scores_t = post[0].get("scores")
                best_idx = int(torch.argmax(scores_t).item()) if scores_t is not None and len(scores_t) > 0 else 0
                score = float(scores_t[best_idx]) if scores_t is not None else 0.7
                mask = post[0]["masks"][best_idx].cpu().numpy().astype(np.uint8) * 255
                if mask.sum() > 0:
                    return mask, score

        except Exception as e:
            logger.warning(f"Point prompt fallback failed: {e}")

    return np.zeros((h, w), dtype=np.uint8), 0.0


def encode_rle(mask: np.ndarray) -> dict:
    """Compact RLE encoding for fast network transfer."""
    flat = mask.flatten()
    if len(flat) == 0:
        return {"rle": [], "shape": list(mask.shape), "starts_with": 0}

    rle, count = [], 1
    starts_with = int(flat[0] > 0)
    current = starts_with

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

    prompts = job_input.get("prompts", [])
    if not prompts:
        return {"error": "No prompts provided"}

    try:
        raw = segment_rooms(model, processor, image, prompts)
    except Exception as e:
        logger.exception("Segmentation error")
        return {"error": f"Segmentation failed: {e}"}

    output_masks = [
        {"id": r["id"], "mask_rle": encode_rle(r["mask"]), "score": round(r["score"], 4)}
        for r in raw
    ]

    elapsed = round(time.time() - t0, 2)
    logger.info(f"Done: {len(prompts)} rooms in {elapsed}s")

    return {"masks": output_masks, "model_used": "sam3", "processing_time_s": elapsed}


if __name__ == "__main__":
    logger.info("Pre-loading SAM3...")
    load_sam3()
    logger.info("SAM3 ready. Starting RunPod serverless worker.")
    runpod.serverless.start({"handler": handler})

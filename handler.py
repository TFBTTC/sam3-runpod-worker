"""
RunPod Serverless Handler — SAM3 Floor Plan Segmentation

SAM3 supporte les text prompts nativement (contrairement à SAM2).
On passe directement le nom de la pièce ("séjour cuisine") → SAM3 segmente.
La bbox GPT-4o est un hint optionnel, pas une nécessité.

API officielle SAM3 (transformers):
    inputs = processor(images=image, text="séjour", return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, threshold=0.5, mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )

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
_model_class = None


def load_sam3():
    """Load SAM3 model and processor once at container startup."""
    global _model, _processor, _model_class

    if _model is not None:
        return _model, _processor

    logger.info(f"Loading SAM3 from {SAM3_MODEL_ID}...")
    t0 = time.time()

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    from transformers import Sam3Processor
    _processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, **kwargs)
    logger.info(f"Processor: {type(_processor).__name__}")

    # Sam3Model is the correct class per official docs
    from transformers import Sam3Model
    _model = Sam3Model.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model_class = "Sam3Model"

    _model = _model.to(DEVICE).eval()
    logger.info(f"SAM3 loaded in {time.time() - t0:.1f}s")
    return _model, _processor


def segment_rooms(model, processor, image: Image.Image, prompts: list) -> list:
    """Run SAM3 inference for each room prompt."""
    results = []
    for prompt in prompts:
        text = prompt.get("text", "room")
        box = prompt.get("box")     # [x1, y1, x2, y2] — optional spatial hint
        point = prompt.get("point") # [cx, cy] — optional fallback
        mask, score = _segment_one(model, processor, image, text, box, point)
        results.append({"id": prompt["id"], "mask": mask, "score": score})
    return results


def _segment_one(model, processor, image, text, box, point):
    """
    Segment one room.

    Attempt order:
    1. Text only      — SAM3 native: understands "séjour cuisine" semantically
    2. Text + box     — text + spatial hint from GPT-4o bbox
    3. Point + box    — pure spatial fallback
    """
    h, w = image.size[1], image.size[0]

    # ── Attempt 1: text prompt only (SAM3 core feature) ────────────────────
    # Official API: text is a plain string, not a list
    # target_sizes comes from inputs["original_sizes"], not manually from (h,w)
    if text:
        try:
            inputs = processor(images=image, text=text, return_tensors="pt")
            inputs = inputs.to(DEVICE)

            if not hasattr(_segment_one, "_input_logged"):
                logger.info(f"Input keys: {list(inputs.keys())}")
                _segment_one._input_logged = True

            with torch.inference_mode():
                outputs = model(**inputs)

            if not hasattr(_segment_one, "_output_logged"):
                logger.info(f"Output type: {type(outputs).__name__}")
                logger.info(f"Output attrs: {[k for k in dir(outputs) if not k.startswith('_')][:20]}")
                _segment_one._output_logged = True

            orig_sizes = inputs.get("original_sizes")
            target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

            mask, score = _extract_mask(outputs, processor, target_sizes, h, w, threshold=0.4)
            if mask is not None and mask.sum() > 0:
                logger.info(f"[text] '{text}' → score={score:.3f}, pixels={mask.sum()}")
                return mask, score

        except Exception as e:
            logger.warning(f"Text-only failed for '{text}': {e}")

    # ── Attempt 2: text + bbox hint ───────────────────────────────────────────
    if text and box:
        try:
            inputs = processor(
                images=image,
                text=text,
                input_boxes=[[[box]]],
                return_tensors="pt",
            )
            inputs = inputs.to(DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs)

            orig_sizes = inputs.get("original_sizes")
            target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

            mask, score = _extract_mask(outputs, processor, target_sizes, h, w, threshold=0.3)
            if mask is not None and mask.sum() > 0:
                logger.info(f"[text+box] '{text}' → score={score:.3f}")
                return mask, score

        except Exception as e:
            logger.warning(f"Text+box failed for '{text}': {e}")

    # ── Attempt 3: point + box (pure spatial) ────────────────────────────
    if point is not None:
        try:
            kwargs = {
                "images": image,
                "input_points": [[[[point[0], point[1]]]]],
                "input_labels": [[[1]]],
                "return_tensors": "pt",
            }
            if box:
                kwargs["input_boxes"] = [[[box]]]

            inputs = processor(**kwargs)
            inputs = inputs.to(DEVICE)

            with torch.inference_mode():
                outputs = model(**inputs)

            orig_sizes = inputs.get("original_sizes")
            target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

            mask, score = _extract_mask(outputs, processor, target_sizes, h, w, threshold=0.3)
            if mask is not None and mask.sum() > 0:
                logger.info(f"[point] '{text}' → score={score:.3f}")
                return mask, score

        except Exception as e:
            logger.warning(f"Point fallback failed for '{text}': {e}")

    logger.warning(f"All attempts failed for '{text}'")
    return np.zeros((h, w), dtype=np.uint8), 0.0


def _extract_mask(outputs, processor, target_sizes, h, w, threshold=0.4):
    """
    Extract best mask from SAM3 outputs.
    Tries post_process_instance_segmentation first (official API),
    then falls back to raw pred_masks interpolation.
    """
    # ── Official API: post_process_instance_segmentation ────────────────────
    try:
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )
        if post and post[0].get("masks") is not None and len(post[0]["masks"]) > 0:
            masks = post[0]["masks"]
            scores = post[0].get("scores")
            best = int(torch.argmax(scores).item()) if (scores is not None and len(scores) > 0) else 0
            score = float(scores[best]) if scores is not None else 0.7
            mask = masks[best].cpu().numpy().astype(np.uint8) * 255
            if mask.sum() > 0:
                return mask, score
    except Exception as e:
        logger.debug(f"post_process_instance_segmentation: {e}")

    # ── Fallback: raw pred_masks upsampled ───────────────────────────────────
    try:
        if hasattr(outputs, "pred_masks"):
            pm = outputs.pred_masks  # (B, N, H_low, W_low)
            iou = getattr(outputs, "iou_scores", None)
            m = pm[0]  # (N, H_low, W_low)
            if m.dim() == 3:
                up = torch.nn.functional.interpolate(
                    m.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(0)
                binary = (up > 0).cpu().numpy().astype(np.uint8) * 255
                if iou is not None:
                    sc = np.atleast_1d(iou[0].squeeze().cpu().numpy())
                    best = int(np.argmax(sc))
                    score = float(sc[best])
                else:
                    best, score = 0, 0.7
                if binary[best].sum() > 0:
                    return binary[best], score
    except Exception as e:
        logger.debug(f"pred_masks fallback: {e}")

    return None, 0.0


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

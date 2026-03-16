"""
RunPod Serverless Handler — SAM3 Floor Plan Segmentation

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
    """Load SAM3 model and processor once at container startup.

    Tries Sam3ForInstanceSegmentation first (has segmentation head),
    falls back to Sam3Model (base) if not available.
    """
    global _model, _processor, _model_class

    if _model is not None:
        return _model, _processor

    logger.info(f"Loading SAM3 from {SAM3_MODEL_ID}...")
    t0 = time.time()

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    from transformers import Sam3Processor
    _processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, **kwargs)
    logger.info(f"Processor loaded: {type(_processor).__name__}")

    # Try segmentation-head model first
    try:
        from transformers import Sam3ForInstanceSegmentation
        _model = Sam3ForInstanceSegmentation.from_pretrained(SAM3_MODEL_ID, **kwargs)
        _model_class = "Sam3ForInstanceSegmentation"
        logger.info("Using Sam3ForInstanceSegmentation")
    except (ImportError, Exception) as e:
        logger.warning(f"Sam3ForInstanceSegmentation not available ({e}), falling back to Sam3Model")
        from transformers import Sam3Model
        _model = Sam3Model.from_pretrained(SAM3_MODEL_ID, **kwargs)
        _model_class = "Sam3Model"
        logger.info("Using Sam3Model")

    _model = _model.to(DEVICE).eval()
    logger.info(f"SAM3 loaded in {time.time() - t0:.1f}s using {_model_class}")
    return _model, _processor


def segment_rooms(model, processor, image: Image.Image, prompts: list) -> list:
    """Run SAM3 inference for each room prompt."""
    h, w = image.size[1], image.size[0]
    results = []

    for prompt in prompts:
        text = prompt.get("text", "room")
        box = prompt.get("box")     # [x1, y1, x2, y2]
        point = prompt.get("point") # [cx, cy]

        mask, score = _segment_one(model, processor, image, text, box, point, h, w)
        results.append({"id": prompt["id"], "mask": mask, "score": score})

    return results


def _segment_one(model, processor, image, text, box, point, h, w):
    """
    Segment one room. Attempt order:
    1. Text + box  (SAM3 native — best quality)
    2. Point + box (spatial only — fallback)
    3. Box only    (last resort)
    """
    # ── Attempt 1: text + box ────────────────────────────────────────────────
    try:
        kwargs = {"images": image, "return_tensors": "pt"}

        # SAM3: text as list of concepts
        if text:
            kwargs["text"] = [text]

        if box:
            kwargs["input_boxes"] = [[[box]]]

        inputs = processor(**kwargs)
        inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

        if not hasattr(_segment_one, "_logged"):
            logger.info(f"Input keys: {list(inputs.keys())}")
            _segment_one._logged = True

        with torch.inference_mode():
            outputs = model(**inputs)

        if not hasattr(_segment_one, "_out_logged"):
            out_keys = [k for k in dir(outputs) if not k.startswith("_") and not callable(getattr(outputs, k, None))]
            logger.info(f"Output attrs: {out_keys[:15]}")
            _segment_one._out_logged = True

        mask, score = _extract_best_mask(outputs, processor, h, w, threshold=0.4)
        if mask is not None and mask.sum() > 0:
            return mask, score

    except Exception as e:
        logger.warning(f"Text+box attempt failed: {e}")

    # ── Attempt 2: point + box ───────────────────────────────────────────────
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
            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            mask, score = _extract_best_mask(outputs, processor, h, w, threshold=0.3)
            if mask is not None and mask.sum() > 0:
                return mask, score

        except Exception as e:
            logger.warning(f"Point+box attempt failed: {e}")

    # ── Attempt 3: box only ──────────────────────────────────────────────────
    if box is not None:
        try:
            kwargs = {
                "images": image,
                "input_boxes": [[[box]]],
                "return_tensors": "pt",
            }
            inputs = processor(**kwargs)
            inputs = {k: v.to(DEVICE) if hasattr(v, "to") else v for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)

            mask, score = _extract_best_mask(outputs, processor, h, w, threshold=0.2)
            if mask is not None and mask.sum() > 0:
                return mask, score

        except Exception as e:
            logger.warning(f"Box-only attempt failed: {e}")

    logger.warning(f"All attempts failed for prompt '{text}' — returning empty mask")
    return np.zeros((h, w), dtype=np.uint8), 0.0


def _extract_best_mask(outputs, processor, h, w, threshold=0.4):
    """
    Extract best mask from model outputs.
    Tries multiple strategies depending on what the model returns.
    """
    # ── Strategy A: post_process_instance_segmentation ───────────────────────
    try:
        post = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=0.5,
            target_sizes=[(h, w)],
        )
        if post and post[0].get("masks") is not None and len(post[0]["masks"]) > 0:
            masks = post[0]["masks"]
            scores = post[0].get("scores")
            best_idx = int(torch.argmax(scores).item()) if (scores is not None and len(scores) > 0) else 0
            score = float(scores[best_idx]) if scores is not None else 0.7
            mask = masks[best_idx].cpu().numpy().astype(np.uint8) * 255
            if mask.sum() > 0:
                logger.info(f"Mask via post_process_instance_segmentation, score={score:.3f}")
                return mask, score
    except Exception as e:
        logger.debug(f"post_process_instance_segmentation failed: {e}")

    # ── Strategy B: raw pred_masks interpolation ──────────────────────────────
    try:
        if hasattr(outputs, "pred_masks"):
            pred_masks = outputs.pred_masks
            iou_scores = getattr(outputs, "iou_scores", None)
            logger.info(f"pred_masks shape: {pred_masks.shape}, iou_scores: {iou_scores.shape if iou_scores is not None else None}")

            # (batch, num_masks, H_low, W_low) → take batch 0
            m = pred_masks[0]  # (num_masks, H, W) or (1, num_masks, H, W)
            if m.dim() == 3:
                masks_up = torch.nn.functional.interpolate(
                    m.unsqueeze(0).float(), size=(h, w), mode="bilinear", align_corners=False
                ).squeeze(0)  # (num_masks, H, W)
                binary = (masks_up > 0).cpu().numpy().astype(np.uint8) * 255

                if iou_scores is not None:
                    sc = iou_scores[0].squeeze().cpu().numpy()
                    sc = np.atleast_1d(sc)
                    best_idx = int(np.argmax(sc))
                    score = float(sc[best_idx])
                else:
                    best_idx, score = 0, 0.7

                if binary[best_idx].sum() > 0:
                    logger.info(f"Mask via pred_masks interpolation, score={score:.3f}")
                    return binary[best_idx], score
    except Exception as e:
        logger.debug(f"pred_masks strategy failed: {e}")

    # ── Strategy C: post_process_masks (SAM2 API) ─────────────────────────────
    try:
        if hasattr(processor, "post_process_masks") and hasattr(outputs, "pred_masks"):
            masks = processor.post_process_masks(
                outputs.pred_masks, [(h, w)], [(h, w)]
            )
            if masks and len(masks[0]) > 0:
                m = masks[0][0]
                if m.dim() == 3:
                    m = m[0]
                binary = m.cpu().numpy().astype(np.uint8) * 255
                if binary.sum() > 0:
                    logger.info("Mask via post_process_masks")
                    return binary, 0.7
    except Exception as e:
        logger.debug(f"post_process_masks strategy failed: {e}")

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
    logger.info(f"Done: {len(prompts)} rooms in {elapsed}s | model: {_model_class}")

    return {"masks": output_masks, "model_used": "sam3", "processing_time_s": elapsed}


if __name__ == "__main__":
    logger.info("Pre-loading SAM3...")
    load_sam3()
    logger.info("SAM3 ready. Starting RunPod serverless worker.")
    runpod.serverless.start({"handler": handler})

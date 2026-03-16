"""
RunPod Serverless Handler — SAM3 Floor Plan Room Segmentation
arXiv:2511.16719 | transformers >= 4.45 | facebook/sam3

Objectif : segmenter les ESPACES des pièces d'un plan d'appartement.

Flux :
  GPT-4o identifie les pièces et leur centre → SAM3 segmente l'espace à ce point.

SAM3 avec prompt POINT segmente la région fermée autour du point donné.
Les murs (lignes épaisses sombres) constituent des frontières naturelles pour SAM3.
C'est exactement le cas d'usage prévu par SAM3 : "segment what is at this point".

Payload:
{
    "image_b64": "<base64 PNG/JPEG>",
    "rooms": [
        {
            "id": 1,
            "type": "chambre",       # label pour le rendu (French)
            "point": [cx, cy],       # centre en pixels — fourni par GPT-4o
            "box":   [x1,y1,x2,y2]  # bbox en pixels — optionnel, améliore la précision
        }
    ]
}

Response:
{
    "masks": [
        {"id": 1, "mask_rle": {"rle":[...], "shape":[h,w], "starts_with":0}, "score": 0.88}
    ],
    "model_used": "sam3",
    "processing_time_s": 3.2
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


def segment_room(model, processor, image: Image.Image, room: dict) -> dict:
    """
    Segment the closed space of one room using SAM3 point prompt.

    SAM3 receives the center point (from GPT-4o) and segments the enclosed
    region at that location — bounded naturally by the wall lines.
    """
    h, w = image.size[1], image.size[0]
    rid = room["id"]
    point = room.get("point")   # [cx, cy] in pixels
    box = room.get("box")       # [x1, y1, x2, y2] in pixels — optional

    if point is None:
        logger.warning(f"Room id={rid} has no point — skipping")
        return {"id": rid, "mask": np.zeros((h, w), dtype=np.uint8), "score": 0.0}

    # ── SAM3 point prompt ─────────────────────────────────────────────────────
    # input_points: [[[cx, cy]]] — batch=1, num_points=1, coords=2
    # input_labels: [[[1]]]     — 1 = foreground point (segment this)
    try:
        kwargs = {
            "images": image,
            "input_points": [[[[float(point[0]), float(point[1])]]]],
            "input_labels": [[[1]]],
            "return_tensors": "pt",
        }
        if box:
            # Box hint improves precision when available
            kwargs["input_boxes"] = [[[ [float(b) for b in box] ]]]

        inputs = processor(**kwargs).to(DEVICE)

        with torch.inference_mode():
            outputs = model(**inputs)

        orig_sizes = inputs.get("original_sizes")
        target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.3,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]

        masks  = result.get("masks",  [])
        scores = result.get("scores", [])

        if len(masks) > 0:
            # Best scoring mask
            if isinstance(scores, torch.Tensor) and len(scores) > 0:
                best = int(torch.argmax(scores).item())
                score = float(scores[best])
            elif isinstance(scores, list) and len(scores) > 0:
                best = int(np.argmax(scores))
                score = float(scores[best])
            else:
                best, score = 0, 0.7

            mask = masks[best].cpu().numpy().astype(np.uint8) * 255

            if mask.sum() > 0:
                logger.info(f"id={rid} '{room.get('type','')}' → {mask.sum()} px, score={score:.3f}")
                return {"id": rid, "mask": mask, "score": score}

        # ── Fallback: raw pred_masks upsampled ────────────────────────────────
        if hasattr(outputs, "pred_masks"):
            pm = outputs.pred_masks[0]  # (N, H_low, W_low)
            if pm.dim() == 3 and pm.shape[0] > 0:
                up = torch.nn.functional.interpolate(
                    pm.unsqueeze(0).float(), size=(h, w),
                    mode="bilinear", align_corners=False
                ).squeeze(0)
                binary = (up > 0).cpu().numpy().astype(np.uint8) * 255

                iou = getattr(outputs, "iou_scores", None)
                if iou is not None:
                    sc = np.atleast_1d(iou[0].squeeze().cpu().numpy())
                    best = int(np.argmax(sc))
                    score = float(sc[best])
                else:
                    best, score = 0, 0.65

                if binary[best].sum() > 0:
                    logger.info(f"id={rid} '{room.get('type','')}' via pred_masks → {binary[best].sum()} px")
                    return {"id": rid, "mask": binary[best], "score": score}

    except Exception as e:
        logger.error(f"SAM3 failed for id={rid}: {e}", exc_info=True)

    logger.warning(f"id={rid} '{room.get('type','')}' — no mask produced")
    return {"id": rid, "mask": np.zeros((h, w), dtype=np.uint8), "score": 0.0}


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

    # Accept "rooms" (canonical) or "prompts" (legacy compat)
    rooms = job_input.get("rooms") or [
        {
            "id": p.get("id", i+1),
            "type": p.get("type", p.get("text", "room")),
            "point": p.get("point"),
            "box": p.get("box"),
        }
        for i, p in enumerate(job_input.get("prompts", []))
    ]

    if not rooms:
        return {"error": "No rooms provided. Send rooms=[{id, type, point:[cx,cy], box:[x1,y1,x2,y2]}]"}

    results = []
    for room in rooms:
        res = segment_room(model, processor, image, room)
        results.append({
            "id": res["id"],
            "mask_rle": encode_rle(res["mask"]),
            "score": round(res["score"], 4),
        })

    elapsed = round(time.time() - t0, 2)
    good = sum(1 for r in results if r["score"] > 0)
    logger.info(f"Done: {len(rooms)} rooms → {good} segmented in {elapsed}s")

    return {"masks": results, "model_used": "sam3", "processing_time_s": elapsed}


if __name__ == "__main__":
    load_sam3()
    runpod.serverless.start({"handler": handler})

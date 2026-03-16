"""
RunPod Serverless Handler — SAM3 Floor Plan Segmentation

Architecture conforme aux recommandations officielles SAM3 :

SAM3 fait du Promptable Concept Segmentation (PCS) :
- text="chambre" → segmente TOUTES les chambres en une passe
- text="séjour" → segmente le séjour
- text="cuisine" → segmente la cuisine
- etc.

On envoie donc UN prompt par TYPE de pièce (pas par instance),
et SAM3 retourne autant de masques qu'il y a d'instances de ce concept.

Payload:
{
    "image_b64": "<base64 PNG/JPEG>",
    "concepts": [
        {"type": "chambre", "text": "chambre"},
        {"type": "séjour",  "text": "séjour"},
        {"type": "cuisine", "text": "cuisine"}
    ]
}

Response:
{
    "segments": [
        {
            "type": "chambre",
            "instances": [
                {"mask_rle": {...}, "score": 0.91, "box": [x1,y1,x2,y2]},
                {"mask_rle": {...}, "score": 0.87, "box": [x1,y1,x2,y2]}
            ]
        }
    ],
    "model_used": "sam3"
}

API officielle SAM3 (transformers >= 4.45, arXiv:2511.16719) :
    inputs = processor(images=image, text="chambre", return_tensors="pt")
    outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.5,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist()
    )[0]
    # results["masks"]  → tous les masques du concept
    # results["boxes"]  → leurs bounding boxes
    # results["scores"] → scores de confiance
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

    kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}

    from transformers import Sam3Processor, Sam3Model

    _processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model = Sam3Model.from_pretrained(SAM3_MODEL_ID, **kwargs)
    _model = _model.to(DEVICE).eval()

    logger.info(f"SAM3 loaded in {time.time() - t0:.1f}s | {_model.__class__.__name__}")
    return _model, _processor


def segment_concept(model, processor, image: Image.Image, text: str) -> list:
    """
    Segment ALL instances of a concept in the image using SAM3 PCS.

    SAM3's Promptable Concept Segmentation finds every occurrence of
    the concept in one forward pass — e.g. text="chambre" returns masks
    for ALL bedrooms simultaneously.

    Returns list of {mask, score, box} for each detected instance.
    """
    h, w = image.size[1], image.size[0]

    try:
        # Official SAM3 API — text as plain string
        inputs = processor(images=image, text=text, return_tensors="pt")
        inputs = inputs.to(DEVICE)

        logger.info(f"[PCS] Concept='{text}' | Input keys: {list(inputs.keys())}")

        with torch.inference_mode():
            outputs = model(**inputs)

        logger.info(f"[PCS] Output type: {type(outputs).__name__}")
        if hasattr(outputs, '__dict__'):
            logger.info(f"[PCS] Output fields: {[k for k in vars(outputs).keys() if not k.startswith('_')][:15]}")

        # Post-process — official API uses original_sizes from inputs
        orig_sizes = inputs.get("original_sizes")
        target_sizes = orig_sizes.tolist() if orig_sizes is not None else [(h, w)]

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.4,
            mask_threshold=0.5,
            target_sizes=target_sizes,
        )[0]

        masks = results.get("masks", [])
        scores = results.get("scores", [])
        boxes = results.get("boxes", [])

        logger.info(f"[PCS] '{text}' → {len(masks)} instance(s) found")

        instances = []
        for i, mask in enumerate(masks):
            m = mask.cpu().numpy().astype(np.uint8) * 255
            if m.sum() == 0:
                continue
            score = float(scores[i]) if i < len(scores) else 0.7
            box = boxes[i].cpu().numpy().tolist() if i < len(boxes) else None
            instances.append({"mask": m, "score": score, "box": box})

        return instances

    except Exception as e:
        logger.error(f"[PCS] Failed for '{text}': {e}", exc_info=True)
        return []


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

    # Support both new "concepts" format and legacy "prompts" format
    concepts = job_input.get("concepts")
    if not concepts:
        # Legacy: convert prompts → concepts (group by unique text)
        prompts = job_input.get("prompts", [])
        if not prompts:
            return {"error": "No concepts or prompts provided"}
        seen = {}
        for p in prompts:
            t = p.get("text", "room")
            if t not in seen:
                seen[t] = {"type": t, "text": t}
        concepts = list(seen.values())

    # Run PCS for each room type concept
    segments = []
    for concept in concepts:
        text = concept.get("text", concept.get("type", "room"))
        instances = segment_concept(model, processor, image, text)

        segments.append({
            "type": concept.get("type", text),
            "text": text,
            "instances": [
                {
                    "mask_rle": encode_rle(inst["mask"]),
                    "score": round(inst["score"], 4),
                    "box": inst["box"],
                }
                for inst in instances
            ],
        })

    elapsed = round(time.time() - t0, 2)
    total_instances = sum(len(s["instances"]) for s in segments)
    logger.info(f"Done: {len(concepts)} concepts → {total_instances} instances in {elapsed}s")

    return {
        "segments": segments,
        "model_used": "sam3",
        "processing_time_s": elapsed,
    }


if __name__ == "__main__":
    logger.info("Pre-loading SAM3...")
    load_sam3()
    logger.info("SAM3 ready. Starting RunPod serverless worker.")
    runpod.serverless.start({"handler": handler})

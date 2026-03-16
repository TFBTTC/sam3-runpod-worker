# SAM3 RunPod Serverless Worker

SAM3 inference worker for floor plan room segmentation. Used by the floor plan segmentation pipeline.

## SAM3 capabilities used
- **Text prompts**: segment by room concept ("séjour cuisine", "chambre", etc.)
- **Box hints**: GPT-4o bounding boxes for spatial grounding
- **4M+ concepts**: understands French architectural terminology

## Setup

1. Add repo secrets on GitHub:
   - `RUNPOD_API_KEY` = your RunPod key

2. Push to main → GitHub Actions builds and pushes to GHCR automatically

3. Add your HuggingFace token to RunPod endpoint environment variables:
   - Key: `HF_TOKEN`
   - Value: `hf_...` (from https://huggingface.co/settings/tokens)

## Image
`ghcr.io/tfbttc/sam3-runpod-worker:latest`

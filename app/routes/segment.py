"""Segmentation + health endpoints."""

from __future__ import annotations

import io

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app import inference
from app.config import settings
from app.schemas import HealthResponse, SegmentResponse
from app.services import get_model, loaded_modalities

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        device=settings.device,
        default_modality=settings.default_modality,
        available_modalities=sorted(settings.registry.keys()),
        loaded_modalities=loaded_modalities(),
    )


@router.post(settings.api_prefix + "/segment", response_model=SegmentResponse, tags=["segment"])
async def segment(
    image: UploadFile = File(..., description="Image file (jpg/png/webp)."),
    modality: str = Form(default=""),
    tta: str = Form(default=""),
) -> SegmentResponse:
    """
    Segment a medical image. `modality` selects the model variant
    (dermoscopy | histology | microscopy); empty falls back to the default.
    `tta` is 'basic' (3 views, fast) or 'full' (8 views, more accurate).
    """
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > settings.request_max_bytes:
        raise HTTPException(status_code=413, detail="Image exceeds size limit.")

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.verify()
        pil = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail=f"Cannot read image: {exc}") from exc

    w, h = pil.size
    if w < 32 or h < 32:
        raise HTTPException(status_code=400, detail=f"Image too small ({w}x{h}); minimum 32x32.")

    tta_mode = tta if tta in ("basic", "full") else settings.default_tta
    model, spec, resolved_modality = get_model(modality or settings.default_modality)

    result = inference.segment(
        image=pil,
        model=model,
        device=settings.device,
        model_name=f"{spec.arch}_n{spec.expansion_n}",
        modality=resolved_modality,
        tta_mode=tta_mode,
        include_raw=False,
    )
    return SegmentResponse(**result)

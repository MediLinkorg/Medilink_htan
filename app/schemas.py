"""Response schemas for the HTAN segmentation service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ComponentDetail(BaseModel):
    component_id: int
    area_px: int
    area_percent: float
    largest_span_px: int
    location: str


class SegmentResponse(BaseModel):
    target_detected: bool = Field(..., description="Whether any region was segmented.")
    segmented_area_px: int
    segmented_area_percent: float = Field(..., description="Segmented area as % of the 256x256 image.")
    relative_size: str = Field(..., description="none | small | medium | large")
    num_components: int
    largest_component_span_px: int
    location: str = Field(..., description="Centroid region, e.g. 'upper-left', 'center'.")
    component_details: list[ComponentDetail]
    severity_estimate: str = Field(..., description="Relative screening bucket: none|mild|moderate|severe|critical.")
    model: str
    modality: str
    tta_mode: str
    image_size: list[int]
    note: str


class HealthResponse(BaseModel):
    status: str
    device: str
    default_modality: str
    available_modalities: list[str]
    loaded_modalities: list[str]

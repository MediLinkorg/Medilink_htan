"""
Smoke tests for the HTAN service. These avoid loading real weights by
monkeypatching the model with a tiny stub that returns logits, so they run
on CPU in CI without GPU or Hugging Face access.
"""

import io

import numpy as np
import torch
from fastapi.testclient import TestClient
from PIL import Image

from app import services
from app.config import ModelSpec, settings
from app.main import app


class _StubModel(torch.nn.Module):
    """Returns a centered positive-logit blob so a region is 'detected'."""

    def forward(self, x):  # noqa: D401
        b, _, h, w = x.shape
        out = torch.full((b, 1, h, w), -6.0)
        out[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 6.0
        return out


def _patch_model(monkeypatch):
    spec = ModelSpec("HTAN_2", 2, "stub.pth", 90.32)
    monkeypatch.setattr(services, "get_model", lambda m: (_StubModel(), spec, "dermoscopy"))
    # route imports get_model by name, so patch there too
    from app.routes import segment as seg_route

    monkeypatch.setattr(seg_route, "get_model", lambda m: (_StubModel(), spec, "dermoscopy"))


def _png_bytes(size=(128, 128)) -> bytes:
    arr = (np.random.rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_segment_detects_region(monkeypatch):
    _patch_model(monkeypatch)
    settings_device = settings.device
    settings.device = "cpu"
    try:
        client = TestClient(app)
        r = client.post(
            settings.api_prefix + "/segment",
            files={"image": ("x.png", _png_bytes(), "image/png")},
            data={"modality": "dermoscopy", "tta": "basic"},
        )
    finally:
        settings.device = settings_device
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["target_detected"] is True
    assert "confidence_score" not in body  # locked: no confidence
    assert body["modality"] == "dermoscopy"


def test_rejects_tiny_image(monkeypatch):
    _patch_model(monkeypatch)
    client = TestClient(app)
    r = client.post(
        settings.api_prefix + "/segment",
        files={"image": ("x.png", _png_bytes((16, 16)), "image/png")},
    )
    assert r.status_code == 400

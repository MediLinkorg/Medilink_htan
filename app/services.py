"""
Model lifecycle for the HTAN service.

Loads a torch model per modality (HTAN_2 / HTAN_1), caches it, and resolves
weights either from a local directory (HTAN_WEIGHTS_DIR) or the Hugging Face
hub (mohamedkhaledmk7/HTAN). State dicts saved as either a raw tensor dict or
a {"model": ...}/{"state_dict": ...} wrapper are both handled.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from threading import Lock

import torch

# Vendored model package lives at the repo root (./models). Make it importable
# regardless of the working directory the server is launched from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.htan.htan import HTAN_1, HTAN_2  # noqa: E402

from app.config import ModelSpec, settings  # noqa: E402

logger = logging.getLogger("medilink.htan")

_ARCHES = {"HTAN_1": HTAN_1, "HTAN_2": HTAN_2}

_CACHE: dict[str, torch.nn.Module] = {}
_LOCK = Lock()


def _resolve_weights(spec: ModelSpec) -> str:
    if settings.weights_dir:
        path = Path(settings.weights_dir) / spec.weights_file
        if not path.exists():
            raise FileNotFoundError(
                f"Weights not found at {path}. Set HTAN_WEIGHTS_DIR correctly or "
                f"unset it to download from Hugging Face."
            )
        return str(path)

    # Lazy import so the package is only required when HF download is used.
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=settings.hf_repo_id,
        filename=spec.weights_file,
        token=settings.hf_token or None,
    )


def _build(spec: ModelSpec) -> torch.nn.Module:
    arch = _ARCHES[spec.arch]
    model = arch(expansion_n=spec.expansion_n, img_size=settings.img_size)

    weights_path = _resolve_weights(spec)
    state = torch.load(weights_path, map_location=settings.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    elif isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]

    model.load_state_dict(state)
    model.to(settings.device)
    model.eval()
    logger.info(
        "Loaded %s (n=%d) weights=%s device=%s",
        spec.arch, spec.expansion_n, spec.weights_file, settings.device,
    )
    return model


def get_model(modality: str) -> tuple[torch.nn.Module, ModelSpec, str]:
    """Return (model, spec, resolved_modality), loading + caching on first use."""
    resolved = modality if modality in settings.registry else settings.default_modality
    spec = settings.registry[resolved]

    if resolved not in _CACHE:
        with _LOCK:
            if resolved not in _CACHE:
                _CACHE[resolved] = _build(spec)
    return _CACHE[resolved], spec, resolved


def warmup() -> list[str]:
    """Preload models per HTAN_PRELOAD_ALL. Returns the list of loaded modalities."""
    targets = list(settings.registry) if settings.preload_all else [settings.default_modality]
    loaded = []
    for modality in targets:
        try:
            get_model(modality)
            loaded.append(modality)
        except Exception as exc:  # noqa: BLE001 — startup should not hard-crash on one variant
            logger.warning("Could not preload modality '%s': %s", modality, exc)
    return loaded


def loaded_modalities() -> list[str]:
    return sorted(_CACHE.keys())

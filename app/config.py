"""
Configuration for the HTAN segmentation service.

The MODEL_REGISTRY maps an imaging modality to the architecture + weights that
won that benchmark. These pairings are locked from the HTAN experiments:

    dermoscopy  -> HTAN_2, n=2   (ISIC, Dice 90.32%)   <- default skin-lesion path
    histology   -> HTAN_1, n=2   (GlaS, Dice 91.67%)
    microscopy  -> HTAN_1, n=2   (Bowl, Dice 92.14%)

Weights are pulled from the Hugging Face repo `mohamedkhaledmk7/HTAN`. The
filename for each variant is configurable so you can rename on HF without
touching code. Confirm the three HF filenames before first deploy (see README).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default).strip()


@dataclass(frozen=True)
class ModelSpec:
    arch: str          # "HTAN_2" | "HTAN_1"
    expansion_n: int
    weights_file: str  # path within HF repo or local WEIGHTS_DIR
    dice: float        # reported test Dice, for /health and logs


@dataclass
class Settings:
    # --- service ---
    host: str = field(default_factory=lambda: _env("HTAN_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(_env("HTAN_PORT", "8001")))
    api_prefix: str = "/api/v1"
    request_max_bytes: int = field(
        default_factory=lambda: int(_env("HTAN_MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
    )

    # --- device / inference ---
    device: str = field(
        default_factory=lambda: _env(
            "HTAN_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
        )
    )
    default_modality: str = field(default_factory=lambda: _env("HTAN_DEFAULT_MODALITY", "dermoscopy"))
    default_tta: str = field(default_factory=lambda: _env("HTAN_DEFAULT_TTA", "basic"))  # basic | full
    img_size: int = 256

    # Preload all model variants at startup (more VRAM) vs. lazy-load on first use.
    preload_all: bool = field(default_factory=lambda: _env("HTAN_PRELOAD_ALL", "false").lower() == "true")

    # --- weights ---
    hf_repo_id: str = field(default_factory=lambda: _env("HTAN_HF_REPO", "mohamedkhaledmk7/HTAN"))
    hf_token: str = field(default_factory=lambda: _env("HF_TOKEN", ""))
    # If set, load weights from this local dir instead of downloading from HF.
    weights_dir: str = field(default_factory=lambda: _env("HTAN_WEIGHTS_DIR", ""))

    registry: dict[str, ModelSpec] = field(
        default_factory=lambda: {
            "dermoscopy": ModelSpec("HTAN_2", 2, _env("HTAN_W_DERMOSCOPY", "htan_2_n2/best_model.pth"), 90.32),
            "histology": ModelSpec("HTAN_1", 2, _env("HTAN_W_HISTOLOGY", "htan_1_n2_glas/best_model.pth"), 91.67),
            "microscopy": ModelSpec("HTAN_1", 2, _env("HTAN_W_MICROSCOPY", "htan_1_n2_bowl/best_model.pth"), 92.14),
        }
    )

    def spec_for(self, modality: str) -> ModelSpec:
        return self.registry.get(modality, self.registry[self.default_modality])


settings = Settings()

"""
Pre-download HTAN weights from Hugging Face into the image / local cache.
Useful in CI or Docker build so the first request isn't slowed by a download.

    python scripts/download_weights.py            # default modality only
    python scripts/download_weights.py --all      # all three variants
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from huggingface_hub import hf_hub_download  # noqa: E402

from app.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Download every modality variant.")
    args = parser.parse_args()

    targets = list(settings.registry) if args.all else [settings.default_modality]
    for modality in targets:
        spec = settings.registry[modality]
        path = hf_hub_download(
            repo_id=settings.hf_repo_id,
            filename=spec.weights_file,
            token=settings.hf_token or None,
        )
        print(f"[{modality:11}] {spec.arch}_n{spec.expansion_n} -> {path}")


if __name__ == "__main__":
    main()

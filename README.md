# MediLink — HTAN Segmentation Service

FastAPI wrapper around the HTAN medical-image segmentation models. One job:
take an image, return structured segmentation/screening features for the
gateway to fold into retrieval and the final answer.

Runs on the **GPU EC2** box (`:8001`). Internal service — only the gateway calls it.

## What it returns

Detection (binary), segmented area as % of image, relative size, component
count, largest component span (pixels), centroid location, a coarse severity
bucket, and the model/modality used. **No physical measurements** (mm/cm²) and
**no confidence score** — both were deliberately dropped (see "Locked decisions").

## Modality → model

The router picks a modality; this service loads the architecture that won that
benchmark:

| modality   | architecture | weights (HF)                  | test Dice |
|------------|--------------|-------------------------------|-----------|
| dermoscopy | HTAN_2 (n=2) | `htan_2_n2/best_model.pth`    | 90.32%    |
| histology  | HTAN_1 (n=2) | `htan_1_n2_glas/best_model.pth` | 91.67%  |
| microscopy | HTAN_1 (n=2) | `htan_1_n2_bowl/best_model.pth` | 92.14%  |

> ⚠️ **Confirm the three HF filenames** in `.env` against your
> `mohamedkhaledmk7/HTAN` repo layout before the first deploy. The dermoscopy
> path matches the existing `inference.py` default; the histology/microscopy
> paths are my best guess and may need renaming.

## API

```
GET  /health
POST /api/v1/segment      multipart: image=<file>, modality=<str>, tta=basic|full
```

Example:

```bash
curl -s -F "image=@lesion.jpg" -F "modality=dermoscopy" -F "tta=basic" \
     http://localhost:8001/api/v1/segment | jq
```

## Run locally

```bash
cp .env.example .env          # set HF_TOKEN only if the HF repo is private
docker compose up --build     # needs nvidia-container-toolkit on the host
```

CPU-only (no GPU host): set `HTAN_DEVICE=cpu` and run
`uvicorn app.main:app --port 8001`. Inference is slow but works for smoke tests.

## Weights

- **From Hugging Face (default):** the service downloads on first use and caches
  in the `hf_cache` volume. Run `python scripts/download_weights.py --all` to
  prefetch.
- **From a local dir:** set `HTAN_WEIGHTS_DIR=/weights`, mount the folder, and
  the `HTAN_W_*` values become paths relative to it.

## Layout

```
app/
  main.py        FastAPI app + startup warmup
  config.py      Settings + MODEL_REGISTRY (modality -> arch + weights)
  services.py    Model loading/caching, HF or local weight resolution
  inference.py   TTA (logits-averaged), post-processing, structured output
  schemas.py     Response models
  routes/segment.py
models/          Vendored HTAN model package (htan/, transattunet/)
scripts/download_weights.py
tests/           CPU smoke tests (stubbed model, no GPU/HF needed)
```

## Locked decisions

- **TTA averages raw logits, then sigmoid once** (matches `trainer.py`).
  Averaging per-augmentation probabilities is not equivalent and is not used.
- **No confidence score** — the model isn't calibrated, so a mean-probability
  "confidence" would mislead. Detection stays binary.
- **No physical measurements** — pixel spacing is unknown without DICOM metadata.

## Tests

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q          # uses a stub model; no weights/GPU required
```

"""HTAN segmentation service — FastAPI entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routes import segment_router
from app.services import warmup

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
logger = logging.getLogger("medilink.htan")


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = warmup()
    logger.info("HTAN service ready on %s. Preloaded: %s", settings.device, loaded or "(lazy)")
    yield


app = FastAPI(
    title="MediLink HTAN Service",
    version="1.0.0",
    description="Medical-image segmentation (dermoscopy / histology / microscopy).",
    lifespan=lifespan,
)
app.include_router(segment_router)

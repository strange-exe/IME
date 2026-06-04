"""FastAPI application entry point for the Shipping Email Classification & Data Extraction Service.

This module sets up the FastAPI application, defines the REST API endpoints, and orchestrates requests by calling classification and extraction logic.
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.models import HealthResponse,ParseEmailRequest,ParseEmailResponse,ResponseMetadata
from app.classifier import classify
from app.extractor import extract_records
from app.trainer import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
_pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the startup and shutdown lifecycle events of the FastAPI application.

    Args:
        app: The FastAPI application instance.
    """
    global _pipeline
    logger.info("Initializing application resources: loading classification pipeline.")
    _pipeline = load_model()
    logger.info("Application resources initialized successfully.")
    yield
    logger.info("Releasing application resources: shutting down.")


app = FastAPI(
    title="Shipping Email Parser API",
    description="Enterprise API for classifying shipping emails (tonnage/cargo_vc/cargo_tc) and extracting structured shipping records.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    """Redirects the root URL to the API documentation."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Performs a health probe verification for the application.

    Returns:
        HealthResponse: The status of the server ("healthy").
    """
    return HealthResponse(status="healthy")


@app.post("/parse-email", response_model=ParseEmailResponse, tags=["Parser"])
async def parse_email(request: ParseEmailRequest) -> ParseEmailResponse:
    """Classifies a shipping email and extracts structured metadata records.

    Args:
        request (ParseEmailRequest): The request payload containing the raw email text.

    Returns:
        ParseEmailResponse: Classified category, confidence score, and extracted records.

    Raises:
        HTTPException: 400 if the body is empty, 500 for internal processing failures.
    """
    if not request.email_body or not request.email_body.strip():
        raise HTTPException(status_code=400, detail="email_body must not be empty")

    start_time = time.perf_counter()

    try:
        category, confidence = classify(request.email_body, _pipeline)
        records: List[Dict[str, Any]] = extract_records(category, request.email_body)
    except Exception as exc:
        logger.error("Failed to parse email body due to an internal exception: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Processing error: {exc}") from exc

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return ParseEmailResponse(
        success=True,
        category=category,
        confidence=confidence,
        records=records,
        metadata=ResponseMetadata(
            records_found=len(records),
            processing_time_ms=elapsed_ms,
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)

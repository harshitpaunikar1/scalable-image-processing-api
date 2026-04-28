"""
FastAPI gateway for the scalable image processing API.
Handles authentication, request validation, rate limiting, and job dispatch.
"""
import hashlib
import logging
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

try:
    from worker import ImageJob, ImageProcessingWorker, ProcessingType
    WORKER_AVAILABLE = True
except ImportError:
    WORKER_AVAILABLE = False


class JobSubmitResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    processing_ms: Optional[float] = None
    error: Optional[str] = None
    created_at: Optional[float] = None


class HealthResponse(BaseModel):
    status: str
    queue_depth: int
    timestamp: float


# Simple in-memory rate limiter
_rate_store: Dict[str, List[float]] = {}
RATE_LIMIT = 30
RATE_WINDOW_S = 60.0


def check_rate_limit(client_id: str) -> bool:
    now = time.time()
    window_start = now - RATE_WINDOW_S
    calls = [t for t in _rate_store.get(client_id, []) if t > window_start]
    _rate_store[client_id] = calls
    if len(calls) >= RATE_LIMIT:
        return False
    _rate_store[client_id].append(now)
    return True


# Valid API keys (in production these would come from a secrets store, never hardcoded)
VALID_API_KEYS = {
    "demo_key_readonly",
    "demo_key_processing",
}


def validate_api_key(api_key: Optional[str]) -> str:
    if not api_key or api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing X-API-Key header.")
    return api_key


_worker: Optional["ImageProcessingWorker"] = None


def get_worker() -> "ImageProcessingWorker":
    global _worker
    if _worker is None and WORKER_AVAILABLE:
        _worker = ImageProcessingWorker(num_threads=4, db_path="jobs.db")
        _worker.start()
    if _worker is None:
        raise HTTPException(status_code=503, detail="Worker not available.")
    return _worker


def create_app() -> "FastAPI":
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI required: pip install fastapi uvicorn python-multipart")

    app = FastAPI(
        title="Scalable Image Processing API",
        description="Asynchronous image processing with classification, detection, and enhancement.",
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    def health():
        w = get_worker()
        return HealthResponse(
            status="ok",
            queue_depth=w._job_queue.qsize(),
            timestamp=time.time(),
        )

    @app.post("/jobs", response_model=JobSubmitResponse)
    async def submit_job(
        file: UploadFile = File(...),
        processing_type: str = Form("classify"),
        priority: int = Form(5),
        brightness: float = Form(1.0),
        contrast: float = Form(1.0),
        sharpness: float = Form(1.0),
        max_size: int = Form(256),
        x_api_key: Optional[str] = Header(default=None),
    ):
        validate_api_key(x_api_key)
        client_id = x_api_key or "anonymous"
        if not check_rate_limit(client_id):
            raise HTTPException(status_code=429, detail="Rate limit exceeded.")

        try:
            pt = ProcessingType(processing_type)
        except ValueError:
            raise HTTPException(status_code=400,
                                detail=f"Invalid processing_type. Valid: {[e.value for e in ProcessingType]}")

        content = await file.read()
        if len(content) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image too large (max 20MB).")

        job_id = hashlib.md5(f"{client_id}{time.time()}{len(content)}".encode()).hexdigest()
        job = ImageJob(
            job_id=job_id,
            processing_type=pt,
            image_bytes=content,
            parameters={"brightness": brightness, "contrast": contrast,
                        "sharpness": sharpness, "max_size": max_size},
            priority=max(1, min(10, priority)),
        )

        w = get_worker()
        accepted = w.submit(job)
        if not accepted:
            raise HTTPException(status_code=503, detail="Server busy; retry later.")

        return JobSubmitResponse(job_id=job_id, status="pending",
                                  message="Job submitted successfully.")

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: str, x_api_key: Optional[str] = Header(default=None)):
        validate_api_key(x_api_key)
        w = get_worker()
        result = w.get_result(job_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Job not found.")
        return JobStatusResponse(
            job_id=result["job_id"],
            status=result["status"],
            result=result.get("result"),
            processing_ms=result.get("processing_ms"),
            error=result.get("error"),
            created_at=result.get("created_at"),
        )

    @app.get("/stats")
    def get_stats(x_api_key: Optional[str] = Header(default=None)):
        validate_api_key(x_api_key)
        w = get_worker()
        return {"stats": w.store.stats(), "queue_depth": w._job_queue.qsize()}

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if not FASTAPI_AVAILABLE:
        print("FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
    else:
        try:
            import uvicorn
            app = create_app()
            print("Starting Image Processing API on http://0.0.0.0:8000")
            uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        except ImportError:
            print("Uvicorn not installed. Run: pip install uvicorn")

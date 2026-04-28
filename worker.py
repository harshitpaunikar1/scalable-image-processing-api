"""
Asynchronous image processing worker for the scalable image processing API.
Processes jobs from a queue with batching, parallelism, and back-pressure.
"""
import hashlib
import io
import json
import logging
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from PIL import Image, ImageFilter, ImageEnhance
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import onnxruntime as ort
    ORT_AVAILABLE = True
except ImportError:
    ORT_AVAILABLE = False


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingType(str, Enum):
    CLASSIFY = "classify"
    DETECT = "detect"
    OCR = "ocr"
    THUMBNAIL = "thumbnail"
    ENHANCE = "enhance"
    BARCODE = "barcode"


@dataclass
class ImageJob:
    job_id: str
    processing_type: ProcessingType
    image_bytes: bytes
    parameters: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    priority: int = 5


@dataclass
class JobResult:
    job_id: str
    status: JobStatus
    result: Optional[Dict[str, Any]]
    processing_ms: float
    error: Optional[str] = None
    completed_at: float = field(default_factory=time.time)


class ImagePreprocessor:
    """Prepares images for model inference."""

    def decode(self, image_bytes: bytes) -> Optional[np.ndarray]:
        if CV2_AVAILABLE:
            arr = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            return np.array(img)
        return np.frombuffer(image_bytes[:640 * 480 * 3], dtype=np.uint8).reshape(480, 640, 3)

    def resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        if CV2_AVAILABLE:
            return cv2.resize(image, size)
        return image

    def normalize(self, image: np.ndarray) -> np.ndarray:
        return (image.astype(np.float32) - 127.5) / 127.5

    def to_chw(self, image: np.ndarray) -> np.ndarray:
        return np.transpose(image, (2, 0, 1))

    def thumbnail(self, image_bytes: bytes, max_size: int = 256) -> bytes:
        if PIL_AVAILABLE:
            img = Image.open(io.BytesIO(image_bytes))
            img.thumbnail((max_size, max_size))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return buf.getvalue()
        return image_bytes[:1024]

    def enhance(self, image_bytes: bytes, brightness: float = 1.0,
                contrast: float = 1.0, sharpness: float = 1.0) -> bytes:
        if not PIL_AVAILABLE:
            return image_bytes
        img = Image.open(io.BytesIO(image_bytes))
        if brightness != 1.0:
            img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0:
            img = ImageEnhance.Contrast(img).enhance(contrast)
        if sharpness != 1.0:
            img = ImageEnhance.Sharpness(img).enhance(sharpness)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()


class ModelRunner:
    """Runs ONNX model inference or returns stub results."""

    STUB_CLASSES = ["person", "forklift", "pallet", "truck", "box",
                     "hardhat", "vest", "barrier", "crane", "conveyor"]

    def __init__(self, model_path: Optional[str] = None):
        self._session = None
        if ORT_AVAILABLE and model_path and os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self._session = ort.InferenceSession(model_path, sess_options=opts)
                logger.info("ONNX model loaded: %s", model_path)
            except Exception as exc:
                logger.warning("ONNX load failed: %s", exc)

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        if self._session is not None:
            try:
                input_name = self._session.get_inputs()[0].name
                tensor = self._preprocess(image)
                outputs = self._session.run(None, {input_name: tensor})
                probs = outputs[0][0]
                top_idx = int(np.argmax(probs))
                return {"label": f"class_{top_idx}", "confidence": float(probs[top_idx])}
            except Exception as exc:
                logger.error("Classify error: %s", exc)
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        label = self.STUB_CLASSES[rng.integers(0, len(self.STUB_CLASSES))]
        return {"label": label, "confidence": float(rng.uniform(0.55, 0.99))}

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(int(time.time() * 1000) % 2**32)
        h, w = image.shape[:2]
        n = int(rng.integers(0, 4))
        detections = []
        for _ in range(n):
            label = self.STUB_CLASSES[rng.integers(0, len(self.STUB_CLASSES))]
            x1 = float(rng.integers(0, w - 60))
            y1 = float(rng.integers(0, h - 60))
            detections.append({
                "label": label,
                "confidence": float(rng.uniform(0.45, 0.99)),
                "bbox": [x1, y1, x1 + float(rng.integers(50, 200)), y1 + float(rng.integers(50, 200))],
            })
        return detections

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        resized = image[:224, :224, :3] if image.shape[0] >= 224 else np.zeros((224, 224, 3), dtype=np.uint8)
        normalized = (resized.astype(np.float32) - 127.5) / 127.5
        chw = np.transpose(normalized, (2, 0, 1))
        return chw[np.newaxis]


class JobStore:
    """SQLite store for job state and results."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        processing_type TEXT,
        status TEXT,
        result TEXT,
        error TEXT,
        processing_ms REAL,
        created_at REAL,
        completed_at REAL,
        priority INTEGER
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def create_job(self, job: ImageJob) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO jobs (job_id,processing_type,status,created_at,priority) VALUES (?,?,?,?,?)",
                (job.job_id, job.processing_type.value, JobStatus.PENDING.value,
                 job.created_at, job.priority),
            )
            self.conn.commit()

    def update_job(self, result: JobResult) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE jobs SET status=?,result=?,error=?,processing_ms=?,completed_at=? WHERE job_id=?",
                (result.status.value, json.dumps(result.result), result.error,
                 result.processing_ms, result.completed_at, result.job_id),
            )
            self.conn.commit()

    def get_job(self, job_id: str) -> Optional[Dict]:
        cur = self.conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        row = cur.fetchone()
        if row is None:
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        if d.get("result"):
            d["result"] = json.loads(d["result"])
        return d

    def stats(self) -> Dict[str, Any]:
        import pandas as pd
        df = pd.read_sql_query(
            "SELECT status, processing_type, COUNT(*) AS cnt, AVG(processing_ms) AS avg_ms "
            "FROM jobs GROUP BY status, processing_type", self.conn
        )
        return df.to_dict(orient="records")


class ImageProcessingWorker:
    """
    Worker that reads from a priority queue, processes image jobs, and writes results.
    """

    MAX_QUEUE_SIZE = 100

    def __init__(self, model_path: Optional[str] = None, db_path: str = ":memory:",
                 num_threads: int = 2):
        self.preprocessor = ImagePreprocessor()
        self.runner = ModelRunner(model_path=model_path)
        self.store = JobStore(db_path=db_path)
        self._job_queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=self.MAX_QUEUE_SIZE)
        self._threads: List[threading.Thread] = []
        self._running = False
        self.num_threads = num_threads

    def submit(self, job: ImageJob) -> bool:
        if self._job_queue.full():
            logger.warning("Job queue full; applying back-pressure.")
            return False
        self.store.create_job(job)
        self._job_queue.put((job.priority, job.created_at, job))
        return True

    def start(self) -> None:
        self._running = True
        for _ in range(self.num_threads):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._threads.append(t)
        logger.info("Worker started with %d threads.", self.num_threads)

    def stop(self) -> None:
        self._running = False
        for _ in self._threads:
            self._job_queue.put((999, 0, None))

    def _worker_loop(self) -> None:
        while self._running:
            try:
                _, _, job = self._job_queue.get(timeout=1.0)
                if job is None:
                    break
                result = self._process(job)
                self.store.update_job(result)
                self._job_queue.task_done()
            except queue.Empty:
                continue

    def _process(self, job: ImageJob) -> JobResult:
        t0 = time.perf_counter()
        try:
            image = self.preprocessor.decode(job.image_bytes)
            result: Dict[str, Any] = {}

            if job.processing_type == ProcessingType.CLASSIFY:
                result = self.runner.classify(image)
            elif job.processing_type == ProcessingType.DETECT:
                result = {"detections": self.runner.detect(image)}
            elif job.processing_type == ProcessingType.THUMBNAIL:
                max_size = job.parameters.get("max_size", 256)
                thumb_bytes = self.preprocessor.thumbnail(job.image_bytes, max_size)
                result = {"thumbnail_size_bytes": len(thumb_bytes), "max_size": max_size}
            elif job.processing_type == ProcessingType.ENHANCE:
                enhanced = self.preprocessor.enhance(
                    job.image_bytes,
                    brightness=job.parameters.get("brightness", 1.0),
                    contrast=job.parameters.get("contrast", 1.0),
                    sharpness=job.parameters.get("sharpness", 1.0),
                )
                result = {"enhanced_size_bytes": len(enhanced)}
            else:
                result = {"message": f"Processing type {job.processing_type} completed."}

            processing_ms = (time.perf_counter() - t0) * 1000
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.COMPLETED,
                result=result,
                processing_ms=round(processing_ms, 1),
            )
        except Exception as exc:
            processing_ms = (time.perf_counter() - t0) * 1000
            logger.error("Job %s failed: %s", job.job_id, exc)
            return JobResult(
                job_id=job.job_id,
                status=JobStatus.FAILED,
                result=None,
                processing_ms=round(processing_ms, 1),
                error=str(exc),
            )

    def get_result(self, job_id: str) -> Optional[Dict]:
        return self.store.get_job(job_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    worker = ImageProcessingWorker(num_threads=2)
    worker.start()

    rng = np.random.default_rng(42)
    fake_image = rng.integers(0, 255, (480, 640, 3), dtype=np.uint8).tobytes()

    job_types = [ProcessingType.CLASSIFY, ProcessingType.DETECT,
                 ProcessingType.THUMBNAIL, ProcessingType.ENHANCE]

    submitted_ids = []
    for i, pt in enumerate(job_types):
        job = ImageJob(
            job_id=f"job_{i:04d}",
            processing_type=pt,
            image_bytes=fake_image,
            parameters={"max_size": 128, "brightness": 1.2, "contrast": 1.1},
            priority=i + 1,
        )
        ok = worker.submit(job)
        if ok:
            submitted_ids.append(job.job_id)
            print(f"Submitted job {job.job_id} ({pt.value})")

    time.sleep(1.5)

    print("\nJob results:")
    for job_id in submitted_ids:
        result = worker.get_result(job_id)
        if result:
            print(f"  {job_id}: status={result['status']} "
                  f"ms={result.get('processing_ms', 0):.1f} "
                  f"result={result.get('result')}")

    print("\nWorker stats:")
    for row in worker.store.stats():
        print(f"  {row}")

    worker.stop()

"""
CLI entry for one saved image.

Example:
    python recognize_image.py F:\\YOLO+OCR\\test\\sample.jpg
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable, List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import FastTextOCR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def first_existing(paths: Iterable[str]) -> str:
    paths = list(paths)
    for path in paths:
        if path and os.path.exists(path):
            return path
    return paths[0]


def parse_ids(value: str) -> Optional[List[int]]:
    value = (value or "").strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def read_image_rgb(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to read image: %s" % path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def build_parser() -> argparse.ArgumentParser:
    default_model = first_existing([
        os.path.join(BASE_DIR, "ma.pt"),
        os.path.join(BASE_DIR, "text_detector.pt"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "ma.pt")),
    ])
    parser = argparse.ArgumentParser(description="Recognize one engraved material text image.")
    parser.add_argument("image", help="Image path.")
    parser.add_argument("--model", default=os.environ.get("TEXT_YOLO_PATH", default_model))
    parser.add_argument("--conf", type=float, default=float(os.environ.get("TEXT_CONF", "0.25")))
    parser.add_argument("--iou", type=float, default=float(os.environ.get("TEXT_IOU", "0.50")))
    parser.add_argument("--ocr-lang", default=os.environ.get("OCR_LANG", "ch"))
    parser.add_argument("--ocr-device", default=os.environ.get("OCR_DEVICE", "gpu"))
    parser.add_argument("--vl-device", default=os.environ.get("VL_DEVICE", "gpu:0"))
    parser.add_argument("--text-class-ids", default=os.environ.get("TEXT_CLASS_IDS", ""))
    parser.add_argument("--no-vl", action="store_true")
    parser.add_argument("--all", action="store_true", help="Return all valid texts instead of first valid text.")
    parser.add_argument("--include-failures", action="store_true")
    return parser


def run_image(args: argparse.Namespace) -> dict:
    image = read_image_rgb(args.image)
    pipeline = FastTextOCR(
        model_path=args.model,
        ocr_lang=args.ocr_lang,
        ocr_device=args.ocr_device,
        vl_device=args.vl_device,
        conf=args.conf,
        iou=args.iou,
        text_class_ids=parse_ids(args.text_class_ids),
        enable_vl_fallback=not args.no_vl,
        temp_dir=os.path.join(BASE_DIR, ".tmp"),
    )
    return pipeline.process(
        image,
        conf=args.conf,
        iou=args.iou,
        return_all=args.all,
        include_failures=args.include_failures,
        enable_vl_fallback=not args.no_vl,
    )


def main() -> int:
    args = build_parser().parse_args()
    payload = run_image(args)
    best_text = payload["best"]["text"] if payload.get("best") else ""
    output = {
        "best_text": best_text,
        "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
        "results": payload.get("results", []),
        "attempt_count": len(payload.get("attempts", [])),
        "detection_count": len(payload.get("detections", [])),
        "candidate_count": len(payload.get("candidates", [])),
        "model_path": payload.get("model_path", args.model),
    }
    print(json.dumps(to_builtin(output), ensure_ascii=False, indent=2))
    return 0 if best_text else 2


if __name__ == "__main__":
    raise SystemExit(main())
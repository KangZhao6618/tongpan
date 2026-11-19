"""
Standalone web/API entry for the fast engraved-text recognition pipeline.

Start:
    python app.py

Model paths:
    Logic1 uses D/danyi.pt. Logic2 uses D/ma.pt.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import traceback
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from engine import FastTextOCR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _first_existing(paths: Iterable[str]) -> str:
    for path in paths:
        if path and os.path.exists(path):
            return path
    return next(iter(paths))


LOGIC1_YOLO_PATH = os.environ.get(
    "LOGIC1_YOLO_PATH",
    _first_existing([
        os.path.join(BASE_DIR, "danyi.pt"),
        os.path.join(BASE_DIR, "text_detector.pt"),
        os.path.abspath(os.path.join(BASE_DIR, "..", "danyi.pt")),
    ]),
)

LOGIC2_YOLO_PATH = os.environ.get(
    "LOGIC2_YOLO_PATH",
    os.environ.get(
        "TEXT_YOLO_PATH",
        _first_existing([
            os.path.join(BASE_DIR, "ma.pt"),
            os.path.join(BASE_DIR, "text_detector.pt"),
            os.path.abspath(os.path.join(BASE_DIR, "..", "ma.pt")),
        ]),
    ),
)

TEXT_YOLO_PATH = LOGIC2_YOLO_PATH

OCR_LANG = os.environ.get("OCR_LANG", "ch")
OCR_DEVICE = os.environ.get("OCR_DEVICE", "gpu")
VL_DEVICE = os.environ.get("VL_DEVICE", "gpu:0")
DEFAULT_CONF = float(os.environ.get("TEXT_CONF", "0.25"))
DEFAULT_IOU = float(os.environ.get("TEXT_IOU", "0.50"))
MAX_TEXT_CANDIDATES = int(os.environ.get("MAX_TEXT_CANDIDATES", "8"))
MAX_VL_CANDIDATES = int(os.environ.get("MAX_VL_CANDIDATES", "3"))
ENABLE_VL_FALLBACK = os.environ.get("ENABLE_VL_FALLBACK", "1").strip() not in ("0", "false", "False")

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response
_pipelines: Dict[str, FastTextOCR] = {}


def _parse_ids(value: str) -> Optional[List[int]]:
    value = (value or "").strip()
    if not value:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_names(value: str) -> Optional[List[str]]:
    value = (value or "").strip()
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def get_pipeline(model_path: str, text_class_ids: Optional[List[int]] = None) -> FastTextOCR:
    key = os.path.abspath(model_path) + "|" + ",".join(str(v) for v in (text_class_ids or []))
    if key not in _pipelines:
        _pipelines[key] = FastTextOCR(
            model_path=model_path,
            ocr_lang=OCR_LANG,
            ocr_device=OCR_DEVICE,
            vl_device=VL_DEVICE,
            conf=DEFAULT_CONF,
            iou=DEFAULT_IOU,
            text_class_ids=text_class_ids if text_class_ids is not None else _parse_ids(os.environ.get("TEXT_CLASS_IDS", "")),
            text_class_names=_parse_names(os.environ.get("TEXT_CLASS_NAMES", "")),
            enable_vl_fallback=ENABLE_VL_FALLBACK,
            max_text_candidates=MAX_TEXT_CANDIDATES,
            max_vl_candidates=MAX_VL_CANDIDATES,
            temp_dir=os.path.join(BASE_DIR, ".tmp"),
        )
    return _pipelines[key]


def _read_image_from_request() -> np.ndarray:
    file = request.files.get("image")
    if file is None:
        raise ValueError("missing image file")
    img = Image.open(io.BytesIO(file.read())).convert("RGB")
    return np.array(img)


def _encode_jpeg(image_rgb: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("failed to encode image")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _draw_box(image: np.ndarray, box: List[int], color: tuple, label: str, thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    if label:
        y_text = max(16, y1 - 6)
        cv2.putText(image, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_overlay(image_rgb: np.ndarray, payload: Dict[str, Any]) -> np.ndarray:
    vis = image_rgb.copy()

    for det in payload.get("detections", []):
        label = "%s %.2f" % (det.get("class_name", ""), float(det.get("confidence", 0.0)))
        _draw_box(vis, det["bbox"], (130, 130, 130), label, thickness=1)

    for cand in payload.get("candidates", []):
        source = str(cand.get("source", "candidate"))
        _draw_box(vis, cand["bbox"], (255, 170, 0), source, thickness=1)

    for item in payload.get("results", []):
        text = item.get("text") or item.get("raw_text") or "result"
        _draw_box(vis, item["bbox"], (0, 220, 0), str(text)[:48], thickness=3)

    return vis


@app.route("/api/recognize_text", methods=["POST", "OPTIONS"])
def api_recognize_text():
    if request.method == "OPTIONS":
        return ("", 204)
    try:
        image = _read_image_from_request()
        conf = float(request.form.get("conf", DEFAULT_CONF))
        iou = float(request.form.get("iou", DEFAULT_IOU))
        return_all = request.form.get("return_all", "0") in ("1", "true", "True")
        include_failures = request.form.get("include_failures", "0") in ("1", "true", "True")
        enable_vl = request.form.get("enable_vl", "1") not in ("0", "false", "False")
        mode = request.form.get("mode", "auto").strip().lower()
        whole_fallbacks = int(request.form.get("whole_fallbacks", "2"))

        classes = None
        classes_raw = request.form.get("classes", "").strip()
        if classes_raw:
            classes = _parse_ids(classes_raw)

        if mode in ("logic1", "ppocr", "fast"):
            pipe = get_pipeline(LOGIC1_YOLO_PATH, text_class_ids=[0])
            payload = pipe.process(
                image,
                classes=classes,
                conf=conf,
                iou=iou,
                return_all=return_all,
                include_failures=include_failures,
                enable_vl_fallback=False,
            )
            payload["mode"] = "logic1"
        elif mode in ("logic2", "vl", "vl_first"):
            pipe = get_pipeline(LOGIC2_YOLO_PATH)
            payload = pipe.process_vl_first(
                image,
                classes=classes,
                conf=conf,
                iou=iou,
                return_all=return_all,
                include_failures=include_failures,
                whole_fallbacks=whole_fallbacks,
            )
        else:
            pipe = get_pipeline(LOGIC2_YOLO_PATH)
            payload = pipe.process(
                image,
                classes=classes,
                conf=conf,
                iou=iou,
                return_all=return_all,
                include_failures=include_failures,
                enable_vl_fallback=enable_vl,
            )
            payload["mode"] = "auto"
        overlay = draw_overlay(image, payload)
        response = {
            "best_text": payload["best"]["text"] if payload.get("best") else "",
            "count": len(payload.get("results", [])),
            "elapsed_seconds": payload.get("elapsed_seconds", 0.0),
            "model_path": payload.get("model_path", ""),
            "mode": payload.get("mode", ""),
            "results": payload.get("results", []),
            "attempt_count": len(payload.get("attempts", [])),
            "detection_count": len(payload.get("detections", [])),
            "candidate_count": len(payload.get("candidates", [])),
            "annotated": _encode_jpeg(overlay),
        }
        return jsonify(_to_builtin(response))
    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


@app.route("/api/model_info", methods=["GET"])
def api_model_info():
    try:
        pipe = get_pipeline(LOGIC2_YOLO_PATH)
        return jsonify({
            "logic1_model_path": LOGIC1_YOLO_PATH,
            "logic2_model_path": LOGIC2_YOLO_PATH,
            "model_path": pipe.model_path,
            "names": _to_builtin(pipe.names),
            "auto_text_class_ids": sorted(pipe.auto_text_class_ids),
            "auto_qr_class_ids": sorted(pipe.auto_qr_class_ids),
            "vl_fallback": pipe.enable_vl_fallback,
        })
    except Exception as exc:
        return jsonify({"error": str(exc), "model_path": TEXT_YOLO_PATH}), 500


@app.route("/", methods=["GET"])
def index():
    return """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fast Text OCR</title>
  <style>
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Arial, "Microsoft YaHei", sans-serif;
      background: #f4f6f8;
      color: #20242a;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      padding: 0 20px;
      border-bottom: 1px solid #d8dde3;
      background: #ffffff;
      font-weight: 700;
    }
    main {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: calc(100vh - 56px);
    }
    aside {
      padding: 16px;
      border-right: 1px solid #d8dde3;
      background: #ffffff;
    }
    section {
      padding: 16px;
      min-width: 0;
    }
    label {
      display: block;
      margin: 14px 0 6px;
      font-size: 13px;
      color: #4d5662;
    }
    input[type="file"],
    input[type="number"],
    input[type="text"] {
      width: 100%;
      padding: 9px 10px;
      border: 1px solid #cbd3dc;
      border-radius: 6px;
      background: #ffffff;
      color: #20242a;
    }
    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      font-size: 13px;
    }
    button {
      width: 100%;
      margin-top: 14px;
      padding: 10px 12px;
      border: 0;
      border-radius: 6px;
      background: #1769aa;
      color: #ffffff;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled {
      background: #9aa8b5;
      cursor: default;
    }
    .status {
      margin-top: 12px;
      font-size: 13px;
      color: #4d5662;
      min-height: 20px;
    }
    .result {
      margin-top: 12px;
      padding: 10px;
      border: 1px solid #d8dde3;
      border-radius: 6px;
      background: #f8fafc;
      font-family: Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      min-height: 80px;
    }
    .image-wrap {
      width: 100%;
      min-height: 420px;
      display: grid;
      place-items: center;
      border: 1px solid #d8dde3;
      background: #111418;
      overflow: auto;
    }
    img {
      max-width: 100%;
      height: auto;
      display: block;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid #d8dde3; }
    }
  </style>
</head>
<body>
  <header>Fast Text OCR</header>
  <main>
    <aside>
      <input id="file" type="file" accept="image/*">
      <div class="row">
        <div>
          <label for="conf">检测置信度</label>
          <input id="conf" type="number" min="0.01" max="1" step="0.01" value="0.25">
        </div>
        <div>
          <label for="iou">框合并阈值</label>
          <input id="iou" type="number" min="0.01" max="1" step="0.01" value="0.50">
        </div>
      </div>
      <label for="classes">类别ID</label>
      <input id="classes" type="text" placeholder="一般留空">
      <label class="check"><input id="vl" type="checkbox" checked> 必要时启用 VL 兜底识别</label>
      <label class="check"><input id="failures" type="checkbox"> 显示失败候选框</label>
      <label class="check"><input id="returnAll" type="checkbox"> 返回所有有效结果</label>
      <button id="run" disabled>开始识别</button>
      <div class="status" id="status">未选择图片</div>
      <div class="result" id="result"></div>
    </aside>
    <section>
      <div class="image-wrap"><img id="preview" alt=""></div>
    </section>
  </main>
  <script>
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    const runBtn = document.getElementById('run');
    const statusEl = document.getElementById('status');
    const resultEl = document.getElementById('result');
    let selectedFile = null;

    fileInput.addEventListener('change', () => {
      selectedFile = fileInput.files[0] || null;
      runBtn.disabled = !selectedFile;
      resultEl.textContent = '';
      if (!selectedFile) {
        preview.removeAttribute('src');
        statusEl.textContent = '未选择图片';
        return;
      }
      preview.src = URL.createObjectURL(selectedFile);
      statusEl.textContent = selectedFile.name;
    });

    runBtn.addEventListener('click', async () => {
      if (!selectedFile) return;
      runBtn.disabled = true;
      statusEl.textContent = '识别中...';
      resultEl.textContent = '';
      const form = new FormData();
      form.append('image', selectedFile, selectedFile.name);
      form.append('conf', document.getElementById('conf').value);
      form.append('iou', document.getElementById('iou').value);
      form.append('enable_vl', document.getElementById('vl').checked ? '1' : '0');
      form.append('include_failures', document.getElementById('failures').checked ? '1' : '0');
      form.append('return_all', document.getElementById('returnAll').checked ? '1' : '0');
      const classes = document.getElementById('classes').value.trim();
      if (classes) form.append('classes', classes);

      try {
        const resp = await fetch('/api/recognize_text', { method: 'POST', body: form });
        const data = await resp.json();
        if (!resp.ok || data.error) {
          statusEl.textContent = '识别失败';
          resultEl.textContent = data.error || '请求失败';
          return;
        }
        preview.src = data.annotated;
        statusEl.textContent = `识别完成，用时 ${data.elapsed_seconds}s`;
        resultEl.textContent = JSON.stringify({
          识别文本: data.best_text,
          count: data.count,
          attempts: data.attempt_count,
          detections: data.detection_count,
          candidates: data.candidate_count,
          results: data.results
        }, null, 2);
      } catch (err) {
        statusEl.textContent = '网络错误';
        resultEl.textContent = String(err.message || err);
      } finally {
        runBtn.disabled = !selectedFile;
      }
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    print("Logic1 model:", LOGIC1_YOLO_PATH)
    print("Logic2 model:", LOGIC2_YOLO_PATH)
    print("API:   http://127.0.0.1:5055")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5055")), debug=False)

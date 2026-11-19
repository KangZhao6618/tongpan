"""
Fast text recognition pipeline for engraved material labels.

Pipeline:
1. YOLO detects engraved text boxes from the full image.
2. PP-OCR runs on cropped text boxes first.
3. PaddleOCR-VL 1.6 runs only as a fallback on a few top candidates.

The target output is normalized to labels such as:
CuP:99.9% 260165200-01
CuMn0.8at%:99.9999% 26030654-01
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import numpy as np


DEFAULT_TEXT_CLASS_NAMES = (
    "engraved_text",
    "text",
    "ocr_text",
    "label_text",
    "mark_text",
    "caption",
    "wenben",
    "kezi",
)

DEFAULT_QR_CLASS_NAMES = (
    "qr",
    "qr_code",
    "qrcode",
    "code",
    "barcode",
)

_BRAND_RE = re.compile(r"G\s*R\s*[I1l|]\s*K\s*[I1l|]\s*N", re.IGNORECASE)
_TARGET_RE = re.compile(
    r"(?:(G\s*R\s*[I1l|]\s*K\s*[I1l|]\s*N)\s*)?"
    r"([A-Za-z][A-Za-z0-9OoQIl|SB\.\+\-%％/ ]{0,40}?)\s*[:：;]\s*"
    r"([0-9OoQIl|SB]{1,3}(?:[\.,][0-9OoQIl|SB]{1,6})?)\s*%?\s*"
    r"([0-9OoQIl|SB]{8,10})\s*[-－—_\s]?\s*([0-9OoQIl|SB]{1,3})",
    re.IGNORECASE,
)

_PURITY_RE = re.compile(r"([0-9OoQIl|SB]{1,3}(?:[\.,][0-9OoQIl|SB]{1,6})?)\s*%")
_SERIAL_RE = re.compile(r"([0-9OoQIl|SB]{8,10})\s*[-－—_\s]?\s*([0-9OoQIl|SB]{1,3})")
_DIGIT_TRANS = str.maketrans({
    "O": "0",
    "o": "0",
    "Q": "0",
    "I": "1",
    "l": "1",
    "|": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "b": "8",
})


def _as_int_set(values: Optional[Iterable[int]]) -> Set[int]:
    if not values:
        return set()
    return {int(v) for v in values}


def _split_names(values: Optional[Iterable[str]]) -> Set[str]:
    if not values:
        return set()
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _fix_digits(value: str) -> str:
    return value.translate(_DIGIT_TRANS)


def _has_brand(value: str) -> bool:
    return bool(_BRAND_RE.search(value or ""))


def _fix_brand(value: str) -> str:
    return _BRAND_RE.sub("GRIKIN", value or "")


def _fix_component(value: str) -> Optional[str]:
    fixed = _fix_brand(value)
    fixed = _BRAND_RE.sub("", fixed)
    fixed = fixed.replace("％", "%")
    fixed = re.sub(r"\s+", "", fixed)
    fixed = re.sub(r"[^A-Za-z0-9.%+\-/]", "", fixed)
    fixed = re.sub(r"[OoQ](?=\.)|(?<=\.)[OoQIl|SB]", lambda m: _fix_digits(m.group(0)), fixed)
    fixed = fixed.strip(".:-;/")
    if not fixed:
        return None

    if re.fullmatch(r"cup", fixed, re.IGNORECASE):
        return "CuP"
    if re.match(r"cumn", fixed, re.IGNORECASE):
        return "CuMn" + fixed[4:]
    if fixed.lower().startswith("cu"):
        return "Cu" + fixed[2:]
    return fixed


def _format_label(component: str, purity: str, serial: Tuple[str, str], brand: bool = False) -> str:
    body = "%s:%s%% %s-%s" % (component, purity, serial[0], serial[1])
    return "GRIKIN " + body if brand else body


def _fix_purity(value: str) -> Optional[str]:
    fixed = _fix_digits(value).replace(",", ".")
    fixed = re.sub(r"[^0-9.]", "", fixed)
    if not fixed:
        return None
    if "." not in fixed and len(fixed) == 3:
        fixed = fixed[:2] + "." + fixed[2:]
    try:
        number = float(fixed)
    except ValueError:
        return None
    if not (0 < number <= 100):
        return None
    return fixed


def _fix_serial(main: str, suffix: str) -> Optional[Tuple[str, str]]:
    main_fixed = re.sub(r"\D", "", _fix_digits(main))
    suffix_fixed = re.sub(r"\D", "", _fix_digits(suffix))
    if not (8 <= len(main_fixed) <= 10):
        return None
    if not (1 <= len(suffix_fixed) <= 3):
        return None
    if len(suffix_fixed) == 1:
        suffix_fixed = "0" + suffix_fixed
    return main_fixed, suffix_fixed


def normalize_target_text(tokens: Sequence[str]) -> Optional[str]:
    """Extract and normalize material composition, purity, serial, and optional GRIKIN brand."""
    raw = " ".join(str(t) for t in tokens if str(t).strip())
    if not raw:
        return None

    normalized = (
        raw.replace("：", ":")
        .replace("－", "-")
        .replace("—", "-")
        .replace("_", "-")
    )
    variants = [
        normalized,
        normalized.replace(" ", ""),
        normalized.replace("|", "").replace(" ", ""),
    ]

    for variant in variants:
        match = _TARGET_RE.search(variant)
        if not match:
            continue
        brand = bool(match.group(1)) or _has_brand(match.group(0))
        component = _fix_component(match.group(2))
        purity = _fix_purity(match.group(3))
        serial = _fix_serial(match.group(4), match.group(5))
        if component and purity and serial:
            return _format_label(component, purity, serial, brand=brand)

    for variant in variants:
        purity_match = _PURITY_RE.search(variant)
        serial_match = _SERIAL_RE.search(variant)
        if not purity_match or not serial_match:
            continue
        prefix = variant[: purity_match.start()]
        brand = _has_brand(prefix)
        component = None
        separator = max(prefix.rfind(":"), prefix.rfind("："), prefix.rfind(";"))
        if separator >= 0:
            component = _fix_component(prefix[:separator])
        purity = _fix_purity(purity_match.group(1))
        serial = _fix_serial(serial_match.group(1), serial_match.group(2))
        if component and purity and serial:
            return _format_label(component, purity, serial, brand=brand)

    return None


def _average_score(scores: Sequence[float]) -> float:
    if not scores:
        return 0.0
    return round(float(sum(scores) / len(scores)), 4)


def _clip_box(box: Sequence[float], width: int, height: int) -> List[int]:
    x1, y1, x2, y2 = box
    x1_i = max(0, min(width, int(round(x1))))
    y1_i = max(0, min(height, int(round(y1))))
    x2_i = max(0, min(width, int(round(x2))))
    y2_i = max(0, min(height, int(round(y2))))
    x1_i, x2_i = sorted((x1_i, x2_i))
    y1_i, y2_i = sorted((y1_i, y2_i))
    return [x1_i, y1_i, x2_i, y2_i]


def _box_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def _safe_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, names.get(str(class_id), "cls_%s" % class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return "cls_%s" % class_id


class FastTextOCR:
    """YOLO text detection plus OCR with optional VL fallback."""

    def __init__(
        self,
        model_path: str,
        ocr_lang: str = "ch",
        ocr_device: str = "gpu",
        vl_device: str = "gpu:0",
        conf: float = 0.25,
        iou: float = 0.5,
        text_class_ids: Optional[Iterable[int]] = None,
        text_class_names: Optional[Iterable[str]] = None,
        qr_class_names: Optional[Iterable[str]] = None,
        enable_vl_fallback: bool = True,
        max_text_candidates: int = 8,
        max_vl_candidates: int = 3,
        temp_dir: Optional[str] = None,
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError("YOLO model not found: %s" % model_path)


        from paddleocr import PaddleOCR
        from ultralytics import YOLO
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.ocr = PaddleOCR(lang=ocr_lang, device=ocr_device)
        self.vl_device = vl_device
        self.default_conf = conf
        self.default_iou = iou
        self.text_class_ids = _as_int_set(text_class_ids)
        self.text_class_names = _split_names(text_class_names) or set(DEFAULT_TEXT_CLASS_NAMES)
        self.qr_class_names = _split_names(qr_class_names) or set(DEFAULT_QR_CLASS_NAMES)
        self.enable_vl_fallback = bool(enable_vl_fallback)
        self.max_text_candidates = int(max_text_candidates)
        self.max_vl_candidates = int(max_vl_candidates)
        self.temp_dir = temp_dir
        self._vl_pipeline = None

        if self.temp_dir:
            os.makedirs(self.temp_dir, exist_ok=True)

        self.names = getattr(self.model, "names", {})
        self.auto_text_class_ids = self._find_class_ids(self.text_class_names)
        self.auto_qr_class_ids = self._find_class_ids(self.qr_class_names)

    def _find_class_ids(self, wanted_names: Set[str]) -> Set[int]:
        found: Set[int] = set()
        names = self.names
        if isinstance(names, dict):
            items = names.items()
        elif isinstance(names, (list, tuple)):
            items = enumerate(names)
        else:
            items = []
        for raw_id, raw_name in items:
            name = str(raw_name).strip().lower()
            if any(wanted == name or wanted in name for wanted in wanted_names):
                found.add(int(raw_id))
        return found

    def _is_text_detection(self, det: Dict[str, Any]) -> bool:
        class_id = int(det.get("class_id", -1))
        class_name = str(det.get("class_name", "")).strip().lower()

        if self.text_class_ids:
            return class_id in self.text_class_ids
        if self.auto_text_class_ids:
            return class_id in self.auto_text_class_ids
        if class_name:
            if any(qr == class_name or qr in class_name for qr in self.qr_class_names):
                return False
            return any(name == class_name or name in class_name for name in self.text_class_names)

        return True

    def _is_qr_detection(self, det: Dict[str, Any]) -> bool:
        class_id = int(det.get("class_id", -1))
        class_name = str(det.get("class_name", "")).strip().lower()

        if self.auto_qr_class_ids and class_id in self.auto_qr_class_ids:
            return True
        if class_name:
            return any(name == class_name or name in class_name for name in self.qr_class_names)
        return False

    def detect(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        max_det: int = 300,
    ) -> List[Dict[str, Any]]:
        conf_value = self.default_conf if conf is None else float(conf)
        iou_value = self.default_iou if iou is None else float(iou)

        results = self.model.predict(
            source=image,
            conf=conf_value,
            iou=iou_value,
            classes=classes,
            max_det=max_det,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        height, width = image.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            cls_arr = (
                result.boxes.cls.cpu().numpy().astype(int)
                if result.boxes.cls is not None
                else np.full(len(boxes), -1)
            )
            conf_arr = (
                result.boxes.conf.cpu().numpy()
                if result.boxes.conf is not None
                else np.zeros(len(boxes))
            )

            for idx in range(len(boxes)):
                class_id = int(cls_arr[idx])
                box = _clip_box(boxes[idx], width, height)
                if _box_area(box) <= 0:
                    continue
                detections.append({
                    "bbox": box,
                    "raw_bbox": box[:],
                    "class_id": class_id,
                    "class_name": _safe_name(self.names, class_id),
                    "confidence": round(float(conf_arr[idx]), 4),
                    "area": _box_area(box),
                })

        detections.sort(key=lambda d: float(d.get("confidence", 0.0)), reverse=True)
        return detections

    def _expand_text_box(self, box: Sequence[int], image_shape: Tuple[int, int]) -> List[int]:
        height, width = image_shape
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_x = max(8, int(bw * 0.10))
        pad_y = max(8, int(bh * 0.35))
        return _clip_box([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)

    def _qr_to_text_box(self, box: Sequence[int], image_shape: Tuple[int, int]) -> List[int]:
        height, width = image_shape
        x1, y1, x2, y2 = box
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        return _clip_box(
            [
                x1 - 2.2 * bw,
                y2 - 0.25 * bh,
                x2 + 3.2 * bw,
                y2 + 1.35 * bh,
            ],
            width,
            height,
        )

    def make_candidates(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        image_shape = image.shape[:2]
        text_dets = [det for det in detections if self._is_text_detection(det)]

        candidates: List[Dict[str, Any]] = []
        for det in text_dets:
            item = dict(det)
            item["source"] = "text_detector"
            item["bbox"] = self._expand_text_box(det["bbox"], image_shape)
            item["area"] = _box_area(item["bbox"])
            candidates.append(item)

        if not candidates:
            for det in detections:
                if not self._is_qr_detection(det):
                    continue
                item = dict(det)
                item["source"] = "qr_roi_fallback"
                item["bbox"] = self._qr_to_text_box(det["bbox"], image_shape)
                item["area"] = _box_area(item["bbox"])
                item["class_name"] = "qr_expanded_text_roi"
                candidates.append(item)

        candidates.sort(
            key=lambda d: (
                1 if d.get("source") == "text_detector" else 0,
                float(d.get("confidence", 0.0)),
                int(d.get("area", 0)),
            ),
            reverse=True,
        )
        return candidates

    def crop_region(
        self,
        image: np.ndarray,
        box: Sequence[int],
        min_short_side: int = 224,
        max_long_side: int = 1800,
    ) -> np.ndarray:
        height, width = image.shape[:2]
        x1, y1, x2, y2 = _clip_box(box, width, height)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return crop

        ch, cw = crop.shape[:2]
        short = max(1, min(ch, cw))
        long_side = max(ch, cw)
        scale = 1.0
        if short < min_short_side:
            scale = max(scale, float(min_short_side) / float(short))
        if max_long_side > 0 and long_side * scale > max_long_side:
            scale = float(max_long_side) / float(long_side)

        if abs(scale - 1.0) > 0.05:
            new_w = max(1, int(round(cw * scale)))
            new_h = max(1, int(round(ch * scale)))
            crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return crop

    def _ocr_variants(self, crop: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        if crop.ndim == 2:
            rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
        else:
            rgb = crop

        variants: List[Tuple[str, np.ndarray]] = [("original", rgb)]
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        variants.append(("clahe", cv2.cvtColor(clahe, cv2.COLOR_GRAY2RGB)))

        blur = cv2.GaussianBlur(clahe, (0, 0), 1.0)
        sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)
        variants.append(("sharp", cv2.cvtColor(sharp, cv2.COLOR_GRAY2RGB)))

        return variants

    def _extract_ppocr_tokens(self, result: Any) -> Tuple[List[str], List[float]]:
        texts: List[str] = []
        scores: List[float] = []
        if not result:
            return texts, scores

        for page in result:
            if isinstance(page, dict):
                page_texts = page.get("rec_texts", []) or []
                page_scores = page.get("rec_scores", []) or []
            else:
                page_texts = getattr(page, "rec_texts", []) or []
                page_scores = getattr(page, "rec_scores", []) or []
            for idx, text in enumerate(page_texts):
                value = str(text).strip()
                if not value:
                    continue
                texts.append(value)
                if idx < len(page_scores):
                    try:
                        scores.append(float(page_scores[idx]))
                    except Exception:
                        scores.append(0.0)
        return texts, scores

    def _ocr_crop_light(self, crop: np.ndarray) -> Dict[str, Any]:
        best: Dict[str, Any] = {
            "engine": "ppocr",
            "variant": "",
            "text": "",
            "raw_text": "",
            "tokens": [],
            "ocr_score": 0.0,
            "valid": False,
        }

        for variant_name, variant in self._ocr_variants(crop):
            try:
                result = self.ocr.predict(variant)
                tokens, scores = self._extract_ppocr_tokens(result)
            except Exception as exc:
                best["error"] = str(exc)
                continue

            raw_text = " ".join(tokens).strip()
            score = _average_score(scores)
            normalized = normalize_target_text(tokens)
            current = {
                "engine": "ppocr",
                "variant": variant_name,
                "text": normalized or "",
                "raw_text": raw_text,
                "tokens": tokens,
                "ocr_score": score,
                "valid": normalized is not None,
            }
            if normalized:
                return current
            if raw_text and score >= float(best.get("ocr_score", 0.0)):
                best = current

        return best

    def _get_vl(self) -> Any:
        if self._vl_pipeline is None:
            from paddleocr import PaddleOCRVL

            self._vl_pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                device=self.vl_device,
                use_layout_detection=False,
            )
        return self._vl_pipeline

    def _extract_vl_tokens(self, result: Any) -> List[str]:
        tokens: List[str] = []
        for page in result or []:
            if isinstance(page, dict):
                blocks = page.get("parsing_res_list", []) or []
            else:
                try:
                    blocks = page.get("parsing_res_list", []) or []
                except Exception:
                    blocks = getattr(page, "parsing_res_list", []) or []
            for block in blocks:
                content = None
                if isinstance(block, dict):
                    content = block.get("content") or block.get("block_content")
                else:
                    content = getattr(block, "content", None) or getattr(block, "block_content", None)
                if content:
                    tokens.append(str(content).strip())
        return tokens

    def _ocr_crop_vl(self, crop: np.ndarray) -> Dict[str, Any]:
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir=self.temp_dir) as file_obj:
                temp_path = file_obj.name
            cv2.imwrite(temp_path, cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
            pipeline = self._get_vl()
            result = pipeline.predict(
                temp_path,
                use_layout_detection=False,
                prompt_label="ocr",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            tokens = self._extract_vl_tokens(result)
            normalized = normalize_target_text(tokens)
            return {
                "engine": "paddleocr_vl",
                "variant": "element",
                "text": normalized or "",
                "raw_text": " ".join(tokens).strip(),
                "tokens": tokens,
                "ocr_score": 1.0 if normalized else 0.0,
                "valid": normalized is not None,
            }
        except Exception as exc:
            return {
                "engine": "paddleocr_vl",
                "variant": "element",
                "text": "",
                "raw_text": "",
                "tokens": [],
                "ocr_score": 0.0,
                "valid": False,
                "error": str(exc),
            }
        finally:
            if temp_path:
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    def _merge_candidate_result(self, candidate: Dict[str, Any], ocr_result: Dict[str, Any], elapsed: float) -> Dict[str, Any]:
        item = dict(candidate)
        item.update({
            "text": ocr_result.get("text", ""),
            "raw_text": ocr_result.get("raw_text", ""),
            "texts": [ocr_result["text"]] if ocr_result.get("text") else [],
            "tokens": ocr_result.get("tokens", []),
            "scores": [ocr_result.get("ocr_score", 0.0)] if ocr_result.get("text") else [],
            "ocr_score": ocr_result.get("ocr_score", 0.0),
            "ocr_engine": ocr_result.get("engine", ""),
            "ocr_variant": ocr_result.get("variant", ""),
            "valid": bool(ocr_result.get("valid", False)),
            "elapsed_seconds": round(elapsed, 3),
        })
        if "error" in ocr_result:
            item["error"] = ocr_result["error"]
        return item

    def process_vl_first(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        return_all: bool = False,
        include_failures: bool = False,
        whole_fallbacks: int = 2,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        detections = self.detect(image, classes=classes, conf=conf, iou=iou)
        candidates = self.make_candidates(image, detections)

        attempts: List[Dict[str, Any]] = []
        valid_results: List[Dict[str, Any]] = []
        text_candidates = candidates[: self.max_text_candidates]
        remaining_vl = max(0, int(self.max_vl_candidates))

        for candidate in text_candidates:
            crop = self.crop_region(image, candidate["bbox"])
            if crop.size == 0:
                continue

            ocr_start = time.perf_counter()
            ocr_result = self._ocr_crop_light(crop)
            item = self._merge_candidate_result(candidate, ocr_result, time.perf_counter() - ocr_start)
            item["pipeline_mode"] = "logic2_yolo_crop_ppocr"
            attempts.append(item)
            if item["valid"]:
                valid_results.append(item)
                if not return_all:
                    break
                continue

            if remaining_vl <= 0:
                continue

            ocr_start = time.perf_counter()
            ocr_result = self._ocr_crop_vl(crop)
            item = self._merge_candidate_result(candidate, ocr_result, time.perf_counter() - ocr_start)
            item["pipeline_mode"] = "logic2_yolo_crop_vl_fallback"
            attempts.append(item)
            remaining_vl -= 1
            if item["valid"]:
                valid_results.append(item)
                if not return_all:
                    break

        max_whole = max(0, int(whole_fallbacks))
        if not valid_results and max_whole > 0:
            height, width = image.shape[:2]
            full_bbox = [0, 0, width, height]
            full_crop = self.crop_region(image, full_bbox, min_short_side=0, max_long_side=1800)
            for variant_name, variant in self._ocr_variants(full_crop)[:max_whole]:
                ocr_start = time.perf_counter()
                ocr_result = self._ocr_crop_vl(variant)
                ocr_result["variant"] = "whole_" + variant_name
                candidate = {
                    "bbox": full_bbox,
                    "raw_bbox": full_bbox[:],
                    "class_id": -1,
                    "class_name": "whole_image",
                    "confidence": 0.0,
                    "area": _box_area(full_bbox),
                    "source": "whole_image_vl_fallback",
                }
                item = self._merge_candidate_result(candidate, ocr_result, time.perf_counter() - ocr_start)
                item["pipeline_mode"] = "logic2_whole_image_vl_fallback"
                attempts.append(item)
                if item["valid"]:
                    valid_results.append(item)
                    if not return_all:
                        break

        valid_results.sort(
            key=lambda item: (
                float(item.get("ocr_score", 0.0)),
                float(item.get("confidence", 0.0)),
                1 if item.get("source") == "text_detector" else 0,
            ),
            reverse=True,
        )

        output_results = valid_results if return_all else valid_results[:1]
        if include_failures and not output_results:
            output_results = attempts

        best = output_results[0] if output_results and output_results[0].get("valid") else None
        return {
            "best": best,
            "results": output_results,
            "attempts": attempts,
            "detections": detections,
            "candidates": candidates,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "model_path": self.model_path,
            "mode": "logic2",
        }
    def process(
        self,
        image: np.ndarray,
        classes: Optional[List[int]] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        return_all: bool = False,
        include_failures: bool = False,
        enable_vl_fallback: Optional[bool] = None,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        detections = self.detect(image, classes=classes, conf=conf, iou=iou)
        candidates = self.make_candidates(image, detections)

        attempts: List[Dict[str, Any]] = []
        valid_results: List[Dict[str, Any]] = []
        light_candidates = candidates[: self.max_text_candidates]

        for candidate in light_candidates:
            crop = self.crop_region(image, candidate["bbox"])
            if crop.size == 0:
                continue
            ocr_start = time.perf_counter()
            ocr_result = self._ocr_crop_light(crop)
            item = self._merge_candidate_result(candidate, ocr_result, time.perf_counter() - ocr_start)
            attempts.append(item)
            if item["valid"]:
                valid_results.append(item)
                if not return_all:
                    break

        should_use_vl = self.enable_vl_fallback if enable_vl_fallback is None else bool(enable_vl_fallback)
        if should_use_vl and not valid_results:
            for candidate in candidates[: self.max_vl_candidates]:
                crop = self.crop_region(image, candidate["bbox"])
                if crop.size == 0:
                    continue
                ocr_start = time.perf_counter()
                ocr_result = self._ocr_crop_vl(crop)
                item = self._merge_candidate_result(candidate, ocr_result, time.perf_counter() - ocr_start)
                attempts.append(item)
                if item["valid"]:
                    valid_results.append(item)
                    if not return_all:
                        break

        valid_results.sort(
            key=lambda item: (
                float(item.get("ocr_score", 0.0)),
                float(item.get("confidence", 0.0)),
                1 if item.get("source") == "text_detector" else 0,
            ),
            reverse=True,
        )

        if return_all:
            output_results = valid_results
        else:
            output_results = valid_results[:1]

        if include_failures and not output_results:
            output_results = attempts

        best = output_results[0] if output_results and output_results[0].get("valid") else None
        return {
            "best": best,
            "results": output_results,
            "attempts": attempts,
            "detections": detections,
            "candidates": candidates,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
            "model_path": self.model_path,
        }

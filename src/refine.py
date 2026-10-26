"""把 DOM 提取的“行盒/词盒”收紧到真实墨迹范围。

Chrome 里 ``Range.getClientRects()`` 返回的是**行盒**（含 leading 与半行距），
比字形本身高出 30%~60%。直接当检测框会让模型学到“框里一半是空白”。
这里用干净页的墨迹（渲染结果本身）把每个框收紧到墨迹外接矩形：

  * 横向：去掉两端对齐产生的尾随空白；
  * 纵向：去掉升部/降部的行距空白（但保留字母上方的附加符号）。

收紧只裁剪“空白”，不会移动/缩放坐标，因此仍是零人工成本的真值。
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

INK_THRESHOLD = 0.25
PAD = 1.0


def load_ink(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (255.0 - img.astype(np.float32)) / 255.0


def _tight_box(ink: np.ndarray, box: list[float]) -> list[float] | None:
    h_img, w_img = ink.shape[:2]
    x, y, w, h = box
    x0 = int(max(0, math.floor(x - PAD)))
    y0 = int(max(0, math.floor(y - PAD)))
    x1 = int(min(w_img, math.ceil(x + w + PAD)))
    y1 = int(min(h_img, math.ceil(y + h + PAD)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    sub = ink[y0:y1, x0:x1] > INK_THRESHOLD
    cols = np.flatnonzero(sub.any(axis=0))
    rows = np.flatnonzero(sub.any(axis=1))
    if cols.size == 0 or rows.size == 0:
        return None
    nx0, nx1 = x0 + int(cols[0]), x0 + int(cols[-1]) + 1
    ny0, ny1 = y0 + int(rows[0]), y0 + int(rows[-1]) + 1
    return [float(nx0), float(ny0), float(nx1 - nx0), float(ny1 - ny0)]


def _apply(objects: list[dict], ink: np.ndarray, min_side: float) -> tuple[list[dict], int]:
    out: list[dict] = []
    dropped = 0
    for obj in objects:
        tight = _tight_box(ink, obj["box"])
        if tight is None or tight[2] < min_side or tight[3] < min_side:
            dropped += 1
            continue
        x, y, w, h = tight
        new = dict(obj)
        new["box"] = [round(v, 1) for v in tight]
        new["polygon"] = [
            [round(x, 1), round(y, 1)],
            [round(x + w, 1), round(y, 1)],
            [round(x + w, 1), round(y + h, 1)],
            [round(x, 1), round(y + h, 1)],
        ]
        out.append(new)
    return out, dropped


def refine_labels(clean_image: Path, labels: dict) -> tuple[dict, dict]:
    """返回 (收紧后的 labels, 统计)。同时把墨迹覆盖率记录进 stats。"""
    ink = load_ink(clean_image)
    lines, dropped_lines = _apply(labels["lines"], ink, min_side=6.0)
    words, dropped_words = _apply(labels["words"], ink, min_side=3.0)

    def coverage(objs: list[dict]) -> float:
        vals = []
        for obj in objs:
            x, y, w, h = obj["box"]
            x, y, w, h = int(x), int(y), int(w), int(h)
            sub = ink[max(0, y) : y + h, max(0, x) : x + w]
            if sub.size:
                vals.append(float((sub > INK_THRESHOLD).mean()))
        return round(float(np.mean(vals)), 4) if vals else 0.0

    stats = {
        "lines": len(lines),
        "words": len(words),
        "dropped_lines": dropped_lines,
        "dropped_words": dropped_words,
        "ink_coverage_lines": coverage(lines),
        "ink_coverage_words": coverage(words),
    }
    result = dict(labels)
    result["lines"] = lines
    result["words"] = words
    result["stats"] = {**labels.get("stats", {}), **stats}
    return result, stats

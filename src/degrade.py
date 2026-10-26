"""合成数据的物理退化：几何畸变 + 光影/纸张/透背 + 成像与传感器退化。

关键约束（AGENTS.md 第 4 节）：几何退化用一个解析映射 F: 源页坐标 -> 目标页坐标，
  * 图像侧：用 F 的数值逆映射做 ``cv2.remap``；
  * 标注侧：直接把框的角点与边中点送进 F，得到弯曲后的多边形；
两者共享同一个 F，因此标注与像素零偏差。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# 难度配比（plans.md 第五节）：60% 正常扫描 / 30% 复杂手机拍照 / 10% 极端挑战
PROFILE_WEIGHTS = {"scan": 0.60, "photo": 0.30, "extreme": 0.10}


def _np_rng(rng: random.Random) -> np.random.Generator:
    """从 python Random 派生 numpy Generator，保证种子可复现。"""
    return np.random.default_rng(rng.getrandbits(63))


def assign_profiles(n: int, rng: random.Random) -> list[str]:
    names = list(PROFILE_WEIGHTS)
    weights = [PROFILE_WEIGHTS[n_] for n_ in names]
    return rng.choices(names, weights=weights, k=n)


# ---------------------------------------------------------------- 几何


@dataclass
class Geometry:
    """解析映射：F = H ∘ C，其中 C 是中缝弯曲（位移场），H 是透视/旋转单应。"""

    H: np.ndarray
    spine_x: float
    k: float
    sigma: float
    sag: float

    def forward(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        u, v = p[:, 0], p[:, 1]
        d = u - self.spine_x
        g = np.exp(-((d / self.sigma) ** 2))
        u2 = u + self.k * d * g
        v2 = v + self.sag * g * np.sin(np.pi * np.clip(v, 0.0, 1.0))
        ones = np.ones((p.shape[0], 1))
        hom = np.hstack([np.stack([u2, v2], 1), ones]) @ self.H.T
        return hom[:, :2] / hom[:, 2:3]

    def inverse_map(self, out_w: int, out_h: int, step: int = 4) -> tuple[np.ndarray, np.ndarray]:
        xs = np.arange(0, out_w, step, dtype=np.float64)
        ys = np.arange(0, out_h, step, dtype=np.float64)
        gx, gy = np.meshgrid(xs, ys)
        q = np.stack([gx.ravel() / out_w, gy.ravel() / out_h], 1)
        p = q.copy()
        for _ in range(30):
            err = q - self.forward(p)
            p = p + 0.9 * err
            if float(np.abs(err).max()) < 1e-6:
                break
        map_x = (p[:, 0] * out_w).reshape(len(ys), len(xs)).astype(np.float32)
        map_y = (p[:, 1] * out_h).reshape(len(ys), len(xs)).astype(np.float32)
        if map_x.shape != (out_h, out_w):
            map_x = cv2.resize(map_x, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
            map_y = cv2.resize(map_y, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
        return map_x, map_y


def _make_geometry(rng: random.Random, profile: str) -> Geometry:
    if profile == "scan":
        margin = rng.uniform(0.004, 0.016)
        jitter = rng.uniform(0.002, 0.010)
        rot = rng.uniform(0.1, 0.9)
        k = rng.uniform(0.015, 0.070)
        sigma = rng.uniform(0.12, 0.26)
        sag = rng.uniform(0.0, 0.004)
    elif profile == "photo":
        margin = rng.uniform(0.012, 0.045)
        jitter = rng.uniform(0.008, 0.032)
        rot = rng.uniform(0.4, 2.4)
        k = rng.uniform(0.14, 0.32)
        sigma = rng.uniform(0.09, 0.20)
        sag = rng.uniform(0.003, 0.014)
    else:  # extreme
        margin = rng.uniform(0.010, 0.040)
        jitter = rng.uniform(0.006, 0.028)
        rot = rng.uniform(1.0, 3.0)
        k = rng.uniform(0.10, 0.30)
        sigma = rng.uniform(0.08, 0.22)
        sag = rng.uniform(0.002, 0.012)

    spine_x = rng.choice([0.0, 0.03, 0.5, 0.97, 1.0])
    if profile == "scan":
        spine_x = rng.choice([0.0, 0.5, 1.0])
    if spine_x >= 1.0:
        spine_x = 0.985

    src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    m = margin
    dst = np.array([[m, m], [1 - m, m], [1 - m, 1 - m], [m, 1 - m]], dtype=np.float64)
    dst = dst + np.array([[rng.gauss(0.0, jitter), rng.gauss(0.0, jitter)] for _ in range(4)])
    if profile != "scan" and rng.random() < 0.6:  # 单侧更大的透视（手持翻拍）
        dst[0, 1] += rng.uniform(0.0, 0.035)
        dst[1, 1] += rng.uniform(0.0, 0.035)
    theta = math.radians(rng.uniform(-rot, rot))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    dst = (dst - 0.5) @ np.array([[cos_t, -sin_t], [sin_t, cos_t]]) + 0.5
    dst = np.clip(dst, 0.002, 0.998).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    return Geometry(H=H, spine_x=spine_x, k=k, sigma=sigma, sag=sag)


# ---------------------------------------------------------------- 光度


def _synth_paper(h: int, w: int, rng: random.Random, profile: str) -> np.ndarray:
    """生成纸张底色（浮点 BGR，0~255）。"""
    if profile == "extreme":
        tone = rng.uniform(0.72, 0.90)
        cast = np.array([rng.uniform(0.58, 0.78), rng.uniform(0.70, 0.88), rng.uniform(0.84, 0.98)])
    elif profile == "photo":
        tone = rng.uniform(0.78, 0.95)
        cast = np.array([rng.uniform(0.72, 0.92), rng.uniform(0.85, 0.97), rng.uniform(0.90, 1.00)])
    else:
        tone = rng.uniform(0.88, 0.99)
        cast = np.array([rng.uniform(0.86, 0.98), rng.uniform(0.92, 0.99), rng.uniform(0.94, 1.00)])

    nprng = _np_rng(rng)
    paper = np.ones((h, w, 3), dtype=np.float32) * (tone * cast * 255.0)

    # 低频斑驳（纸浆不均 / 水渍）
    cell = 16
    small = nprng.normal(0.0, 1.0, (max(4, h // cell), max(4, w // cell))).astype(np.float32)
    small = cv2.GaussianBlur(small, (0, 0), rng.uniform(1.0, 3.0))
    lf = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    lf /= float(np.abs(lf).max()) + 1e-6
    paper *= (1.0 + rng.uniform(0.03, 0.08) * lf)[..., None]

    # 纸张纤维：横向拖影
    if rng.random() < 0.7:
        fib = nprng.normal(0.0, 1.0, (max(4, h // 8), max(4, w // 8))).astype(np.float32)
        fib = cv2.resize(fib, (w, h), interpolation=cv2.INTER_LINEAR)
        fib = cv2.blur(fib, (rng.randint(3, 9) | 1, 1))
        paper += (rng.uniform(1.5, 5.0) * fib)[..., None]

    paper += nprng.normal(0.0, rng.uniform(1.0, 3.0), (h, w, 1)).astype(np.float32)
    return paper


def _lighting_field(h: int, w: int, rng: random.Random, profile: str) -> np.ndarray:
    """光照图（乘性），含单侧台灯、暗角、中缝阴影。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / max(w - 1, 1)
    v = yy / max(h - 1, 1)

    if profile == "extreme":
        lo, hi = rng.uniform(0.55, 0.78), rng.uniform(0.92, 1.04)
        vig = rng.uniform(0.12, 0.28)
    elif profile == "photo":
        lo, hi = rng.uniform(0.52, 0.82), rng.uniform(0.98, 1.12)
        vig = rng.uniform(0.12, 0.32)
    else:
        lo, hi = rng.uniform(0.82, 0.95), rng.uniform(1.00, 1.06)
        vig = rng.uniform(0.03, 0.13)

    cx, cy = rng.uniform(0.15, 0.85), rng.uniform(0.10, 0.70)
    sx, sy = rng.uniform(0.35, 1.00), rng.uniform(0.30, 1.20)
    g = np.exp(-(((u - cx) ** 2) / (2 * sx * sx) + ((v - cy) ** 2) / (2 * sy * sy)))
    lum = lo + (hi - lo) * g

    r2 = (u - 0.5) ** 2 + (v - 0.5) ** 2
    lum = lum * (1.0 - vig * 2.2 * r2)
    return lum.astype(np.float32)


def _apply_shadows(img: np.ndarray, rng: random.Random, profile: str, spine_x: float) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / max(w - 1, 1)
    v = yy / max(h - 1, 1)

    # 中缝/装订线阴影
    if profile != "scan" or rng.random() < 0.5:
        width = rng.uniform(0.012, 0.06)
        amp = rng.uniform(0.10, 0.42) if profile != "scan" else rng.uniform(0.04, 0.15)
        band = np.exp(-(((u - spine_x) / width) ** 2))
        img = img * (1.0 - amp * band)[..., None]

    # 手指/手机投影的软边暗块
    if profile != "scan" and rng.random() < 0.45:
        cx, cy = rng.uniform(0.05, 0.95), rng.uniform(0.0, 1.0)
        rx, ry = rng.uniform(0.10, 0.35), rng.uniform(0.15, 0.45)
        ang = rng.uniform(0, math.pi)
        ca, sa = math.cos(ang), math.sin(ang)
        du = (u - cx) * ca + (v - cy) * sa
        dv = -(u - cx) * sa + (v - cy) * ca
        blob = np.exp(-((du / rx) ** 2 + (dv / ry) ** 2) * 2.5)
        blob = cv2.GaussianBlur(blob, (0, 0), rng.uniform(6, 18))
        img = img * (1.0 - rng.uniform(0.15, 0.45) * blob)[..., None]

    # 强反光（极端场景）
    if profile == "extreme" and rng.random() < 0.5:
        cx, cy = rng.uniform(0.1, 0.9), rng.uniform(0.05, 0.6)
        rx, ry = rng.uniform(0.08, 0.25), rng.uniform(0.05, 0.2)
        glare = np.exp(-(((u - cx) / rx) ** 2 + ((v - cy) / ry) ** 2) * 2.0)
        glare = cv2.GaussianBlur(glare, (0, 0), rng.uniform(8, 25))
        img = img + (255.0 - img) * (rng.uniform(0.12, 0.40) * glare)[..., None]
    return img


def _motion_kernel(rng: random.Random) -> np.ndarray:
    length = rng.randint(3, 11)
    angle = rng.uniform(0, math.pi)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), math.degrees(angle), 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def _sensor_noise(img: np.ndarray, rng: random.Random, profile: str) -> np.ndarray:
    h, w = img.shape[:2]
    nprng = _np_rng(rng)
    if profile == "extreme":
        sigma = rng.uniform(6.0, 14.0)
        dust = rng.randint(6, 40)
    elif profile == "photo":
        sigma = rng.uniform(3.0, 8.0)
        dust = rng.randint(2, 14)
    else:
        sigma = rng.uniform(1.0, 4.0)
        dust = rng.randint(0, 5)

    img = img + nprng.normal(0.0, sigma, (h, w, 1)).astype(np.float32)
    img = img + nprng.normal(0.0, sigma * 0.35, (h, w, 3)).astype(np.float32)

    for _ in range(dust):
        x, y = rng.randrange(w), rng.randrange(h)
        r = rng.randint(1, 3)
        val = rng.uniform(0, 90)
        cv2.circle(img, (x, y), r, (val, val, val), -1, lineType=cv2.LINE_AA)

    # 扫描仪/传感器拉丝线
    if profile != "scan" and rng.random() < 0.35:
        x = rng.randrange(w)
        img[:, max(0, x) : x + rng.randint(1, 2)] *= rng.uniform(0.90, 1.10)
    return img


# ---------------------------------------------------------------- 主流程


def degrade_page(
    clean_path: Path,
    partner_path: Path | None,
    profile: str,
    rng: random.Random,
) -> tuple[np.ndarray, Geometry]:
    clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
    if clean is None:
        raise FileNotFoundError(clean_path)
    h, w = clean.shape[:2]

    geo = _make_geometry(rng, profile)
    map_x, map_y = geo.inverse_map(w, h)

    # 1) 文本墨迹层（白底黑字 -> 墨量 0~1）
    ink_src = 1.0 - clean.astype(np.float32) / 255.0
    ink = cv2.remap(
        ink_src, map_x, map_y, interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
    )

    # 2) 透背/透墨：取另一页翻转、模糊、低透明度
    bleed = np.zeros_like(ink)
    if partner_path is not None and rng.random() < 0.85:
        partner = cv2.imread(str(partner_path), cv2.IMREAD_GRAYSCALE)
        if partner is not None:
            partner_ink = 1.0 - cv2.flip(partner, 1).astype(np.float32) / 255.0
            warped = cv2.remap(
                partner_ink, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
            )
            warped = cv2.GaussianBlur(warped, (0, 0), rng.uniform(2.0, 5.0))
            alpha = rng.uniform(0.05, 0.15)
            bleed = warped * alpha

    # 3) 二值化断笔（扫描仪自动锐化的副作用）
    if profile == "extreme" and rng.random() < 0.6:
        speck = (_np_rng(rng).random((h, w)).astype(np.float32) > rng.uniform(0.03, 0.12)).astype(np.float32)
        speck = cv2.GaussianBlur(speck, (0, 0), 0.8)
        ink = ink * speck

    # 4) 纸张 + 墨色合成
    paper = _synth_paper(h, w, rng, profile)
    if profile == "extreme":
        ink_strength = rng.uniform(0.38, 0.62)
        ink_rgb = np.array([rng.uniform(0.24, 0.40)] * 3, dtype=np.float32)
    elif profile == "photo":
        ink_strength = rng.uniform(0.55, 0.85)
        ink_rgb = np.array(
            [rng.uniform(0.10, 0.22), rng.uniform(0.10, 0.20), rng.uniform(0.08, 0.18)], dtype=np.float32
        )
    else:
        ink_strength = rng.uniform(0.75, 0.95)
        ink_rgb = np.array(
            [rng.uniform(0.06, 0.16), rng.uniform(0.06, 0.15), rng.uniform(0.05, 0.13)], dtype=np.float32
        )

    density = np.clip(ink * ink_strength, 0.0, 1.0)
    img = paper * (1.0 - density[..., None] * (1.0 - ink_rgb))
    img = img * (1.0 - np.clip(bleed, 0, 1)[..., None] * 0.55)

    # 5) 光照 / 阴影 / 反光
    lum = _lighting_field(h, w, rng, profile)
    img = img * lum[..., None]
    img = _apply_shadows(img, rng, profile, geo.spine_x)

    # 6) 对焦 / 抖动模糊
    if profile == "extreme":
        sigma = rng.uniform(0.4, 1.6)
    elif profile == "photo":
        sigma = rng.uniform(0.0, 1.6)
    else:
        sigma = rng.uniform(0.0, 0.8)
    if sigma > 0.15:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    if profile != "scan" and rng.random() < 0.25:
        img = cv2.filter2D(img, -1, _motion_kernel(rng))

    # 7) 传感器噪声 / 灰尘
    img = _sensor_noise(img, rng, profile)

    # 8) JPEG 压缩伪影
    if profile == "extreme":
        quality = rng.randint(35, 58)
    elif profile == "photo":
        quality = rng.randint(55, 80)
    else:
        quality = rng.randint(86, 96)
    ok, buf = cv2.imencode(".jpg", np.clip(img, 0, 255).astype(np.uint8), [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img, geo


def _poly_points(box: list[float], extra_midpoints: bool) -> list[tuple[float, float]]:
    x, y, w, h = box
    corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
    if not extra_midpoints:
        return corners
    mids = [
        ((corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2),
        ((corners[1][0] + corners[2][0]) / 2, (corners[1][1] + corners[2][1]) / 2),
        ((corners[2][0] + corners[3][0]) / 2, (corners[2][1] + corners[3][1]) / 2),
        ((corners[3][0] + corners[0][0]) / 2, (corners[3][1] + corners[0][1]) / 2),
    ]
    return [corners[0], mids[0], corners[1], mids[1], corners[2], mids[2], corners[3], mids[3]]


def transform_objects(labels: dict, geo: Geometry, src_w: int, src_h: int) -> tuple[list, list]:
    """把干净页上的 box/polygon 用同一个 F 变换到退化后的像素坐标。"""

    def convert(objects: list[dict], extra_midpoints: bool) -> list[dict]:
        out = []
        for obj in objects:
            pts = np.array(_poly_points(obj["box"], extra_midpoints), dtype=np.float64)
            norm = pts / np.array([src_w, src_h])
            warped = geo.forward(norm) * np.array([src_w, src_h])
            if not np.isfinite(warped).all():
                continue
            xs, ys = warped[:, 0], warped[:, 1]
            x0, y0, x1, y1 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(src_w - 1.0, x1), min(src_h - 1.0, y1)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if not (0 <= cx < src_w and 0 <= cy < src_h):
                continue
            new = dict(obj)
            new["box"] = [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)]
            new["polygon"] = [[round(px, 1), round(py, 1)] for px, py in warped.tolist()]
            out.append(new)
        return out

    return convert(labels["lines"], True), convert(labels["words"], False)


def draw_debug(img: np.ndarray, labels: dict, out_path: Path) -> None:
    canvas = img.copy()
    for obj in labels["lines"]:
        pts = np.array(obj["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 0, 255), 1, lineType=cv2.LINE_AA)
    for obj in labels["words"]:
        pts = np.array(obj["polygon"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [pts], True, (0, 200, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)

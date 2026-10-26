"""字体池：扫描 ``fonts/`` 并按 plans.md 的比例挑选字族。

约束（AGENTS.md 第 1 节）：字体只允许来自 ``fonts/``；必须覆盖 ar/fa/ps 扩展字母，
因此只使用已确认覆盖的 Google/SIL 字族。
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT / "fonts"

# 正文字族分组（Naskh 类占 70%~80%，Nastaliq 15%~20%）
GROUP_DIRS: dict[str, list[str]] = {
    "naskh": ["Noto_Naskh_Arabic", "Scheherazade_New", "Lateef", "Amiri", "Noto_Sans_Arabic"],
    "nastaliq": ["Noto_Nastaliq_Urdu"],
    "title": ["Aref_Ruqaa", "Noto_Kufi_Arabic"],
}

WEIGHT_TOKENS = {
    "thin": 100,
    "extralight": 200,
    "light": 300,
    "regular": 400,
    "medium": 500,
    "semibold": 600,
    "bold": 700,
    "extrabold": 800,
    "black": 900,
}

# 印刷书籍里罕见的压缩字宽变体一律排除
SKIP_TOKENS = ("condensed", "variablefont", "italic")


@dataclass(frozen=True)
class FontFace:
    css_family: str
    rel_path: str  # 相对仓库根，浏览器以 /fonts/... 访问
    weight: int
    group: str
    family: str  # 字族目录名

    def as_dict(self) -> dict:
        return {
            "css_family": self.css_family,
            "path": "/" + self.rel_path.replace("\\", "/"),
            "weight": self.weight,
            "group": self.group,
            "family": self.family,
        }


def _parse_weight(stem: str) -> int:
    low = stem.lower()
    best = 400
    for token, weight in WEIGHT_TOKENS.items():
        if token in low:
            best = weight
    return best


def _css_family(stem: str) -> str:
    return "SD-" + re.sub(r"[^A-Za-z0-9]+", "-", stem)


def load_faces(font_dir: Path = FONT_DIR) -> dict[str, list[FontFace]]:
    """返回 {group: [FontFace, ...]}，纯函数，无缓存副作用。"""
    groups: dict[str, list[FontFace]] = {}
    for group, dirs in GROUP_DIRS.items():
        faces: list[FontFace] = []
        for name in dirs:
            base = font_dir / name
            if not base.is_dir():
                continue
            for path in sorted(base.rglob("*.ttf")):
                stem = path.stem
                low = stem.lower()
                if any(tok in low for tok in SKIP_TOKENS):
                    continue
                faces.append(
                    FontFace(
                        css_family=_css_family(stem),
                        rel_path=str(path.relative_to(ROOT)),
                        weight=_parse_weight(stem),
                        group=group,
                        family=name,
                    )
                )
        groups[group] = faces
    return groups


def _pick(faces: list[FontFace], rng: random.Random, want: tuple[int, ...]) -> FontFace:
    """在期望字重区间内挑一个，缺失则退回全体。"""
    pool = [f for f in faces if f.weight in want] or faces
    # 先随机字族再随机字重，避免某个字族字重多就占更高比例
    families = sorted({f.family for f in pool})
    family = rng.choice(families)
    return rng.choice([f for f in pool if f.family == family])


def pick_body(rng: random.Random, faces: dict[str, list[FontFace]]) -> FontFace:
    """正文/主排版字体：Naskh 82%，Nastaliq 18%。"""
    use_nastaliq = rng.random() < 0.18 and faces.get("nastaliq")
    group = "nastaliq" if use_nastaliq else "naskh"
    return _pick(faces[group], rng, (400, 500))


def pick_title(rng: random.Random, faces: dict[str, list[FontFace]], body: FontFace) -> FontFace:
    """标题字体：45% 用 Ruq'ah/Kufic 标题体，55% 用正文字族的粗体。"""
    if rng.random() < 0.45 and faces.get("title"):
        return _pick(faces["title"], rng, (700, 800, 900))
    same_family = [f for f in faces[body.group] if f.family == body.family and f.weight >= 700]
    if same_family:
        return rng.choice(same_family)
    return _pick(faces[body.group], rng, (700, 800, 900))


def metrics(face: FontFace, rng: random.Random) -> dict:
    """按字形风格给出字号/行高（Nastaliq 需要更矮的行高比与更大字号）。"""
    if face.group == "nastaliq":
        return {
            "size": round(rng.uniform(19.0, 22.5), 1),
            "line_height": round(rng.uniform(2.70, 3.05), 2),
        }
    return {
        "size": round(rng.uniform(16.5, 20.5), 1),
        "line_height": round(rng.uniform(1.85, 2.25), 2),
    }

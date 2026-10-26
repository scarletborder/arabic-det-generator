"""文本池：按 ``语言-位置-编号.txt`` 约定装载 texts/ 下的语料。

命名与格式约束见 AGENTS.md 第 3 节：UTF-8 无 BOM、一行一条记录、位置代号固定。
"""

from __future__ import annotations

import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXTS_DIR = ROOT / "texts"

POSITIONS = ("title", "subtitle", "body", "note", "header", "page", "caption", "list")
NAME_RE = re.compile(r"^(?P<lang>[a-z]{2})-(?P<pos>[a-z]+)-(?P<idx>\d{3})\.txt$")
SENT_SPLIT_RE = re.compile(r"(?<=[.؟!?۔])\s+")

# 页码数字字形：ar 用东阿拉伯数字，fa/ps 用波斯变体
NUMERALS = {
    "ar": "٠١٢٣٤٥٦٧٨٩",
    "fa": "۰۱۲۳۴۵۶۷۸۹",
    "ps": "۰۱۲۳۴۵۶۷۸۹",
}

# plans.md 第 4 节：正文按 5%~10% 概率混入拉丁数字/英文词
LATIN_TERMS = ["1998", "2024", "UNESCO", "DNA", "km", "CD", "PDF", "GPS", "No. 12", "A4", "5%", "ISBN", "20°"]


class TextPool:
    def __init__(self, texts_dir: Path = TEXTS_DIR) -> None:
        self.texts_dir = Path(texts_dir)
        self.pools: dict[tuple[str, str], list[str]] = {}
        self.sources: dict[tuple[str, str], list[str]] = {}
        self._load()

    # ---------- 装载 ----------
    def _load(self) -> None:
        if not self.texts_dir.is_dir():
            return
        for path in sorted(self.texts_dir.rglob("*.txt")):
            m = NAME_RE.match(path.name)
            if not m:
                continue  # 命名不合规的文件不参与合成（包括 plans/AGENTS 等）
            lang, pos = m.group("lang"), m.group("pos")
            if pos not in POSITIONS:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            key = (lang, pos)
            self.pools.setdefault(key, []).extend(lines)
            self.sources.setdefault(key, []).append(path.name)

    # ---------- 查询 ----------
    def langs(self) -> list[str]:
        return sorted({lang for (lang, _pos) in self.pools})

    def has(self, lang: str, pos: str) -> bool:
        return bool(self.pools.get((lang, pos)))

    def count(self, lang: str, pos: str) -> int:
        return len(self.pools.get((lang, pos), []))

    def pick(self, rng: random.Random, lang: str, pos: str) -> str | None:
        lines = self.pools.get((lang, pos))
        if not lines:
            return None
        return rng.choice(lines)

    def pick_many(self, rng: random.Random, lang: str, pos: str, n: int) -> list[str]:
        lines = self.pools.get((lang, pos), [])
        if not lines:
            return []
        return [rng.choice(lines) for _ in range(n)]

    # ---------- 段落构造 ----------
    def paragraph(self, rng: random.Random, lang: str, min_words: int = 80, max_words: int = 300) -> str:
        """拼出 80~300 词、以真实句末标点收尾的段落。"""
        text = ""
        guard = 0
        while len(text.split()) < min_words and guard < 8:
            guard += 1
            nxt = self.pick(rng, lang, "body")
            if nxt is None:
                break
            text = f"{text} {nxt}".strip() if text else nxt
        words = text.split()
        if len(words) > max_words:
            sentences = [s for s in SENT_SPLIT_RE.split(text) if s]
            out: list[str] = []
            for sent in sentences:
                if len(" ".join(out).split()) + len(sent.split()) > max_words and out:
                    break
                out.append(sent)
            text = " ".join(out) if out else " ".join(words[:max_words])
        return text.strip()

    def short_text(self, rng: random.Random, lang: str, pos: str, lo: int, hi: int) -> str | None:
        """从指定位置池取一条长度合适的记录。"""
        lines = self.pools.get((lang, pos), [])
        cands = [ln for ln in lines if lo <= len(ln.split()) <= hi]
        if not cands:
            cands = lines
        return rng.choice(cands) if cands else None


def inject_latin(rng: random.Random, text: str, prob: float = 0.075) -> str:
    """按 5%~10% 概率插入拉丁数字/英文词，避免模型只认纯阿文字形。"""
    if rng.random() > prob:
        return text
    words = text.split()
    if len(words) < 12:
        return text
    idx = rng.randrange(4, len(words) - 2)
    term = rng.choice(LATIN_TERMS)
    if rng.random() < 0.5:
        words[idx] = f"{words[idx]} ({term})"
    else:
        words.insert(idx, term)
    return " ".join(words)


def page_number(rng: random.Random, lang: str, number: int) -> str:
    table = NUMERALS.get(lang, NUMERALS["ar"])
    digits = "".join(table[int(d)] for d in str(number))
    return f"- {digits} -" if rng.random() < 0.4 else digits

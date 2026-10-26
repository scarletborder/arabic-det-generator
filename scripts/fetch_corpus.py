"""抓取阿拉伯语系文本语料，按 AGENTS.md 的命名规则落盘。

来源（均经 HTTP 代理，代理从环境变量读取）：
  - 维基百科  : huggingface datasets-server -> wikimedia/wikipedia 20231101.<lang>   -> texts/wiki/
  - 网页语料  : huggingface datasets-server -> allenai/c4 <lang>                      -> texts/corpus/

产出文件命名：``{lang}-{pos}-{NNN}.txt``（UTF-8 无 BOM，一行一条记录）。

用法::

    export HTTPS_PROXY=http://127.0.0.1:7890
    python scripts/fetch_corpus.py --langs ar,fa,ps --articles 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://datasets-server.huggingface.co/rows"

LANGS = ("ar", "fa", "ps")
C4_INVALID = {"ar": "ar", "fa": "fa", "ps": "ps"}

# 阿拉伯语系常用标点（用于过滤无标点的低质段落）
AR_PUNCT = "،؛؟.,:!؟؛"
LATIN_RE = re.compile(r"[A-Za-z]")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
REF_RE = re.compile(r"\[\d+\]|\[[^\]]{0,40}\]")
CTRL_RE = re.compile(r"[\u0000-\u0008\u000b-\u001f\u007f\u200b\u200e\u200f\u202a-\u202e\ufeff]")
SPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a]+")
HEADING_RE = re.compile(r"^\s*=+\s*(.+?)\s*=+\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[.؟!?۔])\s+")
WORD_RE = re.compile(r"\S+")


def proxy_opener() -> urllib.request.OpenerDirector:
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "http://127.0.0.1:7890"
    handler = urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    return urllib.request.build_opener(handler)


def fetch_rows(dataset: str, config: str, offset: int, length: int, opener) -> list[dict]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": "train", "offset": offset, "length": length}
    )
    url = f"{API}?{query}"
    for attempt in range(5):
        try:
            with opener.open(url, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            time.sleep(0.4)  # 尊重 datasets-server 的限流
            return [row["row"] for row in payload.get("rows", [])]
        except Exception as exc:  # noqa: BLE001 - 网络/限流问题退避重试
            wait = 6 * (attempt + 1)
            print(f"  ! {dataset}/{config} offset={offset} attempt={attempt + 1}: {exc} (sleep {wait}s)")
            time.sleep(wait)
    return []


def clean_line(text: str) -> str:
    text = CTRL_RE.sub("", text)
    text = URL_RE.sub(" ", text)
    text = REF_RE.sub(" ", text)
    text = text.replace("\u0640", "")  # tatweel 会拉长字形，真实书籍少见
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def is_valid_paragraph(text: str) -> bool:
    words = text.split()
    if len(words) < 12:
        return False
    if len(LATIN_RE.findall(text)) > 0.2 * max(len(text), 1):
        return False
    if not any(ch in AR_PUNCT for ch in text):
        return False
    if sum(ch.isalpha() for ch in text) < 0.55 * len(text):
        return False
    return True


def is_valid_heading(text: str) -> bool:
    words = text.split()
    if not 1 <= len(words) <= 8:
        return False
    if URL_RE.search(text) or REF_RE.search(text):
        return False
    return sum(ch.isalpha() for ch in text) >= 0.7 * max(len(text), 1)


def split_article(raw: str) -> tuple[list[str], list[str]]:
    """返回 (段落列表, 小节标题列表)。"""
    paras: list[str] = []
    headings: list[str] = []
    buf: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        m = HEADING_RE.match(line)
        if m:
            if buf:
                paras.append(clean_line(" ".join(buf)))
                buf = []
            head = clean_line(m.group(1))
            if is_valid_heading(head):
                headings.append(head)
            continue
        if not line or line.startswith(("{{", "|", "[[File:", "[[ملف:", "[[پرونده:", "[[دوتنه:", "!", "#", "*")):
            if buf:
                paras.append(clean_line(" ".join(buf)))
                buf = []
            continue
        buf.append(line)
    if buf:
        paras.append(clean_line(" ".join(buf)))
    return [p for p in paras if is_valid_paragraph(p)], headings


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def write_pool(path: Path, lang: str, pos: str, lines: list[str], per_file: int = 300) -> int:
    path.mkdir(parents=True, exist_ok=True)
    for old in sorted(path.glob(f"{lang}-{pos}-*.txt")):
        old.unlink()
    written = 0
    for idx in range(0, len(lines), per_file):
        chunk = lines[idx : idx + per_file]
        name = path / f"{lang}-{pos}-{idx // per_file + 1:03d}.txt"
        name.write_text("\n".join(chunk) + "\n", encoding="utf-8", newline="\n")
        written += 1
    return written


def build_lang(lang: str, articles: int, texts_dir: Path, opener, seed_tag: str) -> dict:
    stats = {"paragraphs": 0, "titles": 0, "headings": 0, "corpus_paras": 0}
    bodies: list[str] = []
    titles: list[str] = []
    headings: list[str] = []
    seen: set[str] = set()

    def add_para(p: str) -> None:
        key = hashlib.md5(p[:120].encode()).hexdigest()
        if key in seen:
            return
        seen.add(key)
        bodies.append(p)

    # ---- 维基百科
    batch = 50
    fetched = 0
    offset = 0
    while fetched < articles and offset < articles * 4:
        rows = fetch_rows("wikimedia/wikipedia", f"20231101.{lang}", offset, batch, opener)
        if not rows:
            break
        offset += batch
        fetched += len(rows)
        for row in rows:
            title = clean_line(str(row.get("title", "")))
            if 1 <= len(title.split()) <= 8:
                titles.append(title)
            paras, heads = split_article(str(row.get("text", "")))
            headings.extend(heads)
            for p in paras:
                add_para(p)

    # ---- 网页语料（C4，限流较严，请求量减半）
    corpus_paras: list[str] = []
    fetched = 0
    offset = 0
    corpus_target = max(60, articles // 3)
    while fetched < corpus_target and offset < corpus_target * 4:
        rows = fetch_rows("allenai/c4", C4_INVALID[lang], offset, 100, opener)
        if not rows:
            break
        offset += batch
        fetched += len(rows)
        for row in rows:
            for p in split_article(str(row.get("text", "")))[0]:
                corpus_paras.append(p)

    # ---- 组装各“位置”池
    body_lines = [p for p in bodies if 20 <= len(p.split()) <= 320]
    corpus_lines = [p for p in corpus_paras if 20 <= len(p.split()) <= 320]

    # 尾注/图注/列表：由短句合成，长度贴近真实书籍
    notes: list[str] = []
    captions: list[str] = []
    lists: list[str] = []
    for src in bodies:
        for sent in split_sentences(src):
            n = len(sent.split())
            if 8 <= n <= 40 and any(ch in AR_PUNCT for ch in sent):
                notes.append(f"{len(notes) + 1}. {sent}")
            if 4 <= n <= 12:
                captions.append(sent.rstrip(".؟!?۔ "))
            if 5 <= n <= 18:
                lists.append(sent)

    uniq = lambda seq: list(dict.fromkeys(seq))  # noqa: E731

    stats["paragraphs"] = write_pool(texts_dir / "wiki", lang, "body", uniq(body_lines))
    stats["titles"] = write_pool(texts_dir / "wiki", lang, "title", uniq(titles))
    stats["headings"] = write_pool(texts_dir / "wiki", lang, "subtitle", uniq(headings))
    write_pool(texts_dir / "wiki", lang, "header", uniq(titles))
    write_pool(texts_dir / "wiki", lang, "note", uniq(notes))
    write_pool(texts_dir / "wiki", lang, "caption", uniq(captions))
    write_pool(texts_dir / "wiki", lang, "list", uniq(lists))
    stats["corpus_paras"] = write_pool(texts_dir / "corpus", lang, "body", uniq(corpus_lines))
    write_pool(texts_dir / "corpus", lang, "title", uniq(titles[: max(1, len(titles) // 3)]))
    stats["lang"] = lang
    stats["tag"] = seed_tag
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="抓取 ar/fa/ps 语料到 texts/{wiki,corpus}")
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--articles", type=int, default=400)
    ap.add_argument("--texts-dir", default=str(ROOT / "texts"))
    args = ap.parse_args()

    opener = proxy_opener()
    texts_dir = Path(args.texts_dir)
    summary = {}
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        if lang not in LANGS:
            print(f"skip unsupported lang {lang}")
            continue
        print(f"[{lang}] fetching ...")
        summary[lang] = build_lang(lang, args.articles, texts_dir, opener, "hf")
        print(f"[{lang}] {summary[lang]}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

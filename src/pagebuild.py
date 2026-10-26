"""页面版式合成：把文本池里的“位置”片段随机组合成符合常识的书页，并生成 HTML。

设计要点（AGENTS.md 第 4 节）：
  * 页面显式 ``dir="rtl"`` + ``lang``，正文 ``text-align: justify`` + 首行缩进；
  * 每个可视文本元素带 ``data-lt``（line type），供渲染阶段从 DOM 直接提取真值；
  * 结构顺序遵循真实书籍：页眉 → 标题 → 小标题 → 正文 → 图/表 → 尾注 → 页码。
"""

from __future__ import annotations

import random
from pathlib import Path
from string import Template

from . import fontpool, textpool

ROOT = Path(__file__).resolve().parent.parent
ILLUS_DIR = ROOT / "texts" / "illustraion"

ARABIC_INDENT = "2em"


def list_illustrations() -> list[str]:
    if not ILLUS_DIR.is_dir():
        return []
    return sorted(p for p in ILLUS_DIR.glob("*.png") if p.is_file())


def _est_chars(text: str) -> int:
    return len(text)


def build_spec(
    rng: random.Random,
    pool: textpool.TextPool,
    faces: dict[str, list[fontpool.FontFace]],
    lang: str,
    page_id: str,
    width: int = 794,
    height: int = 1123,
    illus_prob: float = 0.05,
    page_no: int = 1,
) -> dict:
    body_face = fontpool.pick_body(rng, faces)
    title_face = fontpool.pick_title(rng, faces, body_face)
    met = fontpool.metrics(body_face, rng)

    margin = {
        "top": rng.randint(34, 68),
        "bottom": rng.randint(46, 84),
        "left": rng.randint(42, 78),
        "right": rng.randint(42, 78),
    }
    two_col = rng.random() < 0.15
    columns = 2 if two_col else 1

    blocks: list[dict] = []

    # ---- 页眉（书名/章节名 + 横线）
    if rng.random() < 0.82:
        head = pool.short_text(rng, lang, "header", 1, 9) or pool.short_text(rng, lang, "title", 1, 6)
        if head:
            blocks.append({"type": "header", "text": head, "align": rng.choice(["start", "center"])})

    # ---- 标题 / 小标题
    want_title = rng.random() < 0.8
    if want_title:
        title = pool.short_text(rng, lang, "title", 1, 8) or pool.short_text(rng, lang, "subtitle", 1, 8)
        if title:
            blocks.append(
                {"type": "title", "text": title, "align": rng.choice(["center", "start", "center"])}
            )
    if rng.random() < 0.55:
        sub = pool.short_text(rng, lang, "subtitle", 1, 9)
        if sub:
            blocks.append({"type": "subtitle", "text": sub})

    # ---- 正文：按容量估算填充，渲染端会做溢出裁剪
    content_w = width - margin["left"] - margin["right"]
    if columns == 2:
        content_w = (content_w - 24) / 2
    chars_per_line = max(20.0, content_w / (met["size"] * 0.52))
    used_chars = 0
    for blk in blocks:
        used_chars += _est_chars(str(blk.get("text", ""))) * 1.0
        if blk["type"] == "title":
            used_chars += chars_per_line * 0.9

    content_h = height - margin["top"] - margin["bottom"] - 46
    capacity_lines = content_h / (met["size"] * met["line_height"])
    capacity_chars = chars_per_line * capacity_lines * (columns if columns == 2 else 1)

    if rng.random() < 0.22 and pool.has(lang, "list"):
        items = pool.pick_many(rng, lang, "list", rng.randint(3, 6))
        if items:
            blocks.append({"type": "list", "items": items})
            used_chars += sum(_est_chars(i) for i in items) * 1.15

    # ---- 插图（小概率）：先决定是否放图，占位约 14 行，插在正文 1~3 段之间
    figure_block: dict | None = None
    figures = list_illustrations()
    if figures and rng.random() < illus_prob:
        img = rng.choice(figures)
        caption = (
            pool.short_text(rng, lang, "caption", 3, 14)
            or pool.short_text(rng, lang, "subtitle", 1, 9)
            or ""
        )
        fig_h = int(height * rng.uniform(0.16, 0.30))
        figure_block = {
            "type": "figure",
            "src": "/" + str(img.relative_to(ROOT)).replace("\\", "/"),
            "caption": caption,
            "height": fig_h,
        }
        used_chars += chars_per_line * (fig_h / (met["size"] * met["line_height"]) + 1.6)

    # 多生成 55% 的“备选段落”（渲染端先隐藏，逐段显示直到刚好填满页面）：
    # 版面容量估算有误差，靠浏览器实测比 Python 估算更可靠。
    target_chars = capacity_chars * 0.95
    max_chars = capacity_chars * 1.55
    body_blocks: list[dict] = []
    body_count = 0
    while used_chars < max_chars and body_count < 40:
        para = pool.paragraph(rng, lang, min_words=70, max_words=260)
        if not para:
            break
        para = textpool.inject_latin(rng, para, prob=0.08)
        # 所有正文段落先隐藏，由渲染端逐段显示 + 断句二分，保证页面刚好填满
        body_blocks.append({"type": "body", "text": para, "extra": True})
        used_chars += _est_chars(para) * 1.06
        body_count += 1

    if figure_block is not None and body_blocks:
        body_blocks.insert(rng.randint(1, min(3, len(body_blocks))), figure_block)
    blocks.extend(body_blocks)

    # ---- 尾注
    if rng.random() < 0.42:
        note = pool.short_text(rng, lang, "note", 8, 55)
        if note:
            blocks.append({"type": "note", "text": note})

    # ---- 页码
    blocks.append({"type": "page", "text": textpool.page_number(rng, lang, page_no)})

    return {
        "page_id": page_id,
        "lang": lang,
        "dir": "rtl",
        "width": width,
        "height": height,
        "columns": columns,
        "margin": margin,
        "font_size": met["size"],
        "line_height": met["line_height"],
        "word_spacing": round(rng.uniform(-0.4, 0.6), 2),
        "font_body": body_face.as_dict(),
        "font_title": title_face.as_dict(),
        "blocks": blocks,
    }


CSS_TEMPLATE = Template("""
$faces
*{margin:0;padding:0;box-sizing:border-box;}
html,body{background:#fff;}
.page{
  position:relative;width:${width}px;height:${height}px;
  padding:${pad_top}px ${pad_right}px ${pad_bottom}px ${pad_left}px;
  background:#ffffff;color:#000;overflow:hidden;
  direction:rtl;text-align:right;
  font-family:'${body_family}',serif;font-weight:${body_weight};
  font-size:${font_size}px;line-height:${line_height};
  word-spacing:${word_spacing}px;
  font-kerning:normal;text-rendering:optimizeLegibility;
  -webkit-font-smoothing:antialiased;
}
.hd{
  font-size:0.86em;line-height:1.6;padding-bottom:5px;margin-bottom:0.7em;
  border-bottom:1px solid #000;letter-spacing:0.02em;
}
.hd.center{text-align:center;}
.hd.start{text-align:right;}
h1.title{
  font-family:'${title_family}',serif;font-weight:${title_weight};
  font-size:${title_size}px;line-height:1.55;
  margin:0.15em 0 0.5em;letter-spacing:-0.005em;
}
h1.title.start{text-align:right;}
h1.title.center{text-align:center;}
h2.subtitle{
  font-family:'${title_family}',serif;font-weight:${subtitle_weight};
  font-size:${subtitle_size}px;line-height:1.6;margin:0.5em 0 0.45em;
}
.cols{column-count:${columns};column-gap:24px;text-align:justify;}
p.body{text-align:justify;text-indent:${indent};margin:0 0 0.5em;break-inside:avoid-column;}
ol.list{margin:0.2em 0 0.8em;padding-inline-start:1.6em;text-align:justify;}
ol.list li{margin-bottom:0.25em;}
.ft{margin-top:0.9em;padding-top:0.45em;border-top:1px solid #000;font-size:0.78em;line-height:1.75;}
figure.fig{margin:0.5em 0 0.8em;text-align:center;break-inside:avoid-column;}
figure.fig img{display:inline-block;max-width:100%;filter:grayscale(1) contrast(1.05);}
figure.fig .cap{font-size:0.78em;line-height:1.7;margin-top:0.3em;text-align:center;}
.pn{
  position:absolute;left:0;right:0;bottom:${page_bottom}px;
  text-align:center;font-size:0.9em;letter-spacing:0.06em;
}
""")


def _font_faces(spec: dict) -> str:
    rules = []
    seen = set()
    for key in ("font_body", "font_title"):
        face = spec[key]
        token = (face["css_family"], face["path"])
        if token in seen:
            continue
        seen.add(token)
        rules.append(
            "@font-face{"
            f"font-family:'{face['css_family']}';"
            f"src:url('{face['path']}') format('truetype');"
            f"font-weight:{face['weight']};font-style:normal;font-display:block;"
            "}"
        )
    return "\n".join(rules)


def to_html(spec: dict) -> str:
    m = spec["margin"]
    body_family = spec["font_body"]["css_family"]
    title_family = spec["font_title"]["css_family"]
    title_weight = spec["font_title"]["weight"]
    body_weight = spec["font_body"]["weight"]

    css = CSS_TEMPLATE.substitute(
        faces=_font_faces(spec),
        width=spec["width"],
        height=spec["height"],
        pad_top=m["top"],
        pad_right=m["right"],
        pad_bottom=m["bottom"],
        pad_left=m["left"],
        body_family=body_family,
        body_weight=body_weight,
        font_size=spec["font_size"],
        line_height=spec["line_height"],
        word_spacing=spec["word_spacing"],
        title_family=title_family,
        title_weight=title_weight,
        title_size=round(spec["font_size"] * 1.95, 1),
        subtitle_weight=max(600, title_weight - 100),
        subtitle_size=round(spec["font_size"] * 1.28, 1),
        columns=spec["columns"],
        indent=ARABIC_INDENT,
        page_bottom=max(16, m["bottom"] // 3),
    )

    html: list[str] = []
    inline: list[str] = []
    for blk in spec["blocks"]:
        kind = blk["type"]
        if kind == "header":
            cls = "hd " + blk.get("align", "start")
            inline.append(f'<div class="{cls}" data-lt="header">{blk["text"]}</div>')
        elif kind == "title":
            cls = "title " + blk.get("align", "center")
            inline.append(f'<h1 class="{cls}" data-lt="title">{blk["text"]}</h1>')
        elif kind == "subtitle":
            inline.append(f'<h2 class="subtitle" data-lt="subtitle">{blk["text"]}</h2>')
        elif kind == "body":
            attrs = ' data-shrink="1"'
            if blk.get("extra"):
                attrs += ' data-extra="1" style="display:none"'
            inline.append(f'<p class="body" data-lt="body"{attrs}>{blk["text"]}</p>')
        elif kind == "list":
            items = "".join(f'<li data-lt="list" data-shrink="1">{t}</li>' for t in blk["items"])
            inline.append(f'<ol class="list" data-lt-group="list">{items}</ol>')
        elif kind == "figure":
            img = f'<img src="{blk["src"]}" style="height:{blk["height"]}px;width:auto;max-width:100%;">'
            cap = f'<div class="cap" data-lt="caption">{blk["caption"]}</div>' if blk["caption"] else ""
            inline.append(f'<figure class="fig">{img}{cap}</figure>')
        elif kind == "note":
            inline.append(f'<div class="ft" data-lt="note">{blk["text"]}</div>')
        elif kind == "page":
            html.append(f'<div class="pn" data-lt="page">{blk["text"]}</div>')

    body_html = "\n".join(inline)
    html.insert(0, f'<div class="cols">\n{body_html}\n</div>')
    inner = "".join(html)

    return (
        "<!DOCTYPE html>\n"
        f'<html dir="rtl" lang="{spec["lang"]}"><head><meta charset="utf-8">'
        f"<style>{css}</style></head>"
        f'<body><div class="page">{inner}</div></body></html>'
    )

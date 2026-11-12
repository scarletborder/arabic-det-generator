"""端到端流水线。"""
import argparse, json, random, shutil, subprocess, sys, time

from tqdm import tqdm
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import degrade, fontpool, pagebuild, refine, textpool

def _reset(p):
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)


def stage_pages(args, pool, faces, langs):
    pages_dir = Path(args.out) / "pages"
    _reset(pages_dir)
    rng = random.Random(args.seed)
    plan = [langs[i % len(langs)] for i in range(args.count)]
    rng.shuffle(plan)
    profiles = degrade.assign_profiles(args.count, rng)
    pages = []
    pages_dir.mkdir(parents=True, exist_ok=True)
    bar = tqdm(zip(plan, profiles), total=len(plan), desc="版式", unit="page", dynamic_ncols=True)
    for idx, (lang, prof) in enumerate(bar):
        bar.set_postfix_str(f"{lang}/{prof}")
        prng = random.Random(rng.getrandbits(32))
        pid = f"{lang}-{args.seed:04d}-{idx:06d}"
        spec = pagebuild.build_spec(prng, pool, faces, lang, pid,
            width=args.width, height=args.height, scale=args.scale,
            illus_prob=args.illust_prob,
            page_no=prng.randint(1, 480))
        d = pages_dir / pid
        d.mkdir(parents=True, exist_ok=True)
        (d / "index.html").write_text(pagebuild.to_html(spec), encoding="utf-8")
        meta = dict(spec)
        meta.pop("blocks", None)
        meta["profile"] = prof
        meta["seed"] = prng.getrandbits(32)
        (d / "meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        pages.append({"page_id": pid, "lang": lang, "dir": spec["dir"],
            "width": spec["width"], "height": spec["height"],
            "columns": spec["columns"], "font_size": spec["font_size"],
            "line_height": spec["line_height"], "margin": spec["margin"],
            "fonts": {"body": spec["font_body"], "title": spec["font_title"]},
            "profile": prof, "seed": meta["seed"]})
    man = {"seed": args.seed, "dpr": args.dpr, "pages": pages}
    (pages_dir / "manifest.json").write_text(json.dumps(man, ensure_ascii=False), encoding="utf-8")
    print(f"[pages] {len(pages)} -> {pages_dir}")
    return pages


def stage_render(args):
    _reset(Path(args.out) / "clean")
    o = Path(args.out)
    cmd = ["node", str(ROOT / "scripts" / "render.mjs")]
    cmd += ["--pages", str(o / "pages"), "--clean", str(o / "clean"), "--dpr", str(args.dpr)]
    if args.chrome: cmd += ["--chrome", args.chrome]
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _collect_dets(d: dict) -> list:
    """把一页的 line 标注整理成 PaddleOCR det 的 detection 列表（与 --paddle 同构）。"""
    dets = []
    for ln in d["lines"]:
        text = (ln.get("text") or "").strip()
        pts = [[round(float(px), 1), round(float(py), 1)] for px, py in ln["polygon"]]
        if len(pts) < 4:
            continue
        dets.append({"transcription": text if text else "###", "points": pts})
    return dets


def stage_paddle(args, pages):
    o = Path(args.out)
    src = o / "final" / "labels"
    dst = o  # PaddleOCR: data_dir = outputs，images/ 与 train_list.txt 同层
    _reset(dst / "images")
    lines = []
    skipped = 0
    for f in tqdm(sorted(src.glob("*.json")), desc="Paddle 导出", unit="img", dynamic_ncols=True):
        d = json.loads(f.read_text(encoding="utf-8"))
        pid = d["page_id"]
        img = o / "final" / "images" / f"{pid}.jpg"
        if not img.exists():
            skipped += 1
            continue
        shutil.copy2(img, dst / "images" / img.name)
        dets = _collect_dets(d)
        if not dets:
            skipped += 1
            continue
        lines.append(f"images/{img.name}\t{json.dumps(dets, ensure_ascii=False)}")

    (dst / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    # --gtsha256 仍依赖 final/ 里的原始产物，故此时保留 final 目录
    if not args.gtsha256:
        shutil.rmtree(o / "final", ignore_errors=True)  # 只保留 Paddle 格式，不保留原生 labels
    print(f"[paddle] {len(lines)} images -> {dst} (skipped {skipped})")
    return dst


def stage_gtsha256(args, pages):
    """在 final/ 下额外生成一个 sha256 内容寻址的标注清单（格式同 --paddle）。

    每行：``sha256:<image_sha256>\t[{"transcription", "points"}]``，
    文件写在 ``final/train_list_sha256.txt``。
    """
    import hashlib

    o = Path(args.out)
    src = o / "final" / "labels"
    img_dir = o / "final" / "images"
    dst = o / "final"
    lines = []
    skipped = 0
    for f in tqdm(sorted(src.glob("*.json")), desc="gtsha256 导出", unit="img", dynamic_ncols=True):
        d = json.loads(f.read_text(encoding="utf-8"))
        pid = d["page_id"]
        img = img_dir / f"{pid}.jpg"
        if not img.exists():
            skipped += 1
            continue
        h = hashlib.sha256(img.read_bytes()).hexdigest()
        dets = _collect_dets(d)
        if not dets:
            skipped += 1
            continue
        lines.append(f"sha256:{h}\t{json.dumps(dets, ensure_ascii=False)}")

    (dst / "train_list_sha256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"[gtsha256] {len(lines)} images -> {dst / 'train_list_sha256.txt'} (skipped {skipped})")
    return dst


def stage_degrade(args, pages):
    o = Path(args.out)
    clean = o / "clean"
    fi, fl = o / "final" / "images", o / "final" / "labels"
    _reset(fi)
    _reset(fl)
    if args.debug:
        _reset(o / "debug")

    import cv2

    rng = random.Random(args.seed * 104729 + 17)
    stats = {"pages": 0, "lines": 0, "words": 0, "dropped_pages": 0}
    cov_l = cov_w = 0.0
    pc = {}
    bar = tqdm(pages, desc="退化", unit="page", dynamic_ncols=True)
    for page in bar:
        bar.set_postfix_str(page["profile"])
        pid = page["page_id"]
        cimg = clean / "images" / f"{pid}.png"
        lpath = clean / "labels" / f"{pid}.json"
        if not cimg.exists() or not lpath.exists():
            stats["dropped_pages"] += 1
            continue

        labels = json.loads(lpath.read_text(encoding="utf-8"))
        labels, rs = refine.refine_labels(cimg, labels)
        lpath.write_text(json.dumps(labels, ensure_ascii=False), encoding="utf-8", newline="\n")
        cov_l += rs["ink_coverage_lines"]
        cov_w += rs["ink_coverage_words"]

        partners = [p for p in pages if p["page_id"] != pid and p["lang"] == page["lang"]]
        partner = clean / "images" / f"{rng.choice(partners)['page_id']}.png" if partners else None
        prng = random.Random(page["seed"] ^ 0x5F3759DF)
        img, geo = degrade.degrade_page(cimg, partner, page["profile"], prng)

        sw, sh = labels["width"], labels["height"]
        tlines, twords = degrade.transform_objects(labels, geo, sw, sh)
        cv2.imwrite(str(fi / f"{pid}.jpg"), img)
        out = {
            "image": f"final/images/{pid}.jpg",
            "clean_image": labels["image"],
            "page_id": pid,
            "lang": page["lang"],
            "dir": labels.get("dir", "rtl"),
            "profile": page["profile"],
            "width": sw,
            "height": sh,
            "dpr": labels.get("dpr", args.dpr),
            "fonts": labels.get("fonts"),
            "geometry": {
                "spine_x": round(float(geo.spine_x), 4),
                "bend_k": round(float(geo.k), 4),
                "bend_sigma": round(float(geo.sigma), 4),
                "sag": round(float(geo.sag), 4),
            },
            "stats": {"lines": len(tlines), "words": len(twords)},
            "lines": tlines,
            "words": twords,
        }
        (fl / f"{pid}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8", newline="\n")
        if args.debug:
            degrade.draw_debug(img, out, o / "debug" / f"{pid}.jpg")
        stats["pages"] += 1
        stats["lines"] += len(tlines)
        stats["words"] += len(twords)
        pc[page["profile"]] = pc.get(page["profile"], 0) + 1

    n = max(1, stats["pages"])
    stats["profiles"] = pc
    stats["ink_cov_lines"] = round(cov_l / n, 4)
    stats["ink_cov_words"] = round(cov_w / n, 4)
    print(f"[degrade] {json.dumps(stats, ensure_ascii=False)}")
    return stats


def main():
    ap = argparse.ArgumentParser(description="合成 ar/fa/ps 书籍扫描/拍照检测数据集")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--langs", default="ar,fa,ps")
    ap.add_argument("--out", default=str(ROOT / "outputs"))
    ap.add_argument("--dpr", type=float, default=1.0, help="截图超采样倍率（默认 1.0）")
    ap.add_argument("--res", default="640x800", help="输出页分辨率 WxH，如 640x800 / 736x736 / 794x1123")
    ap.add_argument("--width", type=int, default=0, help="覆盖 --res 的宽（兼容旧用法）")
    ap.add_argument("--height", type=int, default=0, help="覆盖 --res 的高（兼容旧用法）")
    ap.add_argument("--illust-prob", type=float, default=0.05)
    ap.add_argument("--stage", choices=["all", "pages", "render", "degrade"], default="all")
    ap.add_argument("--chrome", default=None)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--paddle",
        action="store_true",
        help="输出 PaddleOCR det 格式：outputs/images/*.jpg + outputs/train_list.txt",
    )
    ap.add_argument(
        "--gtsha256",
        action="store_true",
        help="在 final/ 下额外生成 sha256 内容寻址清单 final/train_list_sha256.txt",
    )
    args = ap.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    for lang in langs:
        if lang not in ("ar", "fa", "ps"):
            ap.error(f"unsupported lang {lang}")

    res = args.res.lower().replace("*", "x").split("x")
    if len(res) != 2 or not all(x.strip().isdigit() for x in res):
        ap.error("--res 需形如 640x800")
    args.width = args.width or int(res[0])
    args.height = args.height or int(res[1])
    args.scale = args.width / 794.0  # 版式按 794px 宽设计

    t0 = time.time()
    o = Path(args.out)
    o.mkdir(parents=True, exist_ok=True)

    pool = textpool.TextPool()
    missing = [x for x in langs if not pool.has(x, "body")]
    if missing:
        print(f"!! texts/ 缺少语料: {missing}，请先运行 scripts/fetch_corpus.py")
        return 2
    faces = fontpool.load_faces()

    man = o / "pages" / "manifest.json"
    if args.stage in ("all", "pages"):
        pages = stage_pages(args, pool, faces, langs)
    else:
        if not man.exists():
            print("!! 缺少 pages/manifest.json，请先执行 --stage pages")
            return 2
        pages = json.loads(man.read_text(encoding="utf-8"))["pages"]

    if args.stage in ("all", "render"):
        stage_render(args)
    if args.stage in ("all", "degrade"):
        stage_degrade(args, pages)
        if args.gtsha256:
            stage_gtsha256(args, pages)
        if args.paddle:
            stage_paddle(args, pages)

    run = {
        "seed": args.seed,
        "count": args.count,
        "langs": langs,
        "dpr": args.dpr,
        "page_size": [args.width, args.height],
        "illust_prob": args.illust_prob,
        "format": ("paddle" + ("+gtsha256" if args.gtsha256 else "")) if args.paddle else ("gtsha256" if args.gtsha256 else "native"),
        "profile_weights": degrade.PROFILE_WEIGHTS,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (o / f"run-{args.seed}.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"[done] {run['elapsed_sec']}s -> {o}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

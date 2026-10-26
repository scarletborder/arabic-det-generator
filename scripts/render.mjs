/**
 * 无头 Chromium 渲染 + DOM 真值标注提取。
 *
 * 输入：outputs/pages/manifest.json（由 scripts/build.py 生成）
 * 输出：outputs/clean/images/<page_id>.png（未退化的干净页）
 *       outputs/clean/labels/<page_id>.json（行级 + 词级 box，像素坐标）
 *
 * 标注红线（AGENTS.md 第 4 节）：坐标只能来自 Range.getClientRects()，禁止 OCR 反推。
 */

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { chromium } from 'playwright-core';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

const DEFAULT_CHROME =
  process.env.CHROME_PATH || 'C:\\Users\\cirno\\AppData\\Local\\Chromium\\Application\\chrome.exe';

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.ttf': 'font/ttf',
  '.otf': 'font/otf',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.txt': 'text/plain; charset=utf-8',
};

function startServer(root) {
  const server = http.createServer((req, res) => {
    const urlPath = decodeURIComponent(new URL(req.url, 'http://127.0.0.1').pathname);
    const filePath = path.join(root, urlPath);
    if (!filePath.startsWith(root)) {
      res.writeHead(403).end('forbidden');
      return;
    }
    fs.readFile(filePath, (err, data) => {
      if (err) {
        res.writeHead(404).end('not found');
        return;
      }
      res.writeHead(200, {
        'Content-Type': MIME[path.extname(filePath).toLowerCase()] || 'application/octet-stream',
        'Cache-Control': 'no-store',
      });
      res.end(data);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, '127.0.0.1', () => {
      const { port } = server.address();
      resolve({ server, port });
    });
  });
}

/** 让页面刚好填满：先逐段显示备选段落，溢出时回退；再裁剪残余溢出。 */
async function fitToPage(page) {
  return page.evaluate(() => {
    const pageEl = document.querySelector('.page');
    const limit = pageEl.clientHeight;
    const overflow = () => pageEl.scrollHeight - limit;

    const splitSentences = (text) =>
      (text || '').split(/(?<=[.؟!?۔])\s+/).filter((x) => x.trim().length);

    // 正文段落默认隐藏：逐段显示；显示不下的那段用“句子/词前缀二分”卡到刚好填满
    const bodyBlocks = [...document.querySelectorAll('[data-extra]')];
    for (const el of bodyBlocks) el.style.display = 'none';

    const fitPrefix = (el) => {
      // 优先按句子切；整段只有一句时退化为按词切（真实书籍页面常在句中断开）
      let units = splitSentences(el.textContent);
      let joiner = ' ';
      if (units.length <= 1) {
        units = (el.textContent || '').split(/\s+/).filter(Boolean);
      }
      if (units.length <= 1) return false;
      let lo = 1;
      let hi = units.length;
      let best = 0;
      const apply = (n) => {
        el.textContent = units.slice(0, n).join(joiner);
      };
      while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        apply(mid);
        if (overflow() <= 1) {
          best = mid;
          lo = mid + 1;
        } else {
          hi = mid - 1;
        }
      }
      if (best > 0) {
        apply(best);
        return true;
      }
      return false;
    };

    for (const el of bodyBlocks) {
      el.style.display = '';
      if (overflow() <= 1) continue; // 还放得下，继续显示下一段
      if (!fitPrefix(el)) el.style.display = 'none'; // 连一个词都放不下，换下一段试试
      else break;
    }

    // 残余溢出（标题/列表/插图过高）兜底裁剪
    let guard = 0;
    while (overflow() > 1 && guard++ < 40) {
      const shrinkables = [...document.querySelectorAll('[data-shrink]')].filter(
        (el) => el.style.display !== 'none',
      );
      if (!shrinkables.length) break;
      shrinkables[shrinkables.length - 1].remove();
    }

    const prect = pageEl.getBoundingClientRect();
    let bottom = 0;
    for (const el of pageEl.querySelectorAll('[data-lt]')) {
      if (el.style.display === 'none' || el.dataset.lt === 'page') continue;
      bottom = Math.max(bottom, el.getBoundingClientRect().bottom - prect.top);
    }
    return {
      overflow: pageEl.scrollHeight - limit,
      contentHeight: pageEl.scrollHeight,
      pageHeight: limit,
      fill: Math.round((bottom / prect.height) * 1000) / 1000,
    };
  });
}

/** 提取词级矩形，并按基线聚类成行。 */
async function extractLabels(page) {
  return page.evaluate(() => {
    const pageEl = document.querySelector('.page');
    const origin = pageEl.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const elements = [...pageEl.querySelectorAll('[data-lt]')];

    const allWords = [];
    const allClusters = [];
    let seq = 0;
    for (const el of elements) {
      const lt = el.dataset.lt;
      const style = getComputedStyle(el);
      const fs = parseFloat(style.fontSize) || 16;
      const nodes = [];
      (function walk(node) {
        for (const child of node.childNodes) {
          if (child.nodeType === 3) nodes.push(child);
          else if (child.nodeType === 1) walk(child);
        }
      })(el);

      const tokens = [];
      for (const tn of nodes) {
        const text = tn.nodeValue || '';
        const re = /\S+/g;
        let m;
        while ((m = re.exec(text)) !== null) {
          const range = document.createRange();
          range.setStart(tn, m.index);
          range.setEnd(tn, m.index + m[0].length);
          const rects = [...range.getClientRects()].filter((r) => r.width > 0 && r.height > 0);
          if (!rects.length) continue;
          const x0 = Math.min(...rects.map((r) => r.left));
          const y0 = Math.min(...rects.map((r) => r.top));
          const x1 = Math.max(...rects.map((r) => r.right));
          const y1 = Math.max(...rects.map((r) => r.bottom));
          tokens.push({
            text: m[0],
            x: x0 - origin.left,
            y: y0 - origin.top,
            w: x1 - x0,
            h: y1 - y0,
            lt,
            fs,
            seq: seq++,
          });
        }
      }

      // 行聚类：纵向重叠 + 横向邻近（避免把双栏的同一高度误并成一行）
      const sorted = [...tokens].sort((a, b) => a.y - b.y || a.x - b.x);
      const clusters = [];
      for (const t of sorted) {
        let target = null;
        for (const c of clusters) {
          const ov = Math.min(c.y1, t.y + t.h) - Math.max(c.y0, t.y);
          if (ov < 0.42 * Math.min(c.y1 - c.y0, t.h)) continue;
          const gap = Math.max(t.x - c.x1, c.x0 - (t.x + t.w));
          if (gap > 1.6 * t.fs) continue;
          target = c;
          break;
        }
        if (!target) {
          clusters.push({ x0: t.x, y0: t.y, x1: t.x + t.w, y1: t.y + t.h, tokens: [t], lt, fs: t.fs });
        } else {
          target.x0 = Math.min(target.x0, t.x);
          target.y0 = Math.min(target.y0, t.y);
          target.x1 = Math.max(target.x1, t.x + t.w);
          target.y1 = Math.max(target.y1, t.y + t.h);
          target.tokens.push(t);
        }
      }
      for (const c of clusters) {
        c.text = c.tokens.sort((a, b) => a.seq - b.seq).map((t) => t.text).join(' ');
      }
      allWords.push(...clusters.flatMap((c) => c.tokens));
      allClusters.push(...clusters);
    }

    const clusters = allClusters;
    const words = allWords;

    // 坐标统一换算到设备像素（= 截图分辨率），避免与退化阶段分辨率不一致
    const round = (v) => Math.round(v * 10) / 10;
    const toBox = (x0, y0, x1, y1) => [
      round(x0 * dpr),
      round(y0 * dpr),
      round((x1 - x0) * dpr),
      round((y1 - y0) * dpr),
    ];
    const toPoly = (x0, y0, x1, y1) => [
      [round(x0 * dpr), round(y0 * dpr)],
      [round(x1 * dpr), round(y0 * dpr)],
      [round(x1 * dpr), round(y1 * dpr)],
      [round(x0 * dpr), round(y1 * dpr)],
    ];

    const wordObjs = words
      .filter((t) => t.w > 0.5 && t.h > 0.5)
      .map((t) => ({
        text: t.text,
        type: t.lt,
        box: toBox(t.x, t.y, t.x + t.w, t.y + t.h),
        polygon: toPoly(t.x, t.y, t.x + t.w, t.y + t.h),
      }));

    const lineObjs = clusters
      .filter((c) => c.x1 - c.x0 > 0.5 && c.y1 - c.y0 > 0.5)
      .sort((a, b) => a.y0 - b.y0 || b.x0 - a.x0)
      .map((c) => ({
        text: c.text,
        type: c.lt,
        box: toBox(c.x0, c.y0, c.x1, c.y1),
        polygon: toPoly(c.x0, c.y0, c.x1, c.y1),
        word_count: c.tokens.length,
      }));

    const blocks = elements.map((el) => {
      const r = el.getBoundingClientRect();
      return {
        type: el.dataset.lt,
        text: (el.textContent || '').trim().slice(0, 200),
        box: toBox(r.left - origin.left, r.top - origin.top, r.right - origin.left, r.bottom - origin.top),
      };
    });

    return {
      width: Math.round(origin.width * dpr),
      height: Math.round(origin.height * dpr),
      dpr,
      page_font: getComputedStyle(pageEl).fontFamily,
      lines: lineObjs,
      words: wordObjs,
      blocks,
    };
  });
}

async function main() {
  const { values } = parseArgs({
    options: {
      pages: { type: 'string', default: path.join(ROOT, 'outputs', 'pages') },
      clean: { type: 'string', default: path.join(ROOT, 'outputs', 'clean') },
      dpr: { type: 'string', default: '2' },
      chrome: { type: 'string', default: DEFAULT_CHROME },
      limit: { type: 'string', default: '0' },
    },
  });

  const pagesDir = path.resolve(values.pages);
  const cleanDir = path.resolve(values.clean);
  const dpr = Number(values.dpr) || 2;
  const manifestPath = path.join(pagesDir, 'manifest.json');
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`manifest not found: ${manifestPath}`);
  }
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  let items = manifest.pages;
  const limit = Number(values.limit) || 0;
  if (limit > 0) items = items.slice(0, limit);

  fs.mkdirSync(path.join(cleanDir, 'images'), { recursive: true });
  fs.mkdirSync(path.join(cleanDir, 'labels'), { recursive: true });

  const { server, port } = await startServer(ROOT);
  const browser = await chromium.launch({
    executablePath: values.chrome,
    args: [
      '--font-render-hinting=none',
      '--disable-lcd-text',
      '--force-color-profile=srgb',
      '--disable-font-subpixel-positioning',
      '--disable-background-timer-throttling',
    ],
  });

  const t0 = Date.now();
  let done = 0;
  const report = [];
  try {
    for (const item of items) {
      const id = item.page_id;
      const context = await browser.newContext({
        viewport: { width: item.width, height: item.height },
        deviceScaleFactor: dpr,
        locale: item.lang === 'fa' ? 'fa-IR' : item.lang === 'ps' ? 'ps-AF' : 'ar-EG',
        colorScheme: 'light',
      });
      const page = await context.newPage();
      try {
        await page.goto(`http://127.0.0.1:${port}/${path.posix.join('outputs', 'pages', id, 'index.html')}`, {
          waitUntil: 'load',
        });
        await page.evaluate(() => document.fonts.ready);
        await page.waitForTimeout(40);
        const fit = await fitToPage(page);
        const data = await extractLabels(page);
        const imgPath = path.join(cleanDir, 'images', `${id}.png`);
        await page.locator('.page').screenshot({ path: imgPath, animations: 'disabled', caret: 'hide' });

        const labels = {
          image: `clean/images/${id}.png`,
          page_id: id,
          lang: item.lang,
          dir: item.dir || 'rtl',
          profile: item.profile,
          width: Math.round(item.width * dpr),
          height: Math.round(item.height * dpr),
          dpr,
          fonts: item.fonts,
          layout: {
            columns: item.columns,
            font_size: item.font_size,
            line_height: item.line_height,
            margin: item.margin,
          },
          page_font: data.page_font,
          stats: {
            lines: data.lines.length,
            words: data.words.length,
            overflow_px: Math.round(fit.overflow),
            fill_ratio: fit.fill,
          },
          lines: data.lines,
          words: data.words,
          blocks: data.blocks,
        };
        fs.writeFileSync(path.join(cleanDir, 'labels', `${id}.json`), JSON.stringify(labels), 'utf8');
        done += 1;
        report.push({
          id,
          lines: data.lines.length,
          words: data.words.length,
          overflow: fit.overflow,
          fill: fit.fill,
        });
        if (done % 5 === 0 || done === items.length) {
          console.log(`  rendered ${done}/${items.length} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
        }
      } catch (err) {
        console.error(`  ! render failed ${id}: ${err.message}`);
      } finally {
        await context.close();
      }
    }
  } finally {
    await browser.close();
    server.close();
  }

  const badOverflow = report.filter((r) => r.overflow > 2);
  const avgFill = report.reduce((s, r) => s + (r.fill || 0), 0) / Math.max(1, report.length);
  console.log(
    `[render] pages=${report.length} avg_lines=${(
      report.reduce((s, r) => s + r.lines, 0) / Math.max(1, report.length)
    ).toFixed(1)} avg_fill=${(avgFill * 100).toFixed(1)}% overflow_pages=${badOverflow.length}`,
  );
  fs.writeFileSync(
    path.join(cleanDir, 'render-report.json'),
    JSON.stringify({ dpr, pages: report }, null, 2),
    'utf8',
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

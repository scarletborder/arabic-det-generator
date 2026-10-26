# AGENTS.md — 合成数据流水线工作约束

本文件是本仓库的**硬性约束**，所有智能体（AI Agent）与人类协作者在读写代码、下载资源、生成数据前必须先阅读并遵守。
方案背景与算法选型见 `plans.md`（方案说明）；本文件只规定**必须做 / 禁止做**的事。

优先级：**用户即时指令 > 本文件 AGENTS.md > plans.md**。若本文件与 `plans.md` 冲突，以本文件为准，并同步修正 `plans.md`。

---

## 0. 任务定义（先明确目标，再动手写代码）

**目标**：合成面向**书籍扫描 / 手机拍照**场景的阿拉伯语系文本检测数据集，用于训练 OCR 文本检测模型（行级 / 词级 bbox 或 polygon）。

**语言范围（仅此三种，不得扩展）**：
| 代号 | 语言 |
| --- | --- |
| `ar` | 阿拉伯语 Arabic |
| `fa` | 波斯语 Persian |
| `ps` | 普什图语 Pashto |

**交付物**：
1. 合成页图像（含真实感退化）；
2. 与之严格对齐的检测标注（推荐 JSON Lines / COCO，字段含 `image_path`、`lang`、`text`、`box`/`polygon`、`line_type`）；
3. 可复现的生成脚本（固定随机种子，记录种子与配置）。

**判定标准**：字形自然（正确连写与 RTL 排版）、标点规范、段落长度符合真实书籍、标注与像素**零偏差**。
文本**语义不重要**——不要花时间追求内容质量，只追求字形与版式的真实感。

**明确不做**：不训练模型、不做 OCR 识别（recognizer）、不做评测，只产数据与标注。

---

## 1. 字体

- **所有字体文件只允许放在 `fonts/` 目录下**（当前已按字体族分好子目录：`Amiri/`、`Aref_Ruqaa/`、`Lateef/`、`Noto_Kufi_Arabic/`、`Noto_Naskh_Arabic/`、`Noto_Nastaliq_Urdu/`、`Noto_Sans_Arabic/`、`Scheherazade_New/`，多数含 `static/` 静态字重子目录）。
- **禁止**把 `.ttf/.otf/.woff` 放到 `src/`、`scripts/`、`texts/` 或仓库根目录；**禁止**在生成 HTML 时把字体复制到临时目录之外的仓库位置（可用绝对 `file://` 路径或本地 HTTP 服务引用）。
- 新字体加入前必须确认 Unicode 覆盖 `ar`/`fa`/`ps` 三者（尤其普什图语扩展字母 `ټ ډ ړ ڼ ښ`），并把许可证文件（`OFL.txt` 等）一并放入该字体子目录。
- 字体使用比例约束（见 `plans.md`）：Naskh 类（Noto Naskh / Scheherazade New / Lateef / Amiri）**70%~80%**；Nastaliq 类（Noto Nastaliq Urdu）**15%~20%**；标题体（Aref Ruqaa / Noto Kufi Arabic）**5%** 左右且只用于标题、大字号。
- **渲染红线**：禁止用 `PIL.ImageDraw.text` 直接绘制阿拉伯文（会产生孤立字符、断连写）。必须走 Chromium/Playwright 渲染，或 `Pillow + libraqm`（`direction="rtl"`、`features`）等带 HarfBuzz + BiDi 的方案。

---

## 2. 网络访问（强制走代理）

本机位于中国大陆，**任何联网行为都必须经 HTTP 代理 `127.0.0.1:7890`**，包括 pip、git、`huggingface_hub`/`datasets`、Playwright 浏览器下载、`web_fetch`、curl/wget 等。

```bash
# 每个新 shell 会话开始前先执行（Bash / Git for Windows）
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
export ALL_PROXY=http://127.0.0.1:7890
export NO_PROXY=localhost,127.0.0.1
```

对应工具写法：
- pip：`pip install --proxy http://127.0.0.1:7890 <pkg>`
- git：`git -c http.proxy=http://127.0.0.1:7890 clone ...`
- Python `requests` / `httpx` / `datasets`：继承上述环境变量即可，**不要**在代码里硬编码代理字符串之外的绕过逻辑。
- Playwright：`HTTPS_PROXY=http://127.0.0.1:7890 python -m playwright install chromium`

约束：
- 代理地址**只允许出现在 `scripts/` 的运行说明或环境变量中**，业务代码需读取 `HTTP_PROXY/HTTPS_PROXY` 环境变量，不得把代理写死进 `src/` 的库函数。
- 若联网失败，先排查代理是否生效，**不要**改走无代理直连或镜像站绕过。
- 大文件（语料、浏览器内核）下载前先估算体积并告知用户。

---

## 3. 目录规划与文件命名

```
fonts/                      # 字体（唯一合法位置，见第 1 节）
src/                        # 源代码：库/模块（可 import，无副作用，不含 argparse 入口）
scripts/                    # 可执行脚本：CLI 入口、一次性任务（thin wrapper，逻辑放 src/）
texts/
  wiki/                     # 维基百科来源文本
  corpus/                   # 其它语料库来源文本（mC4 / OSCAR / Tanzil / Ganjoor / 电子书等）
  illustraion/              # 插图（现有本地素材，勿改名、勿删除）
plans.md                    # 方案说明
AGENTS.md                   # 本约束文件
```

### 3.1 文本文件命名（强制）

格式：**`语言代号-位置-编号.txt`**（全小写 ASCII，连字符分隔，禁止中文、空格、下划线）

- **语言代号**：`ar` | `fa` | `ps`（见第 0 节）。
- **位置代号**（仅允许以下取值，新增须先在本节登记）：
  | 代号 | 含义 |
  | --- | --- |
  | `title` | 文档大标题 / 章节标题 |
  | `subtitle` | 小标题 / 子标题 |
  | `body` | 正文段落 |
  | `note` | 尾注 / 脚注 |
  | `header` | 页眉（含书名、章节名） |
  | `page` | 页码（东阿拉伯数字） |
  | `caption` | 插图说明 |
  | `list` | 列表 / 引文条目 |
- **编号**：三位零填充十进制，从 `001` 递增，同一 `语言-位置` 组合内唯一。
- 示例：`ar-title-001.txt`、`fa-body-014.txt`、`ps-note-002.txt`

### 3.2 文本内容格式

- 编码固定 **UTF-8（无 BOM）**，换行 `\n`。
- **一条记录 = 一行 = 一个自然段落/标题/页眉**（不要把整篇文章塞成一行，也不要一行里塞多段）。行内不得含制表符与换行。
- 存**逻辑序**字符（Unicode 逻辑顺序），**禁止**预先做视觉序重排。
- 必须清洗：剔除 HTML 标签、URL、残留不可见字符（`\u200b`、孤立 `\u200e/\u200f` 除必要方向标记）、残缺控制符；确保阿语标点 `،` `؛` `؟` 正常保留。
- 同一 `语言-编号` 的文件族应成组抽样；生成文档时从 `wiki/` 与 `corpus/` 中**按位置代号随机组合**，拼出符合常识的页面结构（页眉 → 标题 → 小标题 → 正文段落… → 尾注 → 页码），**不要**出现"正文里出现标题字体""页码在页首"等反常识组合。

### 3.3 插图

- 只从 `texts/illustraion/` 读取（`*.png`，含 `1.png`、`2.png`、`3.png` 等）。
- 合成时以**小概率**（默认 **5%** 左右的页面，或按页内元素数 3%~8%）随机放置 1 张，并生成对应 `caption` 行文本。
- 插图必须参与几何退化（与页面同变换），但**不得**为其生成文本框标注（如需图区标注，另设 `figure` 类别）。

### 3.4 生成产物

- 生成结果统一写到 `outputs/`（`outputs/images/`、`outputs/labels/`、`outputs/debug/`），**不要**污染 `src/`、`scripts/`、`texts/`、`fonts/`。
- 产物目录需可被整体删除重生成；每次运行附带 `outputs/run-<seed>.json`（种子、字体、比例、退化参数）。
- `--paddle` 时训练数据改为 PaddleOCR det 格式：`outputs/images/*.jpg` + `outputs/train_list.txt`（每行 `相对路径\t[{"transcription","points"}]`，行级多边形），并**不保留** `outputs/final/`；此时 `data_dir` 指 `outputs/`，`label_file_list` 指 `outputs/train_list.txt`。

---

## 4. 生成流水线约束（源自 plans.md，必须遵守）

1. **排版引擎**：HTML/CSS + 无头 Chromium（Playwright）。页面必须显式声明 `dir="rtl"`、`lang="ar|fa|ps"`，`text-align: justify`，首行缩进 `text-indent`。
2. **真值来源**：标注只能由 `element.getBoundingClientRect()` / `Range.getClientRects()` 在 DOM 中**直接提取**，禁止用 OCR 或图像分析反推标注。
3. **顺序**：先取标注（未变换的干净坐标）→ 再对图像施加几何变换 → 用**同一变换矩阵**同步变换坐标 → 最后施加像素级退化。任何"先退化成图再猜坐标"的做法禁止。
4. **结构比例**：标题 2~6 词加粗；正文段落 80~300 词；列表/引文用 `1.` `-` 或东阿拉伯数字编号。
5. **字符混入**：正文按 **5%~10%** 概率插入拉丁数字/英文词；页码按语言使用 `٠١٢٣٤٥٦٧٨٩`（`ar`）或 `۰۱۲۳۴۵۶۷۸۹`（`fa`/`ps`）。
6. **退化**：`Augraphy`（BookBinding / BleedThrough / PageBorder / Dust / LightingGradient）为主，`Albumentations`（Perspective / MotionBlur / ColorJitter / ImageCompression）补充；几何畸变、透背、光照不均、模糊、噪点按 `plans.md` 第三、四节实现。
7. **难度配比**：60% 正常扫描 / 30% 复杂手机拍照 / 10% 极端挑战。混合比例写在配置里，不要散落在代码中。
8. **可复现**：所有随机性必须来自显式 `seed`（`random.Random(seed)` / `np.random.default_rng(seed)`），禁止裸 `random.random()`。

---

## 5. 编码规范与禁止事项

- 语言：Python（渲染/DOM 真值阶段用 Node.js，仅限 `scripts/render.mjs`）。路径统一用 `pathlib.Path`，**相对仓库根目录解析**，禁止写死 `d:\...` 绝对路径。
- 依赖集中声明在仓库根的 `requirements.txt`（新增依赖须写清版本下限）。
- `src/` 内模块必须**无副作用**（import 不读盘、不渲染、不联网）；所有"跑起来"的动作放 `scripts/`。
- CLI 脚本必须支持 `--seed`、`--count`、`--out`、`--langs` 参数，且 `--help` 可用。
- **禁止**：修改/删除 `texts/illustraion/` 中用户已有素材；删除 `fonts/` 中文件；删除 `.codebuddy/` 目录；在仓库内提交大体积语料（应在 `.gitignore` 中排除 `outputs/` 与 `texts/` 下的原始语料包）。
- 遇到不确定的目录归属或命名，**先在本文件登记规则再实现**，不要临时造新目录。

---

## 6. 协作流程

1. 先读 `plans.md` + 本文件，确认任务边界；
2. 输出简要实施方案（目录、文件、命名、比例）→ 再写代码；
3. 改完代码用少量样本（如 5 页）自检：字形连写正确、RTL 无误、标注框贴合文字；
4. 交付时说明：种子、命令、产物路径、已知问题；
5. 若方案有变更，**同时更新 `plans.md` 与本文件**，保持二者一致。

---

## 7. 已实现流水线

1. `scripts/fetch_corpus.py`：HF datasets-server（代理）→ `texts/wiki`、`texts/corpus`。
2. `scripts/build.py --stage pages`：`src/pagebuild.py` 生成 HTML + 备选段落（默认隐藏）。
3. `scripts/render.mjs`：Chromium 渲染，逐段显示填满页面，`getClientRects` 提取真值。
4. `src/refine.py`：把行/词框收紧到真实墨迹范围（去掉行距与两端对齐空白）。
5. `src/degrade.py`：几何映射 F + 数值逆映射 remap，标注走同一个 F。

产物：`outputs/pages/<id>/`、`outputs/clean/{images,labels}/`、`outputs/final/{images,labels}/`、`outputs/debug/`、`outputs/run-<seed>.json`。

运行（先开代理）：`python scripts/fetch_corpus.py` → `python scripts/build.py --count 30 --seed 7 --debug`。
导出 PaddleOCR det 训练集：在末尾加 `--paddle`（`stage_paddle()` 负责转换并清理 `final/`）。

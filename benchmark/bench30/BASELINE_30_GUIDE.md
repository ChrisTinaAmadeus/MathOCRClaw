# MathDoc 小规模 Baseline（约 30 张）工作指导

本文档整理自项目启动阶段的工作规划，并结合当前仓库实际数据状态更新。目标是：**从 MathDoc 中选出约 30 张高质量试卷图片，用强多模态模型（推荐千问 3 VL）识别题干与手写内容，产出 30 份可评估的 Markdown baseline**；流程需可标准化迁移到全量约 3000+ 张。

---

## 0. 当前仓库状态（检查结论）

| 路径 | 状态 | 说明 |
|------|------|------|
| 评估代码（`main.py` / `pipeline.py` / `core/`） | 已就绪 | 用于 GT vs Pred 评估，**不是**提取工具 |
| `conda` 环境 `mathdoc` | 已就绪 | 需含 `numpy`、`python-Levenshtein` |
| `raw_data/hf/*.zip` | 已下载 | `Result_GT.zip` ~33MB，`Test_images.zip` ~550MB |
| `raw_data/Test_images/` | 已解压 | **464** 张原始试卷图，分 10 组 `Test_image_image1`…`10` |
| `raw_data/Result_GT/` | 已解压 | **461** 份已有 GT `.md`（含插图 jpg），分 10 组 `Result_qwenGT_image1`…`10` |
| `data/` | 尚未按评估规范组织 | 下一步再搭 `image_1`…`image_30` |
| `benchmark/` | 尚未创建 | 放 manifest 与选中图片副本 |

**数据角色澄清：**

- `Test_images`：待识别的原始扫描/拍照试卷（本任务的输入）。
- `Result_GT`：仓库已提供的 Ground Truth Markdown（格式已接近评估器要求，可用作参考与对齐目标）。
- 本仓库主程序：比较 GT 与 Pred 的评估流水线；**baseline 提取脚本需另写**。

**GT Markdown 样例特征（已有 Result_GT）：**

- 可选状态标签：`[格式正常]`
- 大题分区：`一、选择题` / `二、填空题` / `三、解答题`
- 小题：`1.` / `12.` 行首题号
- 答案：`【答案：C】`；无法识别：`[无法识别的选择题]`
- 插图：`![插图](../xxx.jpg)`（相对路径）

---

## 1. 环境（摘要）

```bash
conda activate mathdoc
cd /home/puppeteer/MathDoc

# 若导入仍缺包：
pip install numpy python-Levenshtein huggingface_hub

# API（评估 / 提取都要用）
# 确保 .env 中有 DASHSCOPE_API_KEY
```

验证：

```bash
python -c "from pipeline import EvaluationPipeline; print('OK')"
```

Hugging Face 下载（国内网络常用镜像；本步已完成，仅作记录）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_TOKEN="$(tr -d '\n\r ' < token.txt)"
hf download chenyue123/Result_GT --repo-type dataset --local-dir raw_data/hf
```

解压（无 `unzip` 时用 Python）：

```bash
python - <<'PY'
import zipfile
from pathlib import Path
Path("raw_data").mkdir(exist_ok=True)
for z in ["raw_data/hf/Result_GT.zip", "raw_data/hf/Test_images.zip"]:
    zipfile.ZipFile(z).extractall("raw_data")
print("done")
PY
```

**注意：** `token.txt` / `.env` 勿提交；已写入 `.gitignore`。若 token 曾泄露，请在 HF 设置中轮换。

---

## 2. 任务目标与成功标准

### 目标

1. 从 `Test_images` 选出约 **30** 张高质量、有代表性的试卷页。
2. 用 **Qwen3-VL**（或同等强力多模态模型）提取题干 + 手写答案，得到 **30 份 Markdown**。
3. 人工校对后形成正式 GT（或与已有 `Result_GT` 对齐校验）。
4. 目录与文件格式符合本仓库评估器约定，便于日后扩到全量。

### 成功标准（可验证）

- [ ] `benchmark/manifest.tsv` 有约 30 行，字段完整。
- [ ] 每张图对应一份 `.md`，格式含分区标题 + 题号 + 答案标签。
- [ ] `data/image_{1..30}/` 目录结构可被 `config.py` 的 `generate_path_pairs` 识别。
- [ ] `python main.py` 能跑通至少一轮（GT vs baseline 或 GT vs GT smoke test）。

---

## 3. 推荐工作目录

```text
MathDoc/
├── raw_data/
│   ├── hf/                 # zip 与 HF 缓存
│   ├── Test_images/        # 原始试卷图（全量）
│   └── Result_GT/          # 已有 GT md + 插图
├── benchmark/
│   ├── manifest.tsv        # 30 张映射表（驱动批量脚本）
│   ├── images/             # 选中图片的副本或软链
│   └── prompts/
│       └── extract_v1.txt  # 固定版本的提取 prompt
├── data/
│   ├── image_1/
│   │   ├── gt/                       # 人工定稿 GT
│   │   └── qwen3_vl_baseline/        # VLM 初稿 Pred
│   └── image_2/ ...
├── scripts/
│   ├── select_candidates.py          # 筛图 / 列清单
│   └── extract_baseline.py           # 批量 VLM 提取（可扩全量）
└── docs/
    └── BASELINE_30_GUIDE.md          # 本文档
```

一键建目录：

```bash
cd /home/puppeteer/MathDoc
mkdir -p benchmark/images benchmark/prompts scripts docs
for i in $(seq 1 30); do
  mkdir -p "data/image_${i}/gt" "data/image_${i}/qwen3_vl_baseline"
done
```

---

## 4. 选出约 30 张「高质量」图片

### 4.1 筛选原则

**优先保留**

- 短边分辨率建议 ≥ 1000px
- 整页完整、少裁切、透视不严重
- 印刷清晰，手写可辨
- 题型覆盖：选择 + 填空 + 解答
- 含公式、几何/函数插图（覆盖图片评估链路）

**优先排除**

- 严重模糊、过曝、大阴影
- 封面 / 注意事项页、几乎无题目
- 大面积污损或空白

**建议配比（约 30 张）**

| 类型 | 数量 | 目的 |
|------|------|------|
| 清晰标准 | ~10 | 建立上限参考 |
| 中等噪声 | ~10 | 贴近真实分布 |
| 困难样本 | ~10 | 含重手写、密公式、部分无法识别 |

**多样性：** 尽量从 `Test_image_image1`…`10` 各组各抽 2–4 张，避免只来自单一批次。

### 4.2 与已有 GT 对齐（强烈推荐）

`Result_GT` 中 md 文件名常与原图 stem 相关，例如：

- 图：`Test_images/Test_image_image6/row_3485_....jpg`
- GT：`Result_GT/Result_qwenGT_image6/markdowns/row_3485_....md`

选图时优先选 **两边都存在** 的样本：便于把 VLM 新 baseline 与官方 GT 做对比评估。

快速列候选（按分辨率）：

```bash
conda activate mathdoc
pip install Pillow   # 若未装

python - <<'PY'
from pathlib import Path
from PIL import Image

root = Path("raw_data/Test_images")
rows = []
for p in root.rglob("*"):
    if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    if "Zone.Identifier" in p.name:
        continue
    try:
        w, h = Image.open(p).size
        rows.append((min(w, h), w * h, str(p)))
    except Exception:
        pass
rows.sort(reverse=True)
for mn, area, p in rows[:60]:
    print(f"{mn:4d}  {area:9d}  {p}")
PY
```

### 4.3 写 manifest（扩全量时只改此表）

`benchmark/manifest.tsv` 建议字段（Tab 分隔）：

```text
id	src_image	gt_md	split	quality	note
1	raw_data/Test_images/Test_image_image1/xxx.jpg	raw_data/Result_GT/Result_qwenGT_image1/markdowns/xxx.md	bench30	clear	选择+填空
2	...				
```

约定：`id` 对应 `data/image_{id}/`。

---

## 5. GT / Pred Markdown 格式（评估器硬性要求）

评估器（`GTAnalyzer` / `Preprocessor`）依赖：

1. **大题标题**（识别题型）：`一、选择题` / `二、填空题` / `三、解答题`
2. **小题题号**：行首 `数字.`（如 `1.`、`18.`）
3. **答案**（任选其一）：
   - `【答案：C】` 或 `[答案：1/2]`
   - 填空下划线：`___4___`
   - 拒答：`7. [无法识别的选择题]`
4. **图片**：`![alt](rel_path)`，相对路径时图片与 md **同目录**（或按已有 GT 的 `../xxx.jpg` 约定放置）

可保留但会被预处理过滤的内容：试卷标题、`注意事项`、`[格式正常]` 等。

---

## 6. 用千问 3 多模态生成 30 份 baseline

### 6.1 模型与接口

- 模型：`qwen3-vl-plus`（与 `config.py` 中 `VISION_MODEL` 一致）
- 接口：DashScope OpenAI 兼容模式  
  `https://dashscope.aliyuncs.com/compatible-mode/v1`
- Key：`.env` 中 `DASHSCOPE_API_KEY`

### 6.2 Prompt 要点（写入 `benchmark/prompts/extract_v1.txt` 并版本化）

1. 完整识别印刷题干、选项、手写答案。
2. 输出 Markdown，保留分区标题与题号。
3. 公式用 LaTeX（`$...$` / `$$...$$`）。
4. 手写答案写入 `【答案：...】`；看不清标 `[无法识别的…]`，禁止编造。
5. 卷内小图：用 `![插图](filename.jpg)` 或占位说明，勿丢位置信息。

### 6.3 批量脚本设计原则（可迁移到 3000+）

`scripts/extract_baseline.py` 应：

- 只读 `manifest.tsv`（或等价 JSONL）
- 参数：`--manifest` `--model` `--out-root` `--workers`
- 输出：`data/image_{id}/qwen3_vl_baseline/{stem}.md`
- 同时备份：`*.raw.md`
- 并发建议 2–4，加短延迟防限流

单张调用骨架：

```python
import base64, os
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def extract_page(image_path: Path, prompt: str, out_md: Path, model="qwen3-vl-plus"):
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    resp = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
            ],
        }],
        temperature=0,
    )
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(resp.choices[0].message.content, encoding="utf-8")
```

安装（若缺）：

```bash
pip install openai
```

---

## 7. 人工校对 → 定稿 GT

```text
qwen3_vl_baseline/*.md  →  人工修正  →  gt/*.md
```

或：若沿用仓库已有 `Result_GT`，可把对应 md **复制**到 `data/image_{id}/gt/`，把 VLM 输出放在 `qwen3_vl_baseline/`，直接评估「新提取 vs 官方 GT」。

校对重点：题号与分区、手写答案、公式、插图路径、拒答标记。

---

## 8. 用本仓库跑评估（验证格式是否合格）

编辑 `config.py`：

```python
PATH_PAIRS = generate_path_pairs(
    start=1,
    end=30,
    gt_template="{data_dir}/image_{num}/gt",
    pred_template="{data_dir}/image_{num}/qwen3_vl_baseline",
)
```

运行：

```bash
conda activate mathdoc
cd /home/puppeteer/MathDoc
export MAX_CONCURRENT=4
python main.py
```

报告目录：`output/reports/`（`summary_report.txt` / `.json` 等）。

合格参考：题号结构可解析、对齐成功率明显 > 0、失败列表无异常堆积。

---

## 9. 扩到全量的标准化清单

小 benchmark 跑通后，**只扩规模，不改流程**：

1. **manifest 驱动**：`id, src_path, gt_path, split, quality`
2. **固定目录模板**：`data/image_{num}/gt/`、`data/image_{num}/{model}/`
3. **固定 prompt 版本**：如 `extract_v1.txt`
4. **参数化脚本**：`--manifest --model --out-dir --workers`
5. **自动校验**：题号 regex、是否有分区标题、图片是否存在
6. **抽样人工 QA**：每 100 张抽 2–3 张

---

## 10. 建议执行顺序（从当前状态起）

```text
[已完成] 环境 + 下载 zip + 解压到 raw_data/
[下一步] ① 建 benchmark/ / data/image_{1..30}/ 目录
         ② 按原则选 ~30 张，写 manifest.tsv（优先有配套 Result_GT）
         ③ 固定 extract_v1.txt，写 scripts/extract_baseline.py
         ④ 批量生成 qwen3_vl_baseline/（30 份 md）
         ⑤ 校对或拷贝官方 GT → data/image_*/gt/
         ⑥ 改 config.py end=30，跑 python main.py
         ⑦ 根据报告迭代 prompt / 格式
         ⑧ 固化后扩全量
```

---

## 11. 附录：全量规模速查

| 集合 | 数量 |
|------|------|
| `Test_images` 有效图片 | ~464（分组约 39–49 / 组） |
| `Result_GT` Markdown | 461 |
| 小 baseline 目标 | ~30 张图 → 30 份 md |

组名对应关系示例：

- `Test_image_image6` ↔ `Result_qwenGT_image6`
- 评估配置中的 `image_{num}` 是**你自己重新编号**的 30 槽位，不必与原始 image6 数字相同。

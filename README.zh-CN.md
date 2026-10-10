# Code Review Agent

[English](README.md) | **简体中文**

一个**基于证据、默认只读**的 AI 代码审查 Agent。它会扫描项目、执行静态检查、
向大语言模型索取结构化的问题列表、**逐条对源码进行验证**，最后生成 Markdown
与 JSON 两种格式的审查报告。

本项目的设计目标不只是"找出 Bug"，而是"让审查过程本身值得信任"：

- 没有真实文件、没有合法行号、或者引用的代码证据在指定行上并不存在的问题，
  会被**丢弃**，而不是被写进报告。
- **未完成的审查，绝不会被表述为审查通过。**
- **被审查的项目永远不会被修改。**

```text
开始 → 校验输入 → 扫描与过滤 → 静态检查 → 大模型审查
     → 验证问题 → 生成报告 → 结束
```

---

## 快速开始

```bash
# 1. 创建环境（需要 Python 3.12+，运行时零第三方依赖）
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2. 完全离线运行：不需要 API Key，不产生任何费用
.venv/bin/code-review ./my-react-app --provider fake

# 3. 或者接入真实模型
export OPENAI_API_KEY=sk-...
.venv/bin/code-review ./my-react-app --model gpt-4o-mini
```

不安装包，直接以源码方式运行：

```bash
PYTHONPATH=src python -m code_review_agent ./my-react-app --provider fake
```

输出示例：

```text
Review completed.

Files scanned: 2
Files reviewed: 2
Findings: 6
Critical: 0
High: 2
Medium: 4
Low: 0
Info: 0

Source: 6 from built-in static analysis, 0 from model review.

Static analysis skipped: eslint
Report: /path/to/project/.code-review/report.md
Report (JSON): /path/to/project/.code-review/report.json
```

> 说明：程序输出、报告字段、规则名与配置项均为英文，正文文档提供中文说明。

报告默认写入 `<项目目录>/.code-review/`（可用 `--output-dir` 覆盖）。该目录会被
扫描器排除，因此报告永远不会成为下一次审查的目标。

## 退出码

| 退出码 | 含义 |
| ------ | ---- |
| 0 | 审查已完成。**仍可能存在发现的问题**——"完成"不等于"通过"。 |
| 1 | 内部未预期错误。 |
| 2 | 输入非法（路径或配置有误）。 |
| 3 | 审查失败：模型调用失败、缺少必需工具、预算耗尽，或没有任何文件被审查。 |
| 4 | 报告写入失败。 |
| 130 | 被中断（Ctrl-C）。 |

只要退出码非 0，程序一定会把原因写到 stderr，方便 CI 直接作为门禁条件。

---

## 功能说明

| 阶段 | 行为 |
| ---- | ---- |
| **扫描** | 遍历项目，保留 `.js/.jsx/.ts/.tsx/.mjs/.cjs`。剪除 `node_modules`、`dist`、`.git`、各类缓存；跳过锁文件、压缩文件、`.d.ts`、二进制文件、超大文件与敏感文件。 |
| **静态检查** | 始终运行内置规则；若检测到 ESLint 则额外运行。ESLint 缺失会被**记录并跳过**，而不是直接失败（除非显式开启严格模式）。 |
| **大模型审查** | 逐文件发送带行号的源码与静态诊断结果，可请求只读工具，且必须返回符合 Schema 的 JSON。格式错误会触发有限次数的修复重试。 |
| **验证** | 检查路径是否在项目内、行号是否存在、引用的证据是否真的出现在该行，并过滤完全重复与近似重复的问题。被丢弃的候选问题会在报告中公开说明。 |
| **报告** | Markdown 面向人阅读，JSON 面向工具消费，包含严重级别统计、注意事项、被跳过文件、被丢弃候选表以及各阶段执行轨迹。 |

### 内置规则（无需任何外部工具链）

`no-debugger`、`assignment-in-condition`（同时覆盖 `if (x = y)` 与
`x => x.p = v`）、`no-eval`、`no-function-constructor`、`no-inner-html`、
`no-document-write`、`no-sync-fs`、`no-loose-equality`、`no-empty-catch`、
`promise-not-handled`、`listener-not-removed`、`effect-missing-cleanup`、
`interval-not-cleared`、`timeout-not-cleared`、`no-console-debug`、`no-alert`、
`no-var`。共 17 条。

这些规则存在的意义是：即使机器上完全没有 Node 工具链，流水线也能产出确定性的、
有据可查的结果；同时让整个系统可以离线测试。规则被刻意写得保守——**一条误报
过多的规则比没有规则更糟**，因为它会污染精确率指标。

---

## 配置

优先级：**命令行参数 > 环境变量 > TOML 配置文件 > 默认值**。

```bash
code-review ./app --config ./code-review.toml --model gpt-4o-mini --max-files 20
```

`code-review.toml`：

```toml
[llm]
provider = "openai"          # openai | openai-compatible | deepseek | anthropic | fake
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"   # 这里填的是「变量名」，绝不是密钥本身
temperature = 0.0
max_retries = 3
max_format_repairs = 2
response_format = "json_object"  # json_schema | json_object | none

[budget]
max_total_tokens = 200000
max_requests = 60
max_cost_usd = 2.00          # 去掉该行即不限制费用
max_files = 40
max_file_bytes = 200000
max_context_chars = 120000

[scanner]
extensions = [".js", ".jsx", ".ts", ".tsx"]
max_files = 500
follow_symlinks = false

[static_analysis]
enabled = true
eslint_command = "auto"      # "auto" 自动探测；也可填路径固定；填 "off" 关闭
fail_on_missing_tool = false
builtin_rules = true

[review]
max_review_rounds_per_file = 2
min_confidence = 0.0
enable_tool_calls = true
fail_on_partial_review = true   # 有文件的模型审查失败时，整次运行判定为失败

[report]
output_dir = ".code-review"

[agent]
log_level = "INFO"
```

常用环境变量：`CODE_REVIEW_PROVIDER`、`CODE_REVIEW_MODEL`、
`CODE_REVIEW_BASE_URL`、`CODE_REVIEW_API_KEY_ENV`、`CODE_REVIEW_MAX_FILES`、
`CODE_REVIEW_MAX_REQUESTS`、`CODE_REVIEW_MAX_TOKENS`、
`CODE_REVIEW_MAX_COST_USD`、`CODE_REVIEW_OUTPUT_DIR`、
`CODE_REVIEW_ESLINT_COMMAND`、`CODE_REVIEW_LOG_LEVEL`。

配置中**未知的段落或键会被直接报错拒绝**，而不是被静默忽略——否则一个拼写错误
就会让人误以为配置生效了。

### 常用命令行参数

```bash
--provider fake          # 完全离线，不联网、不产生费用
--skip-static-analysis   # 内置规则仍然运行，仅跳过 ESLint
--eslint-command off     # 显式关闭 ESLint
--fail-on-missing-tool   # 把「工具缺失」升级为致命错误
--no-tool-calls          # 关闭 Agent 的只读工具调用阶段
--format json            # stdout 输出机器可读摘要
-v / -q                  # 调试日志 / 只输出错误
```

---

## 安全设计

Agent 处理的是**可能包含恶意指令的源码**（例如注释里写着"忽略你之前收到的所有
指令"），因此安全边界是按此前提设计的：

- **默认只读。** 除报告目录外，没有任何代码路径会写入被审查项目。该性质由测试
  保证：在运行前后对项目内每个文件做快照比对。
- **路径限制。** 所有文件访问都经过 `PathGuard`，它会拒绝 `..` 越权跳转、项目
  之外的绝对路径，以及指向项目外的符号链接——**包括根目录本身是符号链接的情况**
  （如 macOS 上 `/var` → `/private/var`）。
- **不做 Shell 拼接。** ESLint 以参数数组形式调用（`shell=False`），模型输出
  永远不会被拼进命令行。
- **敏感信息。** `.env` 与各类密钥文件被排除在审查之外；形似凭据的字符串会在
  提示词、证据、日志与报告中被替换为 `[REDACTED]`。
- **提示词注入防护。** 系统提示词明确声明：代码、注释、文档与诊断结果都属于
  **不可信数据**；源码被包裹在显式的 `<SOURCE_FILE>` 分隔符内。
- **受限的自主性。** 模型只能调用三个只读工具，参数有类型校验，每个文件有调用
  次数上限，返回结果有大小限制。

---

## 架构

```text
src/code_review_agent/
  cli.py                 命令行入口、退出码、stdout 摘要
  security.py            路径沙箱、文件过滤、敏感信息脱敏
  validate.py            零依赖的 JSON Schema 子集校验
  models.py              ReviewFinding / ReviewRun / 响应 Schema
  errors.py              类型化错误体系
  config/settings.py     TOML + 环境变量 + 命令行配置
  llm/
    client.py            LLMClient、FakeLLMClient、BudgetGuard、重试
    parsing.py           容错 JSON 提取 + 修复提示词
  tools/
    scanner.py           项目遍历与过滤
    reader.py            源码快照与 1-based 行号映射
    rules.py             内置静态规则
    static_analysis.py   ESLint 适配器与调度
  agent/
    prompts.py           系统提示词与上下文拼装
    tools.py             沙箱化只读工具集
    reviewer.py          逐文件审查循环
    validator.py         证据 / 行号 / 重复性验证
  workflow/orchestrator.py   阶段编排与状态流转
  reporting/generator.py     Markdown + JSON 报告
```

核心设计原则：

- **确定性优先。** 凡是静态检查器或 Schema 能定论的事，都不交给模型判断。
- **验证器是唯一闸门。** 模型自评的 confidence 仅作参考并如实记录，**永远不会
  覆盖证据校验的结果**。
- **状态流转是显式的。** `pending → running → completed | failed | cancelled`，
  并记录每个阶段的耗时与错误。

## 测试

```bash
.venv/bin/python -m pytest tests/ -q
```

**443 个测试，全部离线**：不联网、不需要 API Key、不产生费用。覆盖范围包括：
安全边界（越权路径、符号链接、脱敏）、扫描过滤、行号映射、JSON Schema 校验、
JSON 修复、重试与预算控制、每条内置规则的正反两个方向、证据验证、报告渲染，
以及通过 `FakeLLMClient` 完成的完整工作流集成测试——其中包含各类失败分支
（输出无法解析、预算耗尽、工具缺失、报告不可写）。

集成测试直接断言验收标准，例如"每条正式发现都有文件路径和行号"、
"运行结束后项目逐字节未发生变化"。

## 开发路线

Phase 1（最小 Agent Loop）至 Phase 3（工作流工程化）已实现。
Phase 4（发现复核、评估数据集、模型对比）与 Phase 5（隔离环境自动修复 +
人工审批）尚未实现：本仓库提供了单元与集成测试，但**没有带标注的评估数据集**，
因此这里不给出任何精确率或召回率声明。

## 当前范围之外

不包含：自动修改源码、自动创建 Pull Request、多 Agent 并行、Web 管理后台、
分布式任务队列、长期记忆。Python 项目支持与更多静态分析工具属于后续扩展；
首期仅面向 React / JavaScript / TypeScript。

## 开源协议

[MIT](LICENSE)

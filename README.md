# Code Review Agent

**English** | [简体中文](README.zh-CN.md)

An evidence-based, read-only AI code review agent. It scans a project, runs
static analysis, asks an LLM for structured findings, **validates every finding
against the source**, and writes Markdown + JSON reports.

The design goal is not just "find bugs" but "be trustworthy about what was
actually reviewed":

- A finding without a real file, a valid line number, and evidence that appears
  in the cited lines is rejected, not reported.
- A review that did not complete is never presented as a passing review.
- The reviewed project is never modified.

```
Start → Validate input → Scan & filter → Static analysis → LLM review
      → Validate findings → Generate report → End
```

---

## Quick start

```bash
# 1. Create an environment (Python 3.12+; no runtime dependencies)
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 2. Run entirely offline, with no API key and no cost
.venv/bin/code-review ./my-react-app --provider fake

# 3. Or point it at a real model
export OPENAI_API_KEY=sk-...
.venv/bin/code-review ./my-react-app --model gpt-4o-mini
```

Without installing the package, run it from the source tree:

```bash
PYTHONPATH=src python -m code_review_agent ./my-react-app --provider fake
```

Example output:

```text
Review completed.

Files scanned: 42
Files reviewed: 35
Findings: 4
Critical: 0
High: 1
Medium: 2
Low: 1
Info: 0

Report: .code-review/report.md
```

Reports are written to `<project>/.code-review/` by default (override with
`--output-dir`). That directory is excluded from scanning, so reports never
become review targets themselves.

## Exit codes

| Code | Meaning |
| ---- | ------- |
| 0 | Review completed. Findings may still exist — completion is not a verdict. |
| 1 | Unexpected internal error. |
| 2 | Invalid input (bad path or config). |
| 3 | Review failed: LLM failure, missing required tool, budget stop, or no file reviewed. |
| 4 | Reports could not be written. |
| 130 | Interrupted. |

A non-zero exit always prints the reason to stderr, so CI can gate on it.

---

## What it does

| Stage | Behaviour |
| ----- | --------- |
| **Scan** | Walks the project, keeping `.js/.jsx/.ts/.tsx/.mjs/.cjs`. Prunes `node_modules`, `dist`, `.git`, caches; skips lockfiles, minified files, `.d.ts`, binaries, oversize files, and secrets. |
| **Static analysis** | Always runs built-in rules; additionally runs ESLint when found. A missing ESLint is recorded and skipped, not fatal (unless you opt in). |
| **LLM review** | Per file, sends numbered source plus diagnostics, may request read-only tools, and must return schema-valid JSON. Format errors trigger bounded repair retries. |
| **Validate** | Checks the path is inside the project, the lines exist, the quoted evidence actually occurs there, and removes duplicates/near-duplicates. Rejections are disclosed in the report. |
| **Report** | Markdown for humans, JSON for tooling, with severity counts, caveats, skipped files, the discarded-candidate table, and the run trace. |

### Built-in rules (no external toolchain required)

`no-debugger`, `assignment-in-condition` (including `if (x = y)` and
`x => x.p = v`), `no-eval`, `no-function-constructor`, `no-inner-html`,
`no-document-write`, `no-sync-fs`, `no-loose-equality`, `no-empty-catch`,
`promise-not-handled`, `listener-not-removed`, `effect-missing-cleanup`,
`interval-not-cleared`, `timeout-not-cleared`, `no-console-debug`, `no-alert`,
`no-var`.

These exist so the pipeline produces deterministic, evidence-backed output even
on a machine with no Node toolchain, and so the whole system is testable
offline. They are deliberately conservative: a noisy rule is worse than no rule,
because it corrupts the precision metric.

---

## Configuration

Priority: **CLI flags > environment variables > TOML file > defaults**.

```bash
code-review ./app --config ./code-review.toml --model gpt-4o-mini --max-files 20
```

`code-review.toml`:

```toml
[llm]
provider = "openai"          # openai | openai-compatible | deepseek | anthropic | fake
model = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"   # the *name* of the variable, never the value
temperature = 0.0
max_retries = 3
max_format_repairs = 2
response_format = "json_object"  # json_schema | json_object | none

[budget]
max_total_tokens = 200000
max_requests = 60
max_cost_usd = 2.00          # omit for no cost ceiling
max_files = 40
max_file_bytes = 200000
max_context_chars = 120000

[scanner]
extensions = [".js", ".jsx", ".ts", ".tsx"]
max_files = 500
follow_symlinks = false

[static_analysis]
enabled = true
eslint_command = "auto"      # "auto" to probe, a path/name to pin, "off" to disable
fail_on_missing_tool = false
builtin_rules = true

[review]
max_review_rounds_per_file = 2
min_confidence = 0.0
enable_tool_calls = true
fail_on_partial_review = true   # a file whose LLM review failed makes the run fail

[report]
output_dir = ".code-review"
```

Common environment variables: `CODE_REVIEW_PROVIDER`, `CODE_REVIEW_MODEL`,
`CODE_REVIEW_BASE_URL`, `CODE_REVIEW_API_KEY_ENV`, `CODE_REVIEW_MAX_FILES`,
`CODE_REVIEW_MAX_REQUESTS`, `CODE_REVIEW_MAX_TOKENS`, `CODE_REVIEW_MAX_COST_USD`,
`CODE_REVIEW_OUTPUT_DIR`, `CODE_REVIEW_ESLINT_COMMAND`, `CODE_REVIEW_LOG_LEVEL`.

Unknown config sections or keys are rejected rather than silently ignored.

### Useful flags

```bash
--provider fake          # fully offline, no network, no cost
--skip-static-analysis   # built-in rules stay on; ESLint is skipped
--eslint-command off     # disable ESLint explicitly
--fail-on-missing-tool   # make a missing ESLint fatal
--no-tool-calls          # disable the agent's read-only tool phase
--format json            # machine-readable stdout summary
-v / -q                  # debug / errors-only logging
```

---

## Safety model

The agent reads potentially hostile input (source code containing "ignore your
instructions" comments) and is built accordingly:

- **Read-only.** No code path writes to the project except the report directory.
  Verified by tests that snapshot every project file before and after a run.
- **Path confinement.** All file access goes through `PathGuard`, which rejects
  `..` traversal, absolute paths outside the project, and symlinks that escape
  the root — including symlinked roots such as `/var` → `/private/var`.
- **No shell interpolation.** ESLint is executed as an argv list with
  `shell=False`; model output is never concatenated into a command.
- **Secrets.** `.env`/key files are excluded from review, and credential-looking
  literals are redacted from prompts, evidence, logs, and reports.
- **Prompt injection.** The system prompt declares code, comments, docs and
  diagnostics to be untrusted data, and the source is fenced inside explicit
  `<SOURCE_FILE>` delimiters.
- **Bounded agency.** The model may only call three read-only tools, with typed
  arguments, a per-file call cap, and size-limited results.

---

## Architecture

```
src/code_review_agent/
  cli.py                 CLI, exit codes, stdout summaries
  security.py            path sandbox, file filtering, secret redaction
  validate.py            dependency-free JSON Schema subset
  models.py              ReviewFinding / ReviewRun / response schema
  errors.py              typed error hierarchy
  config/settings.py     TOML + env + CLI configuration
  llm/
    client.py            LLMClient, FakeLLMClient, BudgetGuard, retries
    parsing.py           tolerant JSON extraction + repair prompt
  tools/
    scanner.py           project walk + filtering
    reader.py            source snapshots and 1-based line mapping
    rules.py             built-in static rules
    static_analysis.py   ESLint adapter + orchestration
  agent/
    prompts.py           system prompt and context assembly
    tools.py             sandboxed read-only toolbelt
    reviewer.py          the per-file review loop
    validator.py         evidence / line / duplicate validation
  workflow/orchestrator.py   phase orchestration and status transitions
  reporting/generator.py     Markdown + JSON reports
```

Key design rules:

- **Deterministic first.** Anything a linter or the schema can settle is not
  left to the model's judgement.
- **The validator is the gate.** Model confidence is advisory and recorded as
  such; it never overrides evidence checks.
- **Status transitions are explicit.** `pending → running → completed | failed |
  cancelled`, recorded with per-phase timings and errors.

## Testing

```bash
.venv/bin/python -m pytest tests/ -q
```

443 tests, all offline: no network, no API key, no cost. Coverage includes the
security boundary (traversal, symlinks, redaction), scanner filtering, line
mapping, JSON Schema validation, JSON repair, retries and budgets, each built-in
rule in both directions, evidence validation, report rendering, and full
workflow integration through a `FakeLLMClient` — including the failure branches
(unparseable output, budget exhaustion, missing tool, unwritable report).

Integration tests assert the acceptance criteria directly, such as
"every reported finding has a file and a line" and "the project is byte-for-byte
unchanged after a run".

## Roadmap

Phase 1 (minimal loop) through Phase 3 (workflow engineering) are implemented.
Phase 4 (finding re-review, evaluation datasets, model comparison) and Phase 5
(isolated auto-fix with human approval) are not: the repository ships unit and
integration tests, but **no labelled evaluation dataset**, so no precision or
recall claim is made here.

## Non-goals (current scope)

No automatic source modification, no pull-request creation, no multi-agent
parallelism, no web UI, no distributed queue, no long-term memory. Python
targets and additional analyzers are future work; the first release targets
React / JavaScript / TypeScript.

## License

[MIT](LICENSE)

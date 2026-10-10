"""Built-in static checks that require no external toolchain.

These rules exist so that the static-analysis phase produces deterministic,
evidence-backed input even on a machine without ESLint installed. They are
deliberately conservative: a heuristic that fires on clean code would poison the
precision metric in section 9.4.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Diagnostic:
    """A single static-analysis result, aligned with ESLint's shape."""

    file: str
    line: int
    column: int
    severity: str  # error | warning | info
    rule_id: str
    message: str
    source: str = "builtin"

    @property
    def is_error(self) -> bool:
        return self.severity == "error"

    def to_dict(self) -> dict[str, object]:
        return {
            "file": self.file,
            "line": self.line,
            "column": self.column,
            "severity": self.severity,
            "rule_id": self.rule_id,
            "message": self.message,
            "source": self.source,
        }


def _strip_strings_and_comments(line: str) -> str:
    """Blank out string/comment content so rules match code, not prose.

    Prevents a rule from firing on the word ``eval`` inside a comment or an
    ``==`` inside a string literal.
    """
    result: list[str] = []
    index = 0
    length = len(line)
    quote: str | None = None
    while index < length:
        char = line[index]
        if quote:
            if char == "\\":
                index += 2
                continue
            if char == quote:
                quote = None
            result.append(" ")
            index += 1
            continue
        if char in "\"'`":
            quote = char
            result.append(" ")
            index += 1
            continue
        if char == "/" and index + 1 < length and line[index + 1] in "/*":
            break  # rest of line is a comment
        result.append(char)
        index += 1
    return "".join(result)


#: rule id -> (severity, message, regex applied to code-only text)
_SIMPLE_RULES: tuple[tuple[str, str, str, re.Pattern[str]], ...] = (
    (
        "no-debugger",
        "error",
        "`debugger` statement left in source",
        # The lookahead keeps identifiers such as `debuggerCount` from matching.
        re.compile(r"(?<![\w.$])debugger(?![A-Za-z0-9_$])\s*;?"),
    ),
    (
        "no-console-debug",
        "warning",
        "console logging left in source (console.{log,debug,dir,trace})",
        re.compile(r"\bconsole\s*\.\s*(log|debug|dir|trace)\s*\("),
    ),
    (
        "no-alert",
        "warning",
        "blocking browser dialog (alert/confirm/prompt)",
        re.compile(r"(?<![\w.])(alert|confirm|prompt)\s*\("),
    ),
    (
        "no-loose-equality",
        "warning",
        "loose equality performs type coercion; prefer === / !==",
        re.compile(r"(?<![\w=!<>])(==|!=)(?!=)"),
    ),
    (
        "no-eval",
        "error",
        "dynamic code execution via eval()",
        re.compile(r"(?<![\w.])eval\s*\("),
    ),
    (
        "no-function-constructor",
        "error",
        "dynamic code execution via the Function constructor",
        re.compile(r"(?<![\w.])new\s+Function\s*\("),
    ),
    (
        "no-document-write",
        "warning",
        "document.write can inject markup and blocks parsing",
        re.compile(r"\bdocument\s*\.\s*write(ln)?\s*\("),
    ),
    (
        "no-inner-html",
        "warning",
        "innerHTML assignment can introduce XSS; prefer textContent or a sanitiser",
        re.compile(r"\.\s*innerHTML\s*="),
    ),
    (
        "no-sync-fs",
        "warning",
        "synchronous filesystem call blocks the event loop",
        re.compile(r"\b(fs\s*\.\s*(readFileSync|writeFileSync|existsSync)|require\s*\(\s*['\"]fs['\"]\s*\)\.\w+Sync)\b"),
    ),
    (
        "no-var",
        "info",
        "`var` is function-scoped; prefer const/let",
        re.compile(r"(?<![\w.])var\s+[A-Za-z_$]"),
    ),
)

_UNUSED_CATCH_RE = re.compile(r"catch\s*(\([^)]*\))?\s*\{\s*\}")
#: An `=` where a comparison was almost certainly intended, inside a condition.
#: The left/right guards exclude `==`, `===`, `<=`, `>=`, `!=`, `+=` and the
#: arrow-function `=>`.
_CONDITION_ASSIGNMENT_RE = re.compile(
    r"(?<![-=!<>+\-*/%&|^])=(?!=|>)"
)
_CONDITIONAL_KEYWORDS_RE = re.compile(r"(?<![\w.$])(if|while)\s*\(")
#: `x => x.prop = value`: an assignment used as a predicate/selector body. This
#: is the same defect as `if (x = y)` -- almost always a typo for `===`.
_ARROW_ASSIGNMENT_RE = re.compile(
    r"=>\s*\(?\s*[A-Za-z_$][\w$]*\s*(?:\.[A-Za-z_$][\w$]*)*\s*(?<![=!<>])=(?!=)"
)
_UNHANDLED_FETCH_RE = re.compile(r"(?<![\w.$])fetch\s*\(")
_AWAITED_FETCH_RE = re.compile(r"(?<![\w.$])(await|return|yield)\s+fetch\s*\(")
_PROMISE_HANDLER_RE = re.compile(r"\.\s*(catch|then|finally)\s*\(")
_SET_INTERVAL_RE = re.compile(r"(?<![\w.])setInterval\s*\(")
_CLEAR_INTERVAL_RE = re.compile(r"(?<![\w.])clearInterval\s*\(")
_SET_TIMEOUT_RE = re.compile(r"(?<![\w.])setTimeout\s*\(")
_CLEAR_TIMEOUT_RE = re.compile(r"(?<![\w.])clearTimeout\s*\(")
_EFFECT_RE = re.compile(r"useEffect\s*\(\s*(?:async\s*)?\([^)]*\)\s*=>\s*\{?")
_ADD_LISTENER_RE = re.compile(r"addEventListener\s*\(")
_REMOVE_LISTENER_RE = re.compile(r"removeEventListener\s*\(")


def analyse_text(relative_path: str, text: str) -> list[Diagnostic]:
    """Apply every built-in rule to one file."""
    diagnostics: list[Diagnostic] = []
    lines = text.splitlines()

    for line_number, raw in enumerate(lines, start=1):
        code = _strip_strings_and_comments(raw)
        if not code.strip():
            continue
        for rule_id, severity, message, pattern in _SIMPLE_RULES:
            match = pattern.search(code)
            if match:
                diagnostics.append(
                    Diagnostic(
                        file=relative_path,
                        line=line_number,
                        column=match.start() + 1,
                        severity=severity,
                        rule_id=rule_id,
                        message=message,
                    )
                )

        if _UNUSED_CATCH_RE.search(code):
            diagnostics.append(
                Diagnostic(
                    file=relative_path,
                    line=line_number,
                    column=1,
                    severity="warning",
                    rule_id="no-empty-catch",
                    message="empty catch block silently swallows the error",
                )
            )

        if _has_condition_assignment(code) or _ARROW_ASSIGNMENT_RE.search(code):
            diagnostics.append(
                Diagnostic(
                    file=relative_path,
                    line=line_number,
                    column=1,
                    severity="error",
                    rule_id="assignment-in-condition",
                    message=(
                        "assignment where a comparison (===) or a predicate was likely intended"
                    ),
                )
            )

        if _ADD_LISTENER_RE.search(code) and not _REMOVE_LISTENER_RE.search(code):
            window = "\n".join(lines[line_number - 1 : line_number + 20])
            if not _REMOVE_LISTENER_RE.search(window):
                diagnostics.append(
                    Diagnostic(
                        file=relative_path,
                        line=line_number,
                        column=1,
                        severity="warning",
                        rule_id="listener-not-removed",
                        message="event listener added but never removed (likely leak)",
                    )
                )

    # File-level heuristics: compare usage counts across the whole file.
    body = "\n".join(_strip_strings_and_comments(line) for line in lines)

    if _SET_INTERVAL_RE.search(body) and not _CLEAR_INTERVAL_RE.search(body):
        line_number = _first_line_matching(lines, _SET_INTERVAL_RE)
        diagnostics.append(
            Diagnostic(
                file=relative_path,
                line=line_number,
                column=1,
                severity="warning",
                rule_id="interval-not-cleared",
                message="setInterval is never cleared; the timer leaks after unmount/teardown",
            )
        )

    if _SET_TIMEOUT_RE.search(body) and not _CLEAR_TIMEOUT_RE.search(body):
        line_number = _first_line_matching(lines, _SET_TIMEOUT_RE)
        diagnostics.append(
            Diagnostic(
                file=relative_path,
                line=line_number,
                column=1,
                severity="info",
                rule_id="timeout-not-cleared",
                message="setTimeout is never cleared; consider cleanup on unmount",
            )
        )

    effect_line = _first_line_matching(lines, _EFFECT_RE)
    if effect_line and _ADD_LISTENER_RE.search(body) and not _REMOVE_LISTENER_RE.search(body):
        diagnostics.append(
            Diagnostic(
                file=relative_path,
                line=effect_line,
                column=1,
                severity="warning",
                rule_id="effect-missing-cleanup",
                message="useEffect subscribes without returning a cleanup function",
            )
        )

    for line_number, raw in enumerate(lines, start=1):
        code = _strip_strings_and_comments(raw)
        if _UNHANDLED_FETCH_RE.search(code) and not _AWAITED_FETCH_RE.search(code):
            # The handler is normally chained further down the same statement,
            # so scan forward from the call rather than only this line.
            following = "\n".join(
                _strip_strings_and_comments(item) for item in lines[line_number - 1 : line_number + 12]
            )
            if _PROMISE_HANDLER_RE.search(following):
                continue
            diagnostics.append(
                Diagnostic(
                    file=relative_path,
                    line=line_number,
                    column=1,
                    severity="warning",
                    rule_id="promise-not-handled",
                    message="fetch() result is neither awaited nor chained to .catch/.then",
                )
            )

    return _dedupe(diagnostics)


def _has_condition_assignment(code: str) -> bool:
    """Detect ``if (x = y)`` / ``while (x = y)`` without flagging comparisons.

    The parenthesised condition is extracted with brace balancing so nested
    calls (``if (f(a) === b)``) do not produce false positives.
    """
    for match in _CONDITIONAL_KEYWORDS_RE.finditer(code):
        start = code.index("(", match.start())
        depth = 0
        for index in range(start, len(code)):
            char = code[index]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    condition = code[start + 1 : index]
                    if _CONDITION_ASSIGNMENT_RE.search(condition):
                        return True
                    break
    return False


def _first_line_matching(lines: list[str], pattern: re.Pattern[str]) -> int:
    for index, line in enumerate(lines, start=1):
        if pattern.search(line):
            return index
    return 1


def _dedupe(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    seen: set[tuple[str, int, str]] = set()
    result: list[Diagnostic] = []
    for diagnostic in diagnostics:
        key = (diagnostic.file, diagnostic.line, diagnostic.rule_id)
        if key in seen:
            continue
        seen.add(key)
        result.append(diagnostic)
    return result

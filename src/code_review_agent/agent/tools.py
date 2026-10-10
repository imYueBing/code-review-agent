"""The controlled tool boundary exposed to the model (design doc 4.2, 8.3).

The agent may *request* tool calls, but every request is validated against a
fixed registry with typed arguments and bounded results. The model can never
name a shell command, a path outside the project, or an unbounded read.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from ..config.settings import AgentConfig
from ..errors import SecurityError, ToolError
from ..security import PathGuard, redact_secrets
from ..tools.reader import CodeReader
from ..tools.scanner import ScannedFile

logger = logging.getLogger(__name__)

_DECLARATION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:default\s+)?"
    r"(?:async\s+)?"
    r"(?P<kind>function|class|const|let|var|interface|type|enum|namespace|abstract\s+class)\s+"
    r"(?P<name>[A-Za-z_$][\w$]*)"
)
_EXPORT_FROM_RE = re.compile(r"^\s*export\s+(?:\*|\{[^}]*\})\s+from\s+['\"](?P<module>[^'\"]+)['\"]")


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    raw: dict[str, Any] | None = None

    @classmethod
    def parse_many(cls, payload: Any, *, limit: int) -> list["ToolCall"]:
        """Parse a model-provided tool-call list, discarding malformed entries."""
        if isinstance(payload, dict):
            payload = payload.get("tool_calls", [])
        if not isinstance(payload, list):
            return []
        calls: list[ToolCall] = []
        for entry in payload:
            if len(calls) >= limit:
                break
            if not isinstance(entry, dict):
                continue
            name = entry.get("tool") or entry.get("name")
            args = entry.get("args") or entry.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            calls.append(cls(tool=name, args=args, raw=entry))
        return calls


class ProjectToolbelt:
    """Sandboxed read-only tools over the scanned project."""

    #: Hard ceilings; the model cannot raise these.
    MAX_RESULT_CHARS = 4_000
    MAX_SEARCH_HITS = 25
    MAX_READ_LINES = 120
    MAX_OUTLINE_ITEMS = 80

    def __init__(
        self,
        guard: PathGuard,
        reader: CodeReader,
        files: list[ScannedFile],
        config: AgentConfig,
    ) -> None:
        self.guard = guard
        self.reader = reader
        self.config = config
        self._files = {scanned.relative_path: scanned for scanned in files}
        self._handlers: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
            "search_symbol": self._search_symbol,
            "read_lines": self._read_lines,
            "file_outline": self._file_outline,
        }

    @property
    def tool_names(self) -> list[str]:
        return sorted(self._handlers)

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_symbol",
                "args": {"query": "string"},
                "description": "Find definitions and uses of an identifier across reviewed files.",
            },
            {
                "name": "read_lines",
                "args": {"file": "string", "start_line": "int", "end_line": "int"},
                "description": f"Read up to {self.MAX_READ_LINES} lines from a project file.",
            },
            {
                "name": "file_outline",
                "args": {"file": "string"},
                "description": "List top-level declarations in a project file.",
            },
        ]

    def execute(self, call: ToolCall) -> dict[str, Any]:
        """Run one validated tool call, returning a JSON-serialisable result."""
        handler = self._handlers.get(call.tool)
        if handler is None:
            return {
                "tool": call.tool,
                "ok": False,
                "error": f"unknown tool {call.tool!r}; available: {', '.join(self.tool_names)}",
            }
        try:
            payload = handler(call.args)
        except (ToolError, SecurityError) as exc:
            return {"tool": call.tool, "ok": False, "args": call.args, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - a bad tool call must not kill the run
            logger.warning("tool %s raised: %s", call.tool, exc)
            return {"tool": call.tool, "ok": False, "args": call.args, "error": f"{type(exc).__name__}: {exc}"}
        payload["tool"] = call.tool
        payload["ok"] = True
        return _truncate_payload(payload, self.MAX_RESULT_CHARS)

    # -- handlers --------------------------------------------------------
    def _search_symbol(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            raise ToolError("search_symbol requires a non-empty 'query'")
        if len(query) > 80:
            raise ToolError("search_symbol query is too long (max 80 characters)")
        if not re.fullmatch(r"[A-Za-z_$][\w$.\-]*", query):
            raise ToolError("search_symbol query must be an identifier-like token")

        needle = re.compile(rf"\b{re.escape(query)}\b")
        hits: list[dict[str, Any]] = []
        files_searched = 0
        for relative_path in sorted(self._files):
            source = self.reader.read_safe(relative_path)
            if source is None:
                continue
            files_searched += 1
            for line in source.lines():
                if needle.search(line.text):
                    hits.append(
                        {
                            "file": relative_path,
                            "line": line.number,
                            "text": redact_secrets(line.text.strip())[:200],
                        }
                    )
                    if len(hits) >= self.MAX_SEARCH_HITS:
                        return {
                            "query": query,
                            "hits": hits,
                            "truncated": True,
                            "files_searched": files_searched,
                        }
        return {"query": query, "hits": hits, "truncated": False, "files_searched": files_searched}

    def _read_lines(self, args: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(args.get("file") or "").strip()
        if not relative_path:
            raise ToolError("read_lines requires 'file'")
        if relative_path not in self._files:
            raise ToolError(f"{relative_path!r} is not one of the reviewed project files")
        try:
            start = int(args.get("start_line") or 1)
            end = int(args.get("end_line") or (start + self.MAX_READ_LINES - 1))
        except (TypeError, ValueError) as exc:
            raise ToolError("read_lines 'start_line'/'end_line' must be integers") from exc
        if start < 1 or end < 1:
            raise ToolError("line numbers must be >= 1")
        if end < start:
            start, end = end, start
        if end - start + 1 > self.MAX_READ_LINES:
            end = start + self.MAX_READ_LINES - 1

        source = self.reader.read(relative_path)
        end = min(end, max(source.line_count, 1))
        return {
            "file": relative_path,
            "start_line": start,
            "end_line": end,
            "content": redact_secrets(source.numbered_text(start=start, end=end)),
        }

    def _file_outline(self, args: dict[str, Any]) -> dict[str, Any]:
        relative_path = str(args.get("file") or "").strip()
        if not relative_path:
            raise ToolError("file_outline requires 'file'")
        if relative_path not in self._files:
            raise ToolError(f"{relative_path!r} is not one of the reviewed project files")
        source = self.reader.read(relative_path)
        items: list[dict[str, Any]] = []
        for line in source.lines():
            match = _DECLARATION_RE.match(line.text)
            if match:
                items.append(
                    {
                        "line": line.number,
                        "kind": match.group("kind"),
                        "name": match.group("name"),
                    }
                )
            else:
                export_match = _EXPORT_FROM_RE.match(line.text)
                if export_match:
                    items.append(
                        {"line": line.number, "kind": "re-export", "name": export_match.group("module")}
                    )
            if len(items) >= self.MAX_OUTLINE_ITEMS:
                break
        return {"file": relative_path, "items": items, "truncated": len(items) >= self.MAX_OUTLINE_ITEMS}


def _truncate_payload(payload: dict[str, Any], limit: int) -> dict[str, Any]:
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) <= limit:
        return payload
    payload = dict(payload)
    payload["truncated"] = True
    # Drop the bulkiest field progressively until the payload fits.
    for key in ("content", "hits", "items"):
        if key in payload and isinstance(payload[key], (list, str)):
            value = payload[key]
            if isinstance(value, list):
                while value and len(json.dumps(payload, ensure_ascii=False)) > limit:
                    value = value[: max(1, len(value) // 2)]
                payload[key] = value
            else:
                payload[key] = value[: max(0, len(value) // 2)]
    text = json.dumps(payload, ensure_ascii=False)
    if len(text) > limit:
        return {"ok": payload.get("ok", True), "tool": payload.get("tool"), "truncated": True,
                "error": "result exceeded size limit and was dropped"}
    return payload

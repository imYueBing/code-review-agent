"""Built-in static rule tests (design doc 1.3, 2.1).

Each rule is checked in both directions: it must fire on a real violation with
the right line number, and it must stay silent on clean code. Precision matters
more than recall here, because a noisy rule poisons the false-positive metric.
"""

from __future__ import annotations

import pytest

from code_review_agent.tools.rules import analyse_text


def rule_ids(text: str) -> list[str]:
    return [d.rule_id for d in analyse_text("f.ts", text)]


def first_line(text: str, rule_id: str) -> int:
    for diagnostic in analyse_text("f.ts", text):
        if diagnostic.rule_id == rule_id:
            return diagnostic.line
    raise AssertionError(f"{rule_id} did not fire")


class TestPositiveDetection:
    def test_debugger_statement(self):
        assert "no-debugger" in rule_ids("function f() {\n  debugger;\n}\n")

    def test_console_log(self):
        assert "no-console-debug" in rule_ids("console.log('x');\n")

    def test_alert(self):
        assert "no-alert" in rule_ids("alert('hi');\n")

    def test_loose_equality(self):
        assert "no-loose-equality" in rule_ids("if (a == b) { go(); }\n")

    def test_loose_inequality(self):
        assert "no-loose-equality" in rule_ids("if (a != b) { go(); }\n")

    def test_eval(self):
        assert "no-eval" in rule_ids("const x = eval('1+1');\n")

    def test_function_constructor(self):
        assert "no-function-constructor" in rule_ids("const f = new Function('a', 'return a');\n")

    def test_document_write(self):
        assert "no-document-write" in rule_ids("document.write('<b>x</b>');\n")

    def test_inner_html_assignment(self):
        assert "no-inner-html" in rule_ids("el.innerHTML = userInput;\n")

    def test_sync_fs_call(self):
        assert "no-sync-fs" in rule_ids("const data = fs.readFileSync('a.txt');\n")

    def test_var_declaration(self):
        assert "no-var" in rule_ids("var count = 0;\n")

    def test_empty_catch(self):
        assert "no-empty-catch" in rule_ids("try { risky(); } catch (e) {}\n")

    def test_set_interval_without_clear(self):
        assert "interval-not-cleared" in rule_ids("setInterval(() => tick(), 1000);\n")

    def test_set_timeout_without_clear(self):
        assert "timeout-not-cleared" in rule_ids("setTimeout(() => tick(), 1000);\n")

    def test_promise_without_catch(self):
        assert "promise-not-handled" in rule_ids("fetch('/api/users');\n")

    def test_listener_never_removed(self):
        ids = rule_ids("el.addEventListener('click', onClick);\n")
        assert "listener-not-removed" in ids

    def test_effect_missing_cleanup(self):
        text = (
            "useEffect(() => {\n"
            "  window.addEventListener('resize', onResize);\n"
            "}, []);\n"
        )
        assert "effect-missing-cleanup" in rule_ids(text)

    def test_assignment_inside_if_condition(self):
        assert "assignment-in-condition" in rule_ids("if (label = null) {\n  return;\n}\n")

    def test_assignment_inside_while_condition(self):
        assert "assignment-in-condition" in rule_ids("while (item = next()) { use(item); }\n")

    def test_assignment_on_member_expression(self):
        assert "assignment-in-condition" in rule_ids("if (item.id = id) { pick(); }\n")

    def test_assignment_as_arrow_predicate(self):
        text = "return items.find(i => i.id = id).label;\n"
        assert "assignment-in-condition" in rule_ids(text)

    def test_assignment_as_arrow_selector(self):
        assert "assignment-in-condition" in rule_ids("items.map(i => i.label = 'x');\n")

    def test_assignment_in_arrow_callback_body(self):
        assert "assignment-in-condition" in rule_ids("onClick={() => (count = 0)}\n")


class TestNegativeDetection:
    """Rules must not fire on correct code."""

    @pytest.mark.parametrize(
        "text",
        [
            "debuggerStatement();\n",
            "console.logging = 1;\n",
            "const debuggerCount = 1;\n",
        ],
    )
    def test_no_debugger_does_not_overreach(self, text):
        assert "no-debugger" not in rule_ids(text)

    def test_strict_equality_is_clean(self):
        assert "no-loose-equality" not in rule_ids("if (a === b) { go(); }\n")

    def test_strict_inequality_is_clean(self):
        assert "no-loose-equality" not in rule_ids("if (a !== b) { go(); }\n")

    def test_arrow_function_is_not_the_function_constructor(self):
        assert "no-function-constructor" not in rule_ids("const f = () => 1;\n")

    def test_console_error_is_not_debug_logging(self):
        assert "no-console-debug" not in rule_ids("console.error('boom');\n")

    def test_catch_that_handles_error_is_clean(self):
        assert "no-empty-catch" not in rule_ids("try { risky(); } catch (e) { report(e); }\n")

    def test_cleared_interval_is_clean(self):
        text = "const id = setInterval(tick, 1000);\nclearInterval(id);\n"
        assert "interval-not-cleared" not in rule_ids(text)

    def test_promise_with_catch_is_clean(self):
        assert "promise-not-handled" not in rule_ids("fetch('/api').catch(handle);\n")

    def test_listener_with_matching_removal_is_clean(self):
        text = (
            "el.addEventListener('click', onClick);\n"
            "return () => el.removeEventListener('click', onClick);\n"
        )
        assert "listener-not-removed" not in rule_ids(text)

    def test_const_and_let_are_clean(self):
        assert "no-var" not in rule_ids("const a = 1;\nlet b = 2;\n")

    def test_inner_html_read_is_clean(self):
        assert "no-inner-html" not in rule_ids("const html = el.innerHTML;\n")

    @pytest.mark.parametrize(
        "text",
        [
            "if (a === b) { go(); }\n",
            "if (a == b) { go(); }\n",
            "if (a !== b) { go(); }\n",
            "if (a >= b) { go(); }\n",
            "if (a <= b) { go(); }\n",
            "if (a != b) { go(); }\n",
            "if (a && (b === c)) { go(); }\n",
            "const f = () => 1;\n",
            "total += 1;\n",
            "count -= 1;\n",
            "items.find(i => i.id === id)\n",
            "items.map(i => i.label)\n",
            "items.forEach(i => total += i);\n",
            "const g = x => x + 1;\n",
        ],
    )
    def test_assignment_rule_does_not_flag_comparisons(self, text):
        # A false positive here would be reported as an 'error' severity
        # finding, so the guard rails matter more than for a warning.
        assert "assignment-in-condition" not in rule_ids(text)

    def test_ordinary_code_is_completely_clean(self):
        text = (
            "export function add(a: number, b: number): number {\n"
            "  if (a === b) {\n"
            "    return 0;\n"
            "  }\n"
            "  return a + b;\n"
            "}\n"
        )
        assert rule_ids(text) == []


class TestFalsePositiveGuards:
    def test_rule_names_inside_strings_do_not_fire(self):
        assert "no-eval" not in rule_ids("const label = 'never call eval() directly';\n")

    def test_rule_names_inside_comments_do_not_fire(self):
        assert "no-eval" not in rule_ids("// do not use eval( here\nconst a = 1;\n")

    def test_loose_equality_inside_string_does_not_fire(self):
        assert "no-loose-equality" not in rule_ids("const op = 'a == b';\n")

    def test_triple_equals_inside_string_does_not_fire(self):
        assert "no-loose-equality" not in rule_ids("const doc = \"use === not ==\";\n")

    def test_keyword_like_identifiers_do_not_fire(self):
        assert "no-alert" not in rule_ids("const alertBox = createAlertBox();\n")


class TestDiagnosticShape:
    def test_line_numbers_are_accurate(self):
        text = "const a = 1;\nconst b = 2;\nconsole.log(b);\n"
        assert first_line(text, "no-console-debug") == 3

    def test_column_is_one_based(self):
        diagnostics = analyse_text("f.ts", "let x = eval('1');\n")
        eval_diagnostic = next(d for d in diagnostics if d.rule_id == "no-eval")
        assert eval_diagnostic.column >= 1

    def test_severity_is_assigned(self):
        diagnostics = analyse_text("f.ts", "debugger;\n")
        assert diagnostics[0].severity == "error"
        assert diagnostics[0].is_error is True

    def test_source_is_builtin(self):
        assert analyse_text("f.ts", "debugger;\n")[0].source == "builtin"

    def test_file_field_matches_input(self):
        assert analyse_text("src/x.tsx", "debugger;\n")[0].file == "src/x.tsx"

    def test_duplicates_on_same_line_are_collapsed(self):
        diagnostics = analyse_text("f.ts", "console.log(a == b);\n")
        keys = [(d.line, d.rule_id) for d in diagnostics]
        assert len(keys) == len(set(keys))

    def test_to_dict_is_json_serialisable(self):
        import json

        json.dumps([d.to_dict() for d in analyse_text("f.ts", "debugger;\n")])

    def test_empty_input_is_safe(self):
        assert analyse_text("f.ts", "") == []

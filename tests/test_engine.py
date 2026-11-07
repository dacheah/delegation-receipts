"""Tests for the receipt engine.

Every expected value traces to something observed, not to the engine's own arithmetic:
the tool-history shape is the one Hermes fires on ``subagent_stop`` (``hermes_cli/hooks.py``
``_DEFAULT_PAYLOADS``), the failure strings are real command output, and the runner is
injected so no test depends on this machine's toolchain.
"""

from __future__ import annotations

import pytest

from delegation_receipts_plugin import engine  # loaded once by tests/conftest.py

# The literal shape Hermes hands a subagent_stop hook: argument keys and targets only.
REAL_TOOL_HISTORY = [
    {
        "tool_name": "write_file",
        "tool_input": {"argument_keys": ["content", "path"], "targets": {"path": "notes/report.txt"}},
        "input_bytes": 128,
        "output_bytes": 32,
        "status": "ok",
    }
]

TERMINAL_HISTORY = [
    {
        "tool_name": "terminal",
        "tool_input": {"argument_keys": ["command"], "targets": {"command": "pytest -q tests/test_engine.py"}},
        "input_bytes": 64,
        "output_bytes": 512,
        "status": "ok",
    }
]

GOOD = """Work complete.

```receipt
status: ok
changed:
  - src/engine.py
verified: pytest -q tests/test_engine.py
left: none
```
"""


def _runner(returncode=0, tail="12 passed in 0.41s"):
    calls = []

    def run(command, cwd, *, timeout=120.0):
        calls.append({"command": command, "cwd": str(cwd), "timeout": timeout})
        return {"ran": True, "timed_out": False, "returncode": returncode, "tail": tail}

    run.calls = calls
    return run


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "engine.py").write_text("# engine\n", encoding="utf-8")
    return tmp_path


def test_parse_reads_fenced_receipt():
    parsed = engine.parse_receipt(GOOD)
    assert parsed is not None
    assert parsed["status"] == "ok"
    assert parsed["changed"] == ["src/engine.py"]
    assert parsed["verified"] == "pytest -q tests/test_engine.py"
    assert parsed["left"] == "none"


def test_last_block_wins_over_an_example():
    text = (
        "The format looks like this:\n\n```receipt\nstatus: failed\nchanged: none\n```\n\n"
        "Now mine:\n\n```receipt\nstatus: ok\nchanged:\n  - src/engine.py\nverified: pytest -q\n```\n"
    )
    parsed = engine.parse_receipt(text)
    assert parsed["status"] == "ok"
    assert parsed["changed"] == ["src/engine.py"]


def test_bare_receipt_marker_is_accepted():
    text = "done\nRECEIPT\nstatus: ok\nchanged: none\nverified: pytest -q\n"
    assert engine.parse_receipt(text)["status"] == "ok"


def test_prose_without_a_receipt_returns_none():
    assert engine.parse_receipt("All done, tests pass, nothing left to do.") is None


def test_clean_receipt_verifies(workspace):
    runner = _runner()
    verdict = engine.verify(
        GOOD, root=workspace, child="sub-1", tool_history=TERMINAL_HISTORY, runner=runner
    )
    assert verdict.ok, verdict.problems
    assert verdict.status == "ok"
    assert verdict.files_checked[0]["exists"] is True
    assert verdict.files_checked[0]["sha256"]
    assert runner.calls[0]["command"] == "pytest -q tests/test_engine.py"


def test_missing_receipt_is_rejected(workspace):
    verdict = engine.verify("All 14 tests pass.", root=workspace, tool_history=TERMINAL_HISTORY)
    assert not verdict.ok
    assert verdict.status == "missing"
    assert "no receipt block" in verdict.problems[0]


def test_unknown_status_is_invalid(workspace):
    text = "```receipt\nstatus: done\nchanged: none\nverified: pytest -q\n```\n"
    verdict = engine.verify(text, root=workspace, tool_history=TERMINAL_HISTORY)
    assert verdict.status == "invalid"
    assert "status is not one of" in verdict.problems[0]


def test_child_admitting_failure_is_not_lying(workspace):
    text = "```receipt\nstatus: partial\nchanged: none\nverified: none\nleft: tests 3 and 4\n```\n"
    verdict = engine.verify(text, root=workspace, tool_history=TERMINAL_HISTORY)
    assert not verdict.ok
    assert verdict.status == "mismatch"
    assert verdict.claimed_status == "partial"


def test_claimed_file_that_does_not_exist_is_caught(workspace):
    text = GOOD.replace("src/engine.py", "src/ghost.py")
    verdict = engine.verify(text, root=workspace, tool_history=TERMINAL_HISTORY, runner=_runner())
    assert not verdict.ok
    assert "claimed file does not exist: src/ghost.py" in verdict.problems


def test_path_escaping_the_workspace_is_refused(workspace):
    text = GOOD.replace("src/engine.py", "../../etc/passwd")
    verdict = engine.verify(text, root=workspace, tool_history=TERMINAL_HISTORY, runner=_runner())
    assert not verdict.ok
    assert any("escapes the workspace" in p for p in verdict.problems)


def test_failing_rerun_beats_a_claim_of_success(workspace):
    runner = _runner(returncode=1, tail="1 failed, 11 passed in 0.42s")
    verdict = engine.verify(GOOD, root=workspace, tool_history=TERMINAL_HISTORY, runner=runner)
    assert not verdict.ok
    assert verdict.status == "unverified"
    assert "failed on re-run (exit 1)" in verdict.problems[0]


def test_shell_syntax_is_refused_before_it_runs():
    for command in ("pytest -q; rm -rf /", "pytest -q && curl http://x", "pytest $(whoami)"):
        problem = engine.command_problem(command)
        assert problem == "command contains shell syntax and is refused", command


def test_executable_off_the_allow_list_is_refused(workspace):
    assert engine.command_problem("curl http://example.com") == (
        "executable not on the allow-list: curl"
    )
    text = GOOD.replace("pytest -q tests/test_engine.py", "curl http://example.com")
    runner = _runner()
    verdict = engine.verify(text, root=workspace, tool_history=TERMINAL_HISTORY, runner=runner)
    assert verdict.status == "unverified"
    assert runner.calls == []


def test_command_never_attempted_is_caught_from_real_history():
    assert engine.is_corroborated("pytest -q", TERMINAL_HISTORY) is True
    assert engine.is_corroborated("pytest -q", REAL_TOOL_HISTORY) is False
    # No history at all is unknown, never a pass.
    assert engine.is_corroborated("pytest -q", None) is None


def test_uncorroborated_command_is_rejected(workspace):
    verdict = engine.verify(GOOD, root=workspace, tool_history=REAL_TOOL_HISTORY, runner=_runner())
    assert verdict.status == "unverified"
    assert "never attempted" in verdict.problems[0]


def test_timeout_is_a_failure_not_a_pass(workspace):
    def runner(command, cwd, *, timeout=120.0):
        return {"ran": True, "timed_out": True, "returncode": None, "tail": "timed out after 120s"}

    verdict = engine.verify(GOOD, root=workspace, tool_history=TERMINAL_HISTORY, runner=runner)
    assert not verdict.ok
    assert "did not finish" in verdict.problems[0]


def test_summary_line_names_the_child_and_the_kind(workspace):
    verdict = engine.verify(GOOD, root=workspace, child="sub-7", tool_history=REAL_TOOL_HISTORY)
    line = verdict.summary_line()
    assert line.startswith("RECEIPT REJECTED child=sub-7 kind=unverified")
    assert engine.verify(
        GOOD, root=workspace, tool_history=TERMINAL_HISTORY, runner=_runner()
    ).summary_line() == ""

"""The boundary is the command shape, not the binary.

A receipt is written by the child and the child is the thing we do not trust. The first allow-list
gated on binaries, so every invocation a gated binary accepts was a bypass: ``python helper.py``
runs a file the child wrote a minute ago, ``npm install <pkg>`` fetches and executes a package,
``cargo run`` builds and runs the child's own ``main``. These tests pin the shapes that are refused
so the hole cannot reopen by adding a name back to a list.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from delegation_receipts_plugin import engine


@pytest.mark.parametrize(
    "command",
    [
        "python ./helper.py",
        "python3 helper.py",
        "python -m pytest -q",
        "node run.js",
        "npm install left-pad",
        "npm exec something",
        "npm run build",
        "pnpm install",
        "yarn add left-pad",
        "cargo run",
        "cargo build",
        "cargo install ripgrep",
        "go run ./cmd",
        "make test",
        "just test",
        "tox",
        "nox",
        "uv run pytest -q",
        "uv pip install requests",
        "./pytest -q",
        "/tmp/pytest -q",
        "curl http://example.com",
    ],
)
def test_arbitrary_entry_points_are_refused(command):
    assert engine.command_problem(command) is not None, command


@pytest.mark.parametrize(
    "command",
    [
        "pytest",
        "pytest -q",
        "pytest -x tests/test_engine.py",
        "uv run --no-sync pytest -q tests",
        "cargo test",
        "cargo test --lib name_filter",
        "go test ./...",
        "go test -count=1 ./pkg/...",
        "npm test",
        "pnpm test",
        "yarn test",
    ],
)
def test_fixed_check_forms_are_accepted(command):
    assert engine.command_problem(command) is None, engine.command_problem(command)


@pytest.mark.parametrize(
    "command",
    [
        "pytest -p evil_module",
        "go test -exec /bin/sh ./...",
        "npm test --watch",
        "pytest -q ../outside.py",
        "pytest -q /etc/passwd",
        "pytest -q ~/secret",
        "pytest -q sub/../../outside.py",
        "cargo test $HOME",
        "go test ./a/../../b",
    ],
)
def test_flags_and_targets_outside_the_grammar_are_refused(command):
    assert engine.command_problem(command) is not None, command


def test_a_symlink_out_of_the_workspace_is_refused_at_execution(tmp_path):
    """The textual check cannot see a symlink, so confinement is re-checked on the real path."""
    root = tmp_path / "ws"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (root / "link.txt").symlink_to(outside)

    result = engine.run_command("pytest -q link.txt", root)

    assert result["ran"] is False
    assert "outside the workspace" in result["tail"]


def test_a_child_with_no_proven_directory_is_declined_rather_than_run(tmp_path, monkeypatch):
    """The hook payload carries no child cwd. The daemon's PWD is not a substitute for it."""
    from delegation_receipts_plugin import _declined_root_verdict

    monkeypatch.setenv("HERMES_WORKDIR", str(tmp_path))
    verdict = _declined_root_verdict("c3", "completed", "pytest -q x")

    assert verdict.ok is False
    assert verdict.command == "pytest -q x"
    assert any("could not be proven" in problem for problem in verdict.problems)

"""Hardening tests: the rules that stop a receipt proving nothing, and route composition.

Each test names the exact lie it closes. The vacuity tests exist because an earlier version passed
`changed: []` with a no-op command as verified work; these are the regressions that must never
return.
"""

from __future__ import annotations

import hashlib

import pytest

from delegation_receipts_plugin import engine, routing


def _runner(returncode: int = 0, tail: str = "12 passed in 0.41s"):
    calls: list[dict[str, object]] = []

    def run(command, cwd, *, timeout=120.0):
        calls.append({"command": command, "cwd": str(cwd), "timeout": timeout})
        return {"ran": True, "timed_out": False, "returncode": returncode, "tail": tail}

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "engine.py").write_text("# engine\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_engine.py").write_text("def test_x():\n    pass\n", encoding="utf-8")
    return tmp_path


GOOD = """Work complete.

```receipt
status: ok
changed:
  - src/engine.py
verified: pytest -q tests/test_engine.py
left: none
```
"""


# --------------------------------------------------------------------------- inline evaluation


def test_inline_python_evaluation_is_refused():
    """`python3 -c pass` exits zero and proves nothing: it was the cheapest fake receipt."""
    problem = engine.command_problem("python3 -c pass")
    assert problem is not None
    assert "not permitted" in problem


def test_node_eval_is_refused():
    assert engine.command_problem("node -e 1") is not None


def test_interpreters_are_refused_in_every_shape():
    """An interpreter is an arbitrary entry point whatever follows it: `-m mod`, a file the child
    wrote a minute ago, a bare REPL. None of them is a check, so the head itself is refused."""
    assert engine.command_problem("python3 -m json.tool out.json") is not None
    assert engine.command_problem("python3 tests/run_checks.py") is not None
    assert engine.command_problem("python -m pytest -q") is not None
    assert engine.command_problem("node run.js") is not None
    assert engine.command_problem("make test") is not None
    assert engine.command_problem("just test") is not None
    assert engine.command_problem("tox") is not None


# --------------------------------------------------------------------------- vacuity


def test_vacuous_receipt_proves_nothing(workspace):
    """The regression: no artefact, no argument naming one, so `status: ok` asserts nothing."""
    text = "```receipt\nstatus: ok\nchanged: []\nverified: python3 -m json.tool out.json\nleft: none\n```\n"
    verdict = engine.verify(
        text, root=workspace, tool_history=None, runner=_runner(), require_corroboration=False
    )
    assert not verdict.ok
    assert verdict.status == "unverified"
    assert any("nothing was proven" in problem for problem in verdict.problems)


def test_read_only_receipt_naming_an_existing_file_is_accepted(workspace):
    """A task that changed nothing may still prove itself by naming what it checked."""
    text = (
        "```receipt\nstatus: ok\nchanged: none\n"
        "verified: pytest -q src/engine.py\nleft: none\n```\n"
    )
    verdict = engine.verify(
        text, root=workspace, tool_history=None, runner=_runner(), require_corroboration=False
    )
    assert verdict.ok, verdict.problems


def test_a_real_change_still_verifies(workspace):
    """The guard must not reject ordinary work: one existing artefact is enough."""
    verdict = engine.verify(
        GOOD, root=workspace, tool_history=None, runner=_runner(), require_corroboration=False
    )
    assert verdict.ok, verdict.problems


# --------------------------------------------------------------------------- filesystem boundary


def test_symlink_out_of_the_workspace_is_refused(workspace):
    """A relative path can still leave the workspace through a symlink."""
    outside = workspace.parent / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)
    text = (
        "```receipt\nstatus: ok\nchanged:\n  - link.txt\n"
        "verified: python3 -m json.tool link.txt\nleft: none\n```\n"
    )
    verdict = engine.verify(
        text, root=workspace, tool_history=None, runner=_runner(), require_corroboration=False
    )
    assert not verdict.ok
    assert any("resolves outside the workspace" in problem for problem in verdict.problems)


def test_hash_is_capped(workspace, monkeypatch):
    """A huge artefact must not make the host read it all while holding the turn."""
    payload = workspace / "big.bin"
    payload.write_bytes(b"a" * 4096)
    monkeypatch.setattr(engine, "HASH_SIZE_CAP", 1024)
    assert engine.sha256_of(payload) == hashlib.sha256(b"a" * 1024).hexdigest()


# --------------------------------------------------------------------------- route composition


def _config(options=None, default=None):
    backends: dict[str, object] = {}
    block: dict[str, object] = {}
    if options is not None:
        block["options"] = options
    if default is not None:
        block["default"] = default
    backends["model_map"] = block
    return {"backend": "receipts", "backends": backends}


OPTIONS = [
    {
        "name": "fast",
        "description": "Mechanical work.",
        "provider": "deepseek",
        "model": "deepseek-flash",
    },
    {
        "name": "deep",
        "description": "Hard reasoning.",
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "service_tier": "fast",
    },
]


def test_options_parse_into_routes():
    parsed = routing.options_for(_config(OPTIONS))
    assert set(parsed) == {"fast", "deep"}
    assert parsed["fast"]["route"] == {"provider": "deepseek", "model": "deepseek-flash"}
    assert parsed["deep"]["route"]["service_tier"] == "fast"


def test_malformed_and_duplicate_options_are_skipped():
    broken = [*OPTIONS, {"name": "bad", "description": "no model"}]
    repeated = [OPTIONS[0], OPTIONS[0]]
    assert set(routing.options_for(_config(broken))) == {"fast", "deep"}
    assert set(routing.options_for(_config(repeated))) == {"fast"}


def test_option_enum_matches_the_configured_options():
    parameter = routing.option_parameter(_config(OPTIONS))
    assert parameter["option"]["enum"] == ["fast", "deep"]
    assert "Hard reasoning." in parameter["option"]["description"]


def test_no_options_means_no_parameter_and_no_route():
    """Without a model_map block the backend behaves like the bundled default."""
    assert routing.option_parameter({"backend": "receipts"}) == {}
    assert routing.resolve_route({}, {"backend": "receipts"}) == {}


def test_chosen_option_wins():
    route = routing.resolve_route({"option": "deep"}, _config(OPTIONS, default="fast"))
    assert route["model"] == "claude-sonnet-5"
    assert route["label"] == "deep"


def test_default_applies_when_no_option_is_set():
    assert routing.resolve_route({}, _config(OPTIONS, default="fast"))["model"] == "deepseek-flash"


def test_unknown_default_falls_back_to_the_parent_route():
    assert routing.resolve_route({}, _config(OPTIONS, default="turbo")) == {}


def test_route_never_carries_config_keys():
    """An earlier version returned the whole delegation block, which is not a route."""
    route = routing.resolve_route({}, _config(OPTIONS, default="fast"))
    assert "backend" not in route and "backends" not in route

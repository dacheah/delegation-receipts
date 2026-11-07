"""Delegation Receipts: verify what a subagent says it did.

Two seams, one product:

* :class:`ReceiptsDelegationProvider` (``delegation.backend: receipts``) imposes the receipt
  contract through ``delegate_task``'s description, which core caches as part of the prompt
  prefix — so the requirement is deterministic and costs no extra turn.
* The ``subagent_stop`` hook receives each child's summary, redacted tool history and status.
  It parses the receipt, checks the claimed files exist, confirms the cited command was really
  attempted, re-runs it with no shell under a timeout, and records a verdict.
* The ``transform_tool_result`` hook appends any non-ok verdicts to the ``delegate_task`` result,
  so the parent model sees the rejection instead of the child's own account of itself.

The ledger is append-only JSONL under the plugin data directory, so a verdict survives the turn
and ``hermes receipts`` can show what was claimed versus what was checked.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping

from .engine import DEFAULT_ALLOWED_COMMANDS, Verdict, verify

logger = logging.getLogger(__name__)

PLUGIN_NAME = "delegation-receipts"
BACKEND_NAME = "receipts"

#: Appended to delegate_task's description when this backend is active. Deterministic for a given
#: config: core caches it as part of the prompt prefix.
CONTRACT = (
    "Every subagent must end its reply with a receipt block, on its own, last:\n"
    "```receipt\nstatus: ok\nchanged:\n  - path/relative/to/workspace.py\n"
    "verified: pytest -q tests/test_that_file.py\nleft: none\n```\n"
    "status is one of ok, partial, blocked, failed. changed lists each file the child wrote. "
    "verified is the single command that proves the work; it is re-run by the host, so it must be "
    "an allow-listed executable (pytest, python, npm, cargo, go, make, ruff, mypy, tsc, ...) with no "
    "shell syntax, and it must exit zero. left is the unfinished remainder, or none. "
    "A receipt is checked against the filesystem and re-executed: status ok means the command "
    "passed, not that the child says it did."
)

#: Verdicts from this process, awaiting a delegate_task result to annotate. Keyed by parent turn
#: so a stale batch can never be attached to a later call.
_PENDING: dict[str, list[dict[str, Any]]] = {}


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def data_dir() -> Path:
    """The plugin's own data directory, following the active profile."""
    path = _hermes_home() / "plugin-data" / PLUGIN_NAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def ledger_path() -> Path:
    return data_dir() / "ledger.jsonl"


def _record(row: Mapping[str, Any]) -> None:
    try:
        with ledger_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")
    except OSError as exc:  # never fail a turn because the ledger could not be written
        logger.warning("delegation-receipts could not append to the ledger: %s", exc)


def _allowed(ctx: Any) -> tuple[str, ...]:
    try:
        configured = ctx.get_config("allowed_commands", None)
    except Exception:
        configured = None
    if isinstance(configured, (list, tuple)) and configured:
        return tuple(str(item) for item in configured)
    return DEFAULT_ALLOWED_COMMANDS


def _timeout(ctx: Any) -> float:
    try:
        value = float(ctx.get_config("verify_timeout_seconds", 120))
    except Exception:
        return 120.0
    return value if 1.0 <= value <= 3600.0 else 120.0


def _provider_class() -> type:
    from agent.delegation_provider import DelegationProvider

    class ReceiptsDelegationProvider(DelegationProvider):
        """Route exactly as the default backend does; the difference is the contract."""

        @property
        def name(self) -> str:
            return BACKEND_NAME

        @property
        def display_name(self) -> str:
            return "Delegation Receipts"

        @property
        def summary(self) -> str:
            return (
                "Subagents must end with a receipt; claimed files and the cited command are "
                "re-checked when the child stops."
            )

        def description(self, config: Mapping[str, Any]) -> str:
            return CONTRACT

        def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
            # Same route resolution as the bundled default backend: this plugin changes what is
            # checked, never which model runs.
            return {k: v for k, v in config.items() if k != "service_tier"}

    return ReceiptsDelegationProvider


def _on_subagent_stop(ctx: Any, **kwargs: Any) -> None:
    """Check one child's receipt the moment it stops."""
    try:
        child = str(kwargs.get("child_session_id") or kwargs.get("child_role") or "child")
        parent_session_id = str(kwargs.get("parent_session_id") or "")
        parent_turn_id = str(kwargs.get("parent_turn_id") or "")
        summary = kwargs.get("child_summary") or ""
        history = kwargs.get("tool_call_history")
        verdict = verify(
            summary if isinstance(summary, str) else str(summary),
            root=_workspace_root(),
            child=child,
            tool_history=history if isinstance(history, list) else None,
            allowed=_allowed(ctx),
            timeout=_timeout(ctx),
        )
        row = {
            "ts": time.time(),
            "parent_session_id": parent_session_id,
            "parent_turn_id": parent_turn_id,
            "child": child,
            "child_role": kwargs.get("child_role"),
            "child_status": kwargs.get("child_status"),
            "duration_ms": kwargs.get("duration_ms"),
            **verdict.to_dict(),
        }
        _record(row)
        if not verdict.ok:
            _PENDING.setdefault(parent_session_id or parent_turn_id or "*", []).append(row)
            logger.info("delegation-receipts rejected a child receipt: %s", verdict.summary_line())
    except Exception:  # a broken verifier must never break delegation
        logger.warning("delegation-receipts failed while checking a child", exc_info=True)
    return None


def _on_transform_tool_result(ctx: Any, **kwargs: Any) -> str | None:
    """Attach any rejected verdicts to the delegate_task result the parent model reads."""
    try:
        if str(kwargs.get("tool_name") or "") != "delegate_task":
            return None
        if not _PENDING:
            return None
        # Prefer verdicts recorded for this session; only fall back to the whole queue when the
        # host gives us no session to match on. A stale batch must never attach to a later call.
        current = str(kwargs.get("session_id") or "") or None
        keys = [k for k in _PENDING if current and k == current] or list(_PENDING)
        rows: list[dict[str, Any]] = []
        for key in keys:
            rows.extend(_PENDING.pop(key, []))
        if not rows:
            return None
        result = kwargs.get("result")
        if not isinstance(result, str) or not result:
            return None
        lines = [
            "",
            "",
            "RECEIPT CHECK (delegation-receipts re-checked the work each child claims):",
        ]
        for row in rows:
            kind = row.get("status")
            detail = "; ".join(row.get("problems") or []) or "no detail"
            lines.append(f"- child {row.get('child')} — {kind}: {detail}")
            rerun = row.get("rerun") or {}
            if rerun.get("tail"):
                tail = str(rerun["tail"]).strip().splitlines()[-6:]
                lines.append("  re-run output:")
                lines.extend(f"  | {line}" for line in tail)
        lines.append(
            "Do not report this delegated work as complete until the rejection above is resolved."
        )
        return result + "\n".join(lines)
    except Exception:
        logger.warning("delegation-receipts could not annotate a delegate_task result", exc_info=True)
        return None


def _workspace_root() -> Path:
    """Where claimed paths are resolved: the agent's working directory."""
    for key in ("HERMES_WORKDIR", "PWD"):
        value = os.environ.get(key)
        if value and Path(value).is_dir():
            return Path(value)
    return Path.cwd()


def _setup_cli(parser: Any) -> None:
    parser.add_argument(
        "action", nargs="?", default="status", choices=["status", "ledger"],
        help="status: one line per recent check; ledger: the last rows in full",
    )
    parser.add_argument("--limit", type=int, default=20, help="how many ledger rows to show")


def _handle_cli(args: Any) -> int:
    path = ledger_path()
    if not path.exists():
        print("delegation-receipts: no checks recorded yet.")
        return 0
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    limit = max(1, int(getattr(args, "limit", 20) or 20))
    recent = rows[-limit:]
    if getattr(args, "action", "status") == "ledger":
        for row in recent:
            print(json.dumps(row, indent=2, sort_keys=True))
        return 0
    total = len(rows)
    rejected = sum(1 for row in rows if not row.get("ok"))
    print(f"delegation-receipts: {total} checks, {rejected} rejected, {total - rejected} clean")
    for row in recent:
        mark = "ok  " if row.get("ok") else "FAIL"
        claimed = row.get("claimed_status") or "-"
        detail = "; ".join(row.get("problems") or [])[:120]
        print(f"  {mark} {str(row.get('child', '?'))[:24]:24} claimed={claimed:8} {detail}")
    return 0


def register(ctx: Any) -> None:
    try:
        provider_cls = _provider_class()
    except ModuleNotFoundError as exc:
        if exc.name != "agent.delegation_provider":
            raise
        logger.warning(
            "delegation-receipts needs a Hermes core with pluggable delegation providers "
            "(agent.delegation_provider); update Hermes. The plugin is inactive until then."
        )
        return

    ctx.register_delegation_provider(provider_cls())

    def subagent_stop(**kwargs: Any) -> None:
        return _on_subagent_stop(ctx, **kwargs)

    def transform_tool_result(**kwargs: Any) -> str | None:
        return _on_transform_tool_result(ctx, **kwargs)

    ctx.register_hook("subagent_stop", subagent_stop)
    ctx.register_hook("transform_tool_result", transform_tool_result)
    ctx.register_cli_command(
        name="receipts",
        help="Delegation receipt checks: claimed status versus what was verified",
        setup_fn=_setup_cli,
        handler_fn=_handle_cli,
        description="Show recent subagent receipt checks recorded by delegation-receipts.",
    )

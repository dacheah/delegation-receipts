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
from typing import Any, Mapping, Sequence

from .engine import DEFAULT_ALLOWED_COMMANDS, Verdict, parse_receipt, verify
from .routing import option_parameter, options_for, resolve_route

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

#: Shown above the contract when the user has model routes configured, so the model knows routing
#: still works under this backend.
ROUTE_NOTE = (
    "Each task may set `option` to run that subagent on one of the listed model routes; the receipt "
    "is required and checked either way."
)

#: Verdicts the parent has not seen yet, keyed by ``(parent_session_id, parent_turn_id)``. A
#: verdict is handed only to the exact turn that produced it. An earlier version fell back to
#: draining every session's queue, which delivered one parent's rejections into another parent's
#: delegate_task result and left the owning parent with nothing.
_PENDING: dict[tuple[str, str], list[dict[str, Any]]] = {}


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
        """Verify every child's receipt, and keep the user's model routes working.

        ``delegation.backend`` is exclusive in Hermes, so replacing ``model_map`` with this backend
        would otherwise cost the user their per-task routing. Instead the provider reads the same
        ``delegation.backends.model_map`` block and answers routes exactly as that backend would,
        then requires the receipt on top. Installing this is a drop-in switch, not a downgrade.
        """

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
                "re-checked when the child stops. Keeps any model_map routes you have configured."
            )

        def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
            # Same `option` enum as model_map, so a task can still choose its route per subagent.
            return option_parameter(config)

        def description(self, config: Mapping[str, Any]) -> str:
            if options_for(config):
                return f"{ROUTE_NOTE}\n\n{CONTRACT}"
            return CONTRACT

        def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
            # The route comes from the user's own model_map block; an empty answer means the child
            # inherits the parent's route, which is core's behaviour for an unset delegation
            # section. This plugin changes what is checked, never which model runs.
            return resolve_route(task, config)

    return ReceiptsDelegationProvider


def _workspace_root() -> Path:
    """Where claimed paths are resolved when the child's own workspace is unknown."""
    for key in ("HERMES_WORKDIR", "PWD"):
        value = os.environ.get(key)
        if value and Path(value).is_dir():
            return Path(value)
    return Path.cwd()


def _root_from_history(changed: Sequence[str], history: Sequence[Mapping[str, Any]] | None) -> Path | None:
    """Infer the child's workspace from an absolute write target in its own history.

    The hook payload carries no child cwd. Claimed paths are workspace-relative, so an absolute
    path this child wrote whose tail matches a claimed path tells us the prefix to strip — which is
    the difference between checking the child's tree and checking the daemon's.
    """
    if not history:
        return None
    for call in history:
        if not isinstance(call, Mapping):
            continue
        tool_input = call.get("tool_input")
        targets = tool_input.get("targets") if isinstance(tool_input, Mapping) else None
        if not isinstance(targets, Mapping):
            continue
        for key in ("path", "file_path", "target_path", "destination_path", "new_path"):
            value = targets.get(key)
            if not isinstance(value, str) or not os.path.isabs(value):
                continue
            written = Path(value).parts
            for claimed in changed:
                parts = Path(claimed).parts
                if len(written) > len(parts) and written[-len(parts):] == parts:
                    return Path(*written[: -len(parts)]) if len(written) > len(parts) else None
    return None


def _render(rows: Sequence[Mapping[str, Any]], *, header: str) -> str:
    """The block the parent model reads: what was claimed, what failed, and what to do next."""
    lines = ["", "", header]
    for row in rows:
        detail = "; ".join(row.get("problems") or []) or "no detail"
        lines.append(f"- child {row.get('child')} — {row.get('status')}: {detail}")
        for note in row.get("notes") or []:
            lines.append(f"  note: {note}")
        rerun = row.get("rerun") or {}
        if rerun.get("tail"):
            tail = str(rerun["tail"]).strip().splitlines()[-6:]
            lines.append("  re-run output:")
            lines.extend(f"  | {line}" for line in tail)
    lines.append(
        "Do not report this delegated work as complete until the rejection above is resolved."
    )
    return "\n".join(lines)


def _on_subagent_stop(ctx: Any, **kwargs: Any) -> None:
    """Check one child's receipt the moment it stops."""
    try:
        child = str(kwargs.get("child_session_id") or kwargs.get("child_role") or "child")
        key = (
            str(kwargs.get("parent_session_id") or ""),
            str(kwargs.get("parent_turn_id") or ""),
        )
        summary = kwargs.get("child_summary") or ""
        summary_text = summary if isinstance(summary, str) else str(summary)
        history = kwargs.get("tool_call_history")
        history = history if isinstance(history, list) else None

        receipt = parse_receipt(summary_text) or {}
        changed = [str(item) for item in receipt.get("changed") or []]
        root = _root_from_history(changed, history) or _workspace_root()

        verdict = verify(
            summary_text,
            root=root,
            child=child,
            child_status=kwargs.get("child_status"),
            tool_history=history,
            allowed=_allowed(ctx),
            timeout=_timeout(ctx),
        )
        row = {
            "ts": time.time(),
            "parent_session_id": key[0],
            "parent_turn_id": key[1],
            "child": child,
            "child_role": kwargs.get("child_role"),
            "child_status": kwargs.get("child_status"),
            "duration_ms": kwargs.get("duration_ms"),
            "root": str(root),
            **verdict.to_dict(),
        }
        _record(row)
        if not verdict.ok:
            if key[0] or key[1]:
                _PENDING.setdefault(key, []).append(row)
            logger.info("delegation-receipts rejected a child receipt: %s", verdict.summary_line())
    except Exception:  # a broken verifier must never break delegation
        logger.warning("delegation-receipts failed while checking a child", exc_info=True)
    return None


def _on_transform_tool_result(ctx: Any, **kwargs: Any) -> str | None:
    """Attach rejections to the delegate_task result the parent reads, inline sessions only."""
    try:
        if str(kwargs.get("tool_name") or "") != "delegate_task":
            return None
        args = kwargs.get("args")
        if isinstance(args, Mapping) and str(args.get("action") or "") in {"list", "steer", "stop"}:
            return None  # a status call is not a delivery of work
        key = (
            str(kwargs.get("session_id") or kwargs.get("parent_session_id") or ""),
            str(kwargs.get("turn_id") or kwargs.get("parent_turn_id") or ""),
        )
        rows = _PENDING.get(key)
        if not rows:
            return None
        result = kwargs.get("result")
        if not isinstance(result, str) or not result:
            return None  # keep the queue: nothing was delivered
        _PENDING.pop(key, None)
        return result + _render(
            rows, header="RECEIPT CHECK (delegation-receipts re-checked the work each child claims):"
        )
    except Exception:
        logger.warning("delegation-receipts could not annotate a delegate_task result", exc_info=True)
        return None


def _on_pre_llm_call(ctx: Any, **kwargs: Any) -> dict[str, Any] | None:
    """Deliver verdicts for background children, which never reach transform_tool_result.

    ``delegate_task`` in background mode returns "dispatched" and the child's result re-enters the
    conversation as a later message, so the inline transform never sees it. This hook hands the
    parent the checks it has not been told about, on the next turn, in the same turn's context.
    """
    try:
        session = str(kwargs.get("session_id") or "")
        if not session:
            return None
        keys = [key for key in list(_PENDING) if key[0] == session]
        if not keys:
            return None
        rows: list[dict[str, Any]] = []
        for key in keys:
            rows.extend(_PENDING.pop(key, []))
        if not rows:
            return None
        text = _render(
            rows,
            header="RECEIPT CHECK (checks that finished after your delegate_task returned):",
        ).strip()
        return {"context": text}
    except Exception:
        logger.warning("delegation-receipts could not deliver pending verdicts", exc_info=True)
        return None


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
        head = f"  {mark} {str(row.get('child', '?'))[:24]:24} claimed={claimed:8} "
        detail = "; ".join(row.get("problems") or [])
        if not detail:
            print(head.rstrip())
            continue
        # Wrap rather than truncate: a verdict cut off mid-sentence reads as a bug on camera and to
        # the parent, and the reason is the whole point of the line.
        words, lines, current = detail.split(" "), [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if len(head) + len(trial) <= 150 or not current:
                current = trial
            else:
                lines.append(current)
                current = word
        lines.append(current)
        print(head + lines[0])
        for extra in lines[1:]:
            print(" " * len(head) + extra)
        for note in row.get("notes") or []:
            print(" " * len(head) + f"note: {note}")
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

    def pre_llm_call(**kwargs: Any) -> dict[str, Any] | None:
        return _on_pre_llm_call(ctx, **kwargs)

    ctx.register_hook("subagent_stop", subagent_stop)
    ctx.register_hook("transform_tool_result", transform_tool_result)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_cli_command(
        name="receipts",
        help="Delegation receipt checks: claimed status versus what was verified",
        setup_fn=_setup_cli,
        handler_fn=_handle_cli,
        description="Show recent subagent receipt checks recorded by delegation-receipts.",
    )

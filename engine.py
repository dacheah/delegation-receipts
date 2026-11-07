"""Receipt parsing, verification and bounded re-execution for delegated subagent work.

Pure functions over plain data: this module imports nothing from Hermes, so the whole
verification engine is importable and testable without the host application on the path.
The glue that talks to Hermes lives in ``__init__.py``.

The contract a subagent must satisfy, appended to its reply::

    ```receipt
    status: ok
    changed:
      - src/engine.py
      - tests/test_engine.py
    verified: pytest -q tests/test_engine.py
    left: none
    ```

``status`` is one of ``ok``, ``partial``, ``blocked``, ``failed``. ``changed`` lists
workspace-relative paths the child claims to have written. ``verified`` is a single
command the child claims proves the work. ``left`` is free text or ``none``.
"""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

CONTRACT_VERSION = 1

STATUSES = ("ok", "partial", "blocked", "failed")

#: Executables a receipt may cite. Anything else is refused before it is run: a receipt is
#: written by the child, and the child is the thing we do not trust.
DEFAULT_ALLOWED_COMMANDS = (
    "pytest",
    "python",
    "python3",
    "uv",
    "tox",
    "nox",
    "npm",
    "pnpm",
    "yarn",
    "node",
    "cargo",
    "go",
    "make",
    "just",
    "ruff",
    "mypy",
    "tsc",
    "jest",
    "vitest",
    "rspec",
)

#: Shell syntax is refused outright: the command is executed as an argv list with no shell,
#: so anything that only means something to a shell is either an error or an injection.
_SHELL_METACHARS = re.compile(r"[;&|<>`$(){}*?~\\\n]")

_FENCE = re.compile(r"```[ \t]*(?:receipt|RECEIPT)[ \t]*\r?\n(?P<body>.*?)```", re.S)
_BARE = re.compile(r"(?:^|\n)[ \t]*RECEIPT[ \t]*\r?\n(?P<body>(?:.*\n?)*)\Z", re.S)

_ITEM = re.compile(r"^[ \t]*[-*][ \t]+(?P<item>.+?)[ \t]*$")
_PAIR = re.compile(r"^[ \t]*(?P<key>[A-Za-z_]+)[ \t]*:[ \t]*(?P<value>.*?)[ \t]*$")


@dataclass
class Verdict:
    """The outcome of checking one child's receipt."""

    child: str = ""
    status: str = "missing"  # ok | missing | invalid | unverified | mismatch | error
    receipt_present: bool = False
    claimed_status: str | None = None
    problems: list[str] = field(default_factory=list)
    files_checked: list[dict[str, Any]] = field(default_factory=list)
    command: str | None = None
    corroborated: bool | None = None
    rerun: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["ok"] = self.ok
        return data

    def summary_line(self) -> str:
        """One line for the parent model, or ``""`` when the receipt checked out."""
        if self.ok:
            return ""
        head = f"RECEIPT REJECTED child={self.child or '?'} kind={self.status}"
        detail = "; ".join(self.problems) if self.problems else "no detail"
        return f"{head}: {detail}"


def parse_receipt(text: str) -> dict[str, Any] | None:
    """Return the parsed receipt from a child summary, or ``None`` when absent.

    The last acceptable block wins: a child that reasons about the format before writing a
    real one must not have its example mistaken for its claim.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    bodies = [m.group("body") for m in _FENCE.finditer(text)]
    if not bodies:
        bare = _BARE.search(text)
        if bare:
            bodies = [bare.group("body")]
    if not bodies:
        return None
    return _parse_body(bodies[-1])


def _parse_body(body: str) -> dict[str, Any] | None:
    receipt: dict[str, Any] = {"changed": []}
    seen_key = False
    list_mode = False
    for raw in body.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        item = _ITEM.match(line)
        if item and list_mode:
            value = item.group("item").strip()
            if value.lower() not in ("none", "-"):
                receipt["changed"].append(value)
            continue
        pair = _PAIR.match(line)
        if not pair:
            continue
        seen_key = True
        key = pair.group("key").lower()
        value = pair.group("value").strip()
        list_mode = key == "changed"
        if key == "changed":
            inline = [v.strip() for v in value.split(",") if v.strip()]
            if len(inline) == 1 and inline[0].lower() in ("none", "[]", "-"):
                inline = []
            receipt["changed"].extend(inline)
        elif key in ("status", "verified", "left"):
            receipt[key] = value
    if not seen_key:
        return None
    receipt["changed"] = [_clean_path(p) for p in receipt["changed"]]
    return receipt


def _clean_path(path: str) -> str:
    path = path.strip().strip("`'\"").strip()
    if path.startswith("./"):
        path = path[2:]
    return path


def _is_safe_relative(path: str) -> bool:
    if not path or path.startswith("/") or path.startswith("~"):
        return False
    if re.match(r"^[A-Za-z]:", path):
        return False
    return ".." not in Path(path).parts


def sha256_of(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def check_files(paths: Sequence[str], root: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], list[str]]:
    """Check each claimed path exists under ``root``. Returns (rows, problems)."""
    base = Path(root)
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for raw in paths:
        row: dict[str, Any] = {"path": raw, "exists": False}
        if not _is_safe_relative(raw):
            row["problem"] = "not a workspace-relative path"
            problems.append(f"changed path escapes the workspace: {raw}")
            rows.append(row)
            continue
        target = base / raw
        try:
            exists = target.is_file()
        except OSError:
            exists = False
        row["exists"] = exists
        if exists:
            row["bytes"] = target.stat().st_size
            row["sha256"] = sha256_of(target)
        else:
            row["problem"] = "missing"
            problems.append(f"claimed file does not exist: {raw}")
        rows.append(row)
    return rows, problems


def command_problem(command: str, allowed: Sequence[str] = DEFAULT_ALLOWED_COMMANDS) -> str | None:
    """Return why ``command`` may not be run, or ``None`` when it is acceptable."""
    if not command or not command.strip():
        return "no command given"
    if _SHELL_METACHARS.search(command):
        return "command contains shell syntax and is refused"
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return f"command does not parse: {exc}"
    if not argv:
        return "no command given"
    head = os.path.basename(argv[0])
    if head not in set(allowed):
        return f"executable not on the allow-list: {head}"
    return None


def is_corroborated(command: str, tool_history: Sequence[Mapping[str, Any]] | None) -> bool | None:
    """Did the child actually attempt this command?

    ``subagent_stop`` hands us a redacted history: tool names, argument keys and a ``targets``
    mapping — never the raw argument text. So corroboration is deliberately generous: the
    command's executable must appear in a terminal-ish tool call. ``None`` means the host gave
    us no history at all, which is *unknown*, never "yes".
    """
    if tool_history is None:
        return None
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    if not argv:
        return False
    head = os.path.basename(argv[0])
    terminal_tools = {"terminal", "execute_code", "bash", "shell", "run_command"}
    for call in tool_history:
        if not isinstance(call, Mapping):
            continue
        name = str(call.get("tool_name") or "").lower()
        if name not in terminal_tools:
            continue
        blob = repr(call.get("tool_input") or {})
        if head in blob:
            return True
    return False


def run_command(
    command: str,
    cwd: str | os.PathLike[str],
    *,
    timeout: float = 120.0,
    env_allow: Sequence[str] = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV", "PYTHONPATH"),
) -> dict[str, Any]:
    """Run an allow-listed command as an argv list, no shell, bounded by ``timeout``."""
    argv = shlex.split(command)
    env = {k: os.environ[k] for k in env_allow if k in os.environ}
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"ran": True, "timed_out": True, "returncode": None, "tail": f"timed out after {timeout}s"}
    except (OSError, ValueError) as exc:
        return {"ran": False, "timed_out": False, "returncode": None, "tail": str(exc)}
    output = ((proc.stdout or "") + (proc.stderr or "")).strip()
    tail = output[-1500:] if len(output) > 1500 else output
    return {"ran": True, "timed_out": False, "returncode": proc.returncode, "tail": tail}


def verify(
    text: str,
    *,
    root: str | os.PathLike[str],
    child: str = "",
    tool_history: Sequence[Mapping[str, Any]] | None = None,
    allowed: Sequence[str] = DEFAULT_ALLOWED_COMMANDS,
    runner: Callable[..., dict[str, Any]] | None = None,
    timeout: float = 120.0,
    require_files: bool = True,
    require_corroboration: bool = True,
) -> Verdict:
    """Check one child's claimed work against the world.

    A verdict is ``ok`` only when the receipt parses, the status is ``ok``, every claimed path
    exists, and the cited command is allow-listed, corroborated by the tool history, and exits
    zero when re-run. Nothing is inferred from prose.
    """
    verdict = Verdict(child=child)
    receipt = parse_receipt(text)
    if receipt is None:
        verdict.status = "missing"
        verdict.problems.append(
            "no receipt block: the reply must end with a ```receipt block naming status, changed, verified"
        )
        return verdict

    verdict.receipt_present = True
    claimed = str(receipt.get("status") or "").strip().lower()
    verdict.claimed_status = claimed or None
    if claimed not in STATUSES:
        verdict.status = "invalid"
        verdict.problems.append(f"status is not one of {', '.join(STATUSES)}: {claimed or 'absent'}")
        return verdict

    if claimed != "ok":
        verdict.status = "mismatch"
        verdict.problems.append(f"child reported status {claimed}, not ok")
        return verdict

    if require_files:
        rows, problems = check_files(receipt.get("changed") or [], root)
        verdict.files_checked = rows
        verdict.problems.extend(problems)

    command = (receipt.get("verified") or "").strip()
    verdict.command = command or None
    blocker = command_problem(command, allowed)
    if blocker:
        verdict.problems.append(blocker)
        verdict.status = "unverified"
        return verdict

    if require_corroboration:
        verdict.corroborated = is_corroborated(command, tool_history)
        if verdict.corroborated is False:
            verdict.problems.append("the cited command was never attempted by this child")
            verdict.status = "unverified"
            return verdict

    run = (runner or run_command)(command, root, timeout=timeout)
    verdict.rerun = run
    if not run.get("ran"):
        verdict.problems.append(f"could not re-run the cited command: {str(run.get('tail', ''))[:200]}")
        verdict.status = "unverified"
    elif run.get("timed_out"):
        verdict.problems.append(f"the cited command did not finish within {timeout}s")
        verdict.status = "unverified"
    elif run.get("returncode") != 0:
        verdict.problems.append(f"the cited command failed on re-run (exit {run.get('returncode')})")
        verdict.status = "unverified"

    if verdict.problems:
        if verdict.status == "ok":
            verdict.status = "invalid"
        return verdict

    verdict.status = "ok"
    return verdict

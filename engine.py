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

import collections
import hashlib
import os
import re
import shlex
import signal
import subprocess
import threading
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

#: An allow-listed interpreter can still be asked to evaluate a string instead of running a file.
#: ``python3 -c pass`` exits zero and proves nothing, and it is the cheapest way to fake a receipt,
#: so inline evaluation is refused: a proof must name a file or drive a test runner.
INLINE_EVAL_HEADS = frozenset({"python", "python3", "node"})
INLINE_EVAL_FLAGS = frozenset({"-c", "-e", "--eval", "--exec", "-"})

#: A claimed file larger than this is hashed by size only, so a receipt cannot make the host read a
#: multi-gigabyte artefact while it holds the turn.
HASH_SIZE_CAP = 64 * 1024 * 1024

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
    notes: list[str] = field(default_factory=list)
    files_checked: list[dict[str, Any]] = field(default_factory=list)
    command: str | None = None
    left: str | None = None
    child_status: str | None = None
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
    """Hash a claimed artefact, capped at :data:`HASH_SIZE_CAP` so a huge file cannot stall a turn."""
    try:
        digest = hashlib.sha256()
        remaining = HASH_SIZE_CAP
        with path.open("rb") as handle:
            while remaining > 0:
                chunk = handle.read(min(1 << 20, remaining))
                if not chunk:
                    break
                digest.update(chunk)
                remaining -= len(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _inside(path: Path, base: Path) -> bool:
    """Whether ``path`` resolves to somewhere under ``base``, following symlinks both ways."""
    try:
        real = path.resolve()
        root = base.resolve()
    except (OSError, RuntimeError):
        return False
    return real == root or root in real.parents


def _command_references_existing(command: str, base: str | os.PathLike[str]) -> bool:
    """Whether any argument of ``command`` names a file that exists.

    This is the vacuity guard's other half: a read-only task cannot claim a changed file, so it is
    allowed to prove itself by naming the artefact it checked instead.
    """
    try:
        argv = shlex.split(command)
    except ValueError:
        return False
    root = Path(base)
    for token in argv[1:]:
        candidate = token.split("=", 1)[-1] if "=" in token else token
        candidate = candidate.strip().strip("'\"")
        if not candidate or candidate.startswith("-"):
            continue
        try:
            if (root / candidate).is_file():
                return True
        except (OSError, ValueError):
            continue
    return False


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
        # A path can be relative and still leave the workspace through a symlink, so the textual
        # check above is not the boundary: the resolved path is.
        if exists and not _inside(target, base):
            row["problem"] = "resolves outside the workspace"
            problems.append(f"claimed path resolves outside the workspace: {raw}")
            rows.append(row)
            continue
        row["exists"] = exists
        if exists:
            row["bytes"] = target.stat().st_size
            row["sha256"] = sha256_of(target)
        else:
            row["problem"] = "missing"
            problems.append(f"claimed file does not exist: {raw}")
        rows.append(row)
    return rows, problems


#: Interpreters and wrappers whose whole purpose is to run a different program, so an allow-listed
#: name would otherwise smuggle an arbitrary one: `env python3 -c ...`, `find . -exec sh -c`.
DEPUTY_HEADS = frozenset({
    "env", "xargs", "find", "timeout", "nice", "nohup", "sudo", "su", "doas", "script",
    "sh", "bash", "zsh", "dash", "fish", "eval", "exec", "command", "systemd-run", "watch",
    "parallel", "setsid", "chroot", "unshare", "busybox",
})

#: Argument flags that hand execution to a named program.
DEPUTY_FLAGS = frozenset({"-exec", "-execdir", "-ok", "--eval", "--exec"})


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
    # A path is not a name. `./pytest` and `/tmp/pytest` both basename to an allowed executable, so
    # the first version let a child drop its own script in the workspace and "prove" itself with it.
    if "/" in argv[0] or "\\" in argv[0]:
        return f"executable must be a bare name resolved from PATH, not a path: {argv[0]}"
    head = os.path.basename(argv[0])
    if head not in set(allowed):
        return f"executable not on the allow-list: {head}"
    if head in DEPUTY_HEADS:
        return f"{head} runs another program and is refused: cite the program itself"
    if head in INLINE_EVAL_HEADS and any(token in INLINE_EVAL_FLAGS for token in argv[1:]):
        return "inline evaluation proves nothing and is refused: cite a file or a test command"
    if any(token in DEPUTY_FLAGS for token in argv[1:]):
        return "command hands control to another program and is refused"
    return None


#: Tools whose call names the file it writes. Core keeps path-style targets in the hook history,
#: which is the only attribution evidence a plugin actually receives.
WRITE_TOOLS = frozenset({
    "write_file", "patch", "edit_file", "apply_patch", "create_file", "append_file",
    "replace_in_file", "str_replace", "multiedit", "notebook_edit", "edit",
})

#: Tools that run something. Core strips the command text from the hook history and keeps only
#: ``cwd``-style targets, so the strongest attribution available is "this child made a run-style
#: call carrying a command or code argument".
RUN_TOOLS = frozenset({"terminal", "execute_code", "bash", "shell", "run_command", "process"})

#: Statuses that mean the child finished its turn rather than being cut short.
COMPLETED_STATUSES = frozenset({"completed", "complete", "ok", "success", "succeeded", "done"})


def _argument_keys(call: Mapping[str, Any]) -> list[str]:
    tool_input = call.get("tool_input")
    keys = tool_input.get("argument_keys") if isinstance(tool_input, Mapping) else None
    return [str(key).lower() for key in keys] if isinstance(keys, list) else []


def _targets(call: Mapping[str, Any]) -> Mapping[str, Any]:
    tool_input = call.get("tool_input")
    targets = tool_input.get("targets") if isinstance(tool_input, Mapping) else None
    return targets if isinstance(targets, Mapping) else {}


def was_attempted(tool_history: Sequence[Mapping[str, Any]] | None) -> bool | None:
    """Did this child run anything at all?

    The hook history never carries command text: core's target-key set is
    ``{cwd, path, url, ...}`` and deliberately excludes ``command``. An earlier version of this
    engine searched the history for the executable name, which could never match a real payload,
    and so rejected every honest child. The evidence below is what the host genuinely provides.
    ``None`` means no history was supplied at all, which is unknown, never yes.
    """
    if tool_history is None:
        return None
    for call in tool_history:
        if not isinstance(call, Mapping):
            continue
        if str(call.get("tool_name") or "").lower() not in RUN_TOOLS:
            continue
        keys = _argument_keys(call)
        if not keys or {"command", "cmd", "code"} & set(keys):
            return True
    return False


def write_targets(tool_history: Sequence[Mapping[str, Any]] | None) -> list[str]:
    """Every path this child's write-style tool calls named, as the host recorded them."""
    found: list[str] = []
    if not tool_history:
        return found
    for call in tool_history:
        if not isinstance(call, Mapping):
            continue
        if str(call.get("tool_name") or "").lower() not in WRITE_TOOLS:
            continue
        targets = _targets(call)
        for key in ("path", "file_path", "target_path", "destination_path", "new_path", "dst"):
            value = targets.get(key)
            if isinstance(value, str) and value.strip():
                found.append(_norm(value))
    return found


def _norm(value: str) -> str:
    text = str(value).strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    return text.rstrip("/")


def _same_path(claimed: str, observed: str) -> bool:
    left, right = _norm(claimed), _norm(observed)
    return bool(left and right) and (left == right or left.endswith("/" + right) or right.endswith("/" + left))


def _attribution(
    rows: Sequence[Mapping[str, Any]], tool_history: Sequence[Mapping[str, Any]] | None
) -> tuple[list[str], str | None]:
    """Compare the files a child claims with the writes its own history records.

    Returns (notes, fatal). A note is a soft observation — the child may have written through a
    script, which leaves no path target. The fatal case is narrow on purpose: the child made
    writes, and not one of them is a file it claims.
    """
    notes: list[str] = []
    observed = write_targets(tool_history)
    claimed = [str(row.get("path")) for row in rows if row.get("exists")]
    if tool_history is None or not observed or not claimed:
        return notes, None
    unmatched = [path for path in claimed if not any(_same_path(path, seen) for seen in observed)]
    for row in rows:
        if row.get("exists"):
            row["written"] = not any(_same_path(str(row.get("path")), seen) for seen in observed)
    if unmatched and len(unmatched) == len(claimed):
        return notes, "none of the claimed files was written by this child: " + ", ".join(unmatched[:4])
    if unmatched:
        notes.append("claimed but not among this child's writes: " + ", ".join(unmatched[:4]))
    unclaimed = [seen for seen in observed if not any(_same_path(claimed_path, seen) for claimed_path in claimed)]
    if unclaimed:
        notes.append("written but not claimed: " + ", ".join(unclaimed[:4]))
    return notes, None


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill the whole process group, so a grandchild cannot outlive the check."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def run_command(
    command: str,
    cwd: str | os.PathLike[str],
    *,
    timeout: float = 120.0,
    env_allow: Sequence[str] = ("PATH", "HOME", "LANG", "LC_ALL", "TZ", "VIRTUAL_ENV", "PYTHONPATH"),
    keep_lines: int = 400,
    line_cap: int = 8192,
) -> dict[str, Any]:
    """Run an allow-listed command as an argv list: no shell, own process group, bounded output.

    A child chooses this command, so it must not be able to decide what this process costs: the
    output is drained through a bounded deque (the first version buffered everything and let a
    verbose run take the host's RSS with it), and the timeout kills the group rather than the
    direct child, so no `sleep` survives a check.
    """
    try:
        argv = shlex.split(command)
    except ValueError as exc:
        return {"ran": False, "timed_out": False, "returncode": None, "tail": str(exc)}
    env = {key: os.environ[key] for key in env_allow if key in os.environ}
    try:
        proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            env=env,
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except (OSError, ValueError) as exc:
        return {"ran": False, "timed_out": False, "returncode": None, "tail": str(exc)}

    tail: collections.deque[str] = collections.deque(maxlen=keep_lines)
    total = 0

    def _drain() -> None:
        nonlocal total
        stream = proc.stdout
        if stream is None:
            return
        for line in stream:
            total += len(line)
            tail.append(line[:line_cap])

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    timed_out = threading.Event()

    def _expire() -> None:
        timed_out.set()
        _kill_group(proc)

    timer = threading.Timer(timeout, _expire)
    timer.start()
    try:
        returncode = proc.wait()
    finally:
        timer.cancel()
        reader.join(timeout=2.0)
        if proc.stdout is not None:
            proc.stdout.close()

    joined = "".join(tail)
    return {
        "ran": True,
        "timed_out": timed_out.is_set(),
        "returncode": None if timed_out.is_set() else returncode,
        "tail": joined[-1500:],
        "output_bytes": total,
    }


def verify(
    text: str,
    *,
    root: str | os.PathLike[str],
    child: str = "",
    child_status: str | None = None,
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
    verdict = Verdict(child=child, child_status=str(child_status) if child_status else None)
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

    # A child that was interrupted, failed or was stopped did not finish: its claim of "ok" is not
    # a lie so much as an artefact of a cut-short run, and the parent must not read it as success.
    if verdict.child_status and verdict.child_status.strip().lower() not in COMPLETED_STATUSES:
        verdict.status = "unverified"
        verdict.problems.append(f"the child ended as {verdict.child_status}, not completed")
        return verdict

    # "ok" with work still outstanding is a contradiction in the receipt itself.
    left = str(receipt.get("left") or "").strip()
    verdict.left = left or None
    if left and left.strip().lower() not in {"none", "-", "nil", "nothing", "n/a"}:
        verdict.status = "mismatch"
        verdict.problems.append(f"status ok but work is outstanding: {left}")
        return verdict

    changed = [str(item) for item in (receipt.get("changed") or [])]
    command = (receipt.get("verified") or "").strip()
    verdict.command = command or None

    if require_files:
        rows, problems = check_files(changed, root)
        verdict.files_checked = rows
        verdict.problems.extend(problems)
        # A receipt must prove *something*. With no artefact on disk and no file named by the cited
        # command, "status: ok" asserts nothing at all: `changed: []` with a passing no-op would
        # otherwise be recorded as verified work.
        if not any(row.get("exists") for row in rows) and not _command_references_existing(command, root):
            verdict.status = "unverified"
            verdict.problems.append(
                "nothing was proven: no claimed file exists and the cited command names no "
                "workspace file"
            )
            return verdict
        # Attribute the claim to the writes this child actually made.
        notes, fatal = _attribution(rows, tool_history)
        verdict.notes.extend(notes)
        if fatal:
            verdict.status = "unverified"
            verdict.problems.append(fatal)
            return verdict

    blocker = command_problem(command, allowed)
    if blocker:
        verdict.status = "unverified"
        verdict.problems.append(blocker)
        return verdict

    if require_corroboration:
        verdict.corroborated = was_attempted(tool_history)
        if verdict.corroborated is False:
            verdict.status = "unverified"
            verdict.problems.append("this child made no run-style tool call, so the cited command was never attempted")
            return verdict
        if verdict.corroborated is None:
            verdict.status = "unverified"
            verdict.problems.append(
                "the host supplied no tool history for this child, so the cited command cannot be "
                "attributed to it"
            )
            return verdict

    run = (runner or run_command)(command, root, timeout=timeout)
    verdict.rerun = run
    if not run.get("ran"):
        verdict.status = "unverified"
        verdict.problems.append(f"could not re-run the cited command: {str(run.get('tail', ''))[:200]}")
    elif run.get("timed_out"):
        verdict.status = "unverified"
        verdict.problems.append(f"the cited command did not finish within {timeout}s")
    elif run.get("returncode") != 0:
        verdict.status = "unverified"
        verdict.problems.append(f"the cited command failed on re-run (exit {run.get('returncode')})")

    if verdict.problems:
        if verdict.status == "ok":
            verdict.status = "invalid"
        return verdict

    verdict.status = "ok"
    return verdict

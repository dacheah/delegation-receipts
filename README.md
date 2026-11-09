# Delegation Receipts

Hermes can hand work to a subagent. It cannot check the answer. A child comes back saying
"all 14 tests pass, three files changed" and the parent — having no filesystem evidence of its
own — repeats it as fact. Every serious delegation failure looks the same from outside: a
plausible sentence that was never true.

This plugin makes the claim machine-checkable and then checks it.

```
parent ──delegate_task──▶ child ──"status: ok, changed: [a.py], verified: pytest -q"──┐
   ▲                                                                                  │
   │        ┌──────────────── subagent_stop ────────────────┐                         │
   │        │─── files exist? · corroborated? · re-run? ────│                         │
   │        └─────────────────── verdict ───────────────────┘                         │
   └──────────────────────────────────────────────────────────────────────────────────┘
```

## What it does

1. **Imposes a contract.** With `delegation.backend: receipts`, the `delegate_task` description
   tells every child to end its reply with a receipt. Core caches the description as part of the
   prompt prefix, so the requirement is deterministic and costs no extra turn.
2. **Checks it when the child stops.** The `subagent_stop` hook receives the child's summary, its
   status and a redacted tool history. The receipt is parsed, every claimed path is resolved and
   hashed, and each claim is **matched against the write targets that child's own history
   records** — so a child cannot take credit for a file that exists because someone else wrote it.
   The cited command is then checked against a closed set of fixed check forms. Only on an install
   that turns host re-run on — it is **off by default** — is it then executed, with no shell, in
   its own process group, with output and time bounded.
3. **Tells the parent.** A non-ok verdict is appended to the `delegate_task` result by
   `transform_tool_result`, with the failing output attached. A child dispatched in the background
   returns `dispatched` before it finishes, so its verdict is delivered on the parent's next turn
   by `pre_llm_call` instead. Either way the parent reads the rejection rather than the child's own
   account of itself.
4. **Keeps the record.** Every check lands in an append-only ledger, readable with
   `hermes receipts`.

```receipt
status: ok
changed:
  - sources/parser.py
  - tests/test_parser.py
verified: pytest -q tests/test_parser.py
left: none
```

`status` is `ok`, `partial`, `blocked` or `failed`. A child that reports `partial` is not penalised
for honesty — the verdict records it as `mismatch` and carries its `left` text through. Only
`status: ok` is checked to destruction.

## Install

```bash
hermes plugins install dacheah/delegation-receipts
hermes config set delegation.backend receipts
```

Then start a new session (the backend is read at session start). `hermes receipts` shows the
ledger; `hermes receipts ledger` prints the last rows in full.

Without the backend set, the hooks still run and still check receipts, but no child is *told* to
write one — so almost everything comes back `missing`.

## What is checked

| Check | Failure verdict |
|---|---|
| A receipt block exists and parses | `missing` |
| `status` is one of the four | `invalid` |
| `status: ok` carries no outstanding `left` | `mismatch` |
| The child finished, rather than being interrupted or stopped | `unverified` |
| Every `changed` path is workspace-relative and exists | `invalid` |
| The claim matches a file this child's own history records writing | `unverified` |
| Something is proven at all: a real artefact, or a command naming one | `unverified` |
| The cited command is a fixed check form, bare-named, shell-free, not a launcher | `unverified` |
| The child made a run-style tool call at all | `unverified` |
| The cited command exits zero when re-run — only when host re-run is on | `unverified` |
| Host re-run is off, so the cited command is inspected but never executed | `unverified`, with the reason attached |

The closed set of check forms is `pytest …`, `uv run --no-sync pytest …`, `cargo test`,
`go test`, `npm test`, `pnpm test`, `yarn test`, each with a small closed flag set
(`-q`/`-x` for pytest, `--lib`/`--tests`/`--bins` for cargo, `-count=1` for go). There is no
override: the shapes are the boundary, and `uv run` without `--no-sync` is refused because a sync
would execute a build backend the child could have authored minutes earlier.

Execution is a separate decision from inspection. `host_rerun` defaults to `false`:

```bash
hermes config set plugins.entries.delegation-receipts.settings.host_rerun true
```

With it off, a clean-looking receipt reads `unverified` rather than `ok`, because the plugin can
prove the claim is consistent with the filesystem and the child's own history but cannot prove the
command passes. With it on, the timeout defaults to 120 seconds (`verify_timeout_seconds`).

## What it deliberately does not do

- **It cannot veto a child.** The `subagent_stop` hook observes; it cannot cancel, rewrite or
  re-run the child. The verdict reaches the parent as text, and the parent decides. Making
  delegation refusal-able needs a core seam that does not exist yet.
- **Corroboration is argument-level, not textual.** Core redacts the hook payload to argument keys
  and a fixed set of `targets` — and that set excludes `command`, so the command text never reaches
  a plugin. "Did the child attempt this" is therefore answered from a run-style call carrying a
  command or code argument, and the executable is proved by the re-run. A child that ran
  `pytest tests/` and cited `pytest -q other/` is corroborated and then caught by the re-run.
- **Attribution is a matching heuristic.** Claimed paths and recorded write targets are compared by
  normalised suffix, because a child's cwd is not in the payload. A child that writes through a
  script leaves no path target, so its claim is reported as unattributed rather than rejected.
- **One command per receipt.** A receipt proves one thing. Multi-claim verification would need a
  list and a per-item verdict, which is scope this version does not take on.
- **Prose is not policed.** The receipt is checked; the paragraphs around it are not. A child can
  still write nonsense above a correct receipt.

## Disclosure

- **When host re-run is on, the cited command is executed with your privileges** in the agent's
  working directory. It is off by default, precisely because a `subagent_stop` re-run never passes
  through Hermes' dangerous-command approval and does not use the child's own terminal backend.
  Once enabled, the
  command comes from the subagent, which is the party being checked, so it is restricted to a
  fixed check form by bare name, shell-free, argv-split, refused if it is a launcher (`env`,
  `xargs`, `find`, `sh`) or an inline evaluation (`python3 -c`, `node -e`), time-bounded (120 s
  default) and given a filtered
  environment (`PATH`, `HOME`, `LANG`, `LC_ALL`, `TZ`, `VIRTUAL_ENV`, `PYTHONPATH`). Output is
  drained through a bounded buffer so a verbose run cannot exhaust memory, and the timeout kills
  the whole process group.
- **The re-run is not a sandbox.** `pytest` executes the workspace's `conftest.py`, and `npm test`
  runs the project's own scripts, so re-running a cited command is choosing to execute code from
  the tree the child worked in. The allow-list bounds *which* programs run, not what they contain.
- The child's reply text and the paths it claims are read locally and never transmitted; the
  plugin makes no network calls and holds no credentials.
- Verdicts are stored locally in `${HERMES_HOME}/plugin-data/delegation-receipts/ledger.jsonl`,
  which follows the active profile.
- The check does not call a model, so nothing is billed twice for it.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install pytest
.venv/bin/python -m pytest        # 17 tests
hermes plugins validate .         # catalogue checks
hermes plugins doctor .           # discovery, manifest, registration
```

The engine (`engine.py`) imports nothing from Hermes, so it is testable standalone. The glue
(`__init__.py`) is the only part that touches host APIs.

MIT licensed.

## Compatibility

Needs a Hermes core with pluggable delegation providers (`agent.delegation_provider`), which is
`requires_hermes: ">=0.22.5"`. On an older core the plugin logs one warning and registers its
hooks only, so it stays inert rather than breaking a turn.

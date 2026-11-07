# Delegation Receipts

Hermes can hand work to a subagent. It cannot check the answer. A child comes back saying
"all 14 tests pass, three files changed" and the parent — having no filesystem evidence of its
own — repeats it as fact. Every serious delegation failure looks the same from outside: a
plausible sentence that was never true.

This plugin makes the claim machine-checkable and then checks it.

```
parent ──delegate_task──▶ child ──"status: ok, changed: [a.py], verified: pytest -q"──┐
   ▲                                                                                  │
   │        ┌──────────────── subagent_stop ─────────────────┐                        │
   └────────┤ parse · files exist? · command run? · re-run it ├────────────────────────┘
            └──────────────── verdict ────────────────────────┘
```

## What it does

1. **Imposes a contract.** With `delegation.backend: receipts`, the `delegate_task` description
   tells every child to end its reply with a receipt. Core caches the description as part of the
   prompt prefix, so the requirement is deterministic and costs no extra turn.
2. **Checks it when the child stops.** The `subagent_stop` hook receives the child's summary, its
   status and a redacted tool history. The receipt is parsed, every claimed path is resolved and
   hashed, the cited command is checked against an allow-list, corroborated against the tool
   history, then **re-run** with no shell under a timeout.
3. **Tells the parent.** Non-ok verdicts are appended to the `delegate_task` result by the
   `transform_tool_result` hook, with the failing output attached, so the parent reads the
   rejection instead of the child's own account of itself.
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
| Every `changed` path is workspace-relative and exists | `invalid` |
| The cited command is allow-listed, shell-free | `unverified` |
| The child actually attempted that command | `unverified` |
| The command exits zero when re-run | `unverified` |

The allow-list is `pytest`, `python`, `python3`, `uv`, `tox`, `nox`, `npm`, `pnpm`, `yarn`, `node`,
`cargo`, `go`, `make`, `just`, `ruff`, `mypy`, `tsc`, `jest`, `vitest`, `rspec`. Override it with
`hermes config set plugins.entries.delegation-receipts.allowed_commands '[...]'`. The timeout
defaults to 120 seconds; `verify_timeout_seconds` changes it.

## What it deliberately does not do

- **It cannot veto a child.** The `subagent_stop` hook observes; it cannot cancel, rewrite or
  re-run the child. The verdict reaches the parent as text, and the parent decides. Making
  delegation refusal-able needs a core seam that does not exist yet.
- **Corroboration is coarse.** The tool history handed to the hook is redacted — argument keys and
  `targets` only — so "did the child attempt this command" is answered from the executable name
  appearing in a terminal-style call. A child that ran `pytest tests/` and cited
  `pytest -q other/` is corroborated and then caught by the re-run instead.
- **One command per receipt.** A receipt proves one thing. Multi-claim verification would need a
  list and a per-item verdict, which is scope this version does not take on.
- **Prose is not policed.** The receipt is checked; the paragraphs around it are not. A child can
  still write nonsense above a correct receipt.

## Disclosure

- The cited command is executed **with your privileges** in the agent's working directory. The
  command comes from the subagent, which is the party being checked, so it is allow-listed,
  shell-free, argv-split, time-bounded (120 s default) and given a filtered environment (`PATH`,
  `HOME`, `LANG`, `LC_ALL`, `TZ`, `VIRTUAL_ENV`, `PYTHONPATH`).
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

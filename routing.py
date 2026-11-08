"""Model-route composition, so the receipts backend is a superset of the routes you already have.

``delegation.backend`` is exclusive in Hermes: exactly one provider answers "which route does this
task take". Installing a verification backend must therefore not cost the user their per-task model
routing, so this module re-implements the documented ``model_map`` settings schema inside the
receipts provider. With it, this plugin is a drop-in replacement:

```yaml
delegation:
  backend: receipts             # was: model_map
  backends:
    model_map:                  # unchanged — the same block keeps working
      options:
        - name: fast
          description: Mechanical work.
          provider: deepseek
          model: deepseek-flash
        - name: deep
          description: Hard reasoning.
          provider: anthropic
          model: claude-sonnet-5
      default: fast
```

The semantics mirror the official ``model_map`` backend shipped by Nous Research
(NousResearch/hermes-official-plugins, MIT). This is a reimplementation rather than an import: the
provider contract is JSON-only, and depending on another plugin's private module would break the
day either side moved. Divergence from the documented schema is a bug here, not a feature.

Pure functions over plain data — no Hermes import, no I/O — so routing is testable standalone.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

#: Where the compatible settings block lives. Written by the user, read by us.
SETTINGS_PATH = "delegation.backends.model_map"

#: Per-task parameter name core will expose on ``delegate_task`` when options are configured.
PARAMETER = "option"

#: Entry keys copied into the route, with the same meaning as the matching core route fields.
ROUTE_FIELDS = (
    "provider",
    "model",
    "base_url",
    "api_key",
    "api_mode",
    "reasoning_effort",
    "service_tier",
)

_warned: set[str] = set()


def _warn_once(message: str, *args: Any) -> None:
    text = message % args
    if text not in _warned:
        _warned.add(text)
        logger.warning(text)


def _text(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    return " ".join(value.split()) if isinstance(value, str) else ""


def _entry_problem(entry: Any) -> str:
    if not isinstance(entry, Mapping):
        return "is not a mapping"
    if not _text(entry, "name") or not _text(entry, "description"):
        return "needs a name and a description"
    if not _text(entry, "model"):
        return "needs a model"
    if not (_text(entry, "provider") or _text(entry, "base_url")):
        return "needs a provider or a base_url"
    return ""


def options_for(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """``{name: {"description": str, "route": {...}}}`` from ``delegation.backends.model_map``.

    Malformed entries and repeated names are skipped with one warning each, so a typo narrows the
    enum rather than breaking delegation.
    """
    backends = config.get("backends") if isinstance(config, Mapping) else None
    settings = backends.get("model_map") if isinstance(backends, Mapping) else None
    if not isinstance(settings, Mapping):
        return {}
    raw = settings.get("options")
    if raw is None:
        return {}
    if not isinstance(raw, list):
        _warn_once(
            "%s.options must be a list of {name, description, provider/base_url, model} entries; "
            "ignoring it.",
            SETTINGS_PATH,
        )
        return {}

    options: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(raw):
        problem = _entry_problem(entry)
        if problem:
            _warn_once("%s.options[%d] %s; skipped.", SETTINGS_PATH, index, problem)
            continue
        name = _text(entry, "name")
        if name in options:
            _warn_once("%s.options[%d] repeats option %r; skipped.", SETTINGS_PATH, index, name)
            continue
        route = {
            key: entry[key].strip()
            for key in ROUTE_FIELDS
            if isinstance(entry.get(key), str) and entry[key].strip()
        }
        if isinstance(entry.get("fallback_providers"), list):
            route["fallback_providers"] = list(entry["fallback_providers"])
        options[name] = {"description": _text(entry, "description"), "route": route}
    return options


def option_parameter(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The ``delegate_task`` schema addition that keeps the model able to choose a route."""
    options = options_for(config)
    if not options:
        return {}
    lines = [
        "Which configured model runs this subagent. Omit for the default subagent route.",
        *(f"{name}: {option['description']}" for name, option in options.items()),
    ]
    return {
        PARAMETER: {
            "type": "string",
            "enum": list(options),
            "description": "\n".join(lines),
        }
    }


def resolve_route(task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """The route for one task: the chosen option, else the configured default, else the parent's.

    An empty dict means "inherit the parent's route", which is what core does for an unset
    ``delegation`` section. A chosen value outside the enum has already been refused by core; a
    ``default`` naming a missing option is warned once and ignored rather than failing the task.
    """
    options = options_for(config)
    if not options:
        return {}
    chosen = task.get(PARAMETER)
    if not isinstance(chosen, str) or not chosen.strip():
        backends = config.get("backends") if isinstance(config, Mapping) else None
        settings = backends.get("model_map") if isinstance(backends, Mapping) else None
        default = settings.get("default") if isinstance(settings, Mapping) else None
        chosen = default.strip() if isinstance(default, str) else ""
        if chosen and chosen not in options:
            _warn_once(
                "%s.default %r is not an option in %s.options; tasks without an option inherit "
                "the parent's route.",
                SETTINGS_PATH,
                chosen,
                SETTINGS_PATH,
            )
            chosen = ""
    if not chosen or chosen not in options:
        return {}
    return {**options[chosen]["route"], "label": chosen}

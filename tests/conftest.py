"""Test bootstrap.

The plugin's entry module is the repo-root ``__init__.py``, which uses a relative import, so
pytest would otherwise collect the root as a test ``Package`` and import it standalone during
setup — every test then errors with "attempted relative import with no known parent package".

Two fixes, both modelled on the official ``advanced-delegation`` plugin:

* load the plugin once under a stable synthetic package name, so ``from <PACKAGE> import engine``
  resolves the relative import inside ``__init__.py``;
* register a global plugin that collects the repo root as a plain ``Dir`` rather than a
  ``Package``. It must be a global plugin, not a conftest hook, because the root directory's
  collector is created through its *parent's* hook proxy, which never sees this conftest.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PACKAGE = "delegation_receipts_plugin"
_ROOT = Path(__file__).resolve().parents[1]

# ``python -m pytest`` from the plugin root puts that root on sys.path, where the plugin's own
# modules would shadow anything the host imports under the same name.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]


def _load_plugin_package():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]
    spec = importlib.util.spec_from_file_location(
        PACKAGE, _ROOT / "__init__.py", submodule_search_locations=[str(_ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


_load_plugin_package()


@pytest.fixture()
def plugin_root() -> Path:
    return _ROOT


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Every test gets its own HOME and HERMES_HOME, so the ledger can never touch the real one."""
    home = tmp_path / "os-home"
    (home / ".hermes").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))


class _RepoRootIsNotATestPackage:
    """Collect the plugin root as a directory, never as a test package."""

    @pytest.hookimpl(tryfirst=True)
    def pytest_collect_directory(self, path, parent):
        if Path(path).resolve() == _ROOT:
            return pytest.Dir.from_parent(parent, path=path)
        return None


def pytest_configure(config):
    if not config.pluginmanager.has_plugin("delegation-receipts-repo-root"):
        config.pluginmanager.register(_RepoRootIsNotATestPackage(), "delegation-receipts-repo-root")

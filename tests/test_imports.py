"""Smoke test: every module of roux imports (catches undefined names and broken imports at module level)."""

import importlib
import pkgutil

import pytest

import roux


def _module_names():
    errors = []
    names = [
        m.name
        for m in pkgutil.walk_packages(
            roux.__path__, prefix="roux.", onerror=errors.append
        )
    ]
    assert not errors, errors
    return sorted(names)


## roux.workflow.utils and roux.workflow.df warn with a DeprecationWarning on import: expected.
@pytest.mark.parametrize("name", _module_names())
def test_import_module(name):
    importlib.import_module(name)

"""`run_tasks`: skipping or selecting the targets with `ROUX_SKIP_TASKS`/`ROUX_ONLY_TASKS` (no notebook is run)."""

from pathlib import Path

import pytest

from roux.workflow import task
from roux.workflow.task import _skipped_by_env

PATH = "/home/u/project/nbs_01_x/043_results_data.ipynb"
OTHER = "/home/u/project/nbs_01_x/044_plots.ipynb"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("ROUX_SKIP_TASKS", raising=False)
    monkeypatch.delenv("ROUX_ONLY_TASKS", raising=False)


def test_unset_not_skipped():
    assert _skipped_by_env(PATH) is None


@pytest.mark.parametrize(
    "pattern",
    [
        "043_results_data",  # stem
        "043_results_data.ipynb",  # name
        "*/nbs_01_x/043_*",  # path glob
        " foo , 043_results_data ,",  # whitespace and empty patterns
    ],
)
def test_skip(monkeypatch, pattern):
    monkeypatch.setenv("ROUX_SKIP_TASKS", pattern)
    assert _skipped_by_env(PATH) == "ROUX_SKIP_TASKS"
    assert _skipped_by_env(OTHER) is None


def test_only(monkeypatch):
    monkeypatch.setenv("ROUX_ONLY_TASKS", "044_*, 045_x")
    assert _skipped_by_env(PATH) == "ROUX_ONLY_TASKS"
    assert _skipped_by_env(OTHER) is None


def test_skip_wins_over_only(monkeypatch):
    monkeypatch.setenv("ROUX_ONLY_TASKS", "04*")
    monkeypatch.setenv("ROUX_SKIP_TASKS", "043_*")
    assert _skipped_by_env(PATH) == "ROUX_SKIP_TASKS"
    assert _skipped_by_env(OTHER) is None


def test_empty_no_effect(monkeypatch):
    monkeypatch.setenv("ROUX_SKIP_TASKS", "")
    monkeypatch.setenv("ROUX_ONLY_TASKS", " , ")
    assert _skipped_by_env(PATH) is None
    assert _skipped_by_env(OTHER) is None


def test_run_tasks_returns_early(monkeypatch, tmp_path):
    script = tmp_path / "043_results_data.py"
    script.write_text("print('ran')\n")
    cache = tmp_path / "cache"

    def _fail(*args, **kws):
        raise AssertionError("not expected to be called")

    ## no setup, no runner
    monkeypatch.setattr(task, "infer_runner", _fail)
    monkeypatch.setattr(task, "pre_params", _fail)
    monkeypatch.setattr(task, "run_tasks_nb", _fail)
    warnings = []
    monkeypatch.setattr(
        task.logging, "warning", lambda msg, *a, **k: warnings.append(msg)
    )

    monkeypatch.setenv("ROUX_SKIP_TASKS", "043_results_data")
    res = task.run_tasks(
        script.as_posix(),
        params=[{"output_path": (tmp_path / "out.txt").as_posix()}],
        runner="py",
        cache_dir_path=cache.as_posix(),
    )
    assert res is None
    assert not cache.exists()
    assert len(warnings) == 1
    assert "ROUX_SKIP_TASKS" in warnings[0] and Path(script).name in warnings[0]

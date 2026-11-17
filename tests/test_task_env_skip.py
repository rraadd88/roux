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


## (ROUX_SKIP_TASKS, ROUX_ONLY_TASKS) -> the variable that skips PATH and OTHER (None: not skipped)
@pytest.mark.parametrize(
    "skip,only,expected",
    [
        (None, None, (None, None)),
        ("043_results_data", None, ("ROUX_SKIP_TASKS", None)),  # stem
        ("043_results_data.ipynb", None, ("ROUX_SKIP_TASKS", None)),  # name
        ("*/nbs_01_x/043_*", None, ("ROUX_SKIP_TASKS", None)),  # path glob
        (
            " foo , 043_results_data ,",
            None,
            ("ROUX_SKIP_TASKS", None),
        ),  # whitespace, empty patterns
        (None, "044_*, 045_x", ("ROUX_ONLY_TASKS", None)),
        ("043_*", "04*", ("ROUX_SKIP_TASKS", None)),  # skip wins over only
        ("", " , ", (None, None)),  # empty: no effect
    ],
)
def test_skipped_by_env(monkeypatch, skip, only, expected):
    for k, v in [("ROUX_SKIP_TASKS", skip), ("ROUX_ONLY_TASKS", only)]:
        if v is not None:
            monkeypatch.setenv(k, v)
    assert (_skipped_by_env(PATH), _skipped_by_env(OTHER)) == expected


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

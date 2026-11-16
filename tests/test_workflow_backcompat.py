"""Back-compat checks for the `roux.workflow` reorganisation.

The names stay importable from their current modules, and `to_pkg` (dry run) lists the same plan.
"""

import importlib
import json
import subprocess
import sys

import pytest

# g: names importable from each module before the reorganisation (master 5b8c3c4): the public
# g: names defined in the module, plus the re-exports and the shared private helpers that other code imports.
OLD_PATHS = {
    "roux.workflow.cfgs": [
        "get_cfgs",
        "resolve_cfg",
        "read_config",
        "read_cfg",
        "read_sub_configs",
        "read_sub_configs_rly",
        "read_metadata",
        "get_cfg_run",
        "to_cfg_run",
        "to_cfg_run_arc",
    ],
    "roux.workflow.df": ["exclude_items"],
    "roux.workflow.function": [
        "import_from_file",
        "call_with_kws",
        "get_quoted_path",
        "get_path",
        "remove_dirs_from_outputs",
        "get_ios",
        "get_name",
        "get_step",
        "to_task",
        "get_global_imports",
    ],
    "roux.workflow.io": [
        "clear_variables",
        "clear_dataframes",
        "to_py",
        "RUFF_SELECT",
        "check_py",
        "post_code",
        "to_mod",
        "to_py_path",
        "to_scr_path",
        "to_scr",
        "to_src",
        "to_pkg",
        "to_scrs",
        "replacestar_ruff",
        "replacestar",
        "set_outputs",
        "check_for_exit",
        # g: re-exports
        "read_config",
        "read_metadata",
        "read_cli_pm",
        "get_source_path",
        # g: shared private helpers (to_pkg, to_scrs, tests)
        "_pkg_plan",
        "_to_scrs",
        "_to_scr_cleaned",
        "_to_scr_running_py",
        "_PKG_DROP_CODE_LINES",
        "_PKG_DROP_HEADERS",
    ],
    "roux.workflow.log": ["print_parameters", "test_params"],
    "roux.workflow.monitor": ["plot_workflow_log"],
    "roux.workflow.nb": [
        "to_nb_cells",
        "read_nb",
        "get_lines",
        "read_nb_md",
        "to_info",
        "to_replaced_nb",
        "to_filtered_nb",
        "to_filter_nbby_patterns",
        "to_clear_unused_cells",
        "to_clear_outputs",
        "to_filtered_outputs",
        "to_clean_nb",
        "valid_post_task_deps",
        "to_html",
        "to_nb_kernel",
    ],
    "roux.workflow.pms": [
        "extract_pms",
        "expand_pms",
        "read_cli_pm",
        "read_pms",
        "to_params",
        "validate_pms",
        "pre_params",
    ],
    "roux.workflow.task": [
        "logging",
        "pre_task",
        "run_task_nb",
        "apply_run_task_nb",
        "run_tasks_nb",
        "get_sq",
        "get_jobsn",
        "is_q_empty",
        "submit_job",
        "SLURMJob",
        "has_slurm",
        "check_tasks",
        "load_module",
        "infer_runner",
        "to_sbatch_script",
        "parse_time",
        "get_tried_job_keys",
        "feed_jobs",
        "pre_cfg_run",
        "post_tasks",
        "run_tasks",
        "run_cells",
    ],
    "roux.workflow.utils": ["to_workflow", "create_workflow_report"],
    "roux.workflow.version": ["git_commit", "get_tag", "get_commit_message"],
    "roux.workflow.workflow": ["get_scripts", "to_scripts"],
}

# g: modules that fail at import today; strict, so the mark has to go once they are fixed
KNOWN_BROKEN = {}


@pytest.mark.parametrize(
    "module_name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(
                strict=True, raises=NameError, reason=KNOWN_BROKEN[name]
            ),
        )
        if name in KNOWN_BROKEN
        else name
        for name in sorted(OLD_PATHS)
    ],
)
def test_old_import_paths(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in OLD_PATHS[module_name] if not hasattr(module, name)]
    assert not missing, f"no longer importable from {module_name}: {missing}"


# g: the modules split out of io.py and task.py (reorganisation, phase 2), importable on their own
NEW_PATHS = {
    "roux.workflow.scr": [
        "to_py",
        "check_py",
        "to_scr",
        "to_src",
        "to_scr_path",
        "replacestar",
        "_to_scrs",
    ],
    "roux.workflow._worktree": ["_git", "_git_check_ignore", "_plan", "_pkg_plan"],
    "roux.workflow.pkg": ["to_pkg", "to_scrs"],
    "roux.workflow.slurm": [
        "SLURMJob",
        "submit_job",
        "feed_jobs",
        "to_sbatch_script",
        "get_sq",
        "is_q_empty",
    ],
}


@pytest.mark.parametrize("module_name", sorted(NEW_PATHS))
def test_new_modules_import_first(module_name):
    # g: a fresh interpreter, so the module is the first roux.workflow module imported (no import cycle)
    code = f"import {module_name} as m; missing = [n for n in {NEW_PATHS[module_name]!r} if not hasattr(m, n)]; assert not missing, missing"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_moved_names_are_the_same_objects():
    import roux.workflow.io as io
    import roux.workflow.pkg as pkg
    import roux.workflow.scr as scr
    import roux.workflow.slurm as slurm
    import roux.workflow.task as task

    assert io.to_scr is scr.to_scr and io.to_src is scr.to_src
    assert io.to_pkg is pkg.to_pkg and io.to_scrs is pkg.to_scrs
    assert task.submit_job is slurm.submit_job and task.SLURMJob is slurm.SLURMJob


## to_pkg, dry run
NOTEBOOK = json.dumps(
    {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "id": "a1",
                "metadata": {},
                "outputs": [],
                "source": ["x = 1"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
)
FILES = {
    ".gitignore": "scr_*/\n",
    "README.md": "# proj\n",
    "to_pkg.cfg": "input_path: '.'\n",
    "modules_nbs/m.py": "x = 1\n",
    "notebooks_x/01_a.ipynb": NOTEBOOK,
    "notebooks_x/run.ipynb": NOTEBOOK,
}


def _git(repo, *args):
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.com",
            *args,
        ],
        check=True,
        capture_output=True,
    )


@pytest.fixture
def notebooks_repo(tmp_path):
    repo = tmp_path / "proj"
    for rel, text in FILES.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    return repo


def test_to_pkg_dry_run(notebooks_repo, tmp_path, capsys):
    from roux.workflow.io import to_pkg

    output_path = tmp_path / "proj-pkg.yaml"
    assert (
        to_pkg(
            input_path=str(notebooks_repo),
            output_path=str(output_path),
            ignore="to_pkg.cfg\nmodules_nbs/*",
            dry_run="True",
        )
        is None
    )
    ## nothing is written: no worktree, no outputs
    assert not output_path.with_suffix("").exists()
    assert not output_path.exists()
    lines = [
        s
        for s in capsys.readouterr().out.splitlines()
        if s.split(" ")[0] in ["convert", "copy", "ignored"]
    ]
    assert lines == [
        "convert ('notebooks_x/01_a.ipynb', 'scr_x/01_a.py')",
        "convert ('notebooks_x/run.ipynb', 'scr_x/run.py')",
        "copy .gitignore",
        "copy README.md",
        "ignored modules_nbs/m.py",
        "ignored to_pkg.cfg",
    ]


## deprecations (reorganisation, phase 3): the old names still work, with a DeprecationWarning
def test_to_src_warns_and_keeps_the_cli_name(tmp_path, monkeypatch):
    from roux.workflow.scr import to_scr, to_src

    # g: argh names the CLI command after __name__, so `roux to-scr` is unchanged
    assert to_src.__name__ == "to_scr" and to_src.__wrapped__ is to_scr
    monkeypatch.chdir(tmp_path)
    with pytest.warns(DeprecationWarning, match="to_src"):
        try:
            to_src("missing.ipynb", "missing.py")
        except Exception:
            pass


def test_pkg_plan_warns_and_equals_plan(notebooks_repo):
    from roux.workflow._worktree import _pkg_plan, _plan

    with pytest.warns(DeprecationWarning, match="_pkg_plan"):
        old = _pkg_plan(notebooks_repo, "to_pkg.cfg")
    assert old == _plan(notebooks_repo, "to_pkg.cfg")


@pytest.mark.parametrize("module_name", ["roux.workflow.utils", "roux.workflow.df"])
def test_legacy_modules_warn_on_import(module_name):
    code = f"import warnings; warnings.simplefilter('always'); import {module_name}"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert f"{module_name} is deprecated" in res.stderr

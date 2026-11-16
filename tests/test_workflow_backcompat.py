"""Back-compat checks for the `roux.workflow` reorganisation.

The names stay importable from their current modules, and `to_pkg` (dry run) lists the same plan.
"""

import importlib
import json
import subprocess

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

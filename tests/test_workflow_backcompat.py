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
# g: the names moved out of io.py and task.py (reorganisation, phase 2) stay importable from there
OLD_PATHS = {
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
        "read_config",
        "read_metadata",
        "read_cli_pm",
        "get_source_path",
        "_pkg_plan",
        "_to_scrs",
        "_to_scr_cleaned",
        "_to_scr_running_py",
        "_PKG_DROP_CODE_LINES",
        "_PKG_DROP_HEADERS",
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
}

# g: modules that fail at import today; strict, so the mark has to go once they are fixed


@pytest.mark.parametrize("module_name", sorted(OLD_PATHS))
def test_old_import_paths(module_name):
    module = importlib.import_module(module_name)
    missing = [name for name in OLD_PATHS[module_name] if not hasattr(module, name)]
    assert not missing, f"no longer importable from {module_name}: {missing}"


# g: the modules split out of io.py and task.py (reorganisation, phase 2), importable on their own


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
def test_deprecated_names_warn(notebooks_repo):
    from roux.workflow._worktree import _pkg_plan, _plan
    from roux.workflow.scr import to_scr, to_src

    # g: argh names the CLI command after __name__, so `roux to-scr` is unchanged
    assert to_src.__name__ == "to_scr" and to_src.__wrapped__ is to_scr
    with pytest.warns(DeprecationWarning, match="_pkg_plan"):
        assert _pkg_plan(notebooks_repo, "to_pkg.cfg") == _plan(
            notebooks_repo, "to_pkg.cfg"
        )
    code = "import warnings; warnings.simplefilter('always'); import roux.workflow.utils, roux.workflow.df"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "roux.workflow.utils is deprecated" in res.stderr
    assert "roux.workflow.df is deprecated" in res.stderr


## pub branch names (TODO 24): warn only
@pytest.mark.parametrize("pub_branch,valid", [("main-pub", True), ("release", False)])
def test_check_pub_branch(pub_branch, valid, caplog):
    from roux.workflow.pkg import _check_pub_branch

    with caplog.at_level("WARNING"):
        assert _check_pub_branch(pub_branch, "main") is valid
    assert ("not one of" in caplog.text) is (not valid)


## the pub worktree (to_pub plan, S1): `_check_pub_worktree`, shared by to_pkg and to_pub
def test_check_pub_worktree(notebooks_repo, tmp_path, capsys):
    from roux.workflow._worktree import _check_pub_worktree

    src, out = notebooks_repo, tmp_path / "proj-pub"
    ## missing, not created: the command is printed
    with pytest.raises(ValueError, match="to be made by the user"):
        _check_pub_worktree(src, out)
    assert "worktree add --orphan -b pub" in capsys.readouterr().out
    assert not out.exists()

    ## created: orphan, no commits yet
    dev_branch, dev_sha, pub_branch = _check_pub_worktree(src, out, create=True)
    assert pub_branch == "pub" and not dev_sha.endswith("+")
    assert (out / ".git").is_file()

    ## the source is itself a worktree (`.git` is a file)
    _git(src, "worktree", "add", "-q", "-b", "dev2", str(tmp_path / "proj-dev2"))
    assert _check_pub_worktree(tmp_path / "proj-dev2", out)[::2] == ("dev2", "pub")

    ## not a worktree
    (tmp_path / "plain").mkdir()
    with pytest.raises(AssertionError, match="not a worktree"):
        _check_pub_worktree(src, tmp_path / "plain")

    ## a worktree of another repository
    other = tmp_path / "other"
    other.mkdir()
    _git(other, "init", "-q")
    _git(other, "commit", "-q", "--allow-empty", "-m", "init")
    _git(
        other,
        "worktree",
        "add",
        "-q",
        "--orphan",
        "-b",
        "pub",
        str(tmp_path / "other-pub"),
    )
    with pytest.raises(AssertionError, match="is not a worktree of"):
        _check_pub_worktree(src, tmp_path / "other-pub")

    ## shared history: not orphan
    _git(src, "worktree", "add", "-q", "-b", "shared", str(tmp_path / "proj-shared"))
    with pytest.raises(ValueError, match="shares history"):
        _check_pub_worktree(src, tmp_path / "proj-shared")
    assert (
        _check_pub_worktree(src, tmp_path / "proj-shared", orphan=False)[2] == "shared"
    )

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync --all-extras                      # dev setup: all extras + the dev group (.venv/)
pytest -vv tests/test_jupytext_input.py   # fast unit tests
pytest -vv tests/test_examples.py::test_roux_lib_df   # one example notebook
uv run tox                                # py3.10-3.13 (CI: py3.12-3.13), extras=all, `pytest -vv --maxfail=3`
ruff check roux && ruff format roux       # lint/format; ruff is pinned (see below)
roux --help                               # CLI (needs the `workflow` extra)
```

- Most tests are `tests/test_examples.py`: each test executes a notebook in `examples/` with papermill
  (kernel `python3`, outputs to `examples/outputs/`, git-ignored). Run pytest from the repository root.
  A failing test points at a cell of that notebook; the examples double as the documentation.
- `ruff==0.15.10` is pinned in three places that must stay in sync: `[dependency-groups] dev` and
  `[tool.ruff] required-version` in `pyproject.toml`, and `RUFF` in `.github/workflows/format.yaml`.
  Newer versions strip `# noqa` (RUF100) and can delete re-exported imports.
- F401 is ignored and unfixable: roux re-exports names (e.g. `roux/global_imports.py`) marked `# noqa`.
  Keep the "unused" imports. E402/E712 findings are known and accepted.
- CI: `build.yml` (tox matrix + `uv build`/twine check) on push/PR; `format.yaml` (ruff autofix commit)
  is manual only. `build.yml` runs only what the changed files need (its `changes` job): `roux/`, `tests/`,
  `examples/`, `pyproject.toml`, `uv.lock`, `tox.ini`, `MANIFEST.in` or `build.yml` -> both jobs;
  `README.md` alone -> the package check (it is the PyPI long description); anything else (e.g. `.claude/`)
  -> neither. Manual runs (`workflow_dispatch`) run both.

## Architecture

Subpackages: `roux.lib` (dataframes, I/O, strings, sys/paths), `roux.stat`, `roux.viz` (matplotlib/seaborn),
`roux.vizi` (interactive, altair), `roux.workflow` (configs, notebooks -> scripts, task running).
Heavy dependencies are optional extras (`data`, `stat`, `viz`, `fast`, `workflow`), so imports of
them are usually local to the functions.

### The `.rd` dataframe accessor
`roux/lib/__init__.py` registers `pd.DataFrame.rd` (and the older `.rs` for Series). Functions
decorated with `@to_rd` (`roux.lib.df`, `df_apply`, `dfs`, ...) are attached as methods, with the
dataframe as the first argument: `def f(df, ...)` becomes `df.rd.f(...)`. Every seaborn `*plot`
function is attached too (`df.rd.scatterplot(...)` returns the df, for piping). Importing the module
that defines a function is what attaches it.

### Notebook-first workflow (`roux.workflow`)
Code is developed in notebooks, typically with `from roux.global_imports import *`. The tooling then
turns them into CLI scripts:
- `roux replacestar`: the star import -> explicit imports.
- `to_scr` (`roux/workflow/scr.py`; alias `to_src`): notebook (`.ipynb` or jupytext percent `.py`) -> a
  script with a `run(<params>)` function and an argh CLI. The parameters come from the cell tagged
  `parameters`, which must start with a `## param...` comment (`split_by_pms` asserts this).
  It writes scratch files to `.to_src/` in the current directory.
- `to_pkg` (`roux/workflow/pkg.py`): a notebooks repository (`notebooks_X/`) -> a package in a git worktree on an orphan branch
  (`scr_X/`), configured by a params file (`to_pkg.cfg`). `to_scrs` does the same conversion into the
  source repository's git-ignored `scr_*/`. They share `_pkg_plan` (`_worktree.py`) and `_to_scrs` (`scr.py`); keep them shared.
- Layers: `pkg` -> `_worktree` -> `scr` -> `nb`, `function`. `io.py` keeps the notebook runtime helpers (`set_outputs`,
  `to_mod`, ...) and re-exports everything that moved out (`from roux.workflow.io import to_pkg` keeps working);
  `tests/test_workflow_backcompat.py` pins the old import paths. Imports between these modules that would form a
  cycle (e.g. `pkg` -> `io.set_outputs`) are local to the functions.
- `run_tasks` (`roux/workflow/task.py`): runs a script/notebook over a list of params. The runner is
  `py`, `bash` or `slurm` (inferred if not given), and `.ipynb` runs go through papermill (`run_tasks_nb`).
  Params and configs are handled by `roux/workflow/pms.py` (`pre_params`) and `cfgs.py` (omegaconf). The SLURM
  part (`SLURMJob`, `submit_job`, `feed_jobs`, `to_sbatch_script`, ...) is in `slurm.py`, re-exported by `task.py`.

### CLI (`roux/run.py`)
The `roux` entry point is `roux.run:parser.dispatch`, an `argh.ArghParser`. To add a command, import the
function and add it to `parser.add_commands`. argh names commands after `__name__` with `_` -> `-`, so an
alias (`to_src = to_scr`) is registered as `to-scr`, and two functions with the same `__name__` conflict.
CLI arguments arrive as strings: bool/list params default to strings like `"False"` and are parsed with
`read_cli_pm` (`roux/workflow/pms.py`).

## Style
Local imports inside functions, `##`/`###` section comments, `logging` (often `roux.lib.log.Logger`),
and docstrings with `Parameters:` / `Returns:` / `Usage:` / `Examples:` sections (used for the CLI help).

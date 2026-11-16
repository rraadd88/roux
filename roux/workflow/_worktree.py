"""
Git helpers and the plan of `to_pkg`/`to_scrs`: which files of a repository are converted, copied or ignored.
"""

import logging

from roux.workflow.scr import to_scr_path


def _git(repo, com, **kws):
    """Run git in a repo, returns the stripped stdout."""
    import shlex

    from roux.lib.sys import run_com

    return run_com(
        f"git -C {shlex.quote(str(repo))} {com}", verbose=False, **kws
    ).strip()


def _git_check_ignore(repo, paths):
    """Paths that the .gitignore of a repo ignores (by the patterns only i.e. also if tracked)."""
    import subprocess

    if len(paths) == 0:
        return set()
    res = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--no-index", "-z", "--stdin"],
        input="\0".join(paths) + "\0",
        capture_output=True,
        text=True,
    )
    assert res.returncode in [0, 1], res.stderr
    return {p for p in res.stdout.split("\0") if p != ""}


def _plan(src, ignore=""):
    """
    Plan of `to_pkg`: the tracked files of a repository to convert (notebooks), copy or ignore.

    Returns:
        plan (dict), patterns (list): `ignore` as fnmatch patterns to `<repository dir name>/<path>`.
    """
    import fnmatch

    patterns = [
        s.strip()
        for s in ignore.split("\n")
        if s.strip() != "" and not s.strip().startswith("#")
    ]
    patterns = [f"*/{s}" for s in patterns]

    def is_ignored(name):
        return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)

    plan = {"convert": [], "copy": [], "ignored": []}
    for rel in _git(src, "ls-files -z").split("\0"):
        if rel == "" or not (src / rel).is_file():
            continue
        if is_ignored(f"{src.resolve().name}/{rel}"):
            plan["ignored"].append(rel)
        elif rel.endswith(".ipynb"):
            plan["convert"].append((rel, to_scr_path(rel)))
        else:
            plan["copy"].append(rel)
    logging.info({k: len(v) for k, v in plan.items()})
    return plan, patterns


def _pkg_plan(src, ignore=""):
    """Deprecated alias of `_plan`."""
    import warnings

    warnings.warn(
        "_pkg_plan is deprecated; use roux.workflow._worktree._plan",
        DeprecationWarning,
        stacklevel=2,
    )
    return _plan(src, ignore)

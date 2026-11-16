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


def _check_pub_worktree(src, out, create=False, pub_branch="pub", orphan=True):
    """
    Check (or make) the pub worktree of a repository: `out` is a worktree of `src` (also if `src` is a worktree).

    Parameters:
        src (str): path to the source repository (the dev branch).
        out (str): path to the pub worktree.
        create (bool): make a missing worktree, on a new orphan branch `pub_branch` (`to_pkg`); else print the
            command and raise (the worktree is made by the user, `to_pub`).
        pub_branch (str): name of the branch for a new worktree.
        orphan (bool): the worktree shares no history with `src` (no merge-base, if it has commits).

    Returns:
        dev_branch (str), dev_sha (str; `+` if the tracked files are modified), pub_branch (str).

    Requirements:
        git>=2.42 (`worktree add --orphan`), if `create`.
    """
    import re
    import shlex
    from pathlib import Path

    src, out = Path(src), Path(out)
    ## `.git` is a directory in a repository, a file in a worktree
    assert (src / ".git").exists(), f"not a repository: {src}"
    dev_branch = _git(src, "rev-parse --abbrev-ref HEAD")
    dirty = _git(src, "status --porcelain --untracked-files=no")
    dev_sha = _git(src, "rev-parse --short HEAD") + ("+" if dirty else "")

    if not out.exists():
        com = f"worktree add --orphan -b {pub_branch} {shlex.quote(str(out.resolve()))}"
        if not create:
            print(
                f"## to make the pub worktree (an orphan branch), from {src}:\ngit {com}"
            )
            raise ValueError(f"not a worktree (to be made by the user): {out}")
        version = tuple(int(x) for x in re.findall(r"\d+", _git(src, "--version"))[:2])
        assert version >= (2, 42), f"git>=2.42 is required, found {version}"
        ## orphan branch i.e. no shared history
        _git(src, com)
    assert (out / ".git").is_file(), f"not a worktree: {out}"

    def common_dir(p):
        return Path(
            _git(p, "rev-parse --path-format=absolute --git-common-dir")
        ).resolve()

    assert common_dir(out) == common_dir(src), f"{out} is not a worktree of {src}"
    ## the branch, also without commits (`rev-parse --abbrev-ref HEAD` fails then)
    pub_branch = _git(out, "symbolic-ref --short -q HEAD", returncodes=[0, 1]) or "HEAD"
    if orphan:
        head = _git(out, "rev-parse --verify -q HEAD", returncodes=[0, 1])
        if head != "" and _git(src, f"merge-base HEAD {head}", returncodes=[0, 1]):
            raise ValueError(f"{out} shares history with {dev_branch} (not orphan)")
    return dev_branch, dev_sha, pub_branch


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

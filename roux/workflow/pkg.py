"""
Notebooks repository to a package (`to_pkg`, in a worktree) or to scripts in the same repository (`to_scrs`).
"""

import logging
import re
from pathlib import Path

from roux.workflow._worktree import _git, _git_check_ignore, _plan
from roux.workflow.pms import read_cli_pm
from roux.workflow.scr import (
    _PKG_DROP_CODE_LINES,
    _PKG_DROP_HEADERS,
    _to_scrs,
)


def to_pkg(
    input_path=None,
    output_path=None,
    kind=None,
    ignore="",
    clean_nbs="True",
    clean_comments="False",
    force_with_pms="True",
    author=None,
    pub_branch="pub",
    gitignore_drop="scr_*/",
    drop_code_lines_containing=None,
    drop_headers_containing=None,
    rm_unused_funcs="False",
    allow_dirty="False",
    dry_run="False",
    no_commit="False",
    force="False",
):
    """
    Notebooks repository to a package, in a worktree of the same repository.

    The package is the output worktree (orphan branch) with the notebooks converted to
    scripts (`notebooks_X/` -> `scr_X/`) and the other files copied. It is only committed
    locally, never pushed.

    Usage:
        On command line, from the root of the repository (branch of the notebooks).

        ## params file, by default `to_pkg.cfg`
        roux to-pkg --input-path to_pkg.cfg

        ## params e.g.
        input_path: '.'
        output_path: '../pkg.yaml' # the worktree is `../pkg/`
        kind: 'code'
        clean_nbs: False # keep the comments
        ignore: |
            to_pkg.cfg
            modules_nbs/*

    Parameters:
        input_path (str): path to the repository; or to the params file(s), if `output_path` is not provided.
        output_path (str): path to the output, `<dir>.yaml`; `<dir>/` is the worktree.
        kind (str): 'code' to clear the outputs of the notebooks.
        ignore (str): patterns (fnmatch to `<repository dir name>/<path>`), one per line (# for comments). The matching files are not converted/copied.
        clean_nbs (bool): clean the notebooks i.e. clear the unused cells and drop the development leftovers (`drop_code_lines_containing`, `drop_headers_containing`).
        clean_comments (bool): drop the excessive (consecutive) comment lines from the scripts, including the header of the notebook. Independent of `clean_nbs`.
        force_with_pms (bool): scripts with the parameters for the `run` notebooks too.
        author (str): author line for the scripts.
        pub_branch (str): name of the branch for a new worktree.
        gitignore_drop (str): lines (comma-separated) of the `.gitignore` not to carry over.
        drop_code_lines_containing (list): patterns of the lines to drop from the notebooks.
        drop_headers_containing (list): patterns of the headers to drop from the notebooks.
        rm_unused_funcs (bool): remove the unused functions from the scripts.
        allow_dirty (bool): allow the modified tracked files in the repository.
        dry_run (bool): only list the actions.
        no_commit (bool): do not commit in the worktree.
        force (bool): rerun even if the output exists.

    Requirements:
        git>=2.42 (`worktree add --orphan`), git user.name and user.email (to commit).
    """
    import shutil

    from roux.lib.io import to_dict

    ## cli
    clean_nbs, clean_comments, force_with_pms, rm_unused_funcs = [
        read_cli_pm(x)
        for x in [clean_nbs, clean_comments, force_with_pms, rm_unused_funcs]
    ]
    allow_dirty, dry_run, no_commit, force = [
        read_cli_pm(x) for x in [allow_dirty, dry_run, no_commit, force]
    ]
    drop_code_lines_containing = read_cli_pm(drop_code_lines_containing)
    drop_headers_containing = read_cli_pm(drop_headers_containing)
    if drop_code_lines_containing is None:
        drop_code_lines_containing = _PKG_DROP_CODE_LINES
    if drop_headers_containing is None:
        drop_headers_containing = _PKG_DROP_HEADERS

    ## params file(s)
    if input_path is None and output_path is None:
        input_path = "to_pkg.cfg"
    if output_path is None:
        from roux.workflow.function import call_with_kws
        from roux.workflow.pms import pre_params

        overrides = {
            k: v
            for k, v in dict(
                author=author,
                allow_dirty=allow_dirty,
                dry_run=dry_run,
                no_commit=no_commit,
            ).items()
            if v
        }
        for kws in pre_params(input_path, force=force):
            call_with_kws(to_pkg, kws, **overrides)
        return

    ## outputs
    from roux.workflow.io import set_outputs

    output_dir_path, _ = set_outputs(output_path=output_path)
    src, out = Path(input_path), Path(output_dir_path)

    ## checks
    assert (src / ".git").is_dir(), f"not a repository: {src}"
    dirty = _git(src, "status --porcelain --untracked-files=no")
    if dirty and not allow_dirty:
        raise ValueError(f"{src} has uncommitted changes (allow_dirty):\n{dirty}")
    if not (dry_run or no_commit):
        for k in ["name", "email"]:
            assert _git(src, f"config user.{k}", returncodes=[0, 1]) != "", (
                f"git config user.{k} is not set"
            )
    dev_branch = _git(src, "rev-parse --abbrev-ref HEAD")
    dev_sha = _git(src, "rev-parse --short HEAD") + ("+" if dirty else "")

    ## worktree
    if out.exists():
        assert (out / ".git").is_file(), f"not a worktree: {out}"
        assert (
            Path(_git(out, "rev-parse --git-common-dir")).resolve()
            == (src / ".git").resolve()
        ), f"{out} is not a worktree of {src}"
        pub_branch = _git(out, "rev-parse --abbrev-ref HEAD")
    elif not dry_run:
        version = tuple(int(x) for x in re.findall(r"\d+", _git(src, "--version"))[:2])
        assert version >= (2, 42), f"git>=2.42 is required, found {version}"
        import shlex

        ## orphan branch i.e. no shared history
        _git(
            src,
            f"worktree add --orphan -b {pub_branch} {shlex.quote(str(out.resolve()))}",
        )
    logging.info(f"{src} [{dev_branch}] -> {out} [{pub_branch}]")

    ## plan
    plan, patterns = _plan(src, ignore)
    if dry_run:
        for k, v in plan.items():
            for x in v:
                print(k, x)
        return

    ## mirror: empty the worktree (but .git)
    for p in out.iterdir():
        if p.name != ".git":
            shutil.rmtree(p) if p.is_dir() else p.unlink()

    ## gitignore
    drop = {s.strip() for s in gitignore_drop.split(",") if s.strip()}
    lines = []
    for s in (src / ".gitignore").read_text().split("\n"):
        if s.strip() in drop:
            ## with the comments attached on the top
            while lines and lines[-1].strip().startswith("#"):
                lines.pop()
            continue
        lines.append(s)
    while lines and lines[-1].strip() == "":
        lines.pop()
    lines += ["", *[s[2:] for s in patterns]]
    (out / ".gitignore").write_text("\n".join(lines) + "\n")

    ## copy; but not the files that the .gitignore ignores (even if tracked in the source)
    copy_paths = [rel for rel in plan["copy"] if rel != ".gitignore"]
    ignored_paths = _git_check_ignore(out, copy_paths)
    if ignored_paths:
        logging.info(f"not copied, ignored by the .gitignore: {sorted(ignored_paths)}")
    for rel in copy_paths:
        if rel not in ignored_paths:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src / rel, out / rel)

    ## Notebooks -> py
    _to_scrs(
        src,
        out,
        plan["convert"],
        rm_unused_funcs=rm_unused_funcs,
        kind=kind,
        clean_nbs=clean_nbs,
        clean_comments=clean_comments,
        force_with_pms=force_with_pms,
        author=author,
        drop_code_lines_containing=drop_code_lines_containing,
        drop_headers_containing=drop_headers_containing,
    )

    ## Version: commit (locally)
    if not no_commit:
        import shlex

        _git(out, "add -A")
        if _git(out, "status --porcelain") != "":
            _git(out, f"commit -m {shlex.quote(f'+ from {dev_branch} @ {dev_sha}')}")
            logging.info(f"committed in {out}")
        else:
            logging.info("no changes to commit")

    ## Outputs
    to_dict({}, output_path)
    return output_path


def to_scrs(
    input_path="to_pkg.cfg",
    kind=None,
    ignore="",
    clean_nbs="True",
    clean_comments="False",
    force_with_pms="True",
    author=None,
    drop_code_lines_containing=None,
    drop_headers_containing=None,
    rm_unused_funcs="False",
    dry_run="False",
):
    """
    Notebooks of a repository to scripts (`notebooks_X/` -> `scr_X/`), in the repository itself.

    The conversion of `to_pkg` (same params file), without the package e.g. to test the scripts in the
    source repository. The `scr_*/` are generated: emptied first, so they must be ignored by git.

    Usage:
        On command line, from the root of the repository.

        ## params of `to_pkg`, by default `to_pkg.cfg`
        roux to-scrs
        roux to-scrs --dry-run True

    Parameters:
        input_path (str): params file of `to_pkg` (its `input_path` is the repository); or the path to the repository.
        kind (str): 'code' to clear the outputs of the notebooks.
        ignore (str): patterns (fnmatch to `<repository dir name>/<path>`), one per line (# for comments). The matching notebooks are not converted.
        clean_nbs (bool): clean the notebooks, as in `to_pkg`.
        clean_comments (bool): drop the excessive (consecutive) comment lines from the scripts.
        force_with_pms (bool): scripts with the parameters for the `run` notebooks too.
        author (str): author line for the scripts.
        drop_code_lines_containing (list): patterns of the lines to drop from the notebooks.
        drop_headers_containing (list): patterns of the headers to drop from the notebooks.
        rm_unused_funcs (bool): remove the unused functions from the scripts.
        dry_run (bool): only list the actions.

        If `input_path` is a params file, the params are read from it (the `to_pkg`-only ones are ignored), but `dry_run`.

    Returns:
        paths (list): of the scripts.
    """
    import inspect
    import shutil

    dry_run = read_cli_pm(dry_run)

    ## params file
    if Path(input_path).is_file():
        from roux.lib.io import read_dict

        kws = dict(read_dict(input_path))
        kws = {
            k: v
            for k, v in kws.items()
            if k in inspect.signature(to_scrs).parameters and k != "dry_run"
        }
        ## the repository, not a params file again
        assert "input_path" in kws and not Path(kws["input_path"]).is_file(), (
            f"input_path of the repository is missing in {input_path}"
        )
        return to_scrs(**kws, dry_run=dry_run)

    ## cli
    clean_nbs, clean_comments, force_with_pms, rm_unused_funcs = [
        read_cli_pm(x)
        for x in [clean_nbs, clean_comments, force_with_pms, rm_unused_funcs]
    ]
    drop_code_lines_containing = read_cli_pm(drop_code_lines_containing)
    drop_headers_containing = read_cli_pm(drop_headers_containing)
    if drop_code_lines_containing is None:
        drop_code_lines_containing = _PKG_DROP_CODE_LINES
    if drop_headers_containing is None:
        drop_headers_containing = _PKG_DROP_HEADERS

    src = Path(input_path)
    assert (src / ".git").is_dir(), f"not a repository: {src}"
    plan, _ = _plan(src, ignore)

    ## generated dirs: of the outputs and the existing ones
    dir_names = {Path(py_rel).parts[0] for _, py_rel in plan["convert"]}
    dir_names |= {p.name for p in src.glob("scr_*") if p.is_dir()}
    not_scr = sorted(s for s in dir_names if not s.startswith("scr_"))
    if not_scr:
        raise ValueError(f"output dirs not starting with scr_: {not_scr}")
    ## generated i.e. ignored by git and not tracked (e.g. in a package worktree)
    paths = [f"{s}/" for s in sorted(dir_names)]
    not_ignored = sorted(set(paths) - _git_check_ignore(src, paths))
    tracked = sorted(
        {
            rel.split("/")[0] + "/"
            for v in plan.values()
            for rel in v
            if isinstance(rel, str) and rel.split("/")[0] in dir_names
        }
    )
    if not_ignored or tracked:
        raise ValueError(
            f"not generated dirs (not ignored by git: {not_ignored}; tracked: {tracked})"
        )
    if dry_run:
        for s in sorted(dir_names):
            print("empty", f"{s}/")
        for nb_rel, py_rel in plan["convert"]:
            print("convert", nb_rel, "->", py_rel)
        return

    for s in dir_names:
        if (src / s).exists():
            shutil.rmtree(src / s)
    _to_scrs(
        src,
        src,
        plan["convert"],
        rm_unused_funcs=rm_unused_funcs,
        kind=kind,
        clean_nbs=clean_nbs,
        clean_comments=clean_comments,
        force_with_pms=force_with_pms,
        author=author,
        drop_code_lines_containing=drop_code_lines_containing,
        drop_headers_containing=drop_headers_containing,
    )
    return [str(src / py_rel) for _, py_rel in plan["convert"]]

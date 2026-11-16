"""
Analysis repository to a paper repository (`to_pub`, in a worktree on an orphan branch).

The paper repository holds, per analysis, the scripts, configs and source data of the figure panels.
"""

import logging
import re
from pathlib import Path

from roux.workflow._worktree import _check_pub_worktree, _git
from roux.workflow.pms import read_cli_pm


def _get_scripts(x: dict, label: str) -> list:
    """
    The script names of a panel, one per plot.

    Parameters:
        x (dict): `{script: (optional), plots: [...]}`; `script` is a name (`<name>_<plot i>` if several plots), a list (one per plot) or missing (the base names of the plot files).
        label (str): of the panel, for the messages.

    Returns:
        names (list): one per plot.
    """
    assert isinstance(x, dict) and isinstance(x.get("plots"), list), (
        f"{label}: expected {{script: (optional), plots: [...]}}, got: {x}"
    )
    ps, s = x["plots"], x.get("script")
    if s is None:
        s = [Path(p).with_suffix("").name for p in ps]
    elif isinstance(s, str):
        s = [s] if len(ps) == 1 else [f"{s}_{i}" for i in range(1, len(ps) + 1)]
    assert len(s) == len(ps), f"{label}: {len(s)} scripts for {len(ps)} plots"
    return s


def read_figs(
    figs: dict,
    tag: str = None,
    tag_pattern: str = None,
    input_path: str = ".",
    check: bool = True,
):
    """
    The plots of the figures, with their ids and the names of their scripts.

    Parameters:
        figs (dict): `{fig: {panel: {script, plots}}}`, or `{fig: {script, plots}}` for a figure without panels (its plots are numbered as panels). See `_get_scripts` for `script`.
        tag (str): the version of the outputs: replaces the matches of `tag_pattern` in the plot paths.
        tag_pattern (str): regular expression of the version in the plot paths e.g. `(?<=outputs/)v\d+(?=/)`; without it, the paths are kept.
        input_path (str): the root of the repository; the plot paths are relative to it.
        check (bool): raise if a plot or its directory (the saved plot inputs) is missing; else the column `exists` marks them.

    Returns:
        df (DataFrame): one row per plot, with the columns
            `fig` (`<fig><panel><plot>` e.g. `1C1`),
            `input path` (the plot path without suffix),
            `script label`, `fig i`, `panel i`, `plot i` (blank if single),
            `plot label` (e.g. `FIG1C plot1`),
            `exists`.

    Usage:
        figs:
            1:
                C:
                    script: counts_paired
                    plots:
                        - outputs/v1/a/dists_counts.pdf
            S2:
                script: counts_per_sample
                plots:
                    - outputs/v1/b/dists_samples
    """
    import pandas as pd

    rows = []
    for figi, panels in figs.items():
        if panels is None:
            continue
        figi = str(figi)
        assert isinstance(panels, dict), f"{figi}: expected a mapping, got: {panels}"
        if "plots" in panels:
            ## a figure without panels: its plots are numbered as panels
            rows += [
                dict(figi=figi, paneli=str(j), ploti="1", plot=p, script=s)
                for j, (p, s) in enumerate(
                    zip(panels["plots"], _get_scripts(panels, figi)), 1
                )
            ]
            continue
        for paneli, x in panels.items():
            if x is None:
                continue
            scripts = _get_scripts(x, f"{figi}{paneli}")
            rows += [
                dict(figi=figi, paneli=str(paneli), ploti=str(j), plot=p, script=s)
                for j, (p, s) in enumerate(zip(x["plots"], scripts), 1)
            ]

    ## unique script names (the files of the paper repository), before anything is read or written
    names = [r["script"] for r in rows]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"script names not unique (set `script` in `figs`): {dups}")

    for r in rows:
        p = r["plot"]
        if tag is not None and tag_pattern is not None:
            p = re.sub(tag_pattern, tag, p)
        p = Path(input_path) / p
        label = f"{r['figi']}{r['paneli']}"
        r["exists"] = p.exists() and p.with_suffix("").exists()
        if not r["exists"]:
            msg = f"not found: {label} {p} (or its directory)"
            if check:
                raise ValueError(msg)
            logging.warning(msg)
        elif not p.with_suffix(".pdf").exists():
            logging.warning(f"pdf not found: {label} {p.with_suffix('.pdf')}")
        r["input path"] = p.with_suffix("").as_posix()

    df = pd.DataFrame(
        [
            {
                "fig": f"{r['figi']}{r['paneli']}{r['ploti']}",
                "input path": r["input path"],
                "script label": r["script"],
                "fig i": r["figi"],
                "panel i": r["paneli"],
                "plot i": r["ploti"],
                "exists": r["exists"],
            }
            for r in rows
        ],
        columns=[
            "fig",
            "input path",
            "script label",
            "fig i",
            "panel i",
            "plot i",
            "exists",
        ],
    )
    if len(df) == 0:
        return df.assign(**{"plot label": []})
    ## ids: blank if single
    df["panel i"] = df.groupby("fig i")["panel i"].transform(
        lambda x: "" if len(x) == 1 else x
    )
    df["plot i"] = df.groupby(["fig i", "panel i"])["plot i"].transform(
        lambda x: "" if len(x) == 1 else x
    )
    ## a figure without panels: the panel numbers are the plot numbers
    df["plot i"] = df.apply(
        lambda x: (
            x["panel i"]
            if (x["panel i"].isdigit() and x["plot i"] == "")
            else x["plot i"]
        ),
        axis=1,
    )
    df["panel i"] = df.apply(
        lambda x: x["panel i"] if (x["panel i"] != x["plot i"]) else "", axis=1
    )
    df["plot label"] = df.apply(
        lambda x: (
            f"FIG{x['fig i']}{x['panel i']}"
            + ("" if x["plot i"] == "" else f" plot{x['plot i']}")
        ),
        axis=1,
    )
    return df


def _get_analyses(ana_to_sis: dict) -> dict:
    """
    The analysis of each figure: the i-th analysis has the main figure i and its SI figures.

    Parameters:
        ana_to_sis (dict): `{analysis name: [SI figure numbers]}`, in the order of the main figures.

    Returns:
        fig_to_ana (dict): `{fig i: analysis name}` e.g. `{'1': 'Analysis A', 'S1': 'Analysis A'}`.
    """
    fig_to_ana = {}
    for i, (name, sis) in enumerate(ana_to_sis.items(), 1):
        for k in [str(i)] + [f"S{s}" for s in (sis or [])]:
            assert k not in fig_to_ana, f"figure {k} in two analyses"
            fig_to_ana[k] = name
    return fig_to_ana


def _pub_plan(df_figs, ana_to_sis: dict, review_path: str):
    """
    The files of the paper repository (and the review files), per plot.

    Parameters:
        df_figs (DataFrame): output of `read_figs`.
        ana_to_sis (dict): see `_get_analyses`.
        review_path (str): the review files in the source repository (e.g. `results/latest`).

    Returns:
        df (DataFrame): `df_figs` with `analysis name`, `analysis label` and the output paths
            (`review path`, `script path`, `config path`, `data path`; the last three relative to `pub_path`).
    """
    from roux.lib.sys import to_path

    fig_to_ana = _get_analyses(ana_to_sis)
    missing = sorted(set(df_figs["fig i"]) - set(fig_to_ana))
    assert not missing, f"figures without an analysis (`ana_to_sis`): {missing}"
    df = df_figs.copy()
    df["analysis name"] = df["fig i"].map(fig_to_ana)
    df["analysis label"] = df["analysis name"].str.lower().apply(to_path)
    df["review path"] = df["fig"].apply(lambda s: f"{review_path}/repo/figs/fig{s}")
    for k, d, ext in [
        ("script", "scripts", "py"),
        ("config", "configs/analyses", "yaml"),
        ("data", "data/analyses", "pqt"),
    ]:
        df[f"{k} path"] = df.apply(
            lambda x: f"{d}/{x['analysis label']}/{x['script label']}.{ext}", axis=1
        )
    return df


def _read_cfg(cfg_path: str) -> dict:
    """
    The params of `to_pub` from a params file: the paths (`*_path` and the sources of `copy`) are relative to its directory (the root of the repository) and are resolved here.
    """
    import os

    from roux.lib.io import read_dict

    root = Path(cfg_path).parent
    kws = read_dict(cfg_path)
    for k, v in kws.items():
        if k.endswith("_path") and isinstance(v, str):
            kws[k] = os.path.normpath(root / v)
    if isinstance(kws.get("copy"), dict):
        kws["copy"] = {os.path.normpath(root / s): d for s, d in kws["copy"].items()}
    return kws


def to_pub(
    input_path=None,
    output_path=None,
    pub_path=None,
    figs=None,
    tag=None,
    tag_pattern=None,
    ana_to_sis=None,
    copy=None,
    gitignore=None,
    cols_desc=None,
    pyproject_drop="tool.uv.sources",
    allow_dirty="False",
    dry_run="False",
    force="False",
):
    """
    Analysis repository to a paper repository, in a worktree of the same repository (orphan branch, made by the user).

    Per analysis, the paper repository holds the scripts (`scripts/<analysis>/<script>.py`), their configs
    (`configs/analyses/<analysis>/<script>.yaml`) and the source data (`data/analyses/<analysis>/<script>.pqt`)
    of the figure panels. Nothing is committed: the git commands are printed.

    Usage:
        On command line, from the root of the repository.

        ## params file, by default `to_pub.cfg`; its paths are relative to its directory
        roux to-pub --input-path to_pub.cfg --dry-run True

        ## params e.g.
        input_path: '.'
        output_path: 'results/latest.yaml' # the review files are in `results/latest/`
        pub_path: '../proj-pub'
        tag: 'v2'
        tag_pattern: '(?<=outputs/)v\d+(?=/)'
        figs: {1: {C: {script: counts, plots: [outputs/v1/a/dists_counts.pdf]}}}
        ana_to_sis: {'Analysis A': [1, 2]}

    Parameters:
        input_path (str): path to the repository; or to the params file, if `output_path` is not provided.
        output_path (str): path to the marker, `<dir>.yaml`; `<dir>/` holds the review files.
        pub_path (str): path to the paper repository, a worktree of the repository on an orphan branch (made by the user: `git worktree add --orphan -b pub <pub_path>`).
        figs (dict): the plots of the figures, see `read_figs`.
        tag (str), tag_pattern (str): the version of the outputs, see `read_figs`.
        ana_to_sis (dict): `{analysis name: [SI figure numbers]}`, in the order of the main figures.
        copy (dict): `{source: destination}` files to copy; the destinations are relative to `pub_path`.
        gitignore (list): lines added to the `.gitignore` of the repository, for the paper repository.
        cols_desc (dict): `{column: description}` for the source data excel.
        pyproject_drop (str): tables (comma-separated) of `pyproject.toml` not to carry over.
        allow_dirty (bool): allow the modified tracked files in the repository.
        dry_run (bool): only list the actions.
        force (bool): remake the outputs that exist.

    Requirements:
        git>=2.42 (`worktree add --orphan`).
    """
    allow_dirty, dry_run, force = [
        read_cli_pm(x) for x in [allow_dirty, dry_run, force]
    ]

    ## params file
    if input_path is None and output_path is None:
        input_path = "to_pub.cfg"
    if output_path is None:
        from roux.workflow.function import call_with_kws

        overrides = {
            k: v
            for k, v in dict(
                allow_dirty=allow_dirty, dry_run=dry_run, force=force
            ).items()
            if v
        }
        return call_with_kws(to_pub, _read_cfg(input_path), **overrides)

    for k, v in dict(pub_path=pub_path, figs=figs, ana_to_sis=ana_to_sis).items():
        assert v is not None, f"`{k}` is required"
    review_path = Path(output_path).with_suffix("").as_posix()
    src, out = Path(input_path), Path(pub_path)

    ## checks
    dirty = _git(src, "status --porcelain --untracked-files=no")
    if dirty and not allow_dirty:
        raise ValueError(f"{src} has uncommitted changes (allow_dirty):\n{dirty}")
    dev_branch, dev_sha, pub_branch = _check_pub_worktree(src, out, create=False)
    logging.info(f"{src} [{dev_branch} @ {dev_sha}] -> {out} [{pub_branch}]")

    ## plan
    df = _pub_plan(
        read_figs(
            figs,
            tag=tag,
            tag_pattern=tag_pattern,
            input_path=input_path,
            check=not dry_run,
        ),
        ana_to_sis=ana_to_sis,
        review_path=review_path,
    )
    if dry_run:
        for _, x in df.iterrows():
            print(
                "panel" if x["exists"] else "missing",
                x["plot label"],
                x["input path"],
                "->",
                x["script path"],
                x["config path"],
                x["data path"],
            )
        for s, d in (copy or {}).items():
            print("copy", s, "->", d)
        for s in ["configs/analyses.yaml", "pyproject.toml", ".gitignore"]:
            print("write", s)
        return

    raise NotImplementedError("to_pub: only `dry_run` for now")

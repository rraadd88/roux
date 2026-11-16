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
        tag_pattern (str): regular expression of the version in the plot paths e.g. `(?<=outputs/)v\\d+(?=/)`; without it, the paths are kept.
        input_path (str): the root of the repository; the plot paths are relative to it.
        check (bool): raise if a plot or its directory (the saved plot inputs) is missing; else the column `exists` marks them.

    Returns:
        df (DataFrame): one row per plot, with the columns
            `fig` (`<fig><panel><plot>`, without the blank ids e.g. `1C`, `1D1`),
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
                "fig": None,
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
    ## the ids of the review files: as the labels e.g. `1C` (FIG1C), `1D1` (FIG1D plot1)
    df["fig"] = df["fig i"] + df["panel i"] + df["plot i"]
    dups = sorted(df.loc[df["fig"].duplicated(), "fig"].unique())
    if dups:
        raise ValueError(f"fig ids not unique: {dups}")
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


def _clean_plot_script(text: str) -> str:
    """
    A plot script, cleaned: without the comments (`#` inside strings kept), the trailing spaces and the consecutive blank lines; a literal `plotp="..."` is replaced by `plotp=__file__`.
    """
    import io
    import tokenize

    tokens = [
        t
        for t in tokenize.generate_tokens(io.StringIO(text).readline)
        if t.type != tokenize.COMMENT
    ]
    text = tokenize.untokenize(tokens)
    text = "\n".join(s.rstrip() for s in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return re.sub(r"plotp\s*=\s*[\"'].*?[\"']\s*,", "plotp=__file__,", text)


def _to_panel(input_path: str, review_dir: str, force: bool = False):
    """
    The review files of a plot saved by `roux.viz.io.to_plot`: `data.pqt`, `config.yaml`, `plot.py` (cleaned) and `stats.yaml` (`ax.stats` of the replotted plot).

    Parameters:
        input_path (str): the plot path without suffix i.e. the directory of the saved plot inputs.
        review_dir (str): the output directory.
        force (bool): remake the files if `data.pqt` exists.

    Returns:
        stats (dict): `ax.stats`, or the saved `stats.yaml` if not remade; None if the data is not tabular (skipped).
    """
    import matplotlib.pyplot as plt

    from roux.lib.io import read_dict, to_dict, to_table
    from roux.viz.io import get_plot_inputs, read_plot

    out = Path(review_dir)
    if (out / "data.pqt").exists() and not force:
        return (
            read_dict(str(out / "stats.yaml")) if (out / "stats.yaml").exists() else {}
        )
    _, data, kws_plot = get_plot_inputs(input_path)
    if "path" in data:
        logging.warning(f"skipped, a column `path` in the data: {input_path}")
        return None
    plt.close("all")
    ax = read_plot(f"{input_path}/plot.py")
    stats = getattr(ax, "stats", None) or {}
    plt.close("all")
    out.mkdir(parents=True, exist_ok=True)
    to_table(data, str(out / "data.pqt"))
    to_dict(kws_plot, str(out / "config.yaml"))
    try:
        to_dict(stats, str(out / "stats.yaml"))
    except Exception as e:
        logging.warning(f"stats not saved ({e}): {input_path}")
    (out / "plot.py").write_text(
        _clean_plot_script(Path(f"{input_path}/plot.py").read_text())
    )
    return stats


def _to_pub_script(p: str, outp: str, src_dir_path: str, inline: list = None) -> str:
    """
    A plot script (`plot.py` of `to_plot`) -> a script of the paper repository.

    Parameters:
        p (str): input path.
        outp (str): output path.
        src_dir_path (str): the root of the source repository, where the packages of `inline` are.
        inline (list): the local packages (top-level names) whose imports are replaced by the source of the imported functions (and the imports of their module), also inside a `try` block (e.g. with `sys.path.append`).

    Returns:
        outp (str): output path.

    Notes:
        The edits for the runner: the lines with `get_plot_inputs` are dropped, `plot_` -> `run`, `kws_plot` -> `config`.
    """
    import ast
    import textwrap

    inline = [inline] if isinstance(inline, str) else (inline or [])
    text = Path(p).read_text()
    lines = text.split("\n")

    def is_local(n):
        return (
            isinstance(n, ast.ImportFrom) and (n.module or "").split(".")[0] in inline
        )

    def get_source(n):
        ## the imported functions and the imports of their module
        mp = Path(src_dir_path) / (n.module.replace(".", "/") + ".py")
        mt = mp.read_text()
        mtree = ast.parse(mt)
        names = [a.name for a in n.names]
        defs = {
            x.name: x
            for x in mtree.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assert all(k in defs for k in names), (mp, names)
        ## not the other module-level names (they are not copied)
        used = {
            x.id for k in names for x in ast.walk(defs[k]) if isinstance(x, ast.Name)
        }
        others = used & (set(defs) - set(names))
        assert not others, (mp, others)
        imps = [
            x
            for x in mtree.body
            if isinstance(x, (ast.Import, ast.ImportFrom)) and not is_local(x)
        ]
        return "\n".join(
            [ast.get_source_segment(mt, x) for x in imps]
            + [f"## from the source: {n.module}"]
            + [ast.get_source_segment(mt, defs[k]) for k in names]
        )

    ## the blocks to replace: (first line, last line, indent, the nodes with the imports)
    blocks = []
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.Try) and any(
            is_local(x) for b in n.body for x in ast.walk(b)
        ):
            nodes = [x for b in n.body for x in ast.walk(b) if is_local(x)]
            blocks.append((n.lineno, n.end_lineno, n.col_offset, nodes))
        elif is_local(n):
            blocks.append((n.lineno, n.end_lineno, n.col_offset, [n]))
    ## the outer blocks only (a `try` contains its imports)
    blocks = [
        b
        for b in blocks
        if not any(o[0] <= b[0] and b[1] <= o[1] and o != b for o in blocks)
    ]
    for start, end, col, nodes in sorted(blocks, key=lambda b: b[0], reverse=True):
        code = textwrap.indent("\n".join(get_source(n) for n in nodes), " " * col)
        lines[start - 1 : end] = code.split("\n")
    text = "\n".join(lines)

    ## the edits for the runner
    text = "\n".join(s for s in text.split("\n") if "get_plot_inputs" not in s)
    text = re.sub(r"def plot_\(", "def run(", text).replace("kws_plot", "config")
    Path(outp).parent.mkdir(parents=True, exist_ok=True)
    Path(outp).write_text(text)
    return outp


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


def to_review(
    input_path=None,
    review_path=None,
    figs=None,
    tag=None,
    tag_pattern=None,
    ana_to_sis=None,
    cols_desc=None,
    excel="tables/source_data.xlsx",
    force=False,
):
    """
    The review files of `to_pub`, in the analysis repository only (no git checks, no paper repository).

    In `review_path`: `figs.tsv`, `figs/fig<id>.pdf`, `repo/figs/fig<id>/` (data, config, cleaned plot script and
    stats of each plot), `stats.yaml` (`<fig>:<key>`), `stats_fmt.json` (`{{<fig>:<key>}}`: the formatted numbers)
    and the source data excel.

    Usage:
        From the root of the repository.

        ## the params file of `to_pub` (by default `to_pub.cfg`): `review_path` is the directory of its `output_path`
        to_review("to_pub.cfg")

    Parameters:
        input_path (str): path to the repository; or to the params file, if `review_path` is not provided.
        review_path (str): the directory of the review files e.g. `results/latest`.
        figs (dict): the plots of the figures, see `read_figs`.
        tag (str), tag_pattern (str): the version of the outputs, see `read_figs`.
        ana_to_sis (dict): `{analysis name: [SI figure numbers]}`, in the order of the main figures.
        cols_desc (dict): `{column: description}` for the source data excel.
        excel (str): the source data excel (a sheet per plot), relative to `review_path`.
        force (bool): remake the outputs that exist.

    Returns:
        pd.DataFrame: the plan (see `_pub_plan`) of the plots with data.
    """
    ## params file
    if review_path is None:
        import inspect

        kws = _read_cfg(input_path or "to_pub.cfg")
        review_path = Path(kws["output_path"]).with_suffix("").as_posix()
        keys = set(inspect.signature(to_review).parameters) - {"review_path", "force"}
        return to_review(
            review_path=review_path,
            force=force,
            **{k: v for k, v in kws.items() if k in keys},
        )
    if input_path is None:
        input_path = "."
    for k, v in dict(figs=figs, ana_to_sis=ana_to_sis).items():
        assert v is not None, f"`{k}` is required"

    df = _pub_plan(
        read_figs(figs, tag=tag, tag_pattern=tag_pattern, input_path=input_path),
        ana_to_sis=ana_to_sis,
        review_path=review_path,
    )

    ## the plots, the panels and their stats
    import shutil

    from roux.lib.io import to_table

    review = Path(review_path)
    (review / "figs").mkdir(parents=True, exist_ok=True)
    to_table(df, str(review / "figs.tsv"))
    stats = {}
    for _, x in df.iterrows():
        logging.info(f">>> {x['plot label']}")
        stats[x["fig"]] = _to_panel(x["input path"], x["review path"], force=force)
        pdf, pdf_out = (
            Path(f"{x['input path']}.pdf"),
            review / f"figs/fig{x['fig']}.pdf",
        )
        if pdf.exists() and (force or not pdf_out.exists()):
            shutil.copy2(pdf, pdf_out)

    ## stats: {fig: ax.stats} -> `<fig>:<key>`, and formatted
    from roux.lib.dict import flatten_dict
    from roux.lib.io import to_dict
    from roux.lib.str import format_number

    stats_flat = flatten_dict(
        {k: flatten_dict(v, sep=":") for k, v in stats.items() if v}, sep=":"
    )
    to_dict(stats_flat, str(review / "stats.yaml"))
    to_dict(
        {
            "{{" + k + "}}": v if isinstance(v, str) else format_number(v)
            for k, v in stats_flat.items()
        },
        str(review / "stats_fmt.json"),
    )

    ## the plots with data (not skipped)
    from roux.lib.io import read_table, to_excel

    df = df.loc[df["review path"].apply(lambda p: (Path(p) / "data.pqt").exists()), :]
    skipped = sorted(set(stats) - set(df["fig"]))
    if skipped:
        logging.warning(f"no data (skipped): {skipped}")
    dfs = {
        x["plot label"]: read_table(f"{x['review path']}/data.pqt", verbose=False)
        for _, x in df.iterrows()
    }

    ## source data excel: a sheet per plot; the column descriptions as comments
    if excel:
        cols = {c for d in dfs.values() for c in d.columns}
        comments = {c: s for c, s in (cols_desc or {}).items() if c in cols and s}
        (review / excel).parent.mkdir(parents=True, exist_ok=True)
        to_excel(
            {k: d.copy() for k, d in dfs.items()},
            str(review / excel),
            comments=comments or None,
            save_input=True,
        )
    return df


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
    inline=None,
    excel="tables/source_data.xlsx",
    pyproject_drop="tool.uv.sources",
    allow_dirty="False",
    dry_run="False",
    force="False",
):
    """
    Analysis repository to a paper repository, in a worktree of the same repository (orphan branch, made by the user).

    The review files (in the directory of `output_path`): `figs.tsv`, `figs/fig<id>.pdf`, `repo/figs/fig<id>/`
    (data, config, cleaned plot script and stats of each plot), `stats.yaml` (`<fig>:<key>`), `stats_fmt.json`
    (`{{<fig>:<key>}}`: the formatted numbers) and the source data excel.

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
        tag_pattern: '(?<=outputs/)v\\d+(?=/)'
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
        inline (list): the local packages of the repository whose functions are copied into the scripts, see `_to_pub_script`.
        excel (str): the source data excel (a sheet per plot), relative to the review directory.
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

    if dry_run:
        ## plan
        df = _pub_plan(
            read_figs(
                figs,
                tag=tag,
                tag_pattern=tag_pattern,
                input_path=input_path,
                check=False,
            ),
            ana_to_sis=ana_to_sis,
            review_path=review_path,
        )
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

    ## review files (in the source repository)
    import shutil

    from roux.lib.io import to_dict

    df = to_review(
        input_path=input_path,
        review_path=review_path,
        figs=figs,
        tag=tag,
        tag_pattern=tag_pattern,
        ana_to_sis=ana_to_sis,
        cols_desc=cols_desc,
        excel=excel,
        force=force,
    )

    ## the paper repository: written to a temporary directory, then moved in (the other files are kept)
    import tempfile

    from roux.lib.io import read_dict

    files = []
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=f".{out.name}.") as tmp:
        tmp = Path(tmp)

        def put(rel):
            files.append(rel)
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            return tmp / rel

        for _, x in df.iterrows():
            _to_pub_script(
                f"{x['review path']}/plot.py",
                str(put(x["script path"])),
                src_dir_path=input_path,
                inline=inline,
            )
            shutil.copy2(f"{x['review path']}/config.yaml", put(x["config path"]))
            shutil.copy2(f"{x['review path']}/data.pqt", put(x["data path"]))
        to_dict(
            df.groupby("analysis name", sort=False)["script label"].agg(list).to_dict(),
            str(put("configs/analyses.yaml")),
        )
        for s, d in (copy or {}).items():
            shutil.copy2(s, put(d))
        ## pyproject.toml, without the tables of `pyproject_drop` (e.g. the local sources)
        if (src / "pyproject.toml").exists():
            drop = {f"[{s.strip()}]" for s in pyproject_drop.split(",") if s.strip()}
            lines, skip = [], False
            for s in (src / "pyproject.toml").read_text().split("\n"):
                if s.strip().startswith("["):
                    skip = s.strip() in drop
                if not skip:
                    lines.append(s)
            put("pyproject.toml").write_text("\n".join(lines).rstrip("\n") + "\n")
        ## .gitignore: of the repository + `gitignore` (the later lines win e.g. `!scripts/`)
        put(".gitignore").write_text(
            (
                (src / ".gitignore").read_text().rstrip("\n")
                if (src / ".gitignore").exists()
                else ""
            )
            + "\n\n## the paper repository (`gitignore` of to_pub)\n"
            + "\n".join(gitignore or [])
            + "\n"
        )
        for rel in files:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp / rel, out / rel)

    ## stale: the generated kinds of files that are not in this run (not removed)
    stale = sorted(
        p.relative_to(out).as_posix()
        for d in ["scripts", "configs/analyses", "data/analyses"]
        for p in (out / d).rglob("*")
        if p.is_file() and p.relative_to(out).as_posix() not in files
    )
    if stale:
        logging.warning(f"in {out} but not generated (stale?): {stale}")

    ## the commit: by the user
    import shlex

    msg = shlex.quote(f"+ from {dev_branch} @ {dev_sha}")
    q = shlex.quote(str(out.resolve()))
    print(
        f"## to commit the paper repository:\ngit -C {q} add -A\ngit -C {q} commit -m {msg}"
    )

    ## marker
    to_dict(
        dict(
            source=dict(branch=dev_branch, sha=dev_sha, tag=tag),
            pub=dict(path=str(out), branch=pub_branch),
            files=files,
        ),
        output_path,
    )
    return output_path

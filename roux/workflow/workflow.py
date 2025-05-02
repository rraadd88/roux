"""For workflow management."""

import logging
from pathlib import Path
from roux.lib.sys import exists, dirname, basename, makedirs, basenamenoext, abspath
from roux.lib.io import read_ps, read_dict
import pandas as pd
import subprocess


def get_scripts(
    ps: list,
    notebook_prefix: str = "\d{2}",
    notebook_suffix: str = "_v\d{2}",
    test: bool = False,
    fast: bool = True,
    cores: int = 6,
    force: bool = False,
    tab: str = "    ",
    **kws,
) -> pd.DataFrame:
    """Get scripts.

    Args:
        ps (list): paths.
        notebook_prefix (str, optional): prefix of the notebook file to be considered as a "task".
        notebook_suffix (str, optional): suffix of the notebook file to be considered as a "task".
        test (bool, optional): test mode. Defaults to False.
        fast (bool, optional): parallel processing. Defaults to True.
        cores (int, optional): cores to use. Defaults to 6.
        force (bool, optional): overwrite the outputs. Defaults to False.
        tab (str, optional): tab in spaces. Defaults to '    '.

    Returns:
        pd.DataFrame: output table.
    """
    d1 = {basenamenoext(p): p for p in ps}
    # print(d1)
    # d1={p:v for p,v in d1.items() if p.startswith(notebook_prefix) and p.endswith(notebook_suffix)}
    import re

    d1 = {
        k: v
        for k, v in d1.items()
        if re.match(notebook_prefix + ".*" + notebook_suffix, k)
    }
    # print(d1)
    if test:
        logging.info(d1)
    assert (
        len(d1) != 0
    ), f"no notebooks with the {notebook_prefix} prefix and {notebook_suffix} suffix found"
    if notebook_suffix != "":
        ## remove suffix
        # d1={p.split(notebook_suffix)[0]:v for p,v in d1.items()}
        d1 = {re.split(notebook_suffix, p)[0].rstrip("_"): v for p, v in d1.items()}
    # print(d1.keys())
    # brk
    df1 = pd.DataFrame(pd.Series(d1, name="notebook path"))
    df1.index.name = "task name"
    df1 = df1.sort_index().reset_index()
    if not df1["task name"].apply(lambda x: x[:1].isdigit()).all():
        logging.warning("notebooks are not numbered")
    # df1=df1.loc[(df1['notebook path'].apply(lambda x: (notebook_suffix in x))),:]
    # print(df1.shape)
    from roux.workflow.function import to_task

    if not fast or df1["notebook path"].nunique() < 5:
        # df2=df1['notebook path'].apply(lambda x: print(x,force,not fast,notebook_suffix,kws))
        df2 = df1.apply(
            lambda x: to_task(
                notebookp=x["notebook path"],
                task=x["task name"],
                force=force,
                verbose=not fast,
                notebook_suffix=notebook_suffix,
                **kws,
            ),
            axis=1,
        )
    else:
        from roux.lib.df import get_name
        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=cores, progress_bar=True)
        df2 = df1.groupby(["notebook path", "task name"]).parallel_apply(
            lambda x: to_task(
                notebookp=get_name(x, ["notebook path"])[0],
                task=get_name(x, ["task name"])[0],
                force=force,
                verbose=not fast,
                notebook_suffix=notebook_suffix,
                **kws,
            )
        )
    assert len(df2) != 0, "no notebooks found"
    df2 = df2.apply(pd.Series).rename(
        columns={0: "rule code", 1: "output paths"}, errors="raise"
    )
    if df2.index.name is None:
        df2.index.name = "notebook path"
    # print(df2)
    df2 = df2.reset_index(0).dropna(subset=["notebook path", "rule code"])
    if df1["notebook path"].nunique() == 1:
        # print(df2.index)
        # print(df2.shape)
        # print(df2)
        df2["notebook path"] = df1["notebook path"].tolist()[0]
    from roux.lib.set import flatten

    # if test:print(d1)
    df2["output paths"] = df2["output paths"].apply(
        lambda x: f",\n{tab}{tab}".join(flatten(x)) if isinstance(x, list) else ""
    )
    df2 = df2.reset_index(drop=True)
    assert len(df2) != 0, "no functions found"
    return df2


def to_scripts(
    # packagen: str,
    packagep: str,
    notebooksdp: str,
    validate: bool = False,
    ps: list = None,
    notebook_prefix: str = "\d{2}",
    notebook_suffix: str = "_v\d{2}",
    scripts: bool = True,
    workflow: bool = True,
    sep_step: str = "## step",
    todos: bool = False,
    git: bool = True,
    clean: bool = False,
    test: bool = False,
    force: bool = True,
    tab: str = "    ",
    **kws,
):
    """To scripts.

    Args:
        # packagen (str): package name.
        packagep (str): path to the package.
        notebooksdp (str, optional): path to the notebooks. Defaults to None.
        validate (bool, optional): validate if functions are formatted correctly. Defaults to False.
        ps (list, optional): paths. Defaults to None.
        notebook_prefix (str, optional): prefix of the notebook file to be considered as a "task".
        notebook_suffix (str, optional): suffix of the notebook file to be considered as a "task".
        scripts (bool, optional): make scripts. Defaults to True.
        workflow (bool, optional): make workflow file. Defaults to True.
        sep_step (str, optional): separator marking the start of a step. Defaults to "## step".
        todos (bool, optional): show todos. Defaults to False.
        git (bool, optional): save version. Defaults to True.
        clean (bool, optional): clean temporary files. Defaults to False.
        test (bool, optional): test mode. Defaults to False.
        force (bool, optional): overwrite outputs. Defaults to True.
        tab (str, optional): tab size. Defaults to '    '.

    Keyword parameters:
        kws: parameters provided to the `get_script` function,
            including `sep_step` and `sep_step_end`

    TODOs:
        1. For version control, use https://github.com/jupyterlab/jupyterlab-git.
    """
    packagen = basename(dirname(abspath(".")))
    # packagescriptsp=f"{packagep}/{packagen}"
    packagescriptsp = notebooksdp
    df_outp = f"{packagescriptsp}/.workflow/info.tsv"
    makedirs(df_outp)
    # if notebooksdp is None:
    #     notebooksdp=f'{packagescriptsp}/notebooks'
    # info(packagescriptsp,notebooksdp)
    if scripts:
        # get all the notebooks
        if ps is not None:
            ps = read_ps(ps)
            ps = [abspath(p) for p in ps]
            make_all = False
        else:
            ps = read_ps(f"{notebooksdp}/*ipynb")[::-1]
            make_all = True
        if test:
            logging.info(len(ps))
        if exists(f"{notebooksdp}/.workflow/config.yaml"):
            cfg = read_dict(f"{notebooksdp}/.workflow/config.yaml")
            if "exclude" in cfg:
                ps = [p for p in ps if basename(p) not in cfg["exclude"]]
                logging.info(f"remaining few paths after excluding: {len(ps)}")
        df2 = get_scripts(
            ps,
            force=force,
            notebook_prefix=notebook_prefix,
            notebook_suffix=notebook_suffix,
            sep_step=sep_step,
            **kws,
        )
        if test:
            logging.info(df2.shape)
        if not make_all and exists(df_outp):
            df_ = pd.read_csv(df_outp, sep="\t").rd.clean()
            df2 = (
                df_.loc[~(df_["notebook path"].isin(ps)), :]
                .append(df2)
                .drop_duplicates()
            )
            df2 = df2.dropna(subset=["notebook path", "rule code"])
            # else:
            #     logging.warning('likely incomplete workflow')
        df2.to_csv(df_outp, sep="\t")
        if test:
            logging.info(df_outp)
        ## make __init__.py if not exists
        initp = Path(f"{notebooksdp}/lib/__init__.py")
        if not initp.exists():
            initp.touch()
    if workflow or (ps is None):
        ## workflow inferred
        if "df2" not in globals():
            df2 = pd.read_csv(df_outp, sep="\t")
        from roux.workflow.io import to_workflow

        to_workflow(df2, workflowp=f"{packagescriptsp}/workflow.py")
    ## make readme
    if todos:
        from roux.workflow.io import to_info

        to_info(
            p=f"{packagescriptsp}/*{notebook_prefix}*{notebook_suffix}*.ipynb",
            outp=f"{packagescriptsp}/README.md",
        )
        from roux.lib.io import read_list

        [
            print(s)
            for s in "\n".join(read_list(f"{packagescriptsp}/README.md")).split(
                sep_step
            )
            if "TODO" in s
        ]
    if git:
        from .version import git_commit

        git_commit(packagep, suffix_message="" if not validate else " (not validated)")
    if clean:
        output = subprocess.check_output(
            f'grep -Hrn "%run " {packagescriptsp}/lib/*.py',
            shell=True,
            universal_newlines=True,
        ).split("\n")
        print(output)

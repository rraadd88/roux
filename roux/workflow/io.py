"""
For input/output of workflow.
"""

## logging
import logging
import re
from pathlib import Path

## data
import pandas as pd

from roux.lib.sys import (
    exists,
    makedirs,
    read_ps,
)

## for backcompatibility
from roux.workflow.cfgs import read_config, read_metadata  ##noqa
from roux.workflow.pms import read_cli_pm


## variables
def clear_variables(
    dtype=None,
    variables=None,
    # globals_dict=None,
):
    """Clear dataframes from the workspace."""
    assert (dtype is None or variables is None) and (
        not (dtype is None and variables is None)
    )
    import gc

    if variables is None:
        variables = [
            k
            for k in globals()
            if isinstance(globals()[k], dtype) and not k.startswith("_")
        ]
    if len(variables) == 0:
        return
    for k in variables:
        del globals()[k]
    gc.collect()
    logging.info(f"cleared {len(variables)} variables: {variables}.")


def clear_dataframes():
    return clear_variables(
        dtype=pd.DataFrame,
        variables=None,
    )


from roux.lib.text import replace_text


## used in ui
from roux.lib.sys import get_source_path  # noqa


def to_mod(
    p=None,  # nb
    outp=None,  # py
    kws_nb_export={},
    ## ruff
    errors="raise",
    **kws_check_py,
):
    if p is None:
        from roux.lib.sys import get_source_path

        p = get_source_path()
        logging.warning(f"p={p}")

    if outp is None:
        outp = f"../{'modules/' if Path(p).stem.count('_') == 1 else '' if Path(p).stem.count('_') == 2 else ValueError(Path(p).stem)}{Path(p).stem.replace('_', '/')}.py"

    kws_nb_export = {
        **dict(
            nbname=p,
            lib_path=Path(outp).parent.as_posix(),
            name=Path(outp).stem,
        ),
        ## override
        **kws_nb_export,
    }
    from nbdev.export import nb_export

    nb_export(
        **kws_nb_export,
    )
    # _py_path=f"{kws_nb_export['lib_path']}/{kws_nb_export['nbname'].split('.')[0].split('_')[-1]}.py"
    _py_path = f"{kws_nb_export['lib_path']}/{kws_nb_export['name']}.py"
    logging.info(_py_path)

    check_py(
        _py_path,
        errors=errors,
        **kws_check_py,
    )

    return outp


## outputs setup
def set_outputs(
    output_path=None,
    output_paths=None,
):
    # if output_path is None:
    #     output_path=locals().get('output_path')
    # if output_paths is None:
    #     output_paths=locals().get('output_paths')

    if isinstance(output_path, str):
        output_dir_path = Path(output_path).with_suffix("").as_posix()
        if isinstance(output_paths, list):
            output_paths = {fn: f"{output_dir_path}/{fn}" for fn in output_paths}
            from roux.lib.log import log_dict

            logging.info("output_paths:\n" + log_dict(output_paths, out=True))
        else:
            logging.info(f"output_dir_path: {output_dir_path}")
            output_paths = {}
    else:
        output_dir_path, output_paths = None, {}
    return output_dir_path, output_paths


## tasks
def check_for_exit(
    ## check this if empty
    p,  # saved table
    ## save this before exiting
    outp,  # output path
    data=None,  # e.g output cfg
    force=False,
):
    """
    Check for early exit in a script.
    """
    exit = force

    if not exit:
        if isinstance(p, str):
            from roux.lib.io import is_table_empty

            exit = is_table_empty(p)
        elif isinstance(p, pd.DataFrame):
            exit = len(p) == 0
            if data is None:
                data = p

    if exit:
        logging.warning("exiting early because table is empty..")
        if data is not None:
            from roux.lib.io import to_data

            to_data(data, outp)
        else:
            Path(outp).touch()
        import sys

        sys.exit(0)


## back-compat: moved out of this module (import them from here or from their new modules)
from roux.workflow.scr import (  # noqa
    RUFF_SELECT,
    _PKG_CHECKS,
    _PKG_DROP_CODE_LINES,
    _PKG_DROP_HEADERS,
    _clean_scr,
    _pms_to_run_args,
    _to_scr_cleaned,
    _to_scr_running_py,
    _to_scrs,
    check_py,
    post_code,
    replacestar,
    replacestar_ruff,
    to_py,
    to_py_path,
    to_scr,
    to_scr_path,
    to_src,
)
from roux.workflow._worktree import _git, _git_check_ignore, _pkg_plan, _plan  # noqa
from roux.workflow.pkg import to_pkg, to_scrs  # noqa

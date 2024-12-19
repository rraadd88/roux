"""For task management."""

## logging
import logging

## internal
from roux.lib.io import read_dict, to_dict
from pathlib import Path
from roux.lib.sys import (
    basenamenoext,
    dirname,
    exists,
    get_datetime,
    makedirs,
    splitext,
)

import pandas as pd
## parallel-processing
import roux.lib.df_apply as rd #noqa

try:
    from tqdm import tqdm
    from roux.lib.sys import is_interactive_notebook

    if not is_interactive_notebook():
        ## progress bar
        tqdm.pandas()
    else:
        from tqdm import notebook

        notebook.tqdm().pandas()
except ImportError:
    logging.warning(
        "ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html"
    )

import papermill as pm

## validators
def validate_params(
    d: dict,
) -> bool:
    return ("input_path" in d) and ("output_path" in d)


## execution
def run_task(
    parameters: dict,
    input_notebook_path: str,
    kernel: str = None,
    output_notebook_path: str = None,
    start_timeout: int = 600,
    verbose=False,
    force=False,
    **kws_papermill,
) -> str:
    """
    Run a single task.

    Prameters:
        parameters (dict): parameters including `output_path`s.
        input_notebook_path (dict): path to the input notebook which is parameterized.
        kernel (str): kernel to be used.
        output_notebook_path: path to the output notebook which is used as a report.
        verbose (bool): verbose.

    Keyword parameters:
        kws_papermill: parameters provided to the `pm.execute_notebook` function.

    Returns:
        Output path.
    """
    if exists(parameters["output_path"]) and not force:
        return parameters["output_path"]
    if not output_notebook_path:
        ## save report i.e. output notebook
        output_notebook_path = f"{splitext(parameters['output_path'])[0]}_reports/{get_datetime()}_{basenamenoext(input_notebook_path)}.ipynb"
        makedirs(output_notebook_path)
    if verbose:
        logging.info(parameters["output_path"], output_notebook_path)
    ## save parameters
    to_dict(parameters, f"{dirname(output_notebook_path)}/parameters.yaml")

    if verbose:
        logging.info(parameters)
    if kernel is None:
        logging.warning("`kernel` name not provided.")

    pm.execute_notebook(
        input_path=input_notebook_path,
        output_path=output_notebook_path,
        parameters=parameters,
        kernel_name=kernel,
        start_timeout=start_timeout,
        report_mode=True,
        allow_errors=False,
        # cwd=None #(str or Path, optional) – Working directory to use when executing the notebook
        # prepare_only (bool, optional) – Flag to determine if execution should occur or not
        **kws_papermill,
    )
    # return parameters['output_path']
    return output_notebook_path

def apply_run_task(
    x: str,
    input_notebook_path: str,
    kernel: str,
    force=False,
    **kws_papermill,
    ):
    try:
        return run_task(
            x,
            input_notebook_path=input_notebook_path,
            kernel=kernel,
            force=force,
            **kws_papermill,
        )
    except:
        # logging.error
        raise RuntimeError(f"tb: check {x}")
        # return 

def run_tasks(
    input_notebook_path: str,
    kernel: str = None,
    inputs: list = None,
    output_path_base: str = None,
    params=None,
    parameters_list=None, # same as params
    fast: bool = False,
    fast_workers: int = 6,
    to_filter_nbby_patterns_kws=None,
    input_notebook_temp_path=None,
    out_paths: bool = True,
    post: bool = False,
    test1: bool = False,
    force: bool = False,
    test: bool = False,
    verbose: bool = False,
    **kws_papermill,
) -> list:
    """
    Run a list of tasks.

    Prameters:
        input_notebook_path (dict): path to the input notebook which is parameterized.
        kernel (str): kernel to be used.
        inputs (list): list of parameters without the output paths, which would be inferred by encoding.
        output_path_base (str): output path with a placeholder e.g. 'path/to/{KEY}/file'.
        parameters_list (list): list of parameters including the output paths.
        out_paths (bool): return paths of the reports (Defaults to True).
        post (bool): post-process (Defaults to False).
        test1 (bool): test only first task in the list (Defaults to False).
        fast (bool): enable parallel-processing.
        fast_workers (bool): number of parallel-processes.
        force (bool): overwrite the outputs.
        test (bool): test-mode.
        verbose (bool): verbose.

    Keyword parameters:
        kws_papermill: parameters provided to the `pm.execute_notebook` function e.g. working directory (cwd=)
        to_filter_nbby_patterns_kws (list): dictionary containing parameters to be provided to `to_filter_nbby_patterns` function (Defaults to None).

    Returns:
        parameters_list (list): list of parameters including the output paths, inferred if not provided.

    TODOs:
        0. Ignore temporary parameters e.g test, verbose etc while encoding inputs.
        1. Integrate with apply_on_paths for parallel processing etc.

    Notes:
    1. To resolve `RuntimeError: This event loop is already running in python` from `multiprocessing`, execute
        import nest_asyncio
        nest_asyncio.apply()
    """
    assert exists(input_notebook_path), input_notebook_path
    if test:
        force = True
    assert not (params is not None and parameters_list is not None)
    if params is not None and parameters_list is None:
        parameters_list=params
        del params
    ## save task in unique directories
    if parameters_list is None:
        ## infer output paths
        from roux.lib.sys import to_output_paths

        parameters_list = to_output_paths(
            inputs=inputs,
            output_path_base=output_path_base,
            encode_short=True,
            key_output_path="output_path",
            verbose=verbose,
            force=force,
        )
        ## save all parameters
        for k, parameters in parameters_list.items():
            ## save parameters
            output_dir_path = output_path_base.split("{KEY}")[0]
            to_dict(
                parameters,
                f"{output_dir_path}/{k.split(output_dir_path)[1].split('/')[0]}/.parameters.yaml",
            )
    # print(parameters_list)
    
    if isinstance(parameters_list, str):
        parameters_list = read_dict(parameters_list)
        
    if len(parameters_list) == 0:
        logging.info("nothing to process. use `force`=True to rerun.")
        return
        
    if isinstance(parameters_list, dict):
        ## input_paths used as keys
        if not any(['input_path' in d for d in parameters_list.values()]):
            logging.warning("setting keys of params as input_path s ..")
            parameters_list={k:{**d,**{'input_path':k}} for k,d in parameters_list.items()}
        if validate_params(
            parameters_list[
                list(parameters_list.keys())[0]
            ]
            ):
            parameters_list = list(parameters_list.values())
        else:
            raise ValueError(parameters_list)
            
    if test:
        logging.info("Aborting run because of the test mode")
        return parameters_list
        
    if isinstance(parameters_list, list):
        before = len(parameters_list)
        ## TODO: use `to_outp`?
        parameters_list = [
            d
            for d in parameters_list
            if (force if force else not exists(d["output_path"]))
        ]
        if not force:
            if before - len(parameters_list) != 0:
                logging.info(
                    f"parameters_list reduced because force=False: {before} -> {len(parameters_list)}"
                )
            if len(parameters_list) == 0:
                return parameters_list
    else:
        raise ValueError(parameters_list)
    ## chech for duplicate output paths
    assert len(set([d["output_path"] for d in parameters_list])) == len(
        parameters_list
    ), (len(set([d["output_path"] for d in parameters_list])), len(parameters_list))
    ## input_path!=output_path
    assert [Path(d["input_path"])!=Path(d["output_path"]) for d in parameters_list], [d for d in parameters_list if Path(d["input_path"])==Path(d["output_path"])]
    
    if to_filter_nbby_patterns_kws is not None:
        logging.info("filtering the notebook")
        if input_notebook_temp_path is None:
            import tempfile

            # input_notebook_temp_file = tempfile.NamedTemporaryFile(delete=False)
            # input_notebook_temp_file.close()
            # input_notebook_temp_path=input_notebook_temp_file.name+'.ipynb'
            input_notebook_temp_path = (
                f"{tempfile.gettempdir()}/{Path(input_notebook_path).name}"
            )
        logging.info(f"temporary notebook file path: {input_notebook_temp_path}")
        ## copy input to the temporary
        import shutil

        shutil.copyfile(input_notebook_path, input_notebook_temp_path)

        from roux.workflow.nb import to_filter_nbby_patterns

        input_notebook_path = to_filter_nbby_patterns(
            input_notebook_temp_path,
            input_notebook_temp_path,
            **to_filter_nbby_patterns_kws,
        )
        input_notebook_path = input_notebook_temp_path
    #     clean=True
    # else:
    #     clean=False

    ## run tasks

    df1 = (
        pd.Series(parameters_list)
        # to df
        .to_frame('params')
    )
        
    if len(df1) == 0:
        logging.warning("No tasks remaining.")
        return 
        
    if test1:
        df1 = df1.head(1)
        logging.warning("testing only the first input.")
        
    if not fast or len(df1)==1:
        df1['nb path'] = getattr(
            df1['params'],
            "progress_apply"
            if hasattr(df1, "progress_apply") and len(df1) > 1
            else "apply",
        )(
            lambda x: apply_run_task(
                x,
                input_notebook_path=input_notebook_path,
                kernel=kernel,
                **kws_papermill,
                force=force,
            )
        )
    else:        
        logging.info(f"running in parallel (cpus={fast_workers})..")
        
        # disable logging
        sorted(list(logging.root.manager.loggerDict.keys()))
        for k in [
            'papermill',
            'papermill.translators',
            'papermill.utils',
        ]:
            logging.getLogger(k).setLevel(logging.CRITICAL)
        
        df1['nb path']=(
            df1
            .rd.apply_async(
                lambda x: 
                    apply_run_task(
                        x['params'],
                        input_notebook_path=input_notebook_path,
                        kernel=kernel,
                        **kws_papermill,
                        force=force,
                    ),
                cpus=fast_workers,
            )
        )
    # return ds2
    if post==True and not fast:        
        from roux.workflow.io import valid_post_task_deps, to_html
        if valid_post_task_deps:
            df1['html path']=(
            df1
            .rd.apply_async(
                lambda x: 
                    to_html(
                        x['nb path'],                    
                    ),
                cpus=fast_workers,
            )
            )
        
    if not out_paths:
        return before
    else:
        return df1

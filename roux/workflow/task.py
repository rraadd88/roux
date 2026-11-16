"""For task management."""

import sys

sys.tracebacklimit = 0  # needs to be at the very top

import os
import time

from tqdm import tqdm

## logging
# try:
from roux.lib.log import Logger

logging = Logger()  # level='INFO'
# except:
# import logging
# logging.basicConfig(level='INFO', force=True)

from pathlib import Path

import pandas as pd

## parallel-processing
import roux.lib.df_apply as rd  # noqa
from roux.lib.io import read_dict, read_ps, to_dict

## internal
from roux.lib.sys import (
    exists,
    get_datetime,
    run_com,
)
from roux.workflow.log import test_params

## SLURM (moved to `roux.workflow.slurm`; imported from here too, for back-compat)
from roux.workflow.slurm import (  # noqa
    SLURMJob,
    feed_jobs,
    get_jobsn,
    get_sq,
    get_tried_job_keys,
    has_slurm,
    is_q_empty,
    parse_time,
    submit_job,
    to_sbatch_script,
)
from roux.workflow.pms import pre_params

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


def pre_task(
    pms,
    cache_dir_path=None,
    test=False,
) -> str:

    if isinstance(pms, list):
        assert len(pms) == 1, pms

    if cache_dir_path is None:
        test = True
        logging.warning("log_dir_path={output_path}_logs")

    if test:
        log_dir_path_ = f"{Path(pms['output_path']).with_suffix('').as_posix()}_logs"
        log_dir_path = f"{log_dir_path_}/{get_datetime()}"
    else:
        from roux.lib.str import encode

        # log_dir_path_=f"{Path(cache_dir_path).expanduser().as_posix()}"
        log_dir_path_ = cache_dir_path
        log_dir_path = (
            f"{log_dir_path_}/{get_datetime()}_{encode(pms['output_path'], short=True)}"
        )

    # else:
    # [tmp dir not found by slurm]
    # log_dir_path=tempfile.mkdtemp(prefix=get_datetime())

    # log_dir_path=tempfile.mkdtemp(prefix=get_datetime())
    try:
        to_dict(pms, f"{log_dir_path}/pms.yaml")
    except Exception as e:
        logging.warning(str(e))
    return log_dir_path


## execution
def run_task_nb(
    parameters: dict,
    script_path: str,
    kernel: str = None,
    cache_dir_path: str = None,
    output_notebook_path: str = None,
    start_timeout: int = 600,
    verbose=False,
    force=False,
    # test=True,
    **kws_papermill,
) -> str:
    """
    Run a single task.

    Prameters:
        parameters (dict): parameters including `output_path`s.
        script_path (dict): path to the input notebook which is parameterized.
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

    # ## save parameters
    # to_dict(parameters, f"{dirname(output_notebook_path)}/pms_latest.yaml")

    if cache_dir_path is None:
        cache_dir_path = pre_task(
            parameters,
            cache_dir_path=cache_dir_path,
            # test=test,
        )
    # print(cache_dir_path)
    if not output_notebook_path:
        ## save report i.e. output notebook
        assert cache_dir_path is not None, cache_dir_path
        output_notebook_path = f"{cache_dir_path}_{Path(script_path).name}"
    ## to open
    output_notebook_path = Path(output_notebook_path).absolute()

    if verbose:
        # logging.warning(cache_dir_path)
        # logging.info(parameters)
        # logging.warning
        print(parameters["output_path"], output_notebook_path)

    if kernel is None:
        logging.warning("`kernel` name not provided.")
    # try:
    pm.execute_notebook(
        input_path=script_path,
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
    # except PapermillExecutionError as e:
    #     e_last=str(e).split('Traceback (most recent call last)')[-1]
    #     logging.error(f"{script_path}\n{e_last}")
    #     test_params(parameters)

    # return parameters['output_path']
    return output_notebook_path


def apply_run_task_nb(
    x: str,
    script_path: str,
    kernel: str,
    force=False,
    **kws_papermill,
):
    from papermill.exceptions import PapermillExecutionError

    try:
        return run_task_nb(
            x,
            script_path=script_path,
            kernel=kernel,
            force=force,
            **kws_papermill,
        )
    except PapermillExecutionError as e:
        test_params(x, logger=logging.error)

        e_last = str(e).split("Traceback (most recent call last)")[-1]
        raise RuntimeError(f"{e_last}\n{x['output_path']}") from None

        sys.exit(0)


def run_tasks_nb(
    script_path: str = None,
    params=None,
    ## kws_run
    kernel: str = None,
    cpus: int = 1,
    pre: bool = True,
    post: bool = False,
    simulate: bool = False,  # ~dry
    test1: bool = False,
    force: bool = False,
    # test: bool = False,
    verbose: bool = False,
    ## make the params
    inputs: list = None,
    output_path_base: str = None,
    to_filter_nbby_patterns_kws=None,
    input_notebook_temp_path=None,
    out_paths: bool = True,
    ## back.c.
    input_notebook_path: str = None,
    parameters_list=None,  # same as params
    fast: bool = None,  ## to be deprecated
    fast_workers: int = None,  ## to be deprecated
    **kws_papermill,
) -> list:
    """
    Run a list of tasks.

    Prameters:
        script_path (dict): path to the input notebook which is parameterized.
        kernel (str): kernel to be used.
        inputs (list): list of parameters without the output paths, which would be inferred by encoding.
        output_path_base (str): output path with a placeholder e.g. 'path/to/{KEY}/file'.
        parameters_list (list): list of parameters including the output paths.
        out_paths (bool): return paths of the reports (Defaults to True).
        post (bool): post-process (Defaults to False).
        test1 (bool): test only first task in the list (Defaults to False).
        fast (bool): enable parallel-processing.
        cpus (bool): number of parallel-processes.
        force (bool): overwrite the outputs.
        test (bool): test-mode.
        verbose (bool): verbose.

    Keyword parameters:
        kws_papermill: parameters provided to the `execute_notebook` function e.g. working directory (cwd=)
        to_filter_nbby_patterns_kws (list): dictionary containing parameters to be provided to `to_filter_nbby_patterns` function (Defaults to None).

    Returns:
        parameters_list (list): list of parameters including the output paths, inferred if not provided.

    TODOs:
        0. Ignore temporary parameters e.g test, verbose etc while encoding inputs.
        1. Integrate with apply_on_paths for parallel processing etc.

    Notes:
    1. To resolve `RuntimeError: This event loop is already running in python` from `multiprocessing`, or `zmq.error.ZMQError: Address already in use` run this before,
        import nest_asyncio
        nest_asyncio.apply()
    """

    def _to_out(
        out_paths,
        params,
        df1,
    ):
        if not out_paths:
            return params
        else:
            if "nb path" in df1:
                df1 = df1.set_index("nb path")
            return df1["params"]  # .apply(pd.Series)

    if script_path is None and input_notebook_path is not None:
        script_path = input_notebook_path
        logging.warning("input_notebook_path will be deprec.")
        del input_notebook_path
    assert exists(script_path), script_path

    if len(params) == 0:
        params = None
    assert not (params is not None and parameters_list is not None), (
        params,
        parameters_list,
    )
    if params is None and parameters_list is not None:
        params = parameters_list
        del parameters_list

    params = pre_params(
        params=params,
        inputs=inputs,
        output_path_base=output_path_base,
        verbose=verbose,
        force=force,
        test1=test1,
        # testn = testn,
    )

    if len(params) == 0:
        # logging.status('exiting ..')
        return

    ## nb
    if to_filter_nbby_patterns_kws is not None:
        logging.info("filtering the notebook")
        if input_notebook_temp_path is None:
            import tempfile

            # input_notebook_temp_file = tempfile.NamedTemporaryFile(delete=False)
            # input_notebook_temp_file.close()
            # input_notebook_temp_path=input_notebook_temp_file.name+'.ipynb'
            input_notebook_temp_path = (
                f"{tempfile.gettempdir()}/{Path(script_path).name}"
            )
        logging.info(f"temporary notebook file path: {input_notebook_temp_path}")
        ## copy input to the temporary
        import shutil

        shutil.copyfile(script_path, input_notebook_temp_path)

        from roux.workflow.nb import to_filter_nbby_patterns

        script_path = to_filter_nbby_patterns(
            input_notebook_temp_path,
            input_notebook_temp_path,
            **to_filter_nbby_patterns_kws,
        )
        script_path = input_notebook_temp_path
    #     clean=True
    # else:
    #     clean=False
    if pre:
        logging.debug("pre-processing nb ..")
        from roux.workflow.nb import to_nb_kernel

        to_nb_kernel(
            script_path,
            kernel=kernel,
        )
    ## run tasks
    ## for log
    from datetime import datetime

    _start_time = datetime.now()

    df1 = (
        pd.Series(params)
        # to df
        .to_frame("params")
    )

    if len(df1) == 0:
        logging.warning("No tasks remaining.")
        # _to_out(
        #         out_paths,
        #         params,
        #         df1,
        #     )
        return

    if test1:
        df1 = df1.head(1)
        logging.warning("testing only the first input.")

    ## back-compatibility
    if fast_workers is not None:
        cpus = fast_workers
    fast = cpus > 1

    if not simulate:
        if (not fast) or len(df1) == 1:
            df1["nb path"] = getattr(
                df1["params"],
                "progress_apply"
                if hasattr(df1, "progress_apply") and len(df1) > 1
                else "apply",
            )(
                lambda x: apply_run_task_nb(
                    x,
                    script_path=script_path,
                    kernel=kernel,
                    **kws_papermill,
                    force=force,
                )
            )
        else:
            logging.info(f"running in parallel (cpus={cpus})..")

            # disable logging
            import logging as logging_base

            sorted(list(logging_base.root.manager.loggerDict.keys()))
            for k in [
                "papermill",
                "papermill.translators",
                "papermill.utils",
            ]:
                logging_base.getLogger(k).setLevel(logging_base.CRITICAL)

            df1["nb path"] = df1.rd.apply_async(
                lambda x: apply_run_task_nb(
                    x["params"],
                    script_path=script_path,
                    kernel=kernel,
                    **kws_papermill,
                    force=force,
                ),
                cpus=cpus,
            )
        # return ds2

        if post and not fast:
            from roux.workflow.nb import to_html, valid_post_task_deps

            if valid_post_task_deps:
                df1["html path"] = df1.rd.apply_async(
                    lambda x: to_html(
                        x["nb path"],
                    ),
                    cpus=cpus,
                )

    ## log
    logging.info(f"Time taken: {datetime.now() - _start_time}")

    return _to_out(
        out_paths,
        params,
        df1,
    )


def check_tasks(
    state="",
    job_name_expr="",
    query_expr="| tail -n 10",
    eff=False,
):
    cols = [
        "JobID",
        "State",
        "Elapsed",
        "End",
        "JobName",
    ]
    # 'COMMAND' sacct: error: Invalid field requested: "COMMAND"
    com = f"sacct -u $USER --format={','.join(cols)}%200 | grep '.*{state}.*roux:.*{job_name_expr}.*' {query_expr}"
    txt = run_com(
        com,
        # text=True,
        verbose=True,
    )
    if "\n" not in txt:
        logging.warning(f"no output from {com}")
        return None
    from io import StringIO

    df_sacct = pd.read_fwf(
        StringIO(txt),
        header=None,
        names=cols,
    ).rename(
        columns={"JobID": "Job ID"},
    )
    if not eff:
        return df_sacct.set_index("Job ID").to_string()

    ds_seff = df_sacct["Job ID"].apply(lambda x: run_com(f"seff {x}"))
    df_seff = ds_seff.apply(
        lambda x: (
            {s.split(": ")[0]: s.split(": ")[1] for s in x.split("\n") if ": " in s}
            if x.startswith("Job ID")
            else {}
        )
    ).apply(pd.Series)

    ## clean
    for c in [
        "Cluster",
        "User/Group",
        "State",
        "Job Wall-clock time",
        "Nodes",
        "Cores per node",
        "CPU Utilized",
        "Memory Utilized",
        "State",
    ]:
        if c in df_seff:
            df_seff = df_seff.drop([c], axis=1)

    return (
        df_seff.astype({"Job ID": int})
        .merge(
            right=df_sacct.astype({"Job ID": int}),
            how="right",
            on="Job ID",
            validate="1:1",
        )
        .set_index("Job ID")
        .to_string()
    )


def load_module(
    m,
    validate=True,
):
    import os
    import subprocess

    if isinstance(m, list):
        for m_ in m:
            load_module(
                m_,
                validate=validate,
            )
        return

    # Load the module and modify the current environment
    # load_module_command = f"source $(realpath ~/.bashrc) && module load {m}"
    import shlex

    load_module_command = f"module load {shlex.quote(m)} && env"
    logging.info(load_module_command)
    result = subprocess.run(
        load_module_command,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )

    # Parse the output and set environment variables
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

    if validate:
        # Verify samtools is accessible
        os.system(f"{shlex.quote(m)} --version")


def infer_runner(
    runner=None,
    script_type=None,
):
    assert runner in ["py", "bash", "slurm", None], runner

    runner_in = runner
    del runner

    if runner_in == "slurm" and not has_slurm():
        runner = "bash"
    elif runner_in is None:
        from roux.lib.sys import is_interactive_notebook

        if script_type in ["ipynb"] and is_interactive_notebook():
            runner = "py"  ## run_tasks_nb
        # if script_type in ['sh','py',None]:
        else:
            runner = "bash"
    else:
        runner = runner_in

    if runner_in is not None and runner != runner_in:
        logging.warning(f"runner={runner}")
    else:
        logging.info(f"runner={runner}")

    return runner


def pre_cfg_run(
    cfg_run,
    script_type=None,
):
    # cfg_run={}
    outps = {}
    for step in cfg_run:
        sp = cfg_run[step]["kws_run"]["script_path"]
        if isinstance(sp, dict):
            if script_type is None:
                st = list(sp.keys())[0]
            sp = sp[st]
            assert isinstance(sp, str), sp
            del st
            cfg_run[step]["kws_run"]["script_path"] = sp
        assert isinstance(cfg_run[step]["kws_run"]["script_path"], str), cfg_run[step][
            "kws_run"
        ]["script_path"]
        outp = cfg_run[step]["pms_run"]["output_path"]
        if outp in outps:
            # dup
            if cfg_run[step]["pms_run"] == outps[outp]:
                # exact dup
                pass
            else:
                raise ValueError(
                    f"found same output for two tasks: {step} {cfg_run[step]['pms_run']} {outps[outp]}"
                )
        outps[outp] = cfg_run[step]["pms_run"]

    ## assert no dups
    outps = list(outps.keys())
    from collections import Counter

    assert len(outps) == len(set(outps)), Counter(outps)

    return cfg_run


def post_tasks(
    params=None,
    arxv=False,
    clean=False,  ## remove logs
    ind=None,
    outp=None,
    simulate=False,
    validate=False,
    verbose=True,
):
    """
    Notes:
        Count files:

            for d in */; do if [ -d "$d" ]; then echo $d": "$(find $d | wc -l) ;fi;done
    """

    if clean:
        assert ind is not None
        if outp is None:
            # from roux.lib.sys import get_datetime
            outp = f".roux/post_tasks_{get_datetime()}.zip"

        outd = Path(outp).parent.mkdir(parents=True, exist_ok=True)

        if validate:
            read_ps(ind, tree_depth=3)

        # com=f"find {ind}"+r" -type f \( -name '*.ipynb' -o -name '*.out'  -o -name '*.err' -o -name '*.log' -o -name 'run.sh' -o -name 'pms.yaml' \) "
        com = f"find {ind}" + r" -type d -name '*_logs' "
        if simulate:
            print(run_com(com))
        else:
            com += f" -print0 | xargs -0 zip -r -m -q {outp}"
            print(run_com(com))

        if validate:
            read_ps(ind, tree_depth=3)

    if arxv:
        assert params is not None
        params = pre_params(
            params,
            flt_output_exists=True,  # completed, output exists
            force=True,  ## do not filter
        )
        from roux.lib.io import to_arxv

        for pms in params:
            to_arxv(
                pms["output_path"],
                outp=None,
                simulate=simulate,
                verbose=verbose,
                force=False,
                wait=True,
                exclude=[
                    "'*.bam'",
                    "'*.fastq'",
                    "'*.fq'",
                    "'*.h5'",
                    "'*.h5mu'",
                    "'*.h5ad'",
                    "'*.adata'",
                ],
            )
            if simulate:
                break

        # return outp


## wrapper
def _skipped_by_env(script_path) -> str | None:
    """
    Whether a `run_tasks` target is skipped by the environment variables `ROUX_SKIP_TASKS`/`ROUX_ONLY_TASKS`.

    Parameters:
        script_path: the target path (as resolved by `run_tasks`).

    Returns:
        the name of the variable that skips it, else None.

    Notes:
        Both variables are comma-separated fnmatch patterns, each matched against the path, the file name and
        the file stem. A match in `ROUX_SKIP_TASKS` wins over `ROUX_ONLY_TASKS`.
    """
    from fnmatch import fnmatch

    def _patterns(name):
        return [s.strip() for s in os.environ.get(name, "").split(",") if s.strip()]

    p = Path(str(script_path))
    keys = [str(script_path), p.name, p.stem]

    def _matches(patterns):
        return any(fnmatch(k, pat) for pat in patterns for k in keys)

    if _matches(_patterns("ROUX_SKIP_TASKS")):
        return "ROUX_SKIP_TASKS"
    only = _patterns("ROUX_ONLY_TASKS")
    if only and not _matches(only):
        return "ROUX_ONLY_TASKS"
    return None


def run_tasks(
    script_path: str,  ## prefix
    params=None,
    runner=None,  ## py, bash, slurm (None:auto)
    cpus: int = 1,
    kernel: str = None,
    ## ipynb
    ## kws_run
    pre: bool = True,
    no_pre: bool = False,
    post_arxv: bool = False,
    no_post_arxv: bool = False,
    post_clean: bool = False,
    post_nb: bool = False,
    ## slurm
    script_pre: str = "",  ## e.g. micromamba run -n env
    ## for slurm if available
    slurm_header="",
    slurm_kws=dict(
        # cpus=1,
        # mem="5gb",
        # time="01:00:00",
    ),
    ## slurm feeding
    # feed_duration : str ='1h', #hr
    # feed_interval : str ='10m',
    # feed_if_jobs_max : float =0.5,
    ## cfg_run
    script_type: str = None,  ## preffix
    ## common
    force_setup: bool = True,
    no_force_setup: bool = False,
    cache_dir_path=None,  ## None: $ROUX_CACHE_DIR, else '~/scratch/.roux'
    wd_path=None,
    ## back-compatible
    parameters_list=None,
    ## exec.
    force: bool = False,
    simulate: bool = False,
    verbose: bool = False,
    log_level: str = "INFO",
    ## pre_params
    flt_input_exists=False,  # should input_path exist
    test1: bool = False,
    testn: int = None,
    test: bool = False,  ## saves more log files
    test_cpus: int = 3,
    **kws_runner,
):
    """
    Run multipliers.

    Args:
        script_path:
        params: params or run_cfg

    Notes:
        Slurm script is created even if runner=='bash'

        `kernel` is used by every runner (for `bash` and `slurm`, as papermill's `--kernel`); if None, the
        notebooks run with the kernel in their metadata.

        `cache_dir_path` (logs and run scripts): if None, the environment variable `ROUX_CACHE_DIR`, else
        '~/scratch/.roux'. Being inherited, the variable also applies to the nested calls (notebooks that call
        `run_tasks`), e.g. to keep the cache inside the working directory:

            ROUX_CACHE_DIR=.cache/roux python runner.py

        Skipping or selecting the targets (also of the nested calls) with environment variables, each a list of
        comma-separated fnmatch patterns, matched against the resolved path, the file name or the file stem:
        `ROUX_SKIP_TASKS` skips the matching targets; `ROUX_ONLY_TASKS`, if set and non-empty, skips the
        targets that match none of its patterns. A match in `ROUX_SKIP_TASKS` wins. A skipped call logs a
        warning and returns None (as when there is nothing to run), e.g.

            ROUX_SKIP_TASKS="043_results_data,*/notebooks_scripts/*" python runner.py

        Back-compatible: input_notebook_path:script_path

            grep -rl 'input_notebook_path=' ../*.ipynb
            sed -i -e 's/input_notebook_path=//g' ../*.ipynb

            sed -i -e 's/parameters_list=/params=/g' ../*.ipynb

    Examples:
        Feeding:
            feed_duration = '99h',
            feed_interval = '1s',
            feed_if_jobs_max = 0.5,
    """
    # arg. dtype
    if testn == "None":
        testn = None

    if simulate:
        test = True
        verbose = True

    ## CLI: negatives override
    if no_post_arxv:
        post_arxv = False
    if no_pre:
        pre = False
    if no_force_setup:
        force_setup = False

    if not test1:
        if cache_dir_path is None:
            ## configurable for the nested calls
            cache_dir_path = os.environ.get("ROUX_CACHE_DIR") or "~/scratch/.roux"
        cache_dir_path = f"{Path(cache_dir_path).expanduser().as_posix()}"
    else:
        cache_dir_path = None
        logging.warning("cache_dir_path=None")

    # ## script_path
    # # g: Imported as an alias to prevent AttributeError if the 'logging' variable was accidentally reassigned to a Logger instance
    # print()
    # if log_level is None:
    import logging as _logging

    log_level_root = _logging.getLevelName(_logging.getLogger().getEffectiveLevel())
    if log_level != log_level_root:
        log_level = log_level_root

    logging.setLevel(level=log_level)

    script_path = script_path.strip().split(" ", 1)
    if len(script_path) == 1:
        sub_com = ""
        script_path = script_path[0]
    else:
        script_path, sub_com = script_path

    if Path(script_path).suffix != "":
        assert Path(script_path).exists(), script_path

    script_type = Path(script_path.split(" ")[0]).suffix[
        1:
    ]  # if not '.py run' in script_path else 'py'
    if verbose:
        logging.info(f"script_type='{script_type}' sub_com='{sub_com}'")

    if script_type != "":
        script_path = Path(script_path).resolve().as_posix()

    if verbose:
        logging.info(f"script_path='{script_path}'")

    ## skip/select by the environment variables (inherited by the nested calls)
    skipped_by = _skipped_by_env(script_path)
    if skipped_by is not None:
        logging.warning(f"skipped by ${skipped_by}: {script_path}")
        return

    cwd_path = Path().cwd().as_posix()

    if wd_path is None:
        wd_path = cwd_path
    else:
        wd_path = Path(wd_path).as_posix()
        os.chdir(wd_path)

    runner = infer_runner(
        runner=runner,
        script_type=script_type,
    )

    ## params
    if params is None and parameters_list is not None:
        params = parameters_list
        del parameters_list
        logging.warning("arg: parameters_list will be depr.d.")
    # if isinstance(params,str):
    #     from roux.lib.io import is_dict
    #     assert is_dict(params), f"expected params in dict format: {params}"
    #     params=read_dict(params)

    # if contains_keys(params,['pms_run','kws_run']):
    from roux.lib.io import is_dict

    if is_dict(script_path):
        # cfg_run:
        # recurse
        cfg_run = read_dict(script_path)
        del script_path

        cfg_run = pre_cfg_run(
            cfg_run,
            script_type=script_type,
        )

        dfs_run = {}
        for step in tqdm(cfg_run, disable=len(cfg_run) <= 1):
            logging.processing(step)

            dfs_run[step] = run_tasks(
                params=cfg_run[step]["pms_run"],
                **{
                    **dict(
                        runner=runner,
                        force_setup=force_setup,  # True,
                        cache_dir_path=cache_dir_path,  # roux/',
                        # wd_path = wd_path,# None,
                        force=force,  # False,
                        simulate=simulate,  #  False,
                        verbose=verbose,  #  False,
                        log_level=log_level,  #  'INFO',
                    ),
                    **cfg_run[step]["kws_run"],
                    # **dict(
                    #     script_path=sp
                    # )
                },
            )
            if not simulate:
                ## wait
                while not Path(cfg_run[step]["pms_run"]["output_path"]).exists():
                    time.sleep(2)
            else:
                if not Path(cfg_run[step]["pms_run"]["output_path"]).exists():
                    test_params([cfg_run[step]["pms_run"]])

        return pd.concat(dfs_run)

    params = pre_params(
        params=params,
        # inputs=inputs,
        # output_path_base=output_path_base,
        verbose=verbose,
        force=force,
        test1=test1,
        testn=testn,
        flt_input_exists=flt_input_exists,
    )

    if len(params) == 0:
        return

    kws_runner["verbose"] = verbose

    # if test:
    #     cache_dir_path=f"{wd_path}/{cache_dir_path}"
    if runner in ["slurm", "bash"] and script_type in ["ipynb"] and kernel is None:
        ## use py because of possible kernel issues
        ## infer py
        from roux.workflow.io import to_py_path

        script_path_ = to_py_path(script_path)
        assert Path(script_path_).exists(), script_path_
        if Path(script_path).exists():
            logging.warning(f"script_path -> {script_path_}")
        script_path = script_path_
        del script_path_
        script_type = "py"
    else:
        ## run nbs
        if kernel is None:
            ## inferring kernel
            kernel = Path(os.environ.get("VIRTUAL_ENV")).parent.stem
            logging.info(f"inferred kernel: {kernel}")

    if runner.startswith("py") and script_type == "ipynb":
        from roux.lib.sys import is_interactive_notebook

        test = is_interactive_notebook()

        res = run_tasks_nb(
            script_path,
            params=params,
            **{
                **dict(
                    ## logs
                    # cache_dir_path=cache_dir_path,
                    kernel=kernel,
                    cpus=cpus,
                    pre=pre,
                    post=post_nb,
                    simulate=simulate,
                    test1=test1,
                    force=force,
                    test=test,
                    verbose=verbose,
                ),
                **kws_runner,
            },
        )
        if wd_path != cwd_path:
            ## back to current work dir
            os.chdir(cwd_path)
        return res

    logging.loading("params from the input_path ..")

    if isinstance(params, dict):
        params = list(params.values())

    _time_start = logging.configuring("paths ..", time=True)

    Path(cache_dir_path).mkdir(parents=True, exist_ok=True)

    if runner == "slurm":
        logging.launching("jobs ..", n=min([cpus, 5]))

    logging.status(f"processing on {runner}, {cpus} at a time ..", n=min([cpus, 5]))

    coms = []
    # if runner in ['slurm','bash']:
    ## recursive
    assert isinstance(cpus, int), cpus

    kws_runner = {
        **dict(
            script_path=" ".join(
                [s for s in [script_path, sub_com] if s != ""]
            ),  # , ## preffix
            script_pre=script_pre,  # ='', ## e.g. micromamba run -n env
            kernel=kernel,
            append_header=slurm_header,
            post_arxv=post_arxv,
            post_clean=post_clean,
        ),
        **kws_runner,
        **slurm_kws,
    }

    ## feed
    ## each round feed cpus jobs from run_tasks
    # from random import shuffle
    # shuffle(params)
    # if isinstance(params,dict):
    #     job_keys_tried=get_tried_job_keys(
    #         cache_dir_path
    #     )
    #     logging.info(f"found {len(job_keys_tried)} tried jobs, they will be deprioritized.")
    #     params={
    #         **{k:d for k,d in params.items() if k not in job_keys_tried},
    #         **{k:d for k,d in params.items() if k in job_keys_tried},
    #     }
    # if not simulate:
    #     if runner in ['slurm']:
    #         feed_jobs(
    #             com=params, ## all
    #             jobs=cpus, ## feed each time
    #             # user=user,
    #             feed_duration=feed_duration,
    #             feed_interval=feed_interval,
    #             feed_if_jobs_max=feed_if_jobs_max,

    #             force= force, #=False,
    #             test= test, #=False,

    #             kws_runner=kws_runner,
    #         )

    #         return

    if runner == "slurm":
        logging.status("q.ing jobs.. ")

    sbatch_paths = []
    params_jobs = {}
    job_ids = []

    _logged = False
    for pms in tqdm(params, disable=len(params) <= 1):
        log_dir_path = pre_task(
            pms,
            cache_dir_path=cache_dir_path,
            test=test,
        )

        # key=encode(
        #     pms,#['output_path']
        #     short=True,
        # )
        sbatch_path = f"{log_dir_path}/run.sh"

        if not Path(sbatch_path).exists() or force_setup:
            if runner != "slurm" and not _logged:
                # logging.warning("forcing setup (re-rewiting the sbatch scripts)..")
                _logged = True
            # job=
            # if not simulate:
            to_sbatch_script(
                sbatch_path=sbatch_path,
                pms=pms,
                log_dir_path=log_dir_path,
                **kws_runner,
            )

        if runner == "slurm":
            if not simulate:
                # submit the jobs
                job_ids.append(submit_job(sbatch_path))

        if runner != "slurm":
            coms.append(
                f"bash '{sbatch_path}'",
            )

        sbatch_paths.append(sbatch_path)
        params_jobs[sbatch_path] = pms

    if len(coms) == 0:
        logging.warning("len(coms)==0")

    if runner != "slurm":
        cpus_bash = min([cpus, test_cpus])

        if cpus > cpus_bash:
            logging.warning(f"using test_cpus = {cpus_bash}")

        if not simulate:
            from multiprocessing import Pool

            # Create a pool of worker processes
            with Pool(processes=cpus_bash) as pool:
                pool.map(run_com, coms)

    # logging.saving(f"params_jobs_path={params_jobs_path}")
    # params_jobs_path=f"{cache_dir_path}/{encode(params_jobs,short=True)}.yaml"
    # to_dict(
    #     params_jobs,
    #     params_jobs_path,
    # )

    # logging.info("jobs submitted.")
    if runner == "slurm":
        logging.info(get_sq())

    # logging.saving('outputs.')
    if not simulate:
        logging.done(Path(script_path).stem, time=_time_start)

    # if wd_path!=cwd_path:
    #     ## back to current work dir
    #     os.chdir(cwd_path)

    ## uniform output
    return pd.Series(params_jobs).to_frame("params")["params"]  # .apply(pd.Series)


## running the cells of a notebook as a script
def _read_dict_arg(d) -> dict:
    """A dict from a dict, a JSON string or the path to a JSON/YAML file (e.g. from the CLI)."""
    if d is None or d == "None":
        return {}
    if isinstance(d, dict):
        return d
    if Path(d).is_file():
        return read_dict(d)
    from roux.workflow.pms import read_cli_pm

    d = read_cli_pm(d)
    assert isinstance(d, dict), d
    return d


def _dry_run_tasks(*args, **kws):
    """Stand-in for `run_tasks`: logs the target, the parameters and whether the path-like values exist."""
    params = kws.pop("params", None) or (args[1] if len(args) > 1 else {})
    logging.info(f"run_tasks (dry): {args[:1]} {kws}")
    for k, pms in params.items() if isinstance(params, dict) else enumerate(params):
        lines = [
            f"  {kk}: {v}"
            + (
                " (exists)"
                if isinstance(v, str) and "/" in v and Path(v).exists()
                else ""
            )
            for kk, v in pms.items()
        ]
        logging.info(f"## {k}\n" + "\n".join(lines))


def run_cells(
    nb_path: str,
    tags: str = None,
    params=None,
    values=None,
    run_tasks_kws=None,
    dry: bool = False,
):
    """Run code cells of a notebook as a plain Python script (no kernel or papermill), e.g. one stage of a
    runner notebook that calls `run_tasks`.

    Parameters:
        nb_path (str): path to the notebook.
        tags (str|list): run the code cells with any of these tags (comma-separated), in order; None: all.
        params (dict|str): set after the cell tagged `parameters`, as papermill does (dict, JSON or a JSON/YAML file).
        values (dict|str): set after the cells tagged `setup` e.g. `{"kernel": "python3", "force": true}`.
        run_tasks_kws (dict|str): added to every `run_tasks` call, over the call's own e.g. `{"runner": "py"}`.
        dry (bool): `run_tasks` only logs its parameters and whether the path-like ones exist.

    Notes:
        Runs from the notebook's folder (the cwd is restored after). Lines starting with `%` or `!` are skipped.
        The folder of the current Python is put first in `PATH`, so a kernel started as `python` (e.g. `python3`)
        uses this environment. `MPLBACKEND` defaults to `Agg`.
        For the cache of the nested `run_tasks` calls, set `ROUX_CACHE_DIR`.

    Usage:
        roux run-cells notebooks/run.ipynb --tags setup,ids_prio --values '{"kernel": "python3"}' --dry
    """
    import json
    import roux.workflow.task as task_module

    from roux.workflow.pms import read_cli_pm

    params, values, run_tasks_kws = [
        _read_dict_arg(d) for d in [params, values, run_tasks_kws]
    ]
    dry = read_cli_pm(dry) if isinstance(dry, str) else dry
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]

    ## cells
    nb_path = Path(nb_path).resolve()
    cells = [c for c in json.load(open(nb_path))["cells"] if c["cell_type"] == "code"]
    if tags:
        cells = [c for c in cells if set(tags) & set(c["metadata"].get("tags", []))]
        tags_found = {t for c in cells for t in c["metadata"].get("tags", [])}
        assert set(tags) <= tags_found, f"tags not found: {set(tags) - tags_found}"
    logging.info(f"{len(cells)} cells of {nb_path.name}")

    ## run_tasks used by the cells
    run_tasks_ = task_module.run_tasks
    if dry:
        run_tasks_cells = _dry_run_tasks
    elif run_tasks_kws:

        def run_tasks_cells(*args, **kws):
            return run_tasks_(*args, **{**kws, **run_tasks_kws})

    else:
        run_tasks_cells = run_tasks_
    modules = [task_module] + [
        sys.modules[k] for k in ["roux.global_imports", "roux.run"] if k in sys.modules
    ]

    os.environ["PATH"] = (
        os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]
    )
    os.environ.setdefault("MPLBACKEND", "Agg")
    cwd_path = os.getcwd()
    g = {"__name__": "__main__"}
    try:
        for m in modules:
            m.run_tasks = run_tasks_cells
        os.chdir(nb_path.parent)
        for i, cell in enumerate(cells):
            if "run_tasks" in g:
                g["run_tasks"] = run_tasks_cells
            source = "\n".join(
                line
                for line in "".join(cell["source"]).splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            exec(compile(source, f"{nb_path.name}[cell {i}]", "exec"), g)
            cell_tags = cell["metadata"].get("tags", [])
            if "parameters" in cell_tags:
                g.update(params)
            if "setup" in cell_tags:
                g.update(values)
    finally:
        for m in modules:
            m.run_tasks = run_tasks_
        os.chdir(cwd_path)

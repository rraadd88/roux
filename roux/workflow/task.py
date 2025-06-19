"""For task management."""

## logging
# try:
from roux.lib.log import Logger
logging = Logger() # level='INFO'
# except:
    # import logging #noqa
    # logging.basicConfig(level='INFO', force=True)

## internal
from roux.lib.io import read_dict, to_dict
from roux.workflow.log import test_params

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

def pre_params(
    params=None,
    inputs=None,
    output_path_base=None,
    verbose=False,
    force=False,
    test1: bool = False,
    testn: int = None,
):
    """
    Unified pre-processing for params, used by both run_tasks_nb and run_tasks.
    Handles conversion, checks, output path inference, and filtering (including test1/testn).
    Returns a list of parameter dicts ready for execution.
    """
    # --- Handle input formats and output path inference ---
    param_list = params

    if param_list is None and inputs is not None and output_path_base is not None:
        from roux.lib.sys import to_output_paths
        param_list = to_output_paths(
            inputs=inputs,
            output_path_base=output_path_base,
            encode_short=True,
            key_output_path="output_path",
            verbose=verbose,
            force=force,
        )
        # Optionally save all parameters (as in run_tasks_nb)
        for k, parameters in param_list.items():
            output_dir_path = output_path_base.split("{KEY}")[0]
            to_dict(
                parameters,
                f"{output_dir_path}/{k.split(output_dir_path)[1].split('/')[0]}/.parameters.yaml",
            )

    if isinstance(param_list, str):
        param_list = read_dict(param_list)

    if not param_list or (isinstance(param_list, (list, dict)) and len(param_list) == 0):
        logging.info("nothing to process. use `force`=True to rerun.")
        return []

    # --- Convert dict to list if needed ---
    if isinstance(param_list, dict):
        if 'input_path' in param_list and 'output_path' in param_list:
            ## pms
            param_list=[param_list]
        else:
            if not any(['input_path' in d for d in param_list.values()]):
                logging.warning("setting keys of params as input_path s ..")
                param_list = {k: {**d, **{'input_path': k}} for k, d in param_list.items()}
            if validate_params(list(param_list.values())[0]):
                param_list = list(param_list.values())
            else:
                raise ValueError(param_list)

    # --- Filtering by output existence, as in flt_params ---
    before = len(param_list)
    param_list = [
        d
        for d in param_list
        if (force if force else not Path(d["output_path"]).exists())
    ]
    if not force:
        if before - len(param_list) != 0:
            logging.info(
                f"parameters_list_flt reduced because force=False: {before} -> {len(param_list)}"
            )

    # --- Filtering by test1 and testn ---
    if test1:
        testn = 1
    if testn is not None:
        param_list = param_list[:testn]
        logging.warning(f"filtered to {len(param_list)} jobs ..")

    if len(param_list) == 0:
        # logging.info("No tasks remaining after filtering.")
        return []

    # --- Final assertions ---
    assert len(set([d["output_path"] for d in param_list])) == len(param_list), \
        "Duplicate output_path found in params."
    assert all([Path(d["input_path"]) != Path(d["output_path"]) for d in param_list]), \
        "Some input_path == output_path in params."

    return param_list

## execution
def run_task_nb(
    parameters: dict,
    script_path: str,
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
    if not output_notebook_path:
        ## save report i.e. output notebook
        output_notebook_path = f"{splitext(parameters['output_path'])[0]}_reports/{get_datetime()}_{basenamenoext(script_path)}.ipynb"
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
    # return parameters['output_path']
    return output_notebook_path

def apply_run_task_nb(
    x: str,
    script_path: str,
    kernel: str,
    force=False,
    **kws_papermill,
    ):
    try:
        return run_task_nb(
            x,
            script_path=script_path,
            kernel=kernel,
            force=force,
            **kws_papermill,
        )
    except Exception:
        raise RuntimeError(f"tb: check {x}")

def run_tasks_nb(
    script_path: str=None,
    params=None,

    ## kws_run
    kernel: str = None,
    cpus: int = 1,
    pre: bool = True,
    post: bool = False,

    simulate: bool = False, #~dry
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
    input_notebook_path: str=None, 
    # parameters_list=None, # same as params
    fast: bool = None, ## to be deprecated
    fast_workers: int = None, ## to be deprecated
    
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
    if script_path is None and input_notebook_path is not None:
        script_path=input_notebook_path
        logging.warning("input_notebook_path will be deprec.")    
        del input_notebook_path
    assert exists(script_path), script_path
    
    # assert not (params is not None and parameters_list is not None)
    # if params is not None and parameters_list is None:
    #     parameters_list=params
    #     del params

    # if test:
    #     force = True
    # ## save task in unique directories
    # if parameters_list is None:
    #     ## infer output paths
    #     from roux.lib.sys import to_output_paths

    #     parameters_list = to_output_paths(
    #         inputs=inputs,
    #         output_path_base=output_path_base,
    #         encode_short=True,
    #         key_output_path="output_path",
    #         verbose=verbose,
    #         force=force,
    #     )
    #     ## save all parameters
    #     for k, parameters in parameters_list.items():
    #         ## save parameters
    #         output_dir_path = output_path_base.split("{KEY}")[0]
    #         to_dict(
    #             parameters,
    #             f"{output_dir_path}/{k.split(output_dir_path)[1].split('/')[0]}/.parameters.yaml",
    #         )
    # # print(parameters_list)
    
    # if isinstance(parameters_list, str):
    #     parameters_list = read_dict(parameters_list)
        
    # if len(parameters_list) == 0:
    #     logging.info("nothing to process. use `force`=True to rerun.")
    #     return
        
    # if isinstance(parameters_list, dict):
    #     ## input_paths used as keys
    #     if not any(['input_path' in d for d in parameters_list.values()]):
    #         logging.warning("setting keys of params as input_path s ..")
    #         parameters_list={k:{**d,**{'input_path':k}} for k,d in parameters_list.items()}
    #     if validate_params(
    #         parameters_list[
    #             list(parameters_list.keys())[0]
    #         ]
    #         ):
    #         parameters_list = list(parameters_list.values())
    #     else:
    #         raise ValueError(parameters_list)
            
    # if test:
    #     logging.info("Aborting run because of the test mode")
    #     return parameters_list
        
    # if isinstance(parameters_list, list):
    #     parameters_list=flt_params(
    #         parameters_list,
    #         force=force,
    #     )
    #     if len(parameters_list) == 0:
    #         return 
    # else:
    #     raise ValueError(parameters_list)
    # ## chech for duplicate output paths
    # assert len(set([d["output_path"] for d in parameters_list])) == len(
    #     parameters_list
    # ), (len(set([d["output_path"] for d in parameters_list])), len(parameters_list))
    # ## input_path!=output_path
    # assert [Path(d["input_path"])!=Path(d["output_path"]) for d in parameters_list], [d for d in parameters_list if Path(d["input_path"])==Path(d["output_path"])]
    
    params=pre_params(
        params=params,
        inputs=inputs,
        output_path_base=output_path_base,
        verbose=verbose,
        force=force,
        test1 = test1,
        # testn = testn,
    )

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
        from roux.workflow.io import to_nb_kernel
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
        .to_frame('params')
    )
        
    if len(df1) == 0:
        logging.warning("No tasks remaining.")
        return 
        
    if test1:
        df1 = df1.head(1)
        logging.warning("testing only the first input.")
    
    ## backcompatibility
    if fast_workers is not None:
        cpus=fast_workers
    fast= cpus > 1
    
    if not simulate:
        if (not fast) or len(df1)==1:
            df1['nb path'] = getattr(
                df1['params'],
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
                'papermill',
                'papermill.translators',
                'papermill.utils',
            ]:
                logging_base.getLogger(k).setLevel(logging_base.CRITICAL)
            
            df1['nb path']=(
                df1
                .rd.apply_async(
                    lambda x: 
                        apply_run_task_nb(
                            x['params'],
                            script_path=script_path,
                            kernel=kernel,
                            **kws_papermill,
                            force=force,
                        ),
                    cpus=cpus,
                )
            )
        # return ds2
        
        if post and not fast:        
            from roux.workflow.io import valid_post_task_deps, to_html
            if valid_post_task_deps:
                df1['html path']=(
                    df1
                    .rd.apply_async(
                        lambda x: 
                            to_html(
                                x['nb path'],                    
                            ),
                        cpus=cpus,
                    )
                )
            
    ## log
    logging.info(f"Time taken: {datetime.now()-_start_time}")
    if not out_paths:
        return params
    else:
        if 'nb path' in df1:
            df1=df1.set_index('nb path')
        return df1['params']#.apply(pd.Series)

## server       
import os
import time

from tqdm import tqdm
# from pprint import pprint

from roux.lib.str import encode
from roux.lib.sys import run_com
# from roux.workflow.task import flt_params

from datetime import datetime, timedelta

# def flt_params(
#     parameters_list,
#     force=False,
#     verbose=False,
#     out_type=None,
# ):
#     before = len(parameters_list)
#     if isinstance(parameters_list,list):
#         ## TODO: use `to_outp`?
#         parameters_list_flt = [
#             d
#             for d in parameters_list
#             if (force if force else not Path(d["output_path"]).exists())
#         ]
#     elif isinstance(parameters_list,dict):
#         parameters_list_flt ={ 
#             k:d
#             for k,d in parameters_list.items()
#             if (force if force else not Path(d["output_path"]).exists())
#         }
#     else:
#         raise ValueError(type(parameters_list))
#     if not force:
#         if before - len(parameters_list_flt) != 0:
#             logging.info(
#                 f"parameters_list_flt reduced because force=False: {before} -> {len(parameters_list_flt)}"
#             )
#     if out_type == 'list':
#         if isinstance(parameters_list_flt,dict):
#             parameters_list_flt=list(parameters_list_flt.values())
# #     if verbose:
# #         # check_tasks
# #         outps=[pms['output_path'] for pms in parameters_list]
# #         outps_flt=[pms['output_path'] for pms in parameters_list_flt]
            
# #         for outp in outps:
# #             print(f"{not outp in outps_flt} :{outp}")
            
#     return parameters_list_flt

# # from functools import partial
# def check_tasks(
#     params,
#     ):
#     if isinstance(params,str):
#         params=read_dict(params)
        
#     if isinstance(params,dict):
#         params=list(params.values())

#     flt_params(
#         params,
#         verbose=True,
#         out_type='list',    
#     )

def get_sq(
    user=None,
    ):
    # Build the squeue command
    # --format="%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %R"
    
    return run_com(
        'sq --format="%.18i%.100j"' + (f'-u {user}' if user is not None else ''),
        verbose=False,
    )
    
def get_jobsn(
    user=None,
    verbose=False,
    ):
    lines = get_sq(
        user=user,
    ).splitlines()
    lines=[s for s in lines if "spawner-jupyte" not in s]
    if verbose:
        print(len(lines),lines)
    return len(lines)

def is_q_empty(
    user=None,
    verbose=False,
    ):
    """Check if the SLURM job queue is empty for a specific user (or globally)."""
    return get_jobsn(
        user=user,
        verbose=verbose,    
    ) == 0  # No jobs, just the header

def submit_job(p):
    """Submit a SLURM job using sbatch."""
    job_name=f"roux:{Path(p).stem}"
    if job_name in get_sq():
        logging.error(f"skipped because already running: {job_name}.")
        return 
    
    com = f'sbatch {p}'
    res=run_com(com)
    # Submitted batch job 3325831
    job_id=res.rsplit(' ')[-1]
    logging.info(f"job_id={job_id}")
    return job_id

class SLURMJob:
    def __init__(
        self, 
        job_name, 
        log_path, 
        
        mem="4G", 
        cpus=4, 
        time="01:00:00", 
        append_header="",
        
        ## not used often
        ntasks=1, 
        partition="default"
        ):
        
        self.job_name = job_name
        self.output = log_path
        
        self.time = time
        self.cpus = cpus
        self.mem = mem
        self.append_header = append_header
        
        self.ntasks = ntasks
        self.partition = partition
        self.commands = []

    def add_command(self, command):
        """Add a command to be executed in the SLURM script"""
        self.commands.append(command)

    def write_script(
        self,
        # com=None,
        outp,
        
        job_pre=None,
        modules=None,
        packages=None,
        
        ):
        """Generate the SLURM script"""
        
        # assert not (com is None and outp is None)
        
        # if outp is None and not com is None:
        #     outp=f".slurm/{encode(com)}.sh"
        modules_load_str=''
        packages_install_str=''
        
        if job_pre is None:
            job_pre='## not job_pre provided'
        
        if modules is not None:
            if len(modules)>0:
                modules_load_str+="module load "+(' '.join(modules))
        
        if packages is not None:
            if len(packages)>0:
                packages_install_str+="pip install "+(' '.join(packages))
        
        with open(outp, 'w') as f:
            f.write(
#SBATCH --partition={self.partition}
f"""#!/bin/bash
#SBATCH --job-name={self.job_name}
#SBATCH --err={self.output}/%j.err
#SBATCH --output={self.output}/%j.out                
#SBATCH --time={self.time}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus}
#SBATCH --mem={self.mem}
{self.append_header}

{job_pre}

{modules_load_str}

{packages_install_str}

""")
            for command in self.commands:
                f.write(f"{command}\n")
                
            # f.write("exit(0)\n")
            
    def submit(self, outp):
        """Submit a SLURM job using sbatch."""
        submit_job(outp)
    
def has_slurm(
    ):
    return run_com('sbatch --help',returncodes=[0,1,127],verbose=False)!=''

def infer_runner(
    runner=None,
    script_type=None,
):
    runner_in=runner
    del runner
    
    if runner_in=='slurm':
        if not has_slurm():
            runner='bash'
    elif runner_in is None:
        from roux.lib.sys import is_interactive_notebook
        if script_type in ['ipynb'] and is_interactive_notebook():
            runner='py' ## run_tasks_nb        
        # if script_type in ['sh','py',None]:
        else:
            runner='bash'
    else:
        runner=runner_in
            
    if runner_in is not None and runner!=runner_in:
        logging.warning(f'runner={runner}')    
    else:
        logging.debug(f'runner={runner}')
        
    return runner

def _expand_pms(
    pms
    ):
    return ' '.join([f"--{k.replace('_','-')} {v}" for k,v in pms.items()])
    
def to_sbatch_script(
    script_path,
    pms,
    sbatch_path, # outp
    script_pre=None,
    
    job_pre=None,
    modules=None,
    packages=None,

    # mem,# "5gb",
    # cpus,# 1,
    # time,# "01:00:00",
    append_header="",

    expand_pms=True, # argh

    force=False,
    test=False,
    verbose=False,
    **kws_runner,
):
    sbatch_dir_path=Path(sbatch_path).with_suffix("").as_posix()
    params_path=f"{sbatch_dir_path}/params.yaml"
    log_dir_path=f"{sbatch_dir_path}/logs/"

    Path(sbatch_dir_path).mkdir(parents=True,exist_ok=True)
    Path(log_dir_path).mkdir(parents=True,exist_ok=True)

    ## save parameters
    to_dict(
        pms,
        params_path,
    )
    if test:
        logging.info(f"params_path={params_path}")

    ## create command
    if script_path.endswith('.sh') or '.sh ' in script_path:
        com=f"{script_pre}{'bash ' if not script_path.startswith('bash ') else ''}{script_path} {params_path}"
        # pass
        # return "under dev"
    else:
        # if not any([s.startswith('python') for s in modules]):
        #     modules.append('python')
        if script_path.endswith('.py') or '.py ' in script_path:
            com=f"{script_pre}{'python ' if not script_path.startswith('python ') else ''}{script_path} "
            if not expand_pms:
                 com+=f" --pms {params_path}"
            else:
                 com+=_expand_pms(pms)   
            # packages.append('argh')
            # pass
            # return "under dev"
            # if verbose:
            #     print(com) 
        
        elif script_path.endswith('.ipynb'):
            log_path=f"{log_dir_path}/{Path(script_path).name}"
            com=(
                f"{script_pre}papermill --parameters_file {params_path} "
                f"--stdout-file {Path(log_path).with_suffix('.out')} --stderr-file {Path(log_path).with_suffix('.err')} "
                # kernel_name=kernel,
                "--start-timeout=600 "
                "--report-mode "
                # "--allow_errors=False "
                f"{script_path} {log_path}"
            )
            # --kernel
            # packages.append('papermill')
            # return "under dev"
            # Using -b or --parameters_base64, users can provide a YAML string, base64-encoded, containing parameter values.
            # $ papermill local/input.ipynb s3://bkt/output.ipynb -b YWxwaGE6IDAuNgpsMV9yYXRpbzogMC4xCg==
        else:
            raise ValueError(script_path)

    if test:
        logging.info(com)

    # write the slurm script 
    job = SLURMJob(
        job_name=f"roux:{Path(sbatch_path).stem}",
        log_path=log_dir_path,

        # mem=mem,#"4G",
        # cpus=cpus,#3,
        # time=time,#"01:00:00",
        append_header=append_header,
        
        **kws_runner,
    )
    # include run_multi()
    # import inspect
    # inspect.getsource(func)        
    job.add_command(com)

    job.write_script(
        sbatch_path,
        job_pre=job_pre,
        modules=modules,
        packages=packages,
    )

    # if verbose:
    #     print(dict(
    #         com=com,
    #         sbatch_path=sbatch_path,
    #         job_pre=job_pre,
    #         modules=modules,
    #         packages=packages,            
    #     ))
    return sbatch_path

def parse_time(duration_str):
    """
    Parse a string like '4h', '30m', or '2d' into a timedelta object.
    """
    time_map = {
        'h': 'hours',
        'm': 'minutes',
        's': 'seconds',
        'd': 'days',
    }
    unit = duration_str[-1]  # Last character indicates the unit
    value = int(duration_str[:-1])  # Everything before the last character is the value

    if unit not in time_map:
        raise ValueError(f"Invalid time unit: {unit}. Use 'h', 'm', 's', or 'd'.")

    return timedelta(**{time_map[unit]: value})

def get_tried_job_keys(
    cache_dir_path, #'.roux/'
    ):
    from roux.lib.io import read_ps
    return list(set([Path(p).parent.parent.stem for p in read_ps(f'{cache_dir_path}/*/logs/*.out')]))

def feed_jobs(
    com,
    
    # user,
    feed_duration,
    feed_interval='10m',
    jobs=200, # feed each time

    feed_if_jobs_max=0.5, ## wait till this many jobs are on q
    
    jobs_max=1000,
    test=False,
    force=False,
    
    kws_runner={},
):
    """Monitor SLURM queue and submit new jobs when queue is empty."""        
    duration = parse_time(feed_duration)
    interval = parse_time(feed_interval)
    
    start_time = datetime.now()
    end_time = start_time + duration

    logging.info(f"Start Time: {start_time}")
    logging.info(f"End Time: {end_time}")
    # logging.status('feeding start',time=True)
    # logging.status('feeding end',time=True)

    i=0
    while datetime.now() < end_time:
        
        remaining = end_time - datetime.now()
        # log only every second
        # if remaining.total_seconds() % 600 == 0:
        logging.status(f"Time remaining: {remaining}", end='\r')  
                    
        if get_jobsn()<=(jobs*feed_if_jobs_max) and get_jobsn()<jobs_max:
            logging.status("submitting new jobs...")

            if isinstance(com,str):
                if com.endswith('.yaml'):

                    logging.info("yaml config found")
                    # params=read_dict(com)
                    com=list(read_dict(com).values())
                    
                else:            
                    if not test:
                        if Path(com).exists():
                            submit_job(com)
                            
                        else:
                            run_com(
                                com
                            )
                    continue
            
            if isinstance(com,list): 
                params=com
                if i*jobs < len(params):
                    run_tasks(
                        params=params[(i)*jobs:(i+1)*jobs],
                        **kws_runner,
                        # **{'testn':jobs},
                    )
                    i+=1
                else:
                    logging.status("\nall jobs processed!")
                    break

        logging.status(f"{get_jobsn()} jobs are still running, waiting for {interval}s and jobs <= {jobs*feed_if_jobs_max} ..")
        time.sleep(interval.total_seconds())
        
    logging.info("\nDuration elapsed!")

from roux.lib.dict import contains_keys
## wrapper
def run_tasks(
    script_path: str, ## preffix
    params=None,

    runner=None, ## py, bash, slurm (None:auto)
    cpus: int = 1,
    kernel: str = None,

    ## ipynb
    ## kws_run
    pre: bool = True,
    post: bool = False,
        
    ## slurm
    script_pre : str ='', ## e.g. micromamba run -n env
    
    ## for slurm if available
    slurm_header= "",
    slurm_kws=dict(
        # cpus=1,
        # mem="5gb",
        # time="01:00:00",
    ),

    ## slurm feeding
    feed_duration : str ='1h', #hr 
    feed_interval : str ='10m',
    feed_if_jobs_max : float =0.5,

    ## cfg_run
    script_type: str=None, ## preffix
    
    ## common
    force_setup : bool =True,
    cache_dir_path='.roux/',
    wd_path=None,    
    
    force : bool =False,
    simulate: bool = False,
    
    verbose: bool = False,
    log_level: str = 'INFO',    
    
    test1 : bool =False,
    testn : int =None,
    test : bool =False,
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
        
    Examples:    
        Feeding:
            feed_duration = '99h',
            feed_interval = '1s',
            feed_if_jobs_max = 0.5,    
    """
    
    ## script_path
    logging.setLevel(level=log_level)
    script_path=Path(script_path).resolve().as_posix()
    script_type=Path(script_path.split(' ')[0]).suffix[1:]# if not '.py run' in script_path else 'py'
    
    runner=infer_runner(
        runner=runner,
        script_type=script_type,
    )
    
    ## params
    
    # if isinstance(params,str):
    #     from roux.lib.io import is_dict
    #     assert is_dict(params), f"expected params in dict format: {params}"
    #     params=read_dict(params)

    # if contains_keys(params,['pms_run','kws_run']):
    from roux.lib.io import is_dict
    if is_dict(script_path):
        # cfg_run:
        # recurse
        cfg_run=read_dict(script_path)
        del script_path
        dfs_run={}
        for step in cfg_run:
            logging.processing(step)
            
            sp=cfg_run[step]['kws_run']['script_path']
            if isinstance(sp,dict):
                if script_type is None:
                    st=list(sp.keys())[0]
                sp=sp[st]
                assert isinstance(sp,str), sp
                del st
                
            dfs_run[step]=run_tasks(
                params=cfg_run[step]['pms_run'],
                **{
                    **dict(
                        runner=runner,
                        
                        force_setup = force_setup,# True,
                        cache_dir_path = cache_dir_path, # roux/',
                        wd_path = wd_path,# None,
                        
                        force  = force,# False,
                        simulate = simulate,#  False,
                        
                        verbose = verbose,#  False,
                        log_level = log_level,#  'INFO',    
                        
                    ),
                    **cfg_run[step]['kws_run'],
                    **dict(
                        script_path=sp
                    )
                    },
                )
            if not simulate:
                ## wait 
                while not Path(cfg_run[step]['pms_run']['output_path']).exists():
                    time.sleep(2)
            else:
                test_params([cfg_run[step]['pms_run']])            
            del sp
            
        return pd.concat(dfs_run)
        
    params=pre_params(
        params=params,
        # inputs=inputs,
        # output_path_base=output_path_base,
        verbose=verbose,
        force=force,
        test1 = test1,
        testn = testn,
    )
    
    if runner.startswith('py'):
        # return
        return run_tasks_nb(
            script_path,
            params=params,
    
            kernel = kernel,
            cpus = cpus,
            pre = pre,
            post = post,

            simulate=simulate,
            test1 = test1,
            force = force,
            test = test,
            verbose=verbose,
            
            **kws_runner,
        )
        
    logging.loading('params from the input_path ..')
            
    if isinstance(params,dict):
        params=list(params.values())
                
    _time=logging.configuring("paths ..",get_time=True)
    
    if wd_path is None:
        wd_path=os.getcwd()
    cache_dir_path=f"{wd_path}/{cache_dir_path}"

    if runner=='slurm':
        logging.launching("jobs ..",n=min([cpus,5]))
    
    logging.processing(f"on {runner}, {cpus} at a time ..",n=min([cpus,5]))
    
    coms=[]
    if runner in ['slurm','bash']:
        ## recursive
        assert isinstance(cpus,int), cpus
        
        kws_runner={
            **dict(
                script_path= script_path, #, ## preffix
                script_pre= script_pre, #='', ## e.g. micromamba run -n env

                append_header=slurm_header,                       
            ),
            **kws_runner,
            **slurm_kws,
        }

        
        ## each round feed cpus jobs from run_tasks
        # from random import shuffle
        # shuffle(params)
        if isinstance(params,dict):
            job_keys_tried=get_tried_job_keys(
                cache_dir_path
            )
            logging.info(f"found {len(job_keys_tried)} tried jobs, they will be deprioritized.")
            params={
                **{k:d for k,d in params.items() if k not in job_keys_tried},
                **{k:d for k,d in params.items() if k in job_keys_tried},
            }
        if not simulate:
            if runner in ['slurm']:
                feed_jobs(
                    com=params, ## all
                    jobs=cpus, ## feed each time
                    # user=user,
                    feed_duration=feed_duration,
                    feed_interval=feed_interval,
                    feed_if_jobs_max=feed_if_jobs_max,
                    
                    force= force, #=False,
                    test= test, #=False,
    
                    kws_runner=kws_runner,
                )
                
                return

    if runner=='slurm':
        logging.status("q.ing jobs.. ")
    
    sbatch_paths=[]
    params_jobs={}
    job_ids=[]
            
    for pms in tqdm(params):
        key=encode(
            pms,#['output_path']
            short=True,
        )
        
        sbatch_path=f"{cache_dir_path}/{key}.sh"
        
        if not Path(sbatch_path).exists() or force_setup:
            if runner!='slurm':
                logging.warning("forcing setup (re-rewiting the sbatch scripts)..")
            # job=
            # if not simulate:
            to_sbatch_script(
                sbatch_path=sbatch_path,
                pms=pms,
                **kws_runner,
            )
    
        if runner=='slurm':
            if not simulate:
                # submit the jobs
                job_ids.append(
                    submit_job(
                        sbatch_path
                    )
                )
            
        if runner!='slurm':
            coms.append(
                f"bash {sbatch_path} &> {Path(sbatch_path).with_suffix('').as_posix()}/stdout",
            )
        
        sbatch_paths.append(sbatch_path)
        params_jobs[sbatch_path]=pms
        
    if len(coms)==0:
        logging.warning("len(coms)==0")
    
    if runner!='slurm':
        cpus_bash=min([cpus,test_cpus])
        
        if cpus>cpus_bash:
            logging.warning(f"using test_cpus = {cpus_bash}")
            
        if not simulate:
            
            from multiprocessing import Pool
            # Create a pool of worker processes
            with Pool(processes=cpus_bash) as pool:
                pool.map(run_com, coms)
        
    params_jobs_path=f"{cache_dir_path}/{encode(params_jobs,short=True)}.yaml"
    
    to_dict(
        params_jobs,
        params_jobs_path,
    )
    
    # logging.info("jobs submitted.")
    if runner=='slurm':
        logging.info(get_sq())  
        
    logging.saving(f"params_jobs_path={params_jobs_path}")
        
    # logging.saving('outputs.')
    logging.done('processing.',time=_time)

    ## uniform output
    return pd.Series(params_jobs).to_frame('params')['params']#.apply(pd.Series)

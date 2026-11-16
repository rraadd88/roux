"""
SLURM: job scripts, submission, queue state and feeding jobs to the queue (used by `run_tasks`).
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from roux.lib.io import read_dict, read_ps, to_dict
from roux.lib.log import Logger
from roux.lib.sys import run_com

logging = Logger()

## server
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

    import shlex

    return run_com(
        'sq --format="%.18i%.100j"'
        + (f" -u {shlex.quote(str(user))}" if user is not None else ""),
        verbose=False,
    )


def get_jobsn(
    user=None,
    verbose=False,
):
    lines = get_sq(
        user=user,
    ).splitlines()
    lines = [s for s in lines if not any([k in s for k in ["JOBID", "spawner-jupyte"]])]
    if verbose:
        print(len(lines), lines)
    return len(lines)


def is_q_empty(
    user=None,
    verbose=False,
):
    """Check if the SLURM job queue is empty for a specific user (or globally)."""
    return (
        get_jobsn(
            user=user,
            verbose=verbose,
        )
        == 0
    )  # No jobs, just the header


def submit_job(
    p,
    verbose=False,
):
    """Submit a SLURM job using sbatch."""
    # job_name=f"roux:{Path(p).stem}"
    job_name = f"roux:{Path(p).as_posix()}"
    if job_name in get_sq():
        logging.error(f"skipped because already running: {job_name}.")
        return

    import shlex

    com = f"sbatch {shlex.quote(str(p))}"
    res = run_com(
        com,
        verbose=verbose,
    )
    # Submitted batch job 3325831
    job_id = res.rsplit(" ")[-1]
    if verbose:
        logging.info(f"job_id={job_id}")
    return job_id


class SLURMJob:
    def __init__(
        self,
        log_path,
        job_name=None,
        mem="4G",
        cpus=4,
        time="01:00:00",
        append_header="",
        ## not used often
        ntasks=1,
        partition="default",
        post_arxv: bool = False,
        post_clean: bool = False,
    ):

        self.job_name = job_name
        self.log_path = log_path

        self.time = time
        self.cpus = cpus
        self.mem = mem

        ## could be a file path
        # self.append_header = (open(append_header).read() if ('/' in append_header and ' ' not in append_header) else append_header.replace(' #SBATCH','\n#SBATCH').replace('module','\nmodule'))
        self.append_header = (
            open(append_header).read()
            if (Path(append_header).is_file() and Path(append_header).exists())
            else append_header.replace(" #SBATCH", "\n#SBATCH").replace(
                "module", "\nmodule"
            )
        )

        self.ntasks = ntasks
        self.partition = partition
        self.commands = []

        self.post_arxv = post_arxv
        self.post_clean = post_clean

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
        verbose=False,
        job_name="name",  #'path',
    ):
        """Generate the SLURM script"""

        # assert not (com is None and outp is None)

        # if outp is None and not com is None:
        #     outp=f".slurm/{encode(com)}.sh"
        # if self.job_name is None:
        #     # try:
        #     #     self.job_name='/'.join(s.rsplit('/',3)[-3:-1])
        #     # except Exception as e:
        #     #     if verbose:
        #     #         logging.error(e)
        #     self.job_name=outp
        # self.job_name=f"roux:{self.job_name}"

        self.job_name = f"roux:{Path(outp).as_posix() if job_name == 'path' else Path(outp).name if job_name == 'name' else job_name}"

        modules_load_str = ""
        packages_install_str = ""

        if job_pre is None:
            job_pre = "## not job_pre provided"

        if modules is not None:
            if len(modules) > 0:
                modules_load_str += "module load " + (" ".join(modules))

        if packages is not None:
            if len(packages) > 0:
                packages_install_str += "pip install " + (" ".join(packages))

        with open(outp, "w") as f:
            f.write(
                # SBATCH --partition={self.partition}
                f"""#!/usr/bin/env bash

#SBATCH --job-name={self.job_name}
#SBATCH --err={self.log_path}/%j.err
#SBATCH --output={self.log_path}/%j.out                
#SBATCH --time={self.time}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus}
#SBATCH --mem={self.mem}

## overriding settings (--slurm-header)
{self.append_header}

## Stop script execution on any error
set -e

{job_pre}

{modules_load_str}

{packages_install_str}

# g: Redirects all subsequent stdout/stderr to both the file and the terminal.
# g: This avoids the pipefail masking issue entirely.     
exec > >(tee '{self.log_path}/stdout') 2>&1; 

"""
            )
            f.writelines(f"{command}\n" for command in self.commands)
            f.write(
                f"""
## archive the subdir (if job completed)
roux post-tasks -p '{self.log_path}/pms.yaml' {"--arxv" if self.post_arxv else ""} {"--clean" if self.post_clean else ""} 
"""
            )
            # f.write("exit(0)\n")

    def submit(self, outp):
        """Submit a SLURM job using sbatch."""
        submit_job(outp)


def has_slurm():
    return run_com("sbatch --help", returncodes=[0, 1, 127], verbose=False) != ""


def to_sbatch_script(
    script_path,
    pms,
    sbatch_path,  # outp
    log_dir_path,
    script_pre=None,  ## e.g. bash
    job_pre=None,  ## bash setup.sh
    modules=None,
    packages=None,
    # mem,# "5gb",
    # cpus,# 1,
    # time,# "01:00:00",
    append_header="",
    expand_pms=False,
    kernel=None,  ## for .ipynb, else the kernel in the notebook's metadata
    force=False,
    test=False,
    verbose=False,
    **kws_runner,
):
    # sbatch_dir_path=Path(sbatch_path).with_suffix("").as_posix()

    pms_path = f"{log_dir_path}/pms.yaml"
    assert Path(pms_path).exists(), "check pre_task()"

    if verbose:
        logging.debug(f"pms_path={pms_path}")

    ## create command
    if script_path.endswith(".sh") or ".sh " in script_path:
        com = f"{script_pre}{'bash ' if not script_path.startswith('bash ') else ''}{script_path} {pms_path}"
        # pass
        # return "under dev"
    else:
        # if not any([s.startswith('python') for s in modules]):
        #     modules.append('python')
        if script_path.endswith(".ipynb"):
            log_path = f"{log_dir_path}/{Path(script_path).name}"
            com = (
                f"{script_pre}papermill --parameters_file '{pms_path}' "
                f"--stdout-file '{Path(log_path).with_suffix('.out')}' --stderr-file '{Path(log_path).with_suffix('.err')}' "
                + (f"--kernel '{kernel}' " if kernel else "")
                + "--start-timeout=600 "
                "--report-mode "
                # "--allow_errors=False "
                f"'{script_path}' '{log_path}'"
                # f' || {{ echo "{script_path} crashed"; exit 1; }}'
            )
            # packages.append('papermill')
            # return "under dev"
            # Using -b or --parameters_base64, users can provide a YAML string, base64-encoded, containing parameter values.
            # $ papermill local/input.ipynb s3://bkt/output.ipynb -b YWxwaGE6IDAuNgpsMV9yYXRpbzogMC4xCg==

        elif script_path.endswith(".py") or ".py " in script_path:
            ## TODO: python is not expected in script_path anyways
            _cmd = script_path.split("python ")[-1].strip()
            com = f"{script_pre}python {_cmd} {'' if ' ' in _cmd else ' run'} "
            if not expand_pms:
                ## safer than expand_pms
                com += f" --input-path {pms_path}"
            else:
                from roux.workflow.pms import expand_pms

                com += expand_pms(pms)
            com = (
                f"{com}"
                # f' || {{ echo "{script_path} crashed"; exit 1; }}'
            )
        else:
            com = script_path
            if not expand_pms:
                ## safer than expand_pms
                com += f" --input-path {pms_path}"
            else:
                from roux.workflow.pms import expand_pms

                com += expand_pms(pms)
            com = (
                f"{com}"
                # f' || {{ echo "{script_path} crashed"; exit 1; }}'
            )
            # raise ValueError(script_path)

    if verbose:
        logging.debug(com)

    # write the slurm script
    job = SLURMJob(
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
        "h": "hours",
        "m": "minutes",
        "s": "seconds",
        "d": "days",
    }
    unit = duration_str[-1]  # Last character indicates the unit
    value = int(duration_str[:-1])  # Everything before the last character is the value

    if unit not in time_map:
        raise ValueError(f"Invalid time unit: {unit}. Use 'h', 'm', 's', or 'd'.")

    return timedelta(**{time_map[unit]: value})


def get_tried_job_keys(
    cache_dir_path,  #'.roux/'
):
    return list(
        set(
            [
                Path(p).parent.parent.stem
                for p in read_ps(f"{cache_dir_path}/*/logs/*.out")
            ]
        )
    )


def feed_jobs(
    com,
    # user,
    feed_duration,
    feed_interval="10m",
    jobs=200,  # feed each time
    feed_if_jobs_max=0.5,  ## wait till this many jobs are on q
    jobs_max=1000,
    test=False,
    force=False,
    kws_runner={},
    verbose=False,
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

    i = 0
    while datetime.now() < end_time:
        remaining = end_time - datetime.now()
        # log only every second
        # if remaining.total_seconds() % 600 == 0:
        logging.status(f"Time remaining: {remaining}", end="\r")

        if get_jobsn() <= (jobs * feed_if_jobs_max) and get_jobsn() < jobs_max:
            logging.status("submitting new jobs...")

            if isinstance(com, str):
                if com.endswith(".yaml"):
                    logging.info("yaml config found")
                    # params=read_dict(com)
                    com = list(read_dict(com).values())

                else:
                    if not test:
                        if Path(com).exists():
                            submit_job(com)

                        else:
                            run_com(
                                com,
                                verbose=verbose,
                            )
                    continue

            if isinstance(com, list):
                from roux.workflow.task import run_tasks

                params = com
                if i * jobs < len(params):
                    run_tasks(
                        params=params[(i) * jobs : (i + 1) * jobs],
                        **kws_runner,
                        # **{'testn':jobs},
                    )
                    i += 1
                else:
                    logging.status("\nall jobs processed!")
                    break

        logging.status(
            f"{get_jobsn()} job/s are still running, waiting for {interval}s and jobs <= {jobs * feed_if_jobs_max} .."
        )
        time.sleep(interval.total_seconds())

    logging.info("\nDuration elapsed!")

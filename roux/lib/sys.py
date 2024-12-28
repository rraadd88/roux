"""For processing file paths for example."""

# (str ->) sys -> io
## for file paths
from os.path import (
    exists,
    dirname,
    basename,
    abspath,
    isdir,
    splitext,
)  ## prefer `pathlib` over `os.path`
from pathlib import Path
from glob import glob
from roux.lib.str import replace_many, encode

#
import subprocess
import sys
import logging
import shutil


## for file paths
def basenamenoext(p):
    """Basename without the extension.

    Args:
        p (str): path.

    Returns:
        s (str): output.
    """
    return splitext(basename(p))[0]


def remove_exts(
    p: str,
):
    """Filename without the extension.

    Args:
        p (str): path.

    Returns:
        s (str): output.
    """
    # if not isinstance(p,str):
    #     p=str(p)
    # if exts is None:
    #     exts=Path(Path(p).name).suffixes
    # if isinstance(exts,(list,tuple)):
    #     e=''.join(Path(p).suffixes)
    # if p.endswith(e):
    #     p=p[:-len(e)]
    # return p
    while "." in Path(p).name:
        p = Path(p).with_suffix("")
    return p


def read_ps(
    ps,
    errors=None,
    tree_depth=None,
    test: bool = True,
    verbose: bool = True,
) -> list:
    """Read a list of paths.

    Parameters:
        ps (list|str): list of paths or a string with wildcard/s.
        test (bool): testing.
        verbose (bool): verbose.

    Returns:
        ps (list): list of paths.
    """
    if isinstance(ps, str):
        if "*" in ps:
            ps = glob(ps)
        else:
            if Path(ps).is_dir() and verbose:
                tree(ps,tree_depth=tree_depth)
            ps = [ps]
    
    if isinstance(ps,list):
        assert isinstance(ps[0],str), ps[0]
    
    ps = sorted(ps)
    if test or verbose:
        import pandas as pd

        ds1 = (
            pd.Series({p: p2time(p) if exists(p) else None for p in ps})
            .sort_values()
            .dropna()
        )
        if len(ds1) > 1:
            from roux.lib.str import get_suffix

            d0 = ds1.iloc[[0, -1]].to_dict()
            for k_, k, v in zip(
                ["oldest", "latest"], get_suffix(*d0.keys(), common=False), d0.values()
            ):
                logging.info(f"{k_}: {k}\t{v}")
        elif len(ds1) == 0:
            if errors=='raise':
                logging.error("paths do not exist.")
                return
            logging.warning("paths do not exist.")
    return ps


def to_path(
    s,
    replacewith="_",
    verbose=False,
    coff_len_escape_replacement=100,
):
    """Normalise a string to be used as a path of file.

    Parameters:
        s (string): input string.
        replacewith (str): replace the whitespaces or incompatible characters with.

    Returns:
        s (string): output string.
    """
    import re

    s = re.sub(r"(/)\1+", r"\1", s)  # remove multiple /'s
    if max([len(s_) for s_ in s.split("/")]) < coff_len_escape_replacement:
        s = (
            re.sub(r"[^\w+/.+-=]", replacewith, s)
            .replace("+", replacewith)
            .strip(replacewith)
        )
        s = re.sub(r"(_)\1+", r"\1", s)  # remove multiple _'s
    else:
        if verbose:
            logging.info("replacements not done; possible long IDs in the path.")
    return s.replace(f"/My{replacewith}Drive/", "/My Drive/")  # google drive


#     return re.sub('\W+',replacewith, s.lower() )

# alias to be deprecated in the future
make_pathable_string = to_path
# get_path=to_path


def makedirs(p: str, exist_ok=True, **kws):
    """Make directories recursively.

    Args:
        p (str): path.
        exist_ok (bool, optional): no error if the directory exists. Defaults to True.

    Returns:
        p_ (str): the path of the directory.
    """
    logging.debug(
        "makedirs will be deprecated in the future releases, use pathlib instead: Path(p).parent.mkdir(parents=True, exist_ok=True)"
    )
    from os import makedirs
    from os.path import isdir

    p_ = p
    if not isdir(p):
        p = dirname(p)
    if p != "":
        makedirs(p, exist_ok=exist_ok, **kws)
    return p_


def to_output_path(ps, outd=None, outp=None, suffix=""):
    """Infer a single output path for a list of paths.

    Parameters:
        ps (list): list of paths.
        outd (str): path of the output directory.
        outp (str): path of the output file.
        suffix (str): suffix of the filename.

    Returns:
        outp (str): path of the output file.
    """
    if outp is not None:
        return outp
    from roux.lib.str import get_prefix

    # makedirs(outd)
    ps = read_ps(ps)
    pre = get_prefix(ps[0], ps[-1], common=True)
    if outd is not None:
        pre = outd + (basename(pre) if basename(pre) != "" else basename(dirname(pre)))
    outp = f"{pre}_{suffix}{splitext(ps[0])[1]}"
    return outp


def to_output_paths(
    input_paths: list = None,
    inputs: list = None,
    output_path_base: str = None,
    encode_short: bool = True,
    replaces_output_path=None,
    key_output_path: str = None,
    force: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Infer a output path for each of the paths or inputs.

    Parameters:
        input_paths (list) : list of input paths. Defaults to None.
        inputs (list) : list of inputs e.g. dictionaries. Defaults to None.
        output_path_base (str) : output path with a placeholder '{KEY}' to be replaced. Defaults to None.
        encode_short: (bool) : short encoded string, else long encoded string (reversible) is used. Defaults to True.
        replaces_output_path : list, dictionary or function to replace the input paths. Defaults to None.
        key_output_path (str) : key to be used to incorporate output_path variable among the inputs. Defaults to None.
        force (bool): overwrite the outputs. Defaults to False.
        verbose (bool) : show verbose. Defaults to False.

    Returns:
        dictionary with the output path mapped to input paths or inputs.

    TODOs:
        1. Placeholders other than {KEY}.
    """
    output_paths = {}
    # path standardisation
    for i, _ in enumerate(inputs):
        for k, v in inputs[i].items():
            if k.endswith("_path") and isinstance(v, str):
                inputs[i][k] = str(Path(v))
            if k.endswith("_paths") and isinstance(v, list):
                inputs[i][k] = [str(Path(s)) for s in v]

    if isinstance(input_paths, list):
        ## transform input path
        l1 = {
            replace_many(
                p, replaces=replaces_output_path, replacewith="", ignore=False
            ): p
            for p in input_paths
        }
        ## test collisions
        assert len(l1) == len(input_paths), "possible duplicated output path"
        output_paths.update(l1)
        output_paths_exist = list(filter(exists, output_paths))
    if isinstance(inputs, list):
        ## infer output_path
        assert "*" not in output_path_base, output_path_base
        assert (
            "{KEY}" in output_path_base
        ), f"placeholder i.e. '{{KEY}}' not found in output_path_base: '{output_path_base}'"
        l2 = {
            output_path_base.format(KEY=encode(d.copy(), short=encode_short)): d.copy()
            for d in inputs
        }
        # if verbose:
        #     logging.info(l2.keys())
        ## test collisions
        assert len(l2) == len(
            inputs
        ), "possible duplicated inputs or collisions of the hashes"
        ## check existing output paths
        output_paths.update(l2)
        output_paths_exist = glob(output_path_base.replace("{KEY}", "*"))
    for k in output_paths:
        ## add output path in the dictionary
        if key_output_path is not None:
            output_paths[k][key_output_path] = k
    if force:
        return output_paths
    else:
        if verbose:
            logging.info(f"output_paths: {list(output_paths.keys())}")
            logging.info(f"output_paths_exist: {output_paths_exist}")

        # output_paths_not_exist=list(set(list(output_paths.keys())) - set(output_paths_exist))
        output_paths_not_exist = list(filter(lambda x: not exists(x), output_paths))
        if verbose:
            logging.info(f"output_paths_not_exist: {output_paths_not_exist}")
        if len(output_paths_not_exist) < len(output_paths):
            logging.info(
                f"size of output paths changed: {len(output_paths)}->{len(output_paths_not_exist)}, because {len(output_paths)-len(output_paths_not_exist)}/{len(output_paths)} paths exist. Use force=True to overwrite."
            )
        return {k: output_paths[k] for k in output_paths_not_exist}


def get_encoding(p):
    """Get encoding of a file.

    Parameters:
        p (str): file path

    Returns:
        s (string): encoding.
    """
    import chardet

    with open(p, "rb") as f:
        result = chardet.detect(f.read())
    return result["encoding"]


# ls
def get_all_subpaths(d=".", include_directories=False):
    """Get all the subpaths.

    Args:
        d (str, optional): _description_. Defaults to '.'.
        include_directories (bool, optional): to include the directories. Defaults to False.

    Returns:
        paths (list): sub-paths.
    """
    import os

    paths = []
    for root, dirs, files in os.walk(d):
        if include_directories:
            for d in dirs:
                path = os.path.relpath(os.path.join(root, d), ".")
                paths.append(path)
        for f in files:
            path = os.path.relpath(os.path.join(root, f), d)
            paths.append(path)
    paths = sorted(paths)
    return paths


def get_env(
    env_name: str,
    return_path: bool = False,
):
    """Get the virtual environment as a dictionary.

    Args:
        env_name (str): name of the environment.

    Returns:
        d (dict): parameters of the virtual environment.
    """
    import sys
    import os

    env = os.environ.copy()
    env_name_current = sys.executable.split("anaconda3/envs/")[1].split("/")[0]
    path = sys.executable.replace(env_name_current, env_name)
    if return_path:
        return dirname(path) + "/"
    env["CONDA_PYTHON_EXE"] = path
    if "anaconda3/envs" in env["PATH"]:
        env["PATH"] = env["PATH"].replace(env_name_current, env_name)
    elif "anaconda" in env["PATH"]:
        env["PATH"] = env["PATH"].replace(
            f"{sys.executable.split('/anaconda3')[0]}/anaconda3/bin",
            f"{sys.executable.split('/anaconda3')[0]}/anaconda3/envs/{env_name}/bin",
        )
    else:
        env["PATH"] = path.replace("/bin/python", "/bin") + ":" + env["PATH"]

    return env


def run_com(
    com: str, 
    env=None, 
    template=None,
    test: bool = False, 
    verbose: bool = True, 
    returncodes: list = [0],
    **kws,
    ):
    """Run a bash command.

    Args:
        com (str): command.
        env (str): environment name.
        test (bool, optional): testing. Defaults to False.

    Returns:
        output: output of the `subprocess.call` function.

    Examples:
        from string import Template
        com_template=Template(f"docker run -d -v {wd}:/data image bash -c '$com'")

    TODOs:
        1. logp
        2. error ignoring
    """
    if verbose:
        logging.info(com)
        
    if template is not None:
        com=template.safe_substitute(com=com)
        
    if env is not None:
        # logging.warning("env is not set.")
        response = subprocess.call(
            com,
            shell=True,
            env=get_env(env) if isinstance(env, str) else env if env is not None else env,
            stderr=subprocess.DEVNULL if not test else None,
            stdout=subprocess.DEVNULL if not test else None,
            **kws,
        )
        assert response == 0, f"Error: {com}" + (
            "\nset `test=True` for more verbose." if not test else ""
        )
    else:
        response = subprocess.run(
            com,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
            )
        # print(res.stdout)
        assert response.returncode in returncodes, response
        return response.stdout
    
# alias to be deprecated in the future
runbash=run_com

def runbash_tmp(
    s1: str,
    env: str,
    df1=None,
    inp="INPUT",
    input_type="df",
    output_type="path",
    tmp_infn="in.txt",
    tmp_outfn="out.txt",
    outp=None,
    force=False,
    test=False,
    **kws,
):
    """Run a bash command in `/tmp` directory.

    Args:
        s1 (str): command.
        env (str): environment name.
        df1 (DataFrame, optional): input dataframe. Defaults to None.
        inp (str, optional): input path. Defaults to 'INPUT'.
        input_type (str, optional): input type. Defaults to 'df'.
        output_type (str, optional): output type. Defaults to 'path'.
        tmp_infn (str, optional): temporary input file. Defaults to 'in.txt'.
        tmp_outfn (str, optional): temporary output file.. Defaults to 'out.txt'.
        outp (_type_, optional): output path. Defaults to None.
        force (bool, optional): force. Defaults to False.
        test (bool, optional): test. Defaults to False.

    Returns:
        output: output of the `subprocess.call` function.
    """
    if exists(outp) and not force:
        return
    import tempfile

    with tempfile.TemporaryDirectory() as p:
        if test:
            p = abspath("test/")
        makedirs(p)
        tmp_inp = f"{p}/{tmp_infn}"
        tmp_outp = f"{p}/{tmp_outfn}"
        s1 = replace_many(
            s1,
            {
                "INPUT": tmp_inp,
                "OUTPUT": tmp_outp,
            },
        )
        if df1 is not None:
            if input_type == "df":
                df1.to_csv(
                    replace_many(
                        inp,
                        {
                            "INPUT": tmp_inp,
                        },
                    ),
                    sep="\t",
                )
            elif input_type == "list":
                from roux.lib.set import to_list

                to_list(df1, replace_many(inp, {"INPUT": tmp_inp}))
        response = runbash(s1, env=env, test=test, **kws)
        if exists(tmp_outp):
            if output_type == "path":
                makedirs(outp)
                shutil.move(tmp_outp, outp)
                return outp
        else:
            logging.error(f"output file not found: {outp} ({tmp_outp})")


def create_symlink(
    p: str,
    outp: str,
    test=False,
    force=False,
):
    """Create symbolic links.

    Args:
        p (str): input path.
        outp (str): output path.
        test (bool, optional): test. Defaults to False.

    Returns:
        outp (str): output path.

    TODOs:
        Use `pathlib`: `Path(p).symlink_to(Path(outp))`
    """
    import os

    if not exists(p):
        logging.error(f"skipped: file does not exists {p}")
        return
    if exists(outp) and not force:
        if os.path.islink(outp):
            if os.readlink(outp) == abspath(p):
                logging.error(f"skipped: symlink exists {outp}")
                return
            else:
                logging.error(f"skipped: wrong symlink {os.readlink(outp)} not {outp}")
                return
        else:
            logging.error(f"skipped: file exists {outp}")
            return
    p, outp = abspath(p), abspath(outp)
    com = f"ln -s {p} {outp}"
    makedirs(outp)
    if test:
        print(com)
    os.system(com)
    return outp


def input_binary(q: str):
    """Get input in binary format.

    Args:
        q (str): question.

    Returns:
        b (bool): response.
    """
    reply = ""
    while reply not in ["y", "n", "o"]:
        reply = input(f"{q}:")
        if reply == "y":
            return True
        if reply == "n":
            return False
    return reply


def is_interactive():
    """Check if the UI is interactive e.g. jupyter or command line."""
    import __main__ as main

    return not hasattr(main, "__file__")


def is_interactive_notebook():
    """Check if the UI is interactive e.g. jupyter or command line.

    Notes:

    Reference:
    """
    return "ipykernel.kernelapp" in sys.modules


def get_excecution_location(depth=1):
    """Get the location of the function being executed.

    Args:
        depth (int, optional): Depth of the location. Defaults to 1.

    Returns:
        tuple (tuple): filename and line number.
    """
    from inspect import getframeinfo, stack

    caller = getframeinfo(stack()[depth][0])
    return caller.filename, caller.lineno


## time
## logging system
def get_datetime(
    outstr: bool = True,
    fmt="%G%m%dT%H%M%S",
):
    """Get the date and time.

    Args:
        outstr (bool, optional): string output. Defaults to True.
        fmt (str): format of the string.

    Returns:
        s : date and time.
    """
    import datetime

    time = datetime.datetime.now()
    if outstr:
        # from roux.lib.io import to_path # potential circular import
        # return to_path(str(time)).replace('-','_').replace('.','_')
        return time.strftime(fmt)
    else:
        return time


def p2time(filename: str, time_type="m"):
    """Get the creation/modification dates of files.

    Args:
        filename (str): filename.
        time_type (str, optional): _description_. Defaults to 'm'.

    Returns:
        time (str): time.
    """
    import os
    import datetime

    if time_type == "m":
        t = os.path.getmtime(filename)
    else:
        t = os.path.getctime(filename)
    return str(datetime.datetime.fromtimestamp(t))


def ps2time(ps: list, **kws_p2time):
    """Get the times for a list of files.

    Args:
        ps (list): list of paths.

    Returns:
        ds (Series): paths mapped to corresponding times.
    """
    import pandas as pd
    from glob import glob

    if isinstance(ps, str):
        if isdir(ps):
            ps = glob(f"{ps}/*")
    return (
        pd.Series({p: p2time(p, **kws_p2time) for p in ps})
        .sort_values()
        .reset_index()
        .rename(columns={"index": "p", 0: "time"})
    )


## logging
def get_logger(program="program", argv=None, level=None, dp=None):
    """Get the logging object.

    Args:
        program (str, optional): name of the program. Defaults to 'program'.
        argv (_type_, optional): arguments. Defaults to None.
        level (_type_, optional): level of logging. Defaults to None.
        dp (_type_, optional): _description_. Defaults to None.
    """
    log_format = "[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s"
    # def initialize_logger(output_dir):
    cmd = "_".join([str(s) for s in argv]).replace("/", "_")
    if dp is None:
        dp = ""
    else:
        dp = dp + "/"
    date = get_datetime()
    logp = f"{dp}.log_{program}_{date}_{cmd}.log"
    #'[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    #     # create error file handler and set level to error
    #     handler = logging.FileHandler(os.path.join(output_dir, "error.log"),"w", encoding=None, delay="true")
    #     handler.setLevel(logging.ERROR)
    #     formatter = logging.Formatter(log_format)
    #     handler.setFormatter(formatter)
    #     logger.addHandler(handler)

    # create debug file handler and set level to debug
    handler = logging.FileHandler(logp)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logp


def tree(
    folder_path: str,
    tree_depth: int=None,
    log=True,
):
    # Run the tree command and capture the output
    result = subprocess.run(
        f"tree {folder_path}"+(f' -L {tree_depth}' if tree_depth is not None else ''),
        shell=True, capture_output=True, text=True
    )
    ## clean
    out = result.stdout.replace("\n\n", "\n").strip("\n")
    if log:
        logging.info(out)
    else:
        return out

def grep(
    p: str,
    checks: list,
    exclude: list = [],
    exclude_str: list = [],
    verbose: bool = True,
) -> list:
    """
    To get the output of grep as a list of strings.

    Parameters:
        p (str): input path
    """
    from roux.lib.set import flatten
    import subprocess

    l2 = []
    for s in checks:
        # The command you want to execute
        command = f'grep -i "{s}" {p}'

        # Use subprocess.run to execute the command and capture the output
        completed_process = subprocess.run(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        lines = [
            s.replace('"', "").strip()
            for s in completed_process.stdout.split('\\n",\n')
        ]
        lines = [
            s for s in lines if s != "" and "#noqa" not in s
        ]  # and not s.startswith('#')]
        for k in exclude_str:
            lines = [s for s in lines if k not in s]  # and not s.startswith('#')]
        lines = flatten([s.split("\n") for s in lines])
        lines = list(set(lines) - set(exclude))
        lines = list(set(lines) - set(l2))
        if len(lines) > 0:
            # print(completed_process.stdout)
            # print(f"'{s}'")
            # if verbose:
            #     logging.info(basename(p), f"{s}: {lines}")
            l2 += lines  # [f"{s}: {lines}"]
    return l2

def resolve_paths(
    key_path, #with "PLACE_HOLDER"
    keys, #"PLACE_HOLDER"s'
    paths=None,
    ):
    if isinstance(keys,list):
        #recursive
        d={}
        paths=glob(replace_many(key_path,{k:'*' for k in keys}))
        assert len(paths)>0, key_path.replace(key,'*') 
        
        for k in keys:
            d[k]=resolve_paths(
                key_path, #with "PLACE_HOLDER"
                keys=k, #"PLACE_HOLDER"s
                paths=paths,
                )
        ## combine
        import pandas as pd
        return (
            pd.DataFrame(d)
            .T.to_dict(orient='list')
        )
            
    elif isinstance(keys,str):
        key=keys
    
    if paths is None:
        paths=glob(key_path.replace(key,'*'))
        assert len(paths)>0, key_path.replace(key,'*') 
    
    before_placeholder, after_placeholder = key_path.split(key)
    ## alt. by slashes
    spliti=before_placeholder.count('/')
    
    extracted_segments = {}
    for path in paths:
    #     # Remove prefix and suffix to isolate the PLACE_HOLDER part
    #     if before_placeholder in path and after_placeholder in path:
    #         start = path.find(before_placeholder) + len(before_placeholder)
    #         end = path.find(after_placeholder)
    #         extracted_segments[path[start:end]]=path
        value=path.split('/')[spliti]
        # extracted_segments[value]=path
        extracted_segments[path]=value
        
    return extracted_segments
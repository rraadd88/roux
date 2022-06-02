"""
io_sys -> io_files
"""
import subprocess
import sys
from os.path import exists,dirname,basename,abspath,isdir,realpath,splitext ## prefer `pathlib` over `os.path`
from pathlib import Path
from glob import glob,iglob
import logging
# from roux.lib.io import makedirs
# from roux.global_imports import info

# walker
def get_all_subpaths(d='.',include_directories=False):
    """Get all the subpaths.

    Args:
        d (str, optional): _description_. Defaults to '.'.
        include_directories (bool, optional): to include the directories. Defaults to False.

    Returns:
        paths (list): sub-paths.
    """
    from glob import glob
    import os
    paths=[]
    for root, dirs, files in os.walk(d):
        if include_directories:
            for d in dirs:
                path=os.path.relpath(os.path.join(root, d), ".")
                paths.append(path)
        for f in files:
            path=os.path.relpath(os.path.join(root, f), d)
            paths.append(path)
    paths=sorted(paths)
    return paths

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
    exts: tuple= None,
    ):
    """Filename without the extension.

    Args:
        p (str): path.
        exts (tuple): extensions.

    Returns:
        s (str): output.
    """
    return str(p).rstrip(''.join(Path(p).suffixes) if exts is None else exts)

def makedirs(p: str,exist_ok=True,**kws):
    """Make directories recursively.

    Args:
        p (str): path.
        exist_ok (bool, optional): no error if the directory exists. Defaults to True.

    Returns:
        p_ (str): the path of the directory.
    """
    from os import makedirs
    from os.path import isdir
    p_=p
    if not isdir(p):
        p=dirname(p)
    makedirs(p,exist_ok=exist_ok,**kws)
    return p_

def get_env(env_name: str):
    """Get the virtual environment as a dictionary.

    Args:
        env_name (str): name of the environment.

    Returns:
        d (dict): parameters of the virtual environment.
    """
    import sys,subprocess, os
    env = os.environ.copy()
    env_name_current=sys.executable.split('anaconda3/envs/')[1].split('/')[0]
    path=sys.executable.replace(env_name_current,env_name)
    env['CONDA_PYTHON_EXE']=path
    if 'anaconda3/envs' in env["PATH"]:
        env["PATH"]=env["PATH"].replace(env_name_current,env_name)
    elif 'anaconda' in env["PATH"]:
        env["PATH"]=env["PATH"].replace(f"{sys.executable.split('/anaconda3')[0]}/anaconda3/bin",
                                        f"{sys.executable.split('/anaconda3')[0]}/anaconda3/envs/{env_name}/bin")
    else:
        env["PATH"]=path.replace('/bin/python','/bin')+':'+env["PATH"]
    return env

def runbash(s1,env,test=False,**kws):
    """Run a bash command. 

    Args:
        s1 (str): command.
        env (str): environment name.
        test (bool, optional): testing. Defaults to False.

    Returns:
        output: output of the `subprocess.call` function.

    TODOs:
        1. logp
        2. error ignoring
    """
    if test:logging.info(s1)
    response=subprocess.call(s1, shell=True,
                           env=get_env(env) if isinstance(env,str) else env,
               stderr=subprocess.DEVNULL if not test else None, 
               stdout=subprocess.DEVNULL if not test else None,
               **kws)
    assert response==0, f"Error: {s1}"+('\nset `test=True` for more verbose.' if not test else '')
    return response

def runbash_tmp(s1: str,
            env: str,
            df1=None,
            inp='INPUT',
            input_type='df',
            output_type='path',
            tmp_infn='in.txt',
            tmp_outfn='out.txt',
            outp=None,
            force=False,
            test=False,
            **kws):
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
    from roux.lib.str import replace_many
    import shutil
    import tempfile
    with tempfile.TemporaryDirectory() as p:
        if test: p=abspath('test/')
        makedirs(p)
        tmp_inp=f"{p}/{tmp_infn}"
        tmp_outp=f"{p}/{tmp_outfn}"
        s1=replace_many(s1,{'INPUT':tmp_inp,
                            'OUTPUT':tmp_outp,
                           })
        if not df1 is None:
            if input_type=='df':
                df1.to_csv(replace_many(inp,{'INPUT':tmp_inp,}),sep='\t')
            elif input_type=='list':
                from roux.lib.set import to_list
                to_list(df1,replace_many(inp,{'INPUT':tmp_inp}))
        response=runbash(s1,env=env,
            test=test,
            **kws) 
        if exists(tmp_outp):
            if output_type=='path':
                makedirs(outp)
                shutil.move(tmp_outp,outp)
                return outp
        else:
            logging.error(f"output file not found: {outp} ({tmp_outp})")
            
def create_symlink(p: str,outp: str,test=False):
    """Create symbolic links.

    Args:
        p (str): input path.
        outp (str): output path.
        test (bool, optional): test. Defaults to False.

    Returns:
        outp (str): output path.
    """
    import os
    p,outp=abspath(p),abspath(outp)
    com=f"ln -s {p} {dirname(outp)}"
    if os.path.islink(outp):
        if os.readlink(outp)==abspath(p):
            return com
        else:
            logging.error(f"skipped: wrong symlink {os.readlink(outp)} not {outp}")
            return
    if exists(outp):
        logging.error(f"skipped: file exists {outp}")
        return
    if not exists(p):
        logging.error(f"skipped: file does not exists {p}")
        return 
    makedirs(abspath(dirname(outp)))
    if test: print(com)
    os.system(com)
    return outp

def input_binary(q:str):
    """Get input in binary format.

    Args:
        q (str): question.

    Returns:
        b (bool): response.
    """
    reply=''
    while not reply in ['y','n','o']:
        reply = input(f"{q}:")
        if reply == 'y':
            return True
        if reply == 'n':
            return False
    return reply

def is_interactive():
    """Check if the UI is interactive e.g. jupyter or command line. 
    """
    # thanks to https://stackoverflow.com/a/22424821/3521099
    import __main__ as main
    return not hasattr(main, '__file__')

def is_interactive_notebook():
    """Check if the UI is interactive e.g. jupyter or command line.     
    
    Notes:
        1. Difference in sys.module of notebook and shell
            'IPython.core.completerlib',
            'IPython.core.payloadpage',
            'IPython.utils.tokenutil',
            '_sysconfigdata_m_linux_x86_64-linux-gnu',
            'faulthandler',
            'imp',
            'ipykernel.codeutil',
            'ipykernel.datapub',
            'ipykernel.displayhook',
            'ipykernel.heartbeat',
            'ipykernel.iostream',
            'ipykernel.ipkernel',
            'ipykernel.kernelapp',
            'ipykernel.parentpoller',
            'ipykernel.pickleutil',
            'ipykernel.pylab',
            'ipykernel.pylab.backend_inline',
            'ipykernel.pylab.config',
            'ipykernel.serialize',
            'ipykernel.zmqshell',
            'storemagic'
        
        Code to find the difference:
            from roux.global_imports import *
            import sys
            with open('notebook.txt','w') as f:
                f.write('\n'.join(sys.modules))

            from roux.global_imports import *
            import sys
            with open('shell.txt','w') as f:
                f.write('\n'.join(sys.modules))
            set(open('notebook.txt','r').read().split('\n')).difference(open('shell.txt','r').read().split('\n'))    

    Reference:
        1. https://stackoverflow.com/a/22424821
    """
    return 'ipykernel.kernelapp' in sys.modules

def get_excecution_location(depth=1):
    """Get the location of the function being executed.

    Args:
        depth (int, optional): Depth of the location. Defaults to 1.

    Returns:
        tuple (tuple): filename and line number.
    """
    from inspect import getframeinfo, stack
    caller = getframeinfo(stack()[depth][0])
    return caller.filename,caller.lineno

## time
def get_time():
    """Gets current time in a form of a formated string. Used in logger function.
    """
    from roux.lib.io import to_path # potential circular import
    import datetime
    time=to_path('%s' % datetime.datetime.now())
    return time.replace('-','_').replace(':','_').replace('.','_')

## logging system
def get_datetime(outstr=True):
    """Get the date and time.

    Args:
        outstr (bool, optional): string output. Defaults to True.

    Returns:
        s : date and time.
    """
    from roux.lib.io import to_path # potential circular import
    import datetime
    time=datetime.datetime.now()
    if outstr:
        return to_path(str(time)).replace('-','_')
    else:
        return time

def p2time(filename: str,time_type='m'):
    """Get the creation/modification dates of files.

    Args:
        filename (str): filename.
        time_type (str, optional): _description_. Defaults to 'm'.

    Returns:
        time (str): time.
    """
    import os
    import datetime
    if time_type=='m':
        t = os.path.getmtime(filename)
    else:
        t = os.path.getctime(filename)
    return str(datetime.datetime.fromtimestamp(t))

def ps2time(ps: list,**kws_p2time):
    """Get the times for a list of files. 

    Args:
        ps (list): list of paths.

    Returns:
        ds (Series): paths mapped to corresponding times.
    """
    import pandas as pd
    from glob import glob
    if isinstance(ps,str):
        if isdir(ps):
            ps=glob(f"{ps}/*")
    return pd.Series({p:p2time(p,**kws_p2time) for p in ps}).sort_values().reset_index().rename(columns={'index':'p',0:'time'})
    

def get_logger(program='program',argv=None,level=None,dp=None):
    """Get the logging object.

    Args:
        program (str, optional): name of the program. Defaults to 'program'.
        argv (_type_, optional): arguments. Defaults to None.
        level (_type_, optional): level of logging. Defaults to None.
        dp (_type_, optional): _description_. Defaults to None.
    """
    log_format='[%(asctime)s] %(levelname)s\tfrom %(filename)s in %(funcName)s(..):%(lineno)d: %(message)s'
# def initialize_logger(output_dir):
    cmd='_'.join([str(s) for s in argv]).replace('/','_')
    if dp is None:
        dp=''
    else:
        dp=dp+'/'
    date=get_datetime()
    logp=f"{dp}.log_{program}_{date}_{cmd}.log"
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

# log
from icecream import ic as info
info.configureOutput(prefix='INFO:icrm:')
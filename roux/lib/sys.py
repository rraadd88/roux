"""
io_sys -> io_files
"""
import subprocess
import sys
from os.path import exists,dirname,basename,abspath,isdir,realpath,splitext
from glob import glob,iglob
import logging
# from roux.lib.io import makedirs
# from roux.global_imports import info

# walker
def get_all_subpaths(d='.',include_directories=False): 
    """
    Get all the subpaths (folders and files) from a path.
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

def basenamenoext(p): return splitext(basename(p))[0]
def makedirs(p,exist_ok=True,**kws):
    from os import makedirs
    from os.path import isdir
    p_=p
    if not isdir(p):
        p=dirname(p)
    makedirs(p,exist_ok=exist_ok,**kws)
    return p_

def get_env(env_name):
    import sys,subprocess, os
    env = os.environ.copy()
    env_name_current=sys.executable.split('anaconda3/envs/')[1].split('/')[0]
    path=sys.executable.replace(env_name_current,env_name)
    env['CONDA_PYTHON_EXE']=path
    if 'anaconda' in env["PATH"]:
        env["PATH"]=env["PATH"].replace(f"{sys.executable.split('/anaconda3')[0]}/anaconda3/bin",
                                        f"{sys.executable.split('/anaconda3')[0]}/anaconda3/envs/{env_name}/bin")
    else:
        env["PATH"]=path.replace('/bin/python','/bin')+':'+env["PATH"]
    return env

def runbash(s1,env,test=False,**kws):
    """
    TODOs:
    1. logp
    2. error ignoring
    """
    if test:logging.info(s1)
    return subprocess.call(s1, shell=True,
                           env=get_env(env) if isinstance(env,str) else env,
               stderr=subprocess.DEVNULL if not test else None, 
               stdout=subprocess.DEVNULL if not test else None,
               **kws)

def runbash_tmp(s1,env,
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
    """
    :param df1: input dataframe to be saved as tsv
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
        runbash(s1,env=env,
            test=test,
            **kws)
        if exists(tmp_outp):
            if output_type=='path':
                makedirs(outp)
                shutil.move(tmp_outp,outp)
                return outp
        else:
            logging.error(f"output file not found: {outp} ({tmp_outp})")
            
def create_symlink(p,outp,test=False):
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
    return com

# import roux.lib.dfs
def get_deps(cfg=None,deps=[]):
    import logging
    """
    Installs conda dependencies.

    :param cfg: configuration dict
    """
    if not cfg is None:
        if not 'deps' in cfg:
            cfg['deps']=deps
        else:
            deps=cfg['deps']
    if not len(deps)==0:
        for dep in deps:
            if not dep in cfg:
                runbashcmd(f'conda install {dep}',
                           test=cfg['test'])
                cfg[dep]=dep
    logging.info(f"{len(deps)} deps installed.")
    return cfg

def input_binary(q): 
    reply=''
    while not reply in ['y','n','o']:
        reply = input(f"{q}:")
        if reply == 'y':
            return True
        if reply == 'n':
            return False
    return reply

def is_interactive():
    """
    Check if the UI is interactive e.g. jupyter or command line. 
    """
    # thanks to https://stackoverflow.com/a/22424821/3521099
    import __main__ as main
    return not hasattr(main, '__file__')

def is_interactive_notebook():
    """
    Check if the UI is interactive e.g. jupyter or command line.     
    
    difference in sys.module of notebook and shell
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
    
    # code
    from roux.global_imports import *
    import sys
    with open('notebook.txt','w') as f:
        f.write('\n'.join(sys.modules))

    from roux.global_imports import *
    import sys
    with open('shell.txt','w') as f:
        f.write('\n'.join(sys.modules))
    set(open('notebook.txt','r').read().split('\n')).difference(open('shell.txt','r').read().split('\n'))    
    """
#     logging.warning("is_interactive_notebook function could misbehave")
    # thanks to https://stackoverflow.com/a/22424821
    return 'ipykernel.kernelapp' in sys.modules

def get_excecution_location(depth=1):
    from inspect import getframeinfo, stack
    caller = getframeinfo(stack()[depth][0])
    return caller.filename,caller.lineno

## time
def get_time():
    """
    Gets current time in a form of a formated string. Used in logger function.

    """
    import datetime
    time=make_pathable_string('%s' % datetime.datetime.now())
    return time.replace('-','_').replace(':','_').replace('.','_')

def p2time(filename,time_type='m'):
    """
    Get the creation/modification dates of files.
    """
    import os
    import datetime
    if time_type=='m':
        t = os.path.getmtime(filename)
    else:
        t = os.path.getctime(filename)
    return str(datetime.datetime.fromtimestamp(t))

def ps2time(ps,**kws_p2time):
    import pandas as pd
    from glob import glob
    if isinstance(ps,str):
        if isdir(ps):
            ps=glob(f"{ps}/*")
    return pd.Series({p:p2time(p,**kws_p2time) for p in ps}).sort_values().reset_index().rename(columns={'index':'p',0:'time'})
    
## logging system
from roux.lib.str import make_pathable_string
def get_datetime(outstr=True):
    import datetime
    time=datetime.datetime.now()
    if outstr:
        return make_pathable_string(str(time)).replace('-','_')
    else:
        return time

def get_logger(program='program',argv=None,level=None,dp=None):
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
# # alias
# info=ic


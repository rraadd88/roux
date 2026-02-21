"""
For input/output of workflow.
"""

## logging
import logging

## data
import pandas as pd

from pathlib import Path

import re

from roux.lib.sys import (
    exists,
    makedirs,
)

## for backcompatibility
from roux.workflow.cfgs import read_config, read_metadata ##noqa

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

## io files
def to_py(
    notebookp: str,
    pyp: str = None,
    force: bool = False,
    **kws_get_lines,
) -> str:
    """To python script (.py).

    Args:
        notebookp (str): path to the notebook path.
        pyp (str, optional): path to the python file. Defaults to None.
        force (bool, optional): overwrite output. Defaults to False.

    Returns:
        str: path of the output.
    """
    if pyp is None:
        pyp = notebookp.replace(".ipynb", ".py")
    if exists(pyp) and not force:
        return
    makedirs(pyp)
    from roux.workflow.nb import get_lines

    l1 = get_lines(notebookp, **kws_get_lines)
    l1 = "\n".join(l1).encode("ascii", "ignore").decode("ascii")
    with open(pyp, "w+") as fh:
        fh.writelines(l1)
    return pyp

## export helper
def check_py(
    p,
    errors='raise',
    ruff_ignore='E402,E722,F841,E741,F401,E712,F541,F841', # while prototyping
):
    """
    E402	pycodestyle	Module level import not at top of file.
    E722	pycodestyle	Do not use bare except: (use except Exception:).
    F841	Pyflakes	Local variable is assigned to but never used.
    E741	pycodestyle	Do not use variables named 'l', 'O', or 'I' (ambiguous characters).
    F401	Pyflakes	Module imported but unused.
    E712	pycodestyle	Comparison to True should be if cond is True: or if cond:.
    F541	Pyflakes	f-string without any placeholders.
    F841    Variable defined but not used
    """
    if p.endswith('.py'):
        init_path=f"{Path(p).parent}/__init__.py"
        if not Path(init_path).exists():
            Path(init_path).touch()
            logging.warning(f'created missing: {init_path}')
        
    com=f"ruff check {p} --ignore {ruff_ignore}"
    from roux.lib.sys import run_com  
    stdout = run_com(com, wait=True,
        returncodes=[0,1], #get stdout regardless 
    )  # Captures both streams  

    if errors=='raise':
        logging.warning(stdout)
        assert 'All checks passed!' in stdout, com
    else:
        logging.warning(f"fix following by running: {com}")
    return stdout

from roux.lib.text import replace_text
def post_code(
    p: str,
    lint: bool,
    format: bool,
    verbose: bool = True,
):
    ## %run
    replace_text(
        p,
        {"get_ipython()":"# get_ipython()"}
        )
        
    ## ruff
    com = ""
    if lint:
        com += f"ruff check --fix {p};" #-unsafe-fixes
    if format:
        com += f"ruff format {p};"
    import subprocess

    res = subprocess.run(
        com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        logging.info(res.stdout)
    return res

## used in ui
from roux.lib.sys import get_source_path #noqa
def to_mod(
    p=None, #nb
    outp=None, #py

    kws_nb_export={},

    ## ruff
    errors='raise',
    **kws_check_py,
    ):
    if p is None:
        from roux.lib.sys import get_source_path
        p=get_source_path()
        logging.warning(f"p={p}")        

    if outp is None:
        outp=f"../{'modules/' if Path(p).stem.count('_')==1 else '' if Path(p).stem.count('_')==2 else ValueError(Path(p).stem)}{Path(p).stem.replace('_','/')}.py"

    kws_nb_export={
        **dict(
            nbname=p,
            lib_path=Path(outp).parent.as_posix(),
            name=Path(outp).stem,
        ),
        ## override
        **kws_nb_export
        }
    from nbdev.export import nb_export
    nb_export(
        **kws_nb_export,
    )
    # _py_path=f"{kws_nb_export['lib_path']}/{kws_nb_export['nbname'].split('.')[0].split('_')[-1]}.py"
    _py_path=f"{kws_nb_export['lib_path']}/{kws_nb_export['name']}.py"
    logging.info(_py_path)

    check_py(
        _py_path,
        errors=errors,
        **kws_check_py,
        )

    return outp

def to_py_path(p):
    return '/'.join(list(Path(p).parts[:-2])+[Path(p).parts[-2].replace('nbs','').replace('notebooks','').strip('_'),Path(p).stem+'.py'])    

def to_scr(
    p,
    outp=None,

    with_pms=True, # py with --pms (preferred)
    mark_end='## END',

    replaces={
        "get_ipython":'#get_ipython',
        "sys.exit()":'return',
        "sys.exit(0)":'return',
        "sys.exit(1)":'return',
    },
    ## ruff
    check=True,
    fix=False, # replacestar+format
    errors='raise',

    verbose=False,
    **kws_check_py,
    ):
    """
    Notebook to command line script.
    
    # roux to-src ipynb py
    # ruff check py --ignore E402    

    TODO:
        Prefer py with --pms
    """

    if outp is None:
        outp=to_py_path(p)
        logging.warning(f"outp={outp}")

    pyp=to_py(
        p,
        pyp=f'.to_src/{Path(p).stem}.py',
        force=True,
        )
    
    t_raw=open(pyp,'r').read()
    # print(t_raw[:100])

    t_raw=t_raw.split(mark_end)[0]
    # print(t_raw[:100])

    ## TODO: use textwrap
    t_tab=t_raw.replace('\n','\n    ')
    # print(t_tab[:100])
    
    def split_by_pms(text):
        splits1=re.split(r"# In\[\s*\d*\s*\]:\n    #+ param", text)

        pre_pms=splits1[0].replace('\n    ','\n')
        
        pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1])[0]
        
        post_pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1],1)[1]
        post_pms=re.sub(r"\n    # In\[\s*\d*\s*\]:", '\n', post_pms)
        return (
            [pre_pms, post_pms],
            pms
        )
    if with_pms:
        t_splits,params=split_by_pms(t_tab)
        ## remove comments
        params_lines=params.split('\n')
        params='\n'.join([s.split('#')[0] for s in params_lines])
        
        from roux.workflow.pms import extract_pms
        params_str=',\n    '.join(
            extract_pms(
                params.split('    ')[1:],
                fmt='str',
            )
        )
        if verbose:
            logging.info(params_str)
    else:
        t_splits=['',t_tab]

    t_def_pre="""
    if input_path is not None and output_path is None:
        ## should be params
        from roux.workflow.task import pre_params
        params=pre_params(input_path)
        if len(params)!=1:
            print(f"warning: {len(params)} pms found")
        for i,kws in enumerate(params):
            run(
                **kws
            )
        return 

    ## for cli
    assert input_path is not None
    assert output_path is not None
    """

    t_def='\n'.join([
    """
import argh
def run(
    """,
    params_str if with_pms else '',
    """
    ):
    """,
    t_def_pre if with_pms else ''
    ])
    
    t_end="""
    return output_path

## CLI-setup
parser = argh.ArghParser()
parser.add_commands(
    [
        run,
    ]
)
if __name__ == "__main__": # and sys.stdin.isatty():
    parser.dispatch()
    """
    
    t_src=t_splits[0]+t_def+t_splits[1]+t_end
    
    from roux.lib.str import replace_many
    t_src=replace_many(
            t_src,
            replaces,
            errors=None,
            use_template=False,
        )    

    Path(outp).parent.mkdir(exist_ok=True)
    open(outp,'w').write(t_src)
    if check and not fix:
        check_py(
            outp,
            errors=errors,
            **kws_check_py,
            )
    if fix:
        replacestar(
            outp,
            output_path=None,
            replace_from="    from roux.global_imports import *",
            method='filter', # select
            errors=errors,
            verbose = verbose,
        )
        post_code(
            outp,
            lint=True,
            format=True,
            verbose = verbose,
        )
    return outp
## alias
# source=script
to_src=to_scr

def replacestar_ruff(
    p: str,
    outp: str,
    replace: str = "from roux.global_imports import *",
    clean=False,
    verbose=True,
) -> str:
    from roux.workflow.function import get_global_imports

    lines=get_global_imports(out_fmt='lines')
    ## remove #noqa
    replace_with='\n'.join(
        [s.replace("#noqa", "") if '#keep' not in s else s for s in lines]
        # lines
        ).replace('\n\n','\n')

    ## indent
    import textwrap
    indent=' '*(len(replace) - len(replace.lstrip(' ')))
    if verbose:
        logging.info(f"indent='{indent}'")
    replace_with=textwrap.indent(replace_with, indent)

    replaced = open(p, "r").read().replace(replace, replace_with)

    import tempfile
    temp = tempfile.NamedTemporaryFile().name + ".py"
    # temp
    open(temp, "w").write(replaced)
    if verbose:
        logging.info(temp)

    post_code(
        p=temp,
        lint=True,
        format=True,
        verbose=verbose,
    )

    replaced_lines = open(temp, "r").read().split("\n")
    # import shutil
    # shutil.move(temp,outp)

    if clean:
        ## remove excessive comments
        drop_lines = []
        s_ = ""  # last
        for i, s in enumerate(replaced_lines):
            s = s.strip()
            if s.startswith("## setting states"):
                break
            if s_.startswith("#") and s.startswith("#"):
                drop_lines.append(i - 1)
            # if s == "":
            #     drop_lines.append(i)
            s_ = s
        replaced_lines = [s for i, s in enumerate(replaced_lines) if i not in drop_lines]

    # print(open(outp,'r').read()[:100])
    replaced_text = "\n".join(replaced_lines)

    open(outp, "w").write(replaced_text)
    return outp

## post-processing
def replacestar(
    input_path,
    output_path=None,
    replace_from="from roux.global_imports import *",
    method='filter', # select
    errors='raise',
    verbose: bool = False,
):
    """
    Post-development, replace wildcard (global) import from roux i.e. 'from roux.global_imports import *' with individual imports with accompanying documentation.

    Usage:
        For notebooks developed using roux.global_imports.

    Parameters
        input_path (str): path to the .py or .ipynb file.
        output_path (str): path to the output.
        py_path (str): path to the intermediate .py file.
        in_place (bool): whether to carry out the modification in place.
        return_replacements (bool): return dict with strings to be replaced.
        attributes (dict): attribute names mapped to their keywords for searching.
        verbose (bool): verbose toggle.
        test (bool): test-mode if output file not provided and in-place modification not allowed.

    Returns:
        output_path (str): path to the modified notebook.

    Examples:
        roux replacestar -i notebook.ipynb
        roux replacestar -i notebooks/*.ipynb
    """
    from roux.workflow.function import get_global_imports
    if output_path is None:
        output_path=input_path

    if method=='select':
        stdout=check_py(input_path,errors=None)
        # Find all F405 errors
        # pattern=': F405 `'
        pattern="\x1b[36m:\x1b[0m \x1b[1;31mF405\x1b[0m `"
        funcs=sorted(list(set([s.split(
            pattern
            )[1].split('`')[0] for s in stdout.split('\n') if pattern in s])))
        logging.warning(f"use roux replacestar for {funcs}")

        if len(funcs)>0:
            imports='\n'.join(
                get_global_imports()
                .query(
                    expr="`function name`=={funcs}"
                )
                ['import statement'].tolist()
                )
            replace_text(
                input_path,
                {replace_from:'\n'+imports}
            )
    else:
        output_path=replacestar_ruff(
            output_path,
            output_path,
            replace_from,
            verbose=verbose,
        )
    check_py(output_path,errors=errors)
    return output_path

## outputs setup
def set_outputs(
    output_path=None,
    output_paths=None,
):
    # if output_path is None:
    #     output_path=locals().get('output_path')
    # if output_paths is None:
    #     output_paths=locals().get('output_paths')
        
    if isinstance(output_path,str):
        output_dir_path=Path(output_path).with_suffix('').as_posix()
        if isinstance(output_paths,list):
            output_paths={fn:f"{output_dir_path}/{fn}" for fn in output_paths}
            from roux.lib.log import log_dict
            logging.info("output_paths:\n"+log_dict(output_paths,out=True))
        else:
            logging.info(f"output_dir_path: {output_dir_path}")        
            output_paths={}
    else:
        output_dir_path,output_paths=None,{}
    return output_dir_path,output_paths
    
## tasks
def check_for_exit(
    ## check this if empty
    p, # saved table
    
    ## save this before exiting
    outp, # output path
    data=None, # e.g output cfg

    force=False,
    ):
    """
    Check for early exit in a script.
    """
    exit=force

    if not exit:
        if isinstance(p,str):
            from roux.lib.io import is_table_empty    
            exit=is_table_empty(p)
        elif isinstance(p,pd.DataFrame):
            exit=(len(p)==0)
            if data is None:
                data=p
        
    if exit:
        logging.warning("exiting early because table is empty..")
        if data is not None:
            from roux.lib.io import to_data
            to_data(data,outp)
        else:
            Path(outp).touch()
        import sys
        sys.exit(0)
"""For input/output of workflow."""
## logging
import logging
## data
import pandas as pd

import shutil # for copying files

from roux.lib.sys import (Path, abspath, basename, basenamenoext, create_symlink, exists, glob, isdir, makedirs, splitext)
from roux.lib.io import read_ps,read_dict,is_dict
from roux.lib.set import flatten

## variables
def clear_variables(
    dtype=None,
    variables=None,
    # globals_dict=None,
    ):
    """Clear dataframes from the workspace."""
    assert (dtype is None or variables is None) and (not (dtype is None and variables is None))
    import gc
    if variables is None:
        variables=[k for k in globals() if isinstance(globals()[k], dtype) and not k.startswith('_')]
    if len(variables)==0: return
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
    pyp: str=None,
    force: bool=False,
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
    if pyp is None: pyp=notebookp.replace('.ipynb','.py')
    if exists(pyp) and not force: return 
    makedirs(pyp)
    from roux.workflow.nb import get_lines
    l1=get_lines(notebookp, **kws_get_lines)
    l1='\n'.join(l1).encode('ascii', 'ignore').decode('ascii')
    with open(pyp, 'w+') as fh:
        fh.writelines(l1)
    return pyp

def to_nb_cells(
    notebook,
    outp,
    new_cells,
    validate_diff=None,
    ):
    """
    Replace notebook cells.
    """
    import nbformat
    print(f"notebook length change: {len(notebook.cells):>2}->{len(new_cells):>2} cells")
    if not validate_diff is None:
        assert len(notebook.cells)-len(new_cells)==validate_diff
    elif validate_diff == '>': # filtering
        assert len(notebook.cells)>len(new_cells)
    elif validate_diff == '<': # appending
        assert len(notebook.cells)<len(new_cells)
    notebook.cells = new_cells
    # Save the modified notebook
    with open(outp, 'w', encoding='utf-8') as new_notebook_file:
        nbformat.write(notebook, new_notebook_file)
    return outp

def import_from_file(
    pyp: str
    ):
    """Import functions from python (`.py`) file.

    Args:
        pyp (str): python file (`.py`).

    """
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(abspath(pyp), abspath(pyp)).load_module()

## io parameters    
def infer_parameters(
    input_value,
    default_value):
    """
    Infer the input values and post warning messages.
    
    Parameters:
        input_value: the primary value.
        default_value: the default/alternative/inferred value.
    
    Returns:
        Inferred value.
    """
    if input_value is None:
        logging.warning(f'input is None; therefore using the the default value i.e. {default_value}.')
        return default_value
    else:
        return input_value
    
def to_parameters(
    f: object,
    test: bool=False
    ) -> dict:
    """Get function to parameters map.

    Args:
        f (object): function.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        dict: output.
    """
    import inspect
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params

def read_config(
    p: str,
    config_base=None,
    inputs=None, #overwrite with
    append_to_key=None,
    convert_dtype:bool=True,
    ):
    """
    Read configuration.

    Parameters:
        p (str): input path. 
        config_base: base config with the inputs for the interpolations
    """
    from omegaconf import OmegaConf
    if isinstance(config_base,str):
        if exists(config_base):
            config_base=OmegaConf.create(read_dict(config_base))
            # logging.info(f"Base config read from: {config_base}")
        else:
            logging.warning(f"Base config path not found: {config_base}")
    ## read config
    d1=read_dict(p)
    ## merge
    if not config_base is None:
        if not append_to_key is None:
            # print(config_base)
            # print(d1)            
            d1={append_to_key:{**config_base[append_to_key],**d1}}        
        # print(config_base)
        # print(d1)
        d1=OmegaConf.merge(
                config_base, ## parent
                d1, ## child overwrite with
                )
    elif not inputs is None:
        d1=OmegaConf.merge(
                d1, ## parent
                inputs, ## child overwrite with
                )
    else:
        ## no-merging
        d1=OmegaConf.create(d1)
    # ## convert data dypes
    if convert_dtype:
        d1=OmegaConf.to_object(d1)
    return d1

## metadata-related
def read_metadata(
    p: str,
    ind: str=None,
    max_paths: int= 30,
    config_path_key:str='config_path',
    config_paths: list=[],
    config_paths_auto=False,
    verbose: bool=False,
    **kws_read_config,
    ) -> dict:
    """Read metadata.

    Args:
        p (str, optional): file containing metadata. Defaults to './metadata.yaml'.
        ind (str, optional): directory containing specific setings and other data to be incorporated into metadata. Defaults to './metadata/'.

    Returns:
        dict: output.

    """
    if not exists(p):
        logging.warning(f'not found: {p}')

    d1=read_config(p,**kws_read_config)
    
    ## read dicts
    keys=d1.keys()
    for k in keys:
        # if isinstance(d1[k],str):
        #     ## merge configs
        #     if d1[k].endswith('.yaml'):
        #         d1=read_config(
        #             d1[k],
        #             config_base=d1,
        #             )
        # el
        if isinstance(d1[k],dict):
            ## read `config_path`s
            # if len(d1[k])==1 and list(d1[k].keys())[0]==config_path_key:
            if config_path_key in list(d1[k].keys()):
                if verbose: logging.info(f"Appending config to {k}")
                if exists(d1[k][config_path_key]):
                    d1=read_config(
                        p=d1[k][config_path_key],
                        config_base=d1,
                        append_to_key=k,
                        )
                else:
                    if verbose: logging.warning(f"not exists: {d1[k][config_path_key]}")
        # elif isinstance(d1[k],list):
        #     ## read list of files
        #     ### check 1st path
        #     if len(d1[k])<max_paths:
        #         if not exists(d1[k][0]):
        #             logging.error(f"file not found: {p}; ignoring a list of `{k}`s.")
        #         d_={}
        #         for p_ in d1[k]:
        #             if isinstance(p_,str):
        #                 if is_dict(p_):
        #                     d_[basenamenoext(p_)]=read_dict(p_)
        #                 else:
        #                     d_[basenamenoext(p_)]=p_
        #                     logging.error(f"file not found: {p_}")
        #         if len(d_)!=0:
        #             d1[k]=d_
    ## read files from directory containing specific setings and other data to be incorporated into metadata
    if config_paths_auto:
        if ind is None:
            ind=splitext(p)[0]+'/'
            if verbose:info(ind)
            config_paths+=glob(f"{ind}/*")
    ## before
    config_size=len(d1)
    ## separate metadata (.yaml) /data (.json) files
    for p_ in config_paths:
        if isdir(p_):
            if len(glob(f'{p_}/*.json'))!=0:
                ## data e.g. stats etc
                if not basename(p_) in d1 and len(glob(f'{p_}/*.json'))!=0:
                    d1[basename(p_)]=read_dict(f'{p_}/*.json')
                elif isinstance(d1[basename(p_)],dict) and len(glob(f'{p_}/*.json'))!=0:
                    d1[basename(p_)].update(read_dict(f'{p_}/*.json'))
                else:
                    logging.warning(f"entry collision, could not include '{p_}/*.json'")
        else:
            if is_dict(p_):
                d1[basenamenoext(p_)]=read_dict(p_)
            else:
                logging.error(f"file not found: {p_}")
    if (len(d1)-config_size)!=0:
        logging.info(f"metadata appended from "+str(len(d1)-config_size)+" separate config/s.")    
    return d1

def to_workflow(
    df2: pd.DataFrame,
    workflowp: str,
    tab: str='    '
    ) -> str:
    """Save workflow file.

    Args:
        df2 (pd.DataFrame): input table.
        workflowp (str): path of the workflow file.
        tab (str, optional): tab format. Defaults to '    '.

    Returns:
        str: path of the workflow file.
    """
    makedirs(workflowp)
    from roux.lib.set import list2str
    with open(workflowp,'w') as f:
        ## add rule all
        f.write("from roux.lib.io import read_dict\nfrom roux.workflow.io import read_metadata\nmetadata=read_metadata()\n"
                +'report: "workflow/report_template.rst"\n'
                +"\nrule all:\n"
                 f"{tab}input:\n"
                 f"{tab}{tab}"
#                     +f",\n{tab}{tab}".join(flatten([flatten(l) for l in df2['output paths'].dropna().tolist()]))
                +f",\n{tab}{tab}".join(df2['output paths'].dropna().tolist())
                +"\n# rules below\n\n"
                +'\n'.join(df2['rule code'].dropna().tolist())\
               )
    return workflowp

def create_workflow_report(
    workflowp: str,
    env: str,
    ) -> int:
    """
    Create report for the workflow run.

    Parameters:
        workflowp (str): path of the workflow file (`snakemake`).
        env (str): name of the conda virtual environment where required the workflow dependency is available i.e. `snakemake`.
    """
    from pathlib import Path
    workflowdp=str(Path(workflowp).absolute().with_suffix(''))+'/'
    ## create a template file for the report
    report_templatep=Path(f"{workflowdp}/report_template.rst")
    if not report_templatep.exists():
        report_templatep.parents[0].mkdir(parents=True, exist_ok=True)
        report_templatep.touch()

    from roux.lib.sys import runbash
    runbash(f"snakemake --snakefile {workflowp} --rulegraph > {workflowdp}/workflow.dot;sed -i '/digraph/,$!d' {workflowdp}/workflow.dot",env=env)
    
    ## format the flow chart
    from roux.lib.set import read_list,to_list
    to_list([s.replace('task','').replace('_step','\n') for s in read_list(f'{workflowdp}/workflow.dot')],
            f'{workflowdp}/workflow.dot')
    
    runbash(f"dot -Tpng {workflowdp}/workflow.dot > {workflowdp}/workflow.png",env=env)
    runbash(f"snakemake -s workflow.py --report {workflowdp}/report.html",env=env)

## post-processing
def replacestar(
    input_path,
    output_path=None,
    replace_from='from roux.global_imports import *',
    in_place: bool=False,
    attributes={
        'pandarallel':['parallel_apply'],
        'rd':['.rd.','.log.']
    },
    verbose: bool=False,
    test: bool=False,
    **kws_fix_code,
    ):
    """
    Post-development, replace wildcard (global) import from roux i.e. 'from roux.global_imports import *' with individual imports with accompanying documentation.
    
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
    """
    from roux.workflow.function import get_global_imports
    ## infer input parameters
    if output_path is None:
        if in_place:
            output_path=input_path
        else:
            verbose=True
    try:
        from removestar.removestar import fix_code, replace_in_nb
    except ImportError as error:
        logging.error(f'{error}: Install needed requirement using command: pip install removestar')

    if input_path.endswith(".py"):    
        with open(input_path, encoding="utf-8") as f:
            code = f.read()
    elif input_path.endswith(".ipynb"):
        with open(input_path) as f:
            import nbformat
            nb = nbformat.reads(f.read(), nbformat.NO_CONVERT)

        ## save as py
        from nbconvert import PythonExporter
        exporter = PythonExporter()
        code, _ = exporter.from_notebook_node(nb)

    import tempfile
    replaces = fix_code(
            code=code,
            file=tempfile.NamedTemporaryFile().name,
            return_replacements=True,
            )
    
    if replace_from in replaces:
        imports=replaces[replace_from]
        if imports!='':
            imports=imports.split(' import ')[1].split(', ')
            if test: logging.info(f"imports={imports}")            
        else:
            imports=[]
            if verbose: logging.warning(f"no function imports found in '{input_path}'")    
    else:
        logging.warning(f"'{replace_from}' not found in '{input_path}'; copying the file, as it is.")
        shutil.copy(input_path,output_path)
        return output_path

    imports_attrs=[k for k,v in attributes.items() if any([s in code for s in v])]
    if len(imports_attrs)==0:
        if verbose: logging.warning(f"no attribute imports found in '{input_path}'")
        
    if len(imports+imports_attrs)==0:
        logging.warning(f"no imports found in '{input_path}'; copying the file, as it is.")    
        shutil.copy(input_path,output_path)
        return output_path
    
    df2=get_global_imports()
    from roux.workflow.nb import get_lines
    def get_lines_replace_with(imports,df2): 
        ds=df2.query(expr=f"`function name` in {imports}").apply(lambda x: f"## {x['function comment']}\n{x['import statement']}",axis=1)
        lines=ds.tolist()
        return '\n'.join(lines)
    
    replace_with=''
    if len(imports)!=0:
        replace_with+=get_lines_replace_with(imports,df2.query("`attribute`==False"))

    if len(imports_attrs)!=0:
        replace_with+='\n'+get_lines_replace_with(imports_attrs,df2.query("`attribute`==True"))
    ## remove duplicate lines
    replace_with='\n'.join(pd.Series(replace_with.split('\n')).drop_duplicates(keep='first').tolist())

    replace_with=replace_with.strip()
    replaces_={**replaces,**{replace_from:replace_with}}

    if verbose:logging.info(f"replace     :\n"+('\n'.join([k for k in replaces_.keys()])))
    if verbose:logging.info(f"replace_with:\n"+('\n'.join([v for v in replaces_.values()])))
    
    if not output_path is None:
        # save files
        if input_path.endswith(".py"):
            from roux.lib.str import replace_many
            new_code=replace_many(code,replaces_)
            
            open(output_path,'w').write(new_code)        
        elif input_path.endswith(".ipynb"):
            from roux.workflow.nb import to_replaced_nb
            to_replaced_nb(
                    input_path,
                    replaces=replaces_,
                    cell_type='code',
                    output_path=output_path,
                    )
    return output_path

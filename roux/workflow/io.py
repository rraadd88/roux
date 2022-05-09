import pandas as pd
from roux.lib.sys import *
from roux.lib.io import read_ps
from roux.lib.set import flatten
from roux.lib.dict import read_dict,is_dict

def get_lines(
    p: str,
    keep_comments: bool=True
    ) -> list:
    """Get lines of code from notebook.

    Args:
        p (str): path to notebook.
        keep_comments (bool, optional): keep comments. Defaults to True.

    Returns:
        list: lines.
    """
    import nbformat
    from nbconvert import PythonExporter
    import os
    if os.path.islink(p):
        p=os.readlink(p)
    nb = nbformat.read(p, nbformat.NO_CONVERT)
    exporter = PythonExporter()
    source, meta = exporter.from_notebook_node(nb)
    lines=source.split('\n')
    lines=[s for s in lines if (isinstance(s,str) and s!='' and len(s)<1000)]
    if not keep_comments:
        lines=[s for s in lines if not s.startswith('#')]
    return lines

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
    l1=get_lines(notebookp, **kws_get_lines)
    l1='\n'.join(l1).encode('ascii', 'ignore').decode('ascii')
    with open(pyp, 'w+') as fh:
        fh.writelines(l1)
    return pyp

def import_from_file(
    pyp: str
    ):
    """Import functions from python (`.py`) file.

    Args:
        pyp (str): python file (`.py`).

    """
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(abspath(pyp), abspath(pyp)).load_module()

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
    sign=inspect.signature(f)
    params={}
    for arg in sign.parameters:
        argo=sign.parameters[arg]
        params[argo.name]=argo.default
    #     break
    return params

def read_nb_md(
    p: str
    ) -> list:
    """Read notebook's documentation in the markdown cells.

    Args:
        p (str): path of the notebook.

    Returns:
        list: lines of the strings.
    """
    import nbformat
    from sys import argv
    l1=[]
    l1.append("# "+basenamenoext(p))
    nb = nbformat.read(p, nbformat.NO_CONVERT)
    l1+=[cell.source for cell in nb.cells if cell.cell_type == 'markdown']
    return l1

## metadata-related
def read_metadata(
    ind: str='metadata'
    ) -> dict:
    """Read metadata.

    Args:
        ind (str, optional): input directory. Defaults to 'metadata'.

    Returns:
        dict: output.

    TODOs:
        1. Metadata files include colors.yaml, database.yaml, constants.yaml etc.
    """
    p=f'{ind}/metadata.yaml'
    for p_ in [ind,p]:
        if not exists(p_):
            logging.warning(f'not found: {ind}')
            return 
    d1=read_dict(p)
    ## read dicts
    for k in d1:
        if isinstance(d1[k],list):
            if len(d1[k])<10:
                d_={}
                for p in d1[k]:
                    if isinstance(p,str):
                        if is_dict(p) and exists(p):
                            d_[basenamenoext(p)]=read_dict(p)
                        else:
                            d_[basenamenoext(p)]=p
                            logging.error(f"file not found: {p}")
                if len(d_)!=0:
                    d1[k]=d_
    for p_ in glob(f"{ind}/*"):
        if isdir(p_):
            if len(glob(f'{p_}/*.json'))!=0:
                if not basename(p_) in d1 and len(glob(f'{p_}/*.json'))!=0:
                    d1[basename(p_)]=read_dict(f'{p_}/*.json')
                elif isinstance(d1[basename(p_)],dict) and len(glob(f'{p_}/*.json'))!=0:
                    d1[basename(p_)].update(read_dict(f'{p_}/*.json'))
                else:
                    logging.warning(f"entry collision, could not include '{p_}/*.json'")
        else:
            if is_dict(p_) and exists(p_):
                d1[basenamenoext(p_)]=read_dict(p_)
            else:
                logging.error(f"file not found: {p_}")                
    logging.info(f"metadata read from {p} (+"+str(len(glob(f'{ind}/*.json')))+" jsons)")
    return d1

## create documentation
def to_info(
    p: str='*_*_v*.ipynb',
    outp: str='README.md') -> str:
    """Save README.md file.

    Args:
        p (str, optional): path of the notebook files that would be converted to "tasks". Defaults to '*_*_v*.ipynb'.
        outp (str, optional): path of the output file. Defaults to 'README.md'.

    Returns:
        str: path of the output file.
    """
    ps=read_ps(p)
    l1=flatten([read_nb_md(p) for p in ps])
    with open(outp,'w') as f:
        f.writelines([f"{s}\n" for s in l1])
    return outp

def make_symlinks(
    d1: dict,
    d2: dict,
    project_path: str,
    data: bool=True,
    notebook_suffix: str='_v',
    test: bool=False
    )->list:
    """Make symbolic links.

    Args: 
        d1 (dict): `project name` to `repo name`.
        d2 (dict): `task name` to tuple containing `from project name` `to project name`.
        project_path (str): path of the repository.
        data (bool, optional): make links for the data. Defaults to True.
        notebook_suffix (str, optional): suffix of the notebook file to be considered as a "task".
        test (bool, optional): test mode. Defaults to False.

    Returns:
        list: list of commands.
    """
    coms=[]
    for k in d2:
        ## notebook
        p=read_ps(f"{project_path}/{d2[k][0]}/code/{d1[d2[k][0]]}/{d1[d2[k][0]]}/{k.split('/')[0]}*{notebook_suffix}*.ipynb")[0]
        if test: print(p)
        outp=f"{project_path}/{d2[k][1]}/code/{d1[d2[k][1]]}/{d1[d2[k][1]]}/{basename(p)}"
        if test: print(outp)
        coms.append(create_symlink(p,outp))
        if data:
            ## data_analysed
            p=f"{project_path}/{d2[k][0]}/data/data_analysed/data{k}"
            outp=f"{project_path}/{d2[k][1]}/data/data_analysed/data{k}"
            coms.append(create_symlink(p,outp))
        # break
    return coms

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
        f.write("from roux.lib.dict import read_dict\nfrom roux.workflow.io import read_metadata\nmetadata=read_metadata()\n"
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

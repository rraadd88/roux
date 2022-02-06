from roux.lib.sys import *
from roux.lib.io import read_ps
from roux.lib.set import flatten
from roux.lib.dict import read_dict

def get_lines(p, 
              keep_comments=True):
    """
    
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

def to_py(notebookp, pyp=None,
          force=False,
         **kws_get_lines,
         ):
    if pyp is None: pyp=notebookp.replace('.ipynb','.py')
    if exists(pyp) and not force: return 
    makedirs(pyp)
    l1=get_lines(notebookp, **kws_get_lines)
    l1='\n'.join(l1).encode('ascii', 'ignore').decode('ascii')
    with open(pyp, 'w+') as fh:
        fh.writelines(l1)
    return pyp

def import_from_file(pyp):
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader (abspath(pyp), abspath(pyp)).load_module()

def read_nb_md(p):
    import nbformat
    from sys import argv
    l1=[]
    l1.append("# "+basenamenoext(p))
    nb = nbformat.read(p, nbformat.NO_CONVERT)
    l1+=[cell.source for cell in nb.cells if cell.cell_type == 'markdown']
    return l1

def read_metadata(ind='metadata'):
    """
    TODOs:
    1. colors.yaml, database.yaml, constants.yaml
    """
    p=f'{ind}/metadata.yaml'
    for p_ in [ind,p]:
        if not exists(p_):
            logging.warning(f'not found: {ind}')
            return 
    d1=read_dict(p)
    ## read jsons
    for k in d1:
        if isinstance(d1[k],list):
            if len(d1[k])<10:
                d_={}
                for p in d1[k]:
                    if isinstance(p,str):
                        if p.endswith('.json'):
                            d_[basenamenoext(p)]=read_dict(p)
                if len(d_)!=0:
                    d1[k]=d_
    for p_ in glob(f"{ind}/*"):
        if isdir(p_):
            if len(glob(f'{p_}/*.json'))!=0:
                if not basename(p_) in d1: 
                    d1[basename(p_)]=read_dict(f'{p_}/*.json')
                elif isinstance(d1[basename(p_)],dict):
                    d1[basename(p_)].update(read_dict(f'{p_}/*.json'))
                else:
                    logging.warning(f"entry collision, could not include '{p_}/*.json'")
        else:
            if p_.endswith('.json'):
                d1[basenamenoext(p_)]=read_dict(p_)
    logging.info(f"metadata read from {p} (+"+str(len(glob(f'{ind}/*.json')))+" jsons)")
    return d1

def to_info(p='*_*_v*.ipynb',
    outp='README.md'):
    ps=read_ps(p)
    l1=flatten([read_nb_md(p) for p in ps])
    with open(outp,'w') as f:
        f.writelines([f"{s}\n" for s in l1])    
    return outp

def make_symlinks(d1,d2,project_path,test=False):
    """
    :params d1: `project name` to `repo name` 
    :params d2: `task name` to tuple containing `from project name` `to project name`
    """
    coms=[]
    for k in d2:
        ## notebook
        p=read_ps(f"{project_path}/{d2[k][0]}/code/{d1[d2[k][0]]}/{d1[d2[k][0]]}/{k.split('/')[0]}*_v*.ipynb")[0]
        if test: print(p)
        outp=f"{project_path}/{d2[k][1]}/code/{d1[d2[k][1]]}/{d1[d2[k][1]]}/{basename(p)}"
        if test: print(outp)
        coms.append(create_symlink(p,outp))
        ## data_analysed
        p=f"{project_path}/{d2[k][0]}/data/data_analysed/data{k}"
        outp=f"{project_path}/{d2[k][1]}/data/data_analysed/data{k}"
        coms.append(create_symlink(p,outp))
        # break
    return coms
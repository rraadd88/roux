"""For operations on jupyter notebooks."""
import logging
import nbformat
from roux.lib.sys import basenamenoext

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

def read_nb_md(
    p: str
    ) -> list:
    """Read notebook's documentation in the markdown cells.

    Args:
        p (str): path of the notebook.

    Returns:
        list: lines of the strings.
    """
    from sys import argv
    l1=[]
    nb = nbformat.read(p, nbformat.NO_CONVERT)
    l1+=[cell.source for cell in nb.cells if cell.cell_type == 'markdown']
    return l1

## create documentation
def to_info(
    p: str,
    outp: str,
    linkd: str='',
    ) -> str:
    """Save README.md file.

    Args:
        p (str, optional): path of the notebook files that would be converted to "tasks".
        outp (str, optional): path of the output file, e.g. 'README.md'.

    Returns:
        str: path of the output file.
    """
    from os.path import basename
    from roux.lib.sys import read_ps
    from roux.lib.set import flatten
    ps=read_ps(p)
    
    l1=[]
    for p in ps:
        l_=read_nb_md(p)
        ## get title if available
        title=None
        if 'title:' in l_[0]:
            title=l_[0].split('title:')[1].split('\n')[0].strip(' ').strip('"')
            l_=l_[1:]
        elif l_[0].startswith('# '):
            title=l_[0].split('# ')[1]
            l_=l_[1:]
        if title is None:
            logging.warning(f'title is None for {p}')
            l1+=[f"# {basename(p)}"]+l_
        else:
            l1+=[f"# [{title}]({linkd}{basename(p)})"]+l_
    with open(outp,'w') as f:
        f.writelines([f"{s}\n" for s in l1])
    return outp

def to_replaced_nb(
    nb_path,
    output_path,
    replaces: dict={},
    cell_type: str='code',
    drop_lines_with_substrings=[],
    test=False,
    ):
    """
    Replace text in a jupyter notebook.
    
    Parameters
        nb: notebook object obtained from `nbformat.reads`.
        replaces (dict): mapping of text to 'replace from' to the one to 'replace with'.
        cell_type (str): the type of the cell.
    
    Returns:
        new_nb: notebook object.      
    """
    from nbconvert import PythonExporter, NotebookExporter
    ## read nb
    with open(nb_path) as fh:
        nb = nbformat.reads(fh.read(), nbformat.NO_CONVERT)
    
    new_nb=nb.copy()
    if test:
        print(f"len(new_nb['cells'])={len(new_nb['cells'])}")
    # break_early= str(nb).count(replace_from)==1
    for i,d in enumerate(new_nb['cells']):
        if d['cell_type']==cell_type:
            for replace_from, replace_to in replaces.items():
                if replace_from in d['source']:
                    d['source']=d['source'].replace(replace_from,replace_to)
            for k in drop_lines_with_substrings:
                if k in d['source']:
                    _lines=d['source'].split('\n')
                    _lines_flt=[s for s in _lines if not k in s]
                    # if len(_lines_flt)<len(_lines):
                    #     print(_lines)
                    d['source']='\n'.join(_lines_flt)
            new_nb['cells'][i]=d
            # if break_early:
                # break
    ## save new nb
    to_nb=NotebookExporter()
    source_nb,_=to_nb.from_notebook_node(new_nb)
    if not test:
        with open(output_path, 'w+') as fh:
            fh.writelines(source_nb)
        return output_path                        
    return output_path

def to_filtered_nb(
    p: str,
    outp: str,
    h: str,
    kind: str='include',
    validate_diff: int=None,
    ):
    """
    Filter a notebook based on markdown heading.        
    """
    # Load the Jupyter Notebook
    with open(p, 'r', encoding='utf-8') as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    def get_hlevel(h): return len(h.split('# ')[0])+1
    hlevel=get_hlevel(h)
    # Iterate through the notebook cells
    for cell in notebook.cells:
        if cell.cell_type == 'markdown':
            if cell.source.split('\n')[0]==h:
                # Markdown cell containing the target heading found
                # Now, look for code cells below it until the next markdown cell
                if kind=='include':
                    new_cells = [cell]
                    for next_cell in notebook.cells[notebook.cells.index(cell) + 1:]:
                        if next_cell.cell_type == 'markdown' and get_hlevel(next_cell.source.split('\n')[0])<=hlevel:
                            break
                        else:
                            new_cells.append(next_cell)
                elif kind=='exclude':
                    new_cells = notebook.cells[:notebook.cells.index(cell)]
                    skip=True
                    for next_cell in notebook.cells[notebook.cells.index(cell) + 1:]:
                        if next_cell.cell_type == 'markdown' and get_hlevel(next_cell.source.split('\n')[0])<=hlevel:
                            skip=False
                        if not skip:
                            new_cells.append(next_cell)
    print(f"notebook length change: {len(notebook.cells):>2}->{len(new_cells):>2} cells")
    if not validate_diff is None:
        assert len(notebook.cells)-len(new_cells)==validate_diff
    else:
        assert len(notebook.cells)>len(new_cells)
    notebook.cells = new_cells

    # Save the modified notebook
    with open(outp, 'w', encoding='utf-8') as new_notebook_file:
        nbformat.write(notebook, new_notebook_file)
    return outp

def to_clear_outputs(
    notebook_path,
    new_notebook_path,
    ):
    # Load the Jupyter Notebook
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Clear all outputs in code cells
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []

    # Save the modified notebook
    with open(new_notebook_path, 'w', encoding='utf-8') as new_notebook_file:
        nbformat.write(notebook, new_notebook_file)
    return new_notebook_path

def to_clear_unused_cells(
    notebook_path,
    new_notebook_path,
    ):
    # Load the Jupyter Notebook
    with open(notebook_path, 'r', encoding='utf-8') as notebook_file:
        notebook = nbformat.read(notebook_file, as_version=4)

    # Clear all outputs in code cells
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []

    # Save the modified notebook
    with open(new_notebook_path, 'w', encoding='utf-8') as new_notebook_file:
        nbformat.write(notebook, new_notebook_file)
    return new_notebook_path

def to_diff_notebooks(
    notebook_paths,
    url_prefix="https://localhost:8888/nbdime/difftool?",
    remove_prefix='file://', # for bash
    verbose=True,
    ) -> list:
    """
    "Diff" notebooks using `nbdiff` (https://nbdime.readthedocs.io/en/latest/)
    
    Start the nb-diff session by running: `nbdiff-web`
    
    Todos:
        1. Deprecate if functionality added to `nbdiff-web`.
    """
    import itertools
    logging.warning('to_diff_notebooks is under development.')
    urls_input=[Path(p).absolute().as_uri() for p in notebook_paths]
    urls_output=[]
    for url_base,url_remote in list(itertools.product(urls_input[:1],urls_input[1:])):
        urls_output.append(f"{url_prefix}base={url_base.replace('file://','')}&remote={url_remote.replace('file://','')}")
    if verbose:
        logging.info('Differences between notebooks:')
        logging.info('\n'.join(urls_output))
    return urls_output

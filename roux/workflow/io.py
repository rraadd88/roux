"""For input/output of workflow."""

## logging
import logging

## data
import pandas as pd

from pathlib import Path
import shutil  # for copying files

import re

from roux.lib.sys import (
    abspath,
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

def extract_kws(
    lines,
    fmt='dict',
    ):
    if fmt=='dict':
        parameters = {}
    else:
        parameters=[]
    for line in lines:
        # Remove comments
        line = re.sub(r'#.*', '', line).strip()
        # Match valid assignments
        match = re.match(r'(\w+)\s*=\s*(.+)', line)
        if match:
            key, value = match.groups()
            # Evaluate value if it's a valid literal, otherwise keep it as a string
            try:
                eval(value)
                string=False
            except:
                string=True
            if fmt=='dict':
                parameters[key] = value.strip() if string else eval(value)
            else:
                parameters.append(
                    f"{key}="+("'"+value+"'" if string else value)
                )
    return parameters
    
def to_src(
    p,
    outp, 
    validate=True,
    verbose=False,
    mark_end='## END'
    ):
    """
    Notebook to command line script.
    """
    pyp=to_py(
        p,
        pyp=f'.to_src/{Path(p).stem}.py',
        force=True,
        )
    
    t_raw=open(pyp,'r').read()

    t_raw=t_raw.split(mark_end)[0]
    
    t_tab=t_raw.replace('\n','\n    ')
    
    def split_by_pms(text):
        splits1=re.split(r"# In\[\s*\d*\s*\]:\n    ## param", text)

        pre_pms=splits1[0].replace('\n    ','\n')
        
        pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1])[0]
        
        post_pms=re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1],1)[1]
        post_pms=re.sub(r"\n    # In\[\s*\d*\s*\]:", '\n', post_pms)
        return (
            [pre_pms, post_pms],
            pms
        )
        
    t_splits,params=split_by_pms(t_tab)
    # t_splits
    
    params_str=',\n    '.join(
        extract_kws(
            params.split('    ')[1:],
            fmt='str',
        )
    )
    if verbose:
        print(params_str)
    
    t_def=(
    """
def run(
    """+
    params_str+
    """
    ):
    """
    )
    
    t_end="""
    
## for recursive operations
run_rec=run

## CLI-setup
import argh
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
    # print(t_src)
    Path(outp).parent.mkdir(exist_ok=True)
    open(outp,'w').write(t_src)

    com=f"ruff check {outp} --ignore E402"
    if validate:
        import os
        res=os.system(com)
        assert res==0, res
    else:
        logging.warning(f"validate by running: {com}")        
    return outp
    
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

    logging.info(
        f"notebook length change: {len(notebook.cells):>2}->{len(new_cells):>2} cells"
    )
    if validate_diff is not None:
        assert len(notebook.cells) - len(new_cells) == validate_diff
    elif validate_diff == ">":  # filtering
        assert len(notebook.cells) > len(new_cells)
    elif validate_diff == "<":  # appending
        assert len(notebook.cells) < len(new_cells)
    notebook.cells = new_cells
    # Save the modified notebook
    with open(outp, "w", encoding="utf-8") as new_notebook_file:
        nbformat.write(notebook, new_notebook_file)
    return outp


def import_from_file(pyp: str):
    """Import functions from python (`.py`) file.

    Args:
        pyp (str): python file (`.py`).

    """
    from importlib.machinery import SourceFileLoader

    return SourceFileLoader(abspath(pyp), abspath(pyp)).load_module()


## io parameters
def infer_parameters(input_value, default_value):
    """
    Infer the input values and post warning messages.

    Parameters:
        input_value: the primary value.
        default_value: the default/alternative/inferred value.

    Returns:
        Inferred value.
    """
    if input_value is None:
        logging.warning(
            f"input is None; therefore using the the default value i.e. {default_value}."
        )
        return default_value
    else:
        return input_value


def to_parameters(f: object, test: bool = False) -> dict:
    """Get function to parameters map.

    Args:
        f (object): function.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        dict: output.
    """
    import inspect

    sign = inspect.signature(f)
    params = {}
    for arg in sign.parameters:
        argo = sign.parameters[arg]
        params[argo.name] = argo.default
    #     break
    return params

def to_nb_kernel(
    p : str,
    kernel : str = None,
    outp : str = None,
    ):
    """
    Because no-kernel means previous kernel. 
    """
    from glob import glob
    if len(glob(p))>1:
        # recursive
        d={}
        for p_ in glob(p):
            d[p_]=to_nb_kernel(
                p_,
                kernel = kernel,# : str = None,
                outp = outp,# : str = None,
            )
            print(p_,d[p_])
        return # d
        
    import nbformat
    
    # Load the notebook
    nb = nbformat.read(p, as_version=nbformat.NO_CONVERT)
    
    # Update the kernelspec (or remove it)
    if "kernelspec" in nb.metadata:
        if kernel is None:
            return nb.metadata.kernelspec.name
        else:
            nb.metadata.kernelspec.name = kernel
            nb.metadata.kernelspec.display_name = kernel
            # Or to remove it entirely:
            # del nb.metadata["kernelspec"]
            # Save the modified notebook
            outp = p if outp is None else outp
            nbformat.write(nb, outp)
            return kernel


def to_workflow(df2: pd.DataFrame, workflowp: str, tab: str = "    ") -> str:
    """Save workflow file.

    Args:
        df2 (pd.DataFrame): input table.
        workflowp (str): path of the workflow file.
        tab (str, optional): tab format. Defaults to '    '.

    Returns:
        str: path of the workflow file.
    """
    makedirs(workflowp)
    with open(workflowp, "w") as f:
        ## add rule all
        f.write(
            "from roux.lib.io import read_dict\nfrom roux.workflow.io import read_metadata\nmetadata=read_metadata()\n"
            + 'report: "workflow/report_template.rst"\n'
            + "\nrule all:\n"
            f"{tab}input:\n"
            f"{tab}{tab}"
            #                     +f",\n{tab}{tab}".join(flatten([flatten(l) for l in df2['output paths'].dropna().tolist()]))
            + f",\n{tab}{tab}".join(df2["output paths"].dropna().tolist())
            + "\n# rules below\n\n"
            + "\n".join(df2["rule code"].dropna().tolist())
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
    workflowdp = str(Path(workflowp).absolute().with_suffix("")) + "/"
    ## create a template file for the report
    report_templatep = Path(f"{workflowdp}/report_template.rst")
    if not report_templatep.exists():
        report_templatep.parents[0].mkdir(parents=True, exist_ok=True)
        report_templatep.touch()

    from roux.lib.sys import runbash

    runbash(
        f"snakemake --snakefile {workflowp} --rulegraph > {workflowdp}/workflow.dot;sed -i '/digraph/,$!d' {workflowdp}/workflow.dot",
        env=env,
    )

    ## format the flow chart
    from roux.lib.set import read_list, to_list

    to_list(
        [
            s.replace("task", "").replace("_step", "\n")
            for s in read_list(f"{workflowdp}/workflow.dot")
        ],
        f"{workflowdp}/workflow.dot",
    )

    runbash(f"dot -Tpng {workflowdp}/workflow.dot > {workflowdp}/workflow.png", env=env)
    runbash(f"snakemake -s workflow.py --report {workflowdp}/report.html", env=env)


## post-processing
def replacestar(
    input_path,
    output_path=None,
    replace_from="from roux.global_imports import *",
    in_place: bool = False,
    attributes={"pandarallel": ["parallel_apply"], "rd": [".rd.", ".log."]},
    verbose: bool = False,
    test: bool = False,
    **kws_fix_code,
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
    from roux.lib.io import read_ps

    input_paths = read_ps(input_path)
    if len(input_paths) > 1:
        ## Recursion-mode
        outps = []
        for p in input_paths:
            outps.append(
                replacestar(
                    p,
                    output_path=None,
                    replace_from=replace_from,
                    in_place=in_place,
                    attributes=attributes,
                    verbose=verbose,
                    test=test,
                    **kws_fix_code,
                )
            )
        return

    from roux.workflow.function import get_global_imports

    ## infer input parameters
    if output_path is None:
        if in_place:
            output_path = input_path
        else:
            verbose = True
    try:
        from removestar.removestar import fix_code, replace_in_nb
    except ImportError as error:
        logging.error(
            f"{error}: Install needed requirement using command: pip install removestar"
        )
        return

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

    # try:
    import tempfile

    replaces = fix_code(
        code=code,
        file=tempfile.NamedTemporaryFile().name,
        return_replacements=True,
    )
    # except:
    #     print(code)

    if replace_from in replaces:
        imports = replaces[replace_from]
        if imports != "":
            imports = imports.split(" import ")[1].split(", ")
            if test:
                logging.info(f"imports={imports}")
        else:
            imports = []
            if verbose:
                logging.warning(f"no function imports found in '{input_path}'")
    else:
        logging.info(f"'{replace_from}' not found in '{input_path}'")
        if output_path is not None and not in_place:
            logging.warning("copying the file, as it is.")
            shutil.copy(input_path, output_path)
        return output_path

    imports_attrs = [k for k, v in attributes.items() if any([s in code for s in v])]
    if len(imports_attrs) == 0:
        if verbose:
            logging.warning(f"no attribute imports found in '{input_path}'")

    if len(imports + imports_attrs) == 0:
        logging.info(f"no imports found in '{input_path}'")
        if output_path is not None and not in_place:
            logging.warning("copying the file, as it is.")
            shutil.copy(input_path, output_path)
        return output_path

    df2 = get_global_imports()

    def get_lines_replace_with(imports, df2):
        ds = df2.query(expr=f"`function name` in {imports}").apply(
            lambda x: f"## {x['function comment']}\n{x['import statement']}", axis=1
        )
        if len(ds) == 0:
            return "\n"
        lines = ds.tolist()
        return "\n".join(lines)

    replace_with = ""
    if len(imports) != 0:
        replace_with += get_lines_replace_with(imports, df2.query("`attribute`==False"))

    if len(imports_attrs) != 0:
        replace_with += "\n" + get_lines_replace_with(
            imports_attrs, df2.query("`attribute`==True")
        )
    ## remove duplicate lines
    replace_with = "\n".join(
        pd.Series(replace_with.split("\n")).drop_duplicates(keep="first").tolist()
    )

    replace_with = replace_with.strip()
    replaces_ = {**replaces, **{replace_from: replace_with}}

    if verbose:
        logging.info("replace     :\n" + ("\n".join([k for k in replaces_.keys()])))
    if verbose:
        logging.info("replace_with:\n\n" + ("\n".join([v for v in replaces_.values()])))

    if output_path is not None:
        # save files
        if input_path.endswith(".py"):
            from roux.lib.str import replace_many

            new_code = replace_many(code, replaces_)

            open(output_path, "w").write(new_code)
        elif input_path.endswith(".ipynb"):
            from roux.workflow.nb import to_replaced_nb

            to_replaced_nb(
                input_path,
                replaces=replaces_,
                cell_type="code",
                output_path=output_path,
            )
    ## to ensure the imports are inferred correctly
    com = f"ruff check {output_path}"
    print(com)
    import subprocess

    res = subprocess.run(
        com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(res.stdout)
    return output_path


def replacestar_ruff(
    p: str,
    outp: str,
    replace: str = "from roux.global_imports import *",
    verbose=True,
) -> str:
    from roux import global_imports

    text = open(global_imports.__file__, "r").read()

    lines = (
        text.split("## begin replacestar")[1]
        .split("## end replacestar")[0]
        .replace("#noqa", "")
        .replace("set_theme", "set_theme #noqa")
        .replace("\n\n", "\n")
        .split("\n")
    )
    lines = [s.strip() for s in lines]
    lines = [s for s in lines if s != ""] + [""]
    # lines
    replace_with = "\n".join(lines)

    replaced = open(p, "r").read().replace(replace, replace_with)

    import tempfile

    temp = tempfile.NamedTemporaryFile().name + ".py"
    # temp

    open(temp, "w").write(replaced)

    from roux.workflow.io import post_code

    post_code(
        p=temp,
        lint=True,
        format=True,
        verbose=verbose,
    )

    replaced_lines = open(temp, "r").read().split("\n")

    # import shutil
    # shutil.move(temp,outp)

    drop_lines = []
    s_ = ""  # last
    for i, s in enumerate(replaced_lines):
        s = s.strip()
        if s.startswith("## setting states"):
            break
        if s_.startswith("#") and s.startswith("#"):
            drop_lines.append(i - 1)
        if s == "":
            drop_lines.append(i)
        s_ = s

    cleaned_lines = [s for i, s in enumerate(replaced_lines) if i not in drop_lines]

    cleaned_text = "\n".join(cleaned_lines)

    open(outp, "w").write(cleaned_text)
    return outp


def post_code(
    p: str,
    lint: bool,
    format: bool,
    verbose: bool = True,
):
    ## ruff
    com = ""
    if lint:
        com += f"ruff check --fix {p};"
    if format:
        com += f"ruff format {p};"
    import subprocess

    res = subprocess.run(
        com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        print(res.stdout)
    return res


def to_clean_nb(
    p,
    outp: str = None,
    in_place: bool = False,
    temp_outp: str = None,
    clear_outputs=False,
    drop_code_lines_containing=[
        ## dev
        r".*%run .*",
        ## unused params
        r"^#\s*.*=.*",
        ## unused strings
        r'^#\s*".*',
        r"^#\s*'.*",
        r'^#\s*f".*',
        r"^#\s*f'.*",
        r"^#\s*df.*",
        r"^#\s*.*kws_.*",
        ## lines with one hashtag (not a comment)
        r"^\s*#\s*$",
        r"^\s*#\s*break\s*$",
        ## unused
        # "\[X", #noqa
        # "\[old ", #noqa
        "#old",
        "# old",
        # "\[not used", #noqa
        "# not used",
        ## development
        "#tmp",
        "# tmp",
        "#temp",
        "# temp",
        "check ",
        "checking",
        "# check",
        # "\[SKIP", #noqa
        "DEBUG ",
        # "#todos","# todos",'todos',
    ],
    drop_headers_containing=[
        "check",
        "[check",
        "old",
        "[old",
        "tmp",
        "[tmp",
    ],
    ## ruff
    fix_stars=False,
    lint=False,
    format=False,
    **kws_fix_code,
) -> str:
    """
    Wraper around the notebook post-processing functions.

    Usage:
        For notebooks developed using roux.global_imports.

        On command line:

        ## single input
        roux to-clean-nb in.ipynb out.ipynb -c -l -f

        ## multiple inputs
        roux to-clean-nb "in*.ipynb" -i -c -l -f

    Parameters:
        temp_outp (str): path to the intermediate output.
    """
    from roux.lib.io import read_ps

    input_paths = read_ps(p)
    if len(input_paths) > 1:
        assert in_place, in_place
        logging.info(f"Processing {len(input_paths)} files ..")
        ## Recursive
        outps = []
        for inp in input_paths:
            logging.info(f"Processing {inp} ..")
            outps.append(
                to_clean_nb(
                    p=inp,
                    outp=outp,
                    in_place=in_place,
                    temp_outp=temp_outp,
                    clear_outputs=clear_outputs,
                    drop_code_lines_containing=drop_code_lines_containing,
                    drop_headers_containing=drop_headers_containing,
                    ## ruff
                    lint=lint,
                    format=format,
                    **kws_fix_code,
                )
            )
        return

    from roux.workflow.nb import (
        to_clear_unused_cells,
        to_clear_outputs,
        to_filtered_outputs,
        to_filter_nbby_patterns,
        to_replaced_nb,
    )
    from roux.lib.sys import grep

    if in_place:
        outp = p
    else:
        # makedirs(outp)
        from pathlib import Path

        Path(outp).parent.mkdir(parents=True, exist_ok=True)

    if temp_outp is None:
        import tempfile

        temp_outp = f"{tempfile.gettempdir()}/to_clean_nb.ipynb"

    # Remove the code blocks that have all commented code and empty lines
    to_clear_unused_cells(
        p,
        temp_outp,
    )

    if clear_outputs:
        to_clear_outputs(
            temp_outp,
            temp_outp,
        )

    to_filtered_outputs(temp_outp, temp_outp)

    to_filter_nbby_patterns(temp_outp, temp_outp, patterns=drop_headers_containing)

    to_replaced_nb(
        nb_path=temp_outp,
        output_path=temp_outp,
        replaces={
            ## to replace the star
            " import * #noqa": " import *",
            " import *  #noqa": " import *",
            'if "metadata" in globals(): del metadata': 'if "metadata" in globals():\n   del metadata #noqa',
        },
        cell_type="code",
        drop_lines_with_substrings=drop_code_lines_containing,
    )

    _l = grep(
        p=temp_outp,
        checks=drop_code_lines_containing,
        exclude=["## backup old files if overwriting (force is True)"],
    )

    assert len(_l) == 0, (p, _l)

    if fix_stars:
        try:    
            __import__("removestar")
        except:
            raise ModuleNotFoundError(
                "Optional interactive-use dependencies missing, install by running: pip install removestar"
            )
        
        res = replacestar(
            input_path=temp_outp,
            output_path=outp,
            replace_from="from roux.global_imports import *",
            in_place=False,
            attributes={"pandarallel": ["parallel_apply"], "rd": [".rd.", ".log."]},
            verbose=False,
            test=False,
            **kws_fix_code,
        )
        if res is None:
            return
    post_code(
        p=outp,
        lint=lint,
        format=format,
    )
    return outp

## post tasks
from roux.lib.sys import run_com
def valid_post_task_deps(
    ):
    return run_com('which quarto',returncodes=[0,1])!=''
    
def to_html(
    p,
    # outp=None,
    env='docs',
    kws="",
    verbose=False,
    ):
    """
    Args:
        verbose: True: include stderr        
    """
    if env is not None:
        pre=f"micromamba run -n {env} "
    else:
        pre=""
    # if outp is None:
    outp=Path(p).with_suffix(".html").as_posix()

    if isinstance(kws,list):
        kws=" -M ".join(kws)
    if verbose:
        kws+=" -M warning:false -M error:false" 
    ## convert
    run_com(
        f"{pre}quarto render {p} --to html --toc -M code-fold:true -M code-summary:'_' -M code-tools:true -M self-contained:true -M mermaid-theme=default "+kws,# --output-dir {Path(outp).parent.as_posix()} --output {Path(outp).name}",
        verbose=verbose,
    )
    ## clean
    # invalid escape sequence '\/'
    # run_com(
    #     "sed -i '' 's/<\/head>/<style>summary { display: none; }<\/style><\/head>/' "+outp,
    #     verbose=verbose,
    # )
    return outp


## tasks

def check_for_exit(
    p, # table
    data, # e.g cfg
    output_path, # in a script
    ):
    """
    Check for early exit.
    """
    from roux.lib.io import is_table_empty, to_data
    if is_table_empty(p):
        logging.warning("exiting early because table is empty..")
        to_data(data,output_path)
        import sys
        sys.exit(0)
"""For input/output of workflow."""

## logging
import logging

## data
import pandas as pd

from pathlib import Path
import shutil  # for copying files

from roux.lib.sys import (
    abspath,
    basename,
    basenamenoext,
    exists,
    glob,
    isdir,
    makedirs,
    splitext,
)
from roux.lib.io import read_dict, is_dict


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


def read_config(
    p: str,
    config_base=None,
    inputs=None,  # overwrite with
    append_to_key=None,
    convert_dtype: bool = True,
    verbose: bool = True,
):
    """
    Read configuration.

    Parameters:
        p (str): input path.
        config_base: base config with the inputs for the interpolations
    """
    from omegaconf import OmegaConf

    if isinstance(config_base, str):
        if exists(config_base):
            config_base = OmegaConf.create(read_dict(config_base))
            # logging.info(f"Base config read from: {config_base}")
        else:
            logging.warning(f"Base config path not found: {config_base}")
    ## read config
    d1 = read_dict(p)
    ## merge
    if config_base is not None:
        if append_to_key is not None:
            # print(config_base)
            # print(d1)
            d1 = {append_to_key: {**config_base[append_to_key], **d1}}
        # print(config_base)
        # print(d1)
        d1 = OmegaConf.merge(
            config_base,  ## parent
            d1,  ## child overwrite with
        )
        if verbose:
            print("base config used.")
    if inputs is not None:
        d1 = OmegaConf.merge(
            d1,  ## parent
            inputs,  ## child overwrite with
        )
        if verbose:
            print("inputs incorporated.")
    if isinstance(d1, dict):
        ## no-merging
        d1 = OmegaConf.create(d1)
    # ## convert data dypes
    if convert_dtype:
        d1 = OmegaConf.to_object(d1)
    return d1


## metadata-related
def read_metadata(
    p: str,
    ind: str = None,
    max_paths: int = 30,
    config_path_key: str = "config_path",
    config_paths: list = [],
    config_paths_auto=False,
    verbose: bool = False,
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
        logging.warning(f"not found: {p}")

    d1 = read_config(p, verbose=verbose, **kws_read_config)

    ## read dicts
    keys = d1.keys()
    for k in keys:
        # if isinstance(d1[k],str):
        #     ## merge configs
        #     if d1[k].endswith('.yaml'):
        #         d1=read_config(
        #             d1[k],
        #             config_base=d1,
        #             )
        # el
        if isinstance(d1[k], dict):
            ## read `config_path`s
            # if len(d1[k])==1 and list(d1[k].keys())[0]==config_path_key:
            if config_path_key in list(d1[k].keys()):
                if verbose:
                    logging.info(f"Appending config to {k}")
                if exists(d1[k][config_path_key]):
                    d1 = read_config(
                        p=d1[k][config_path_key],
                        config_base=d1,
                        append_to_key=k,
                        verbose=verbose,
                    )
                else:
                    if verbose:
                        logging.warning(f"not exists: {d1[k][config_path_key]}")
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
            ind = splitext(p)[0] + "/"
            if verbose:
                logging.info(ind)
            config_paths += glob(f"{ind}/*")
    ## before
    config_size = len(d1)
    ## separate metadata (.yaml) /data (.json) files
    for p_ in config_paths:
        if isdir(p_):
            if len(glob(f"{p_}/*.json")) != 0:
                ## data e.g. stats etc
                if basename(p_) not in d1 and len(glob(f"{p_}/*.json")) != 0:
                    d1[basename(p_)] = read_dict(f"{p_}/*.json")
                elif (
                    isinstance(d1[basename(p_)], dict)
                    and len(glob(f"{p_}/*.json")) != 0
                ):
                    d1[basename(p_)].update(read_dict(f"{p_}/*.json"))
                else:
                    logging.warning(f"entry collision, could not include '{p_}/*.json'")
        else:
            if is_dict(p_):
                d1[basenamenoext(p_)] = read_dict(p_)
            else:
                logging.error(f"file not found: {p_}")
    if (len(d1) - config_size) != 0:
        logging.info(
            "metadata appended from "
            + str(len(d1) - config_size)
            + " separate config/s."
        )
    # if verbose and 
    if 'version' in d1:
        logging.info(f"version: {str(d1['version'])}")        
    return d1


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
        "\[X",
        "\[old ",
        "#old",
        "# old",
        "\[not used",
        "# not used",
        ## development
        "#tmp",
        "# tmp",
        "#temp",
        "# temp",
        "check ",
        "checking",
        "# check",
        "\[SKIP",
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
    try:
        __import__("removestar")
        __import__("ruff")
    except:
        raise ModuleNotFoundError(
            "Optional interactive-use dependencies missing, install by running: pip install roux[workflow]"
        )

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
    env=None,
    verbose=False,
    ):
    if env is not None:
        pre=f"micromamba run -n {env} "
    else:
        pre=""
    # if outp is None:
    outp=Path(p).with_suffix(".html").as_posix()
        
    ## convert
    run_com(
        f"{pre}quarto render {p} --to html --toc -M code-fold:true -M code-summary:'_' -M code-tools:true -M self-contained:true",# --output-dir {Path(outp).parent.as_posix()} --output {Path(outp).name}",
        verbose=verbose,
    )
    ## clean
    run_com(
        "sed -i '' 's/<\/head>/<style>summary { display: none; }<\/style><\/head>/' "+outp,
        verbose=verbose,
    )
    return outp

"""For function management."""

import logging
from roux.lib.sys import isdir, exists, dirname, basename, makedirs, basenamenoext
from roux.lib.str import replace_many
from roux.lib.io import to_path
from roux.lib.set import unique, dropna
import pandas as pd


def get_quoted_path(s1: str) -> str:
    """Quoted paths.

    Args:
        s1 (str): path.

    Returns:
        str: quoted path.
    """
    s1 = f'"{s1}"'
    if "{metadata" in s1:
        s1 = "f" + s1
        s1 = replace_many(s1, {"[": "['", "]": "']"})
    return s1


def get_path(
    s: str, validate: bool, prefixes=["data/", "metadata/", "plot/"], test=False
) -> str:
    """Extract pathsfrom a line of code.

    Args:
        s (str): line of code.
        validate (bool): validate the output.
        prefixes (list, optional): allowed prefixes. Defaults to ['data/','metadata/','plot/'].
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: path.

    TODOs:
        1. Use wildcards i.e. *'s.
    """
    if "=" in s:  # and ((not "='" in s) or (not '="' in s)):
        s = s[s.find("=") + 1 :]
    if ")" in s:
        s = s.split(")")[0]
    l0 = [s_ for s_ in s.split(",") if "/" in s_]
    l0 = [
        dirname(s.split("{")[0]) if (("{" in s) and ("{metadata" not in s)) else s
        for s in l0
    ]
    l0 = [dirname(s.split("*")[0]) if "*" in s else s for s in l0]
    if test:
        logging.info(l0)
    if len(l0) != 1:
        if validate:
            assert len(l0) == 1
        else:
            s1 = ""
    else:
        s = l0[0]
        s = replace_many(
            s,
            [
                "read_table",
                "to_table",
                "read_plot",
                "to_plot",
                " ",
                "f'",
                'f"',
                '"',
                "'",
                "(",
                ")",
            ],
            "",
            ignore=True,
        )
        # if test:info(s)
        if any([s.startswith(s_) for s_ in prefixes]):
            s1 = s
        else:
            s1 = ""
    if test:
        logging.info(s1)
    s1 = get_quoted_path(s1)
    s1 = replace_many(s1, {"//": "/", "///": "/"}, ignore=True)
    return s1


def remove_dirs_from_outputs(outputs: list, test: bool = False) -> list:
    """Remove directories from the output paths.

    Args:
        outputs (list): output paths.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        list: paths.
    """
    l_ = [s.replace('"', "") for s in outputs if not s.startswith("f")]
    if any([isdir(p) for p in l_]) and any([not isdir(p) for p in l_]):
        # if filepath is available remove the directory paths
        outputs = [f'"{p}"' for p in l_ if not isdir(p)]
        if test:
            logging.info("directory paths removed")
    if test:
        print(outputs)
    return outputs


def get_ios(l: list, test=False) -> tuple:
    """Get input and output (IO) paths.

    Args:
        l (list): list of lines of code.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        tuple: paths of inputs and outputs.
    """
    ios = [s_ for s_ in l if (("data/" in s_) or ("plot/" in s_) or ("figs/" in s_))]
    if test:
        logging.info(ios)
    inputs = [
        f"{get_path(s,validate=False,test=test)}"
        for s in ios
        if ("read_" in s) or (s.lstrip().startswith("p="))
    ]
    outputs = [
        f"{get_path(s,validate=False,test=test)}"
        for s in ios
        if (("to_" in s) or (s.lstrip().startswith("outp="))) and ("prefix" not in s)
    ]
    outputs = remove_dirs_from_outputs(outputs, test=test)
    inputs, outputs = (
        [p for p in inputs if p != '""'],
        [p for p in outputs if p != '""'],
    )
    return unique(inputs), unique(outputs)


def get_name(
    s: str,
    i: int,
    sep_step: str = "## step",
) -> str:
    """Get name of the function.

    Args:
        s (str): lines in markdown format.
        sep_step (str, optional): separator marking the start of a step. Defaults to "## step".
        i (int): index of the step.

    Returns:
        str: name of the function.
    """
    assert s.startswith(f"# {sep_step}")
    assert s.count(sep_step) == 1
    s1 = to_path(s.replace(f"# {sep_step}", f"step{i:02}")).lower().replace("/", "_")
    s1 = s1 if len(s1) >= 80 else s1[:80]
    s1 = replace_many(s1, {" ": "_", ".": "_"}, ignore=True)
    return s1


def get_step(
    l: list,
    name: str,
    sep_step: str = "## step",
    sep_step_end: str = "## tests",
    test=False,
    tab="    ",
) -> dict:
    """Get code for a step.

    Args:
        l (list): list of lines of code
        name (str): name of the function.
        test (bool, optional): test mode. Defaults to False.
        tab (str, optional): tab format. Defaults to '    '.

    Returns:
        dict: step name to code map.
    """
    # to_fun():
    if test:
        logging.info(name, l[-1])
    docs = []
    code_with_comments = []
    get_docs = True
    get_code = False
    ## in markdown cell
    for i, s in enumerate(l):
        if (s.startswith("#") and not s.startswith("# In[")) and get_docs:
            if i == 0:
                # info(s)
                s = (
                    "# "
                    + s.split(sep_step.split("## ")[0] + " ")[1][0].upper()
                    + s.split(sep_step.split("## ")[0] + " ")[1][1:]
                    + "."
                )
            docs.append(s)
        else:
            get_docs = False
            get_code = True
        if get_code:
            code_with_comments.append(s)

    if test:
        logging.info(docs)
    ## [X: gets commented code] steps marked as '## #1', '## #2' ..
    # docs+=[s for s in l if s.startswith('## #')]
    ## remove leading #
    docs = "\n".join([s[2:] for s in docs])

    code = [s for s in code_with_comments if not s.startswith("#")]
    #     info(code)
    inputs, outputs = get_ios(code, test=test)
    if len(set(inputs) & set(outputs)) != 0:
        inputs = [s for s in inputs if s not in outputs]
    #     if name.endswith('network_networkx'):
    #         info(inputs,outputs)
    if test:
        logging.info(inputs, outputs)
    if "plot" in name:
        output_type = "plots"
    elif "figure" in name:
        output_type = "figures"
    else:
        output_type = "data"
    inputs_str = f",\n{tab*3}".join(inputs)
    outputs_str = f",\n{tab*3}".join(outputs)
    if output_type != "data" and (
        not any([isdir(p.replace('"', "")) for p in outputs])
    ):
        outputs_str = (
            f'report([\n{tab*3}{outputs_str}],\n{tab*3}category="{output_type}")'
        )
    #     elif output_type in ['plots','figures']:
    #         outputs_str=''
    ## snakemake rule
    config = [
        f"    rule {name}:",
        "        input:",
        f"            {inputs_str}",
        "        output:",
        f"            {outputs_str}",
        "        run:",
        f"            from lib.{name.split('_step')[0]} import {name}",
        f"            {name}(metadata=metadata)",
    ]
    config_str = "\n".join(config) + "\n"
    quotes = '"""'
    function = [
        f"def {name}(metadata=None):",
        f"    {quotes}{docs}",
        "",
        "    Parameters:",
        "        metadata (dict): Dictionary containing information required for the analysis. Metadata files are located in `metadata` folder are read using `roux.workflow.io.read_metadata` function.",
        "",
        "    Snakemake rule:",
        "\n".join([tab + s + tab for s in config]),
        f"    {quotes}",
        "    " + "\n    ".join(code_with_comments),
    ]
    if test:
        logging.info(function[0])
    function = "\n".join(function)
    return {
        "function": function,
        "config": config_str,
        "inputs": inputs,
        "outputs": outputs,
    }


def to_task(
    notebookp,
    task=None,
    sep_step: str = "## step",
    sep_step_end: str = "## tests",
    notebook_suffix: str = "_v",
    force=False,
    validate=False,
    path_prefix=None,
    verbose=True,
    test=False,
) -> str:
    """Get the lines of code for a task (script to be saved as an individual `.py` file).

    Args:
        notebookp (_type_): path of the notebook.
        sep_step (str, optional): separator marking the start of a step. Defaults to "## step".
        sep_step_end (str, optional): separator marking the end of a step. Defaults to "## tests".
        notebook_suffix (str, optional): suffix of the notebook file to be considered as a "task".
        force (bool, optional): overwrite output. Defaults to False.
        validate (bool, optional): validate output. Defaults to False.
        path_prefix (_type_, optional): prefix to the path. Defaults to None.
        verbose (bool, optional): show verbose. Defaults to True.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        str: lines of the code.
    """
    if not sep_step.startswith("## "):
        raise ValueError(f"{sep_step} should start with '## '")
    # from roux.lib.str import removesuffix
    if notebook_suffix != "":
        pyp = f"{dirname(notebookp)}/lib/task{basenamenoext(notebookp).split(notebook_suffix)[0] if task is None else task}.py"
    else:
        pyp = f"{dirname(notebookp)}/lib/task{basenamenoext(notebookp) if task is None else task}.py"
    # info(notebookp,pyp)
    # brk
    if exists(pyp) and not force and not test:
        return
    if verbose:
        logging.info(basename(notebookp))
    from roux.workflow.io import to_py, get_lines

    if not test:
        to_py(notebookp, pyp=pyp.replace("/lib/", "/.lib/"), force=force)
    l0 = get_lines(notebookp, keep_comments=True)
    # print(l0)
    # info(sep_step,sep_step_end)
    if path_prefix is not None:
        l0 = [replace_many(s, {"data/": "../data/"}, ignore=True) for s in l0]
    taskn = basenamenoext(pyp)
    if test:
        logging.info(taskn)
    d0 = {}
    get = False
    get_header = True
    l1 = []  # start of the file
    l2 = []  # start of code
    for s in l0:
        if s.startswith(f"# {sep_step}"):
            l2 = []  # start of code
            get_header = False
            get = True
            k = s
        elif s.startswith(sep_step_end) and len(l2) != 0:
            get = False
            stepn = f"{taskn}_{get_name(k,i=len(d0.keys())+1,sep_step=sep_step)}"
            d0[stepn] = get_step(
                l2,
                name=stepn,
                sep_step=sep_step,
                sep_step_end=sep_step_end,
                test=test or verbose,
            )
            l2 = []
            stepn = None
        elif s.startswith("# In"):
            continue
        elif get_header:
            if "get_ipython" in s:
                continue
            l1.append(s)
        if get:
            l2.append(s)
    l3 = []
    for s in l1:
        l3.append(s)
        header = s.startswith("#")
        if ("import" not in s and not header) or s.startswith("# ## "):
            break
    df0 = pd.DataFrame(d0).T
    df0.index.name = "step name"
    df0 = df0.reset_index()
    df0.index.name = "step #"
    df0 = df0.reset_index()
    if len(df0) == 0 and not validate:
        # if verbose:
        logging.warning("no functions found")
        return None, None
    if not test:
        makedirs(pyp)
        with open(pyp, "w") as f:
            f.write("\n".join(l3) + "\n\n" + "\n\n".join(df0["function"].tolist()))
    return "\n".join(df0["config"].tolist()).replace("\n    ", "\n").replace(
        "    rule", "rule"
    ), df0["outputs"].tolist()


def get_global_imports() -> pd.DataFrame:
    """
    Get the metadata of the functions imported from `from roux import global_imports`.
    """
    from roux import global_imports

    lines = open(global_imports.__file__, "r").read().split("\n")

    def clean_(s):
        if ("import " in s or s.startswith("## ")) and not s.startswith("# "):
            # s=s.split(' # ')[0].strip()
            if s not in [""]:
                return s.strip()

    # lines=
    lines = dropna(list(map(clean_, lines)))

    lines_grouped = {}
    k = None
    i = 0
    for s in lines:
        if s.startswith("## "):
            i += 1
            k = (i, f"{s[3:]}")
            lines_grouped[k] = []
            continue
        else:
            lines_grouped[k].append(s)

    df1 = (
        pd.Series(lines_grouped)
        .explode()
        .to_frame("import statement")
        .rename_axis(["rank", "function comment"])
        .reset_index()
        .dropna(subset=["import statement"])
        .query(expr="~(`import statement`.str.strip().str.startswith('#'))")
    )
    # df1

    def get_function_name(s):
        try:
            s = s.split("import ")[1]
            if " as " in s:
                s = s.split(" as ")[1]
            s = s.split(";")[0].split("#")[0]
            if "," not in s:
                return s.strip()
            else:
                return [s_.strip() for s_ in s.split(",")]
        except:
            print(s)

    def clean_import_statements(s, attribute, function_name):
        if not attribute:
            s = s.split("# ")[0]
            if "," in s:
                s = f"{s.split(' import ')[0]} import {function_name}"
        return s

    # return df1
    df2 = (
        df1.assign(
            **{
                "internal": lambda df: df["function comment"].apply(
                    lambda x: "roux" in x
                ),
                "function type": lambda df: df["function comment"].str.split(
                    " ", expand=True
                )[0],
                "rank": lambda df: df.groupby("function type")["rank"].transform("min"),
                "attribute": lambda df: df["import statement"].apply(
                    lambda x: "# attribute" in x
                ),
                "function name": lambda df: df["import statement"].apply(
                    get_function_name
                ),
            }
        )
        .explode("function name")
        .assign(
            **{
                "import statement": lambda df: df.apply(
                    lambda x: clean_import_statements(
                        x["import statement"],
                        x["attribute"],
                        x["function name"],
                    ),
                    axis=1,
                ),
            }
        )
        .sort_values(
            ["rank", "function type", "internal", "attribute", "function name"]
        )
    )
    return df2

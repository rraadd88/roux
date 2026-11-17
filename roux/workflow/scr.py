"""
Notebooks to scripts (`to_scr`), and the conversion helpers shared by `to_pkg` and `to_scrs`.
"""

import functools
import logging
import re
from pathlib import Path

from roux.lib.sys import exists, makedirs, read_ps
from roux.lib.text import replace_text
from roux.workflow.pms import read_cli_pm


## io files
def to_py(
    notebookp: str,
    pyp: str = None,
    clean=False,
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
    # g: for .py (jupytext percent) input the default output path equals the input path
    assert Path(pyp).resolve() != Path(notebookp).resolve(), (
        f"output path is the same as the input path ({notebookp}); pass `pyp`."
    )
    if exists(pyp) and not force:
        return
    makedirs(pyp)

    from roux.workflow.nb import get_lines

    l1 = get_lines(notebookp, **kws_get_lines)
    if clean:
        l1 = [s for s in l1 if not s.strip().startswith("# In")]

    l1 = "\n".join(l1).encode("ascii", "ignore").decode("ascii")
    with open(pyp, "w+") as fh:
        fh.writelines(l1)
    return pyp


## export helper
## rules to check and fix: the former defaults of ruff (newer versions enable many more)
RUFF_SELECT = "E4,E7,E9,F"


def _ruff():
    """The `ruff` of the running environment (also if it is not activated), else the one on `PATH`; None if neither."""
    import shutil
    import sys

    p = Path(sys.executable).parent / "ruff"
    return str(p) if p.exists() else shutil.which("ruff")


def check_py(
    p,
    errors="raise",
    ruff_ignore="E402,E722,F841,E741,F401,E712,F541,F841,E721",  # while prototyping
):
    """
    E402	pycodestyle	Module level import not at top of file.
    E721	pycodestyle	Do not compare types, use isinstance().
    E722	pycodestyle	Do not use bare except: (use except Exception:).
    F841	Pyflakes	Local variable is assigned to but never used.
    E741	pycodestyle	Do not use variables named 'l', 'O', or 'I' (ambiguous characters).
    F401	Pyflakes	Module imported but unused.
    E712	pycodestyle	Comparison to True should be if cond is True: or if cond:.
    F541	Pyflakes	f-string without any placeholders.
    F841    Variable defined but not used
    """
    if p.endswith(".py"):
        init_path = f"{Path(p).parent}/__init__.py"
        if not Path(init_path).exists():
            Path(init_path).touch()
            logging.warning(f"created missing: {init_path}")

    import shlex

    ruff = _ruff()
    if ruff is None:
        logging.warning(f"ruff not found: not checked: {p}")
        return ""
    com = f"{shlex.quote(ruff)} check {shlex.quote(str(p))} --select {RUFF_SELECT} --ignore {ruff_ignore}"
    from roux.lib.sys import run_com

    stdout = run_com(
        com,
        wait=True,
        returncodes=[0, 1],  # get stdout regardless
    )  # Captures both streams

    if errors == "raise":
        logging.warning(stdout)
        assert (stdout == "") or ("All checks passed!" in stdout), (com, stdout)
    else:
        logging.warning(f"fix following by running: {com}")
    return stdout


def post_code(
    p: str,
    lint: bool,
    format: bool,
    verbose: bool = True,
):
    ## %run
    replace_text(p, {"get_ipython()": "# get_ipython()"})

    ## ruff
    import shlex

    ruff = _ruff()
    if ruff is None:
        logging.warning(f"ruff not found: not linted/formatted: {p}")
        return None
    ruff = shlex.quote(ruff)
    com = ""
    if lint:
        com += f"{ruff} check --select {RUFF_SELECT} --fix {shlex.quote(str(p))};"  # -unsafe-fixes
    if format:
        com += f"{ruff} format {shlex.quote(str(p))};"
    import subprocess

    res = subprocess.run(
        com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if verbose:
        logging.info(res.stdout)
    return res


def to_py_path(p):
    return "/".join(
        list(Path(p).parts[:-2])
        + [
            Path(p).parts[-2].replace("nbs", "").replace("notebooks", "").strip("_"),
            Path(p).stem + ".py",
        ]
    )


def to_scr_path(p):
    """
    Path of a notebook to the path of the script in the `scr_` layout.

    e.g. `notebooks_diffs/01_x.ipynb` -> `scr_diffs/01_x.py`
    """
    from roux.lib.str import replace_many

    outp = replace_many(
        str(p),
        {
            ".ipynb": ".py",
            "nbs_": "scr_",
            "notebooks_": "scr_",
            "_nbs": "",
            "_notebooks": "",
        },
        use_template=False,
    )
    if "scr_" not in outp:
        pp = Path(outp)
        outp = (pp.parent.parent / f"scr_{pp.parent.stem}" / pp.name).as_posix()
    return outp


def _pms_to_run_args(code):
    """
    Parameters cell -> the arguments of `run()` and the lines that restore the non-literal defaults.

    Parameters:
        code (str): source of the parameters cell.

    Returns:
        args (list): `name=default` strings. A literal default (accepted by `ast.literal_eval`, e.g. numbers,
            strings, lists, dicts, `None`) is rendered with `repr()`. A non-literal default (e.g. an f-string,
            a name, a call or an expression on other parameters) becomes `None`, because it may refer to the
            notebook's variables, which a function default can't.
        pre (list): statements for the beginning of `run()`, `if <name> is None: <name> = <expression>`, in
            the order of the cell. They restore the non-literal defaults at run time, where the expressions
            can use the module-level variables and the other parameters. Passing `None` also gets them.
            The expression is the cell's source text, not `ast.unparse`, whose output depends on the
            Python version (e.g. 3.12.3 reuses the outer quotes inside f-strings, valid only on >=3.12).
            A multi-line expression keeps its continuation lines as they are (inside brackets or strings,
            so they need no indentation).
    """
    import ast

    tree = ast.parse(code)
    pms = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    pms[target.id] = node.value

    args, pre = [], []
    for k, v in pms.items():
        try:
            args.append(f"{k}={ast.literal_eval(v)!r}")
        except Exception:
            args.append(f"{k}=None")
            expr = ast.get_source_segment(code, v)
            if expr is None:
                expr = ast.unparse(v)
            pre.append(f"if {k} is None: {k} = {expr}")
    return args, pre


def to_scr(
    p,
    outp=None,
    with_pms="True",  # py with --pms (preferred)
    mark_end="## END",
    pre_clean=None,
    clean=False,
    replaces={
        "get_ipython": "#get_ipython",
        "sys.exit()": "return",
        "sys.exit(0)": "return",
        "sys.exit(1)": "return",
    },
    ## ruff
    check=True,
    fix=False,  # replacestar+format
    clean_comments="True",  # with fix: drop the excessive comments (see `replacestar_ruff`)
    errors="raise",
    verbose=False,
    **kws_check_py,
):
    """
    Notebook to command line script.

    # roux to-src ipynb py
    # ruff check py --ignore E402

    Args:
        p (str): path to the notebook, either .ipynb or a jupytext percent-format .py (`# %%` cells).
        outp (str, optional): path to the output script. Required when it would otherwise equal `p`.

    Note:
        Papermill-based execution (`task.py`) still needs an .ipynb; keep a paired notebook with
        `jupytext --set-formats ipynb,py:percent`.

        With `with_pms`, the assignments of the parameters cell become the arguments of `run()` (see
        `_pms_to_run_args`). Literal defaults are kept (`repr()`). Non-literal ones (e.g.
        `p=f"../out/{cfg['version']}.yaml"`) default to `None` and are assigned at the beginning of `run()`
        if `None`, i.e. evaluated at run time, as in the notebook.

    TODO:
        Prefer py with --pms
    """

    if outp is None:
        outp = to_py_path(p)
        logging.warning(f"outp={outp}")
    # g: guard against overwriting a .py (jupytext percent) input
    assert Path(outp).resolve() != Path(p).resolve(), (
        f"output path is the same as the input path ({p}); pass `outp`."
    )

    ## TODO: use decorator
    with_pms = read_cli_pm(with_pms)
    pre_clean = read_cli_pm(pre_clean)

    import tempfile

    ## scratch files in a temporary directory (before: `.to_src/` in the current directory)
    with tempfile.TemporaryDirectory(prefix="roux_to_scr_") as scratch_dir_path:
        if pre_clean:
            if pre_clean == True:
                pre_clean = {}
            assert isinstance(pre_clean, dict), pre_clean

            from roux.workflow.nb import to_clean_nb

            p = to_clean_nb(
                p,
                f"{scratch_dir_path}/{Path(p).name}",
                **{
                    **dict(
                        clear_outputs=False,
                        drop_code_lines_containing=[],
                        drop_headers_containing=[],
                    ),
                    **pre_clean,
                },
            )

        pyp = to_py(
            p,
            pyp=f"{scratch_dir_path}/{Path(p).stem}.py",
            clean=clean,
            force=True,
        )

        t_raw = open(pyp, "r").read()
    # print(t_raw[:100])

    t_raw = t_raw.split(mark_end)[0]
    # print(t_raw[:100])

    ## TODO: use textwrap
    t_tab = t_raw.replace("\n", "\n    ")
    # print(t_tab[:100])

    def split_by_pms(text):
        splits1 = re.split(r"# In\[\s*\d*\s*\]:\n    #+ param", text)
        # splits1=re.split(r"# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # splits1 = re.split(r"[ \t]*# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # splits1=re.split(r"\s*# In\[\s*\d*\s*\]:\n\s*#+\s*params?", text)
        # print(text[:200])
        assert len(splits1) == 2, text[:200].replace("\n", "\\n") + " .."

        pre_pms = splits1[0].replace("\n    ", "\n")

        pms = re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1])[0]

        post_pms = re.split(r"\n    # In\[\s*\d*\s*\]:", splits1[1], 1)[1]
        post_pms = re.sub(r"\n    # In\[\s*\d*\s*\]:", "\n", post_pms)
        return ([pre_pms, post_pms], pms)

    t_pms_pre = ""
    if with_pms:
        t_splits, params = split_by_pms(t_tab)
        ## the cell's code without the rest of the '## param..' line and the indentation of `t_tab`
        params_code = "\n".join(
            [s[4:] if s.startswith("    ") else s for s in params.split("\n")[1:]]
        )
        try:
            params_args, params_pre = _pms_to_run_args(params_code)
        except SyntaxError as e:
            logging.warning(f"parameters not parsed ({e}); using extract_pms")
            ## remove comments
            params_lines = params.split("\n")
            params = "\n".join([s.split("#")[0] for s in params_lines])

            from roux.workflow.pms import extract_pms

            params_args = extract_pms(params.split("    ")[1:], fmt="str")
            params_pre = []
        params_str = " " * 4 + ",\n    ".join(params_args)
        if len(params_pre) > 0:
            ## indent only the first line: the continuation lines are kept as they are
            t_pms_pre = "\n".join(
                [
                    " " * 4 + s
                    for s in ["## the defaults that are not literals"] + params_pre
                ]
            )
        if verbose:
            logging.info(params_str)
    else:
        t_splits = ["", t_tab]

    t_def_pre = """
    if input_path is not None and output_path is None:
        ## should be params
        from roux.workflow.task import pre_params
        params=pre_params(input_path)
        # if len(params)!=1:
        #     logging.info(f"processing {len(params)} pms")
        from roux.workflow.function import call_with_kws
        for i,_kws in enumerate(params):
            call_with_kws(
                run,
                _kws,
                **{k:v for k,v in dict(verbose=locals().get('verbose'),force=locals().get('force'),).items() if v is not None}
            )
        return 

    ## for cli
    assert input_path is not None, 'check docs using --help'
    assert output_path is not None
    """

    t_def = "\n".join(
        [
            """
import argh
def run(""",
            params_str if with_pms else "",
            """):""",
            t_pms_pre,
            t_def_pre if with_pms else "",
        ]
    )

    t_end = """
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

    t_src = t_splits[0] + t_def + t_splits[1] + t_end

    from roux.lib.str import replace_many

    t_src = replace_many(
        t_src,
        replaces,
        errors=None,
        use_template=False,
    )

    Path(outp).parent.mkdir(exist_ok=True)
    open(outp, "w").write(t_src)
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
            method="filter",  # select
            method_kws={"clean": read_cli_pm(clean_comments)},
            errors=errors,
            verbose=verbose,
        )
        post_code(
            outp,
            lint=True,
            format=True,
            verbose=verbose,
        )
    return outp


## alias
# source=script


## deprecated alias; `functools.wraps` keeps `__name__` (`to_scr`) and the signature
@functools.wraps(to_scr)
def to_src(*args, **kws):
    import warnings

    warnings.warn("to_src is deprecated; use to_scr", DeprecationWarning, stacklevel=2)
    return to_scr(*args, **kws)


## notebooks repo -> package
## development leftovers to flag in the notebooks
_PKG_CHECKS = [
    "#tmp",
    "# tmp",
    "#temp",
    "# temp",
    "check ",
    "checking",
    "# check",
    "#todos",
    "# todos",
    "todos",
    "[SKIP",
    "[DEBUG ",
    "[X",
    "[old ",
    "#old",
    "# old",
    "[not used",
    "# not used",
    "[unused",
]
## default patterns to drop from the notebooks
_PKG_DROP_CODE_LINES = [
    ## unused
    r"^#\s*.*=.*",
    ## unused strings
    r'^#\s*".*',
    r"^#\s*'.*",
    r'^#\s*f".*',
    r"^#\s*f'.*",
    r"^#\s*df.*",
    r"^#\s*.*kws_.*",
    ## lines with one hashtag
    r"^\s*#\s*$",
    r"^\s*#\s*break\s*$",
    "#old",
    "# old",
    "# not used",
    ## development
    "#tmp",
    "# tmp",
    "#temp",
    "# temp",
    "check ",
    "checking",
    "# check",
    "DEBUG ",
]
_PKG_DROP_HEADERS = ["[check", "[old", "[tmp", "[temp", "[skip", "[unused"]


def _to_scr_running_py(p, outp):
    """Use .py in the tasks i.e. point the mentions of notebooks to the scripts."""
    from roux.lib.str import replace_many

    lines = []
    for s in open(p, "r"):
        if ".ipynb" in s:
            s = replace_many(
                s,
                {
                    ".ipynb": ".py",
                    "nbs_": "scr_",
                    "notebooks_": "scr_",
                    "_nbs": "",
                    "_notebooks": "",
                },
                use_template=False,
            )
        lines.append(s)
    text = re.sub(r"\n{1,2}", "\n", "\n".join(lines))
    open(outp, "w").write(text)


def _to_scr_cleaned(
    nb_path,
    py_path,
    nb_cleaned_path,
    py_cleaned_path,
    kind=None,
    clean_nbs=True,
    clean_comments=False,
    force_with_pms=False,
    author=None,
    drop_code_lines_containing=[],
    drop_headers_containing=[],
):
    """Check the notebook for the issues, clean it and convert it to a script."""
    import inspect

    from roux.lib.sys import grep
    from roux.workflow.nb import to_clean_nb

    nb_path, py_path = str(nb_path), str(py_path)

    #### Check for issues
    lines = grep(
        p=nb_path,
        checks=_PKG_CHECKS + list(drop_code_lines_containing),
        exclude=["## backup old files if overwriting (force is True)"],
    )
    logging.warning(lines)

    if clean_nbs:
        to_clean_nb(
            nb_path,
            nb_cleaned_path,
            clear_outputs=(kind == "code"),
            drop_code_lines_containing=list(drop_code_lines_containing),
            drop_headers_containing=list(drop_headers_containing),
        )
    else:
        nb_cleaned_path = nb_path

    ### To py (`nbs_` -> `scr_`)
    replaces = dict(inspect.signature(to_scr).parameters["replaces"].default)
    if author:
        #### Add the authorship
        replaces["\nimport argh"] = f"# Author: {author}\n\nimport argh"
    to_scr(
        nb_cleaned_path,
        py_cleaned_path,
        with_pms=Path(nb_path).stem != "run" or force_with_pms,
        replaces=replaces,
        fix=True,
        clean_comments=str(bool(clean_comments)),
        errors="raise",
        verbose=True,
    )

    #### Use .py in run_tasks
    Path(py_path).parent.mkdir(parents=True, exist_ok=True)
    _to_scr_running_py(py_cleaned_path, py_path)
    init_path = Path(py_path).parent / "__init__.py"
    if not init_path.exists():
        init_path.touch()
    return py_path


def _clean_scr(inp, outp, target_functions=None):
    """Remove the unused functions (by default, top-level) from a script."""
    import ast

    tree = ast.parse(open(inp, "r").read())
    if target_functions is None:
        target_functions = [
            n.name
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
    used = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    removable = [f for f in target_functions if f not in used]

    class FunctionRemover(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            return None if node.name in removable else self.generic_visit(node)

        visit_AsyncFunctionDef = visit_FunctionDef

    tree = FunctionRemover().visit(tree)
    ast.fix_missing_locations(tree)
    logging.info(
        {"removed": removable, "kept": [f for f in target_functions if f in used]}
    )
    open(outp, "w").write(ast.unparse(tree))
    return outp


def _to_scrs(src, out, convert, rm_unused_funcs=False, **kws):
    """Notebooks to scripts (`_to_scr_cleaned`), for the `convert` list of `_plan`."""
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir_path:
        for nb_rel, py_rel in convert:
            logging.info(f"{nb_rel} -> {py_rel}")
            _to_scr_cleaned(
                src / nb_rel,
                out / py_rel,
                nb_cleaned_path=f"{temp_dir_path}/00_cleaning.ipynb",
                py_cleaned_path=f"{temp_dir_path}/00_cleaning.py",
                **kws,
            )
            if rm_unused_funcs:
                _clean_scr(out / py_rel, out / py_rel)


def replacestar_ruff(
    p: str,
    outp: str,
    replace: str = "from roux.global_imports import *",
    clean=True,
    indent=None,
    verbose=True,
) -> str:
    """
    Replace the wildcard (global) import with the import lines of `roux.global_imports`.

    Parameters:
        replace (str): the statement to replace.
        indent (str): indentation of the import lines. If None, the statement is replaced at any indentation
            (e.g. at the module level and inside `run()` of a script by `to_scr`), each with its own.
    """
    import re
    import textwrap

    from roux.workflow.function import get_global_imports

    lines = get_global_imports(out_fmt="lines")
    ## remove
    replace_with = "\n".join(
        [s.replace("#noqa", "") if "#keep" not in s else s for s in lines]
        # lines
    ).replace("\n\n", "\n")

    text = open(p, "r").read()
    ## indent
    if indent is None:
        if verbose:
            logging.info("indent: as found")
        replaced = re.sub(
            r"(?m)^([ \t]*)" + re.escape(replace.strip()),
            lambda m: textwrap.indent(replace_with, m.group(1)),
            text,
        )
    else:
        if verbose:
            logging.info(f"indent='{indent}'")
        replaced = text.replace(replace, textwrap.indent(replace_with, indent))

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
            if s_.startswith("#") and (s.startswith("#") or s == ""):
                drop_lines.append(i - 1)
            # if s == "":
            #     drop_lines.append(i)
            s_ = s
        replaced_lines = [
            s for i, s in enumerate(replaced_lines) if i not in drop_lines
        ]

    # print(open(outp,'r').read()[:100])
    replaced_text = "\n".join(replaced_lines)

    open(outp, "w").write(replaced_text)
    return outp


## post-processing
def replacestar(
    input_path,
    output_path=None,
    replace_from="from roux.global_imports import *",
    method="filter",  # select
    method_kws={},
    errors="raise",
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
    if len(read_ps(input_path)) > 1:
        assert output_path is None, "only replacing allowed .."
        for p in read_ps(input_path):
            # recurse
            replacestar(
                p,
                replace_from=replace_from,
                method=method,  # select
                method_kws=method_kws,
                errors=errors,
                verbose=verbose,
            )
        return

    from roux.workflow.function import get_global_imports

    if output_path is None:
        output_path = input_path

    if method == "select":
        stdout = check_py(input_path, errors=None)
        # Find all F405 errors
        # pattern=': F405 `'
        pattern = "\x1b[36m:\x1b[0m \x1b[1;31mF405\x1b[0m `"
        funcs = sorted(
            list(
                set(
                    [
                        s.split(pattern)[1].split("`")[0]
                        for s in stdout.split("\n")
                        if pattern in s
                    ]
                )
            )
        )
        logging.warning(f"use roux replacestar for {funcs}")

        if len(funcs) > 0:
            imports = "\n".join(
                get_global_imports()
                .query(expr="`function name`=={funcs}")["import statement"]
                .tolist()
            )
            replace_text(input_path, {replace_from: "\n" + imports})
    else:
        output_path = replacestar_ruff(
            output_path,
            output_path,
            replace_from,
            verbose=verbose,
            **method_kws,
        )
    check_py(output_path, errors=errors)
    return output_path

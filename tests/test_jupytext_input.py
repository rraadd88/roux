"""Tests for reading jupytext percent-format .py files as notebooks, and converting notebooks to scripts."""

import ast
import py_compile
from pathlib import Path

import pytest

from roux.workflow.io import (
    _PKG_DROP_CODE_LINES,
    _PKG_DROP_HEADERS,
    _to_scr_cleaned,
    to_py,
    to_scr,
)
from roux.workflow.nb import read_nb

input_dir_path = "examples/inputs"
py_path = f"{input_dir_path}/demo_percent.py"
ipynb_path = f"{input_dir_path}/demo_percent.ipynb"


def test_read_nb_same_cells():
    nb_py = read_nb(py_path)
    nb_ipynb = read_nb(ipynb_path)
    assert [(c.cell_type, c.source) for c in nb_py.cells] == [
        (c.cell_type, c.source) for c in nb_ipynb.cells
    ]


def test_to_scr_py_equals_ipynb(tmp_path):
    outp_py = to_scr(py_path, outp=f"{tmp_path}/from_py/demo.py")
    outp_ipynb = to_scr(ipynb_path, outp=f"{tmp_path}/from_ipynb/demo.py")
    t_py = Path(outp_py).read_text()
    assert "def run(" in t_py
    assert t_py == Path(outp_ipynb).read_text()


def test_read_nb_unsupported_suffix(tmp_path):
    p = f"{tmp_path}/demo.txt"
    Path(p).write_text("")
    with pytest.raises(ValueError):
        read_nb(p)


def test_same_path_asserts():
    with pytest.raises(AssertionError, match="pass `outp`"):
        to_scr(py_path, outp=py_path)
    with pytest.raises(AssertionError, match="pass `pyp`"):
        to_py(py_path)


## notebook -> script (`to_scr` via `_to_scr_cleaned`, as in `to_scrs`/`to_pkg`)
def _to_scr_from_cells(tmp_path, monkeypatch, pre, pms, post):
    """Script from a notebook with the `pre` cells, a parameters cell (`pms` lines) and the `post` cells."""
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    nb.cells = (
        [nbf.v4.new_code_cell(s) for s in pre]
        + [
            nbf.v4.new_code_cell(
                "## parameters\n" + "\n".join(pms), metadata={"tags": ["parameters"]}
            )
        ]
        + [nbf.v4.new_code_cell(s) for s in post]
    )
    nb_path = tmp_path / "nb.ipynb"
    nbf.write(nb, nb_path)
    ## `to_scr` writes .to_src/ in the current directory
    monkeypatch.chdir(tmp_path)
    outp = _to_scr_cleaned(
        nb_path,
        tmp_path / "out" / "nb.py",
        nb_cleaned_path=str(tmp_path / "cleaned.ipynb"),
        py_cleaned_path=str(tmp_path / "cleaned.py"),
        force_with_pms=True,
        drop_code_lines_containing=list(_PKG_DROP_CODE_LINES),
        drop_headers_containing=list(_PKG_DROP_HEADERS),
    )
    py_compile.compile(outp, doraise=True)
    return Path(outp).read_text()


def test_global_imports_multiline():
    from roux.workflow.function import get_global_imports

    lines = get_global_imports(out_fmt="lines")
    for s in lines:
        ast.parse(s)  # each line is a whole statement
    text = "\n".join(lines)
    assert "read_dict" in text and "check_non_overlaps_with" in text
    names = get_global_imports()["function name"].tolist()
    assert "read_dict" in names and "dropna" in names
    assert not any(s.startswith("(") or s == "" for s in names), names


@pytest.mark.parametrize("where", ["module", "run"])
def test_to_scr_star_import(tmp_path, monkeypatch, where):
    ## the star import before (module level) or after (inside `run()`) the parameters cell
    star = ["from roux.global_imports import *"]
    t = _to_scr_from_cells(
        tmp_path,
        monkeypatch,
        pre=star if where == "module" else [],
        pms=["input_path=None", "output_path=None", "x=1"],
        post=(star if where == "run" else []) + ["print(read_dict, dropna([x, None]))"],
    )
    assert "def run(" in t and "import *" not in t
    imported = {
        a.name
        for n in ast.walk(ast.parse(t))
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    }
    assert {"read_dict", "dropna"} <= imported, imported


def test_to_scr_pms_defaults(tmp_path, monkeypatch):
    t = _to_scr_from_cells(
        tmp_path,
        monkeypatch,
        pre=["cfg = {'version': 'v1'}"],
        pms=[
            "input_path=None",
            "output_path=None",
            "x=1",
            "f=0.5",
            "b=True",
            "d={'a': 1}",
            "l=['a', \"b\"]",
            "expr=\"`col` != 'all'\"",
            's="it\'s"',
            "s1='1'",
            "c='#fff'  # a colour",
            "p=f\"../out/{cfg['version']}.yaml\"",
            "y=x * 2",
        ],
        post=["print(x, f, b, d, l, expr, s, s1, c, p, y)"],
    )
    run = [
        n
        for n in ast.parse(t).body
        if isinstance(n, ast.FunctionDef) and n.name == "run"
    ][0]
    defaults = {
        a.arg: ast.literal_eval(d)
        for a, d in zip(run.args.args[-len(run.args.defaults) :], run.args.defaults)
    }
    assert defaults == {
        "input_path": None,
        "output_path": None,
        "x": 1,
        "f": 0.5,
        "b": True,
        "d": {"a": 1},
        "l": ["a", "b"],
        "expr": "`col` != 'all'",
        "s": "it's",
        "s1": "1",
        "c": "#fff",
        ## not literals: assigned in `run()` if None
        "p": None,
        "y": None,
    }
    body = [ast.unparse(n) for n in run.body[:2]]
    assert body == [
        "if p is None:\n    p = f\"../out/{cfg['version']}.yaml\"",
        "if y is None:\n    y = x * 2",
    ], body

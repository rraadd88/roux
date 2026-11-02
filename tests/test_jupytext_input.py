"""Tests for reading jupytext percent-format .py files as notebooks."""

from pathlib import Path

import pytest

from roux.workflow.io import to_py, to_scr
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

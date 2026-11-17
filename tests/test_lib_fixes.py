"""Regression tests for fixed code paths in roux.lib (TODO 1, TODO 2)."""

import inspect

import numpy as np
import pandas as pd
import pytest


def test_to_sparse_cols():
    from roux.lib.df import to_sparse

    df1 = pd.DataFrame({"a": [0, 0, 1], "b": [0.0, np.nan, 2.0], "c": ["x", "y", "z"]})
    df2 = to_sparse(df1, cols=["a", "b"])
    assert isinstance(df2["a"].dtype, pd.SparseDtype)
    assert df2["a"].dtype.fill_value == 0
    assert isinstance(df2["b"].dtype, pd.SparseDtype)


def test_filter_rows_dict_test_mode():
    from roux.lib.df import filter_rows

    df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "x"]})
    df2 = filter_rows(df1, {"b": "x"}, test=True)
    assert df2["b"].tolist() == ["x", "x"]


def test_merge_by_subset_no_overlap():
    from roux.lib.dfs import merge_by_subset

    df1 = pd.DataFrame({"on": [[1, 2]], "v1": [1]})
    df2 = pd.DataFrame({"on": [[3]], "v2": [2]})
    df3 = merge_by_subset(df1, df2, on="on", suffixes=("_1", "_2"))
    assert len(df3) == 0
    assert {"v1", "v2"} <= set(df3.columns)


def test_resolve_paths_no_match(tmp_path):
    from roux.lib.sys import resolve_paths

    with pytest.raises(AssertionError):
        resolve_paths(f"{tmp_path}/KEY1/KEY2.txt", keys=["KEY1", "KEY2"])


def test_query_default_unchanged():
    """TODO 2: a call does not change the default of the next call."""
    from roux.lib.df import query

    default = inspect.signature(query).parameters["kws_log"].default
    df1 = pd.DataFrame({"a": [1, 2, 3]})
    query(df1, expr="`a` > 1", subset=["a"])
    assert inspect.signature(query).parameters["kws_log"].default == default == {}


@pytest.mark.parametrize(
    "code",
    [
        ## TODO 45: `to_plot(` inside a block
        "if True:\n    fig,ax=plt.subplots()\n    to_plot(ax)\n",
        ## TODO 48: the line already has `if ax is None:`
        "if ax is None: fig,ax=plt.subplots()\nto_plot(ax)\n",
    ],
)
def test_to_script_saves_valid_code(code, tmp_path, monkeypatch):
    import ast

    import roux.viz.io as vio

    ## read the plot's code from a log, as in a notebook
    monkeypatch.setattr(vio, "is_interactive_notebook", lambda: True)
    logp = tmp_path / "log.py"
    logp.write_text("begin_plot()\n" + code)
    srcp = vio.to_script(
        srcp=str(tmp_path / "plot.py"), plotp="plot.png", logp=str(logp)
    )
    ast.parse(open(srcp).read())

"""For comparison related stats."""

## logging
import logging

## data
import numpy as np
import pandas as pd
## internal


def get_comparison(
    df1: pd.DataFrame,
    d1: dict = None,
    coff_p: float = 0.05,
    between_ys: bool = False,
    verbose: bool = False,
    **kws,
):
    """
    Compare the x and y columns.

    Parameters:
        df1 (pd.DataFrame): input table.
        d1 (dict): columns dict, output of `get_cols_x_for_comparison`.
        between_ys (bool): compare y's

    Notes:
        Column information:
            d1={'cols_index': ['id'],
           'cols_x': {'cont': [], 'desc': []},
           'cols_y': {'cont': [], 'desc': []}}

        Comparison types:
            1. continuous vs continuous -> correlation
            2. decrete vs continuous -> difference
            3. decrete vs decrete -> FE or chi square
    """
    # ##
    if d1 is None:
        from roux.stat.preprocess import get_cols_x_for_comparison

        d1 = get_cols_x_for_comparison(
            df1,
            **kws,
        )
        if verbose:
            logging.info(d1)

    ## gather stats in a dictionary
    if between_ys:
        for dtype in ["desc", "cont"]:
            d1["cols_x"][dtype] = list(
                np.unique(d1["cols_x"][dtype] + d1["cols_y"][dtype])
            )

    from roux.lib.set import get_alt

    d2 = {}
    ## 1. correlations
    if len(d1["cols_y"]["cont"]) != 0 and len(d1["cols_x"]["cont"]) != 0:
        from roux.stat.corr import get_corrs

        d2["correlation x vs y"] = get_corrs(
            df1=df1,
            method="spearman",
            cols=d1["cols_y"]["cont"],
            cols_with=d1["cols_x"]["cont"],
            coff_inflation_min=50,
        )

    ## 2. difference
    from roux.stat.diff import get_diff

    for k in ["x", "y"]:
        if (
            len(d1[f"cols_{k}"]["desc"]) != 0
            and len(d1[f"cols_{get_alt(['x','y'],k)}"]["cont"]) != 0
        ):
            d2[f"difference {k} vs {get_alt(['x','y'],k)}"] = get_diff(
                df1,
                cols_x=d1[f"cols_{k}"]["desc"],
                cols_y=d1[f"cols_{get_alt(['x','y'],k)}"]["cont"],
                cols_index=d1["cols_index"],
                cols=["variable x", "variable y"],
                coff_p=coff_p,
            )
        else:
            logging.warning(
                f"not len(d1[f'cols_{k}']['desc'])!=0 and len(d1[f'cols_{get_alt(['x','y'],k)}']['cont'])!=0"
            )
    ## 3. association
    if len(d1["cols_x"]["desc"]) != 0 and len(d1["cols_y"]["desc"]) != 0:
        from roux.stat.diff import compare_classes_many

        d2["association x vs y"] = compare_classes_many(
            df1=df1,
            cols_y=d1["cols_y"]["desc"],
            cols_x=d1["cols_x"]["desc"],
        )

    ## rename to uniform column names
    if "correlation x vs y" in d2:
        d2["correlation x vs y"] = (
            d2["correlation x vs y"]
            .rename(
                columns={
                    "variable1": "variable x",
                    "variable2": "variable y",
                    "$r_s$": "stat",
                },
                errors="raise",
            )
            .assign(**{"stat type": "$r_s$"})
        )
    for k in ["x vs y", "y vs x"]:
        if f"difference {k}" in d2:
            d2[f"difference {k}"] = (
                d2[f"difference {k}"]
                .rename(
                    columns={
                        "P (MWU test)": "P",
                        "stat (MWU test)": "stat",
                    },
                    errors="raise",
                )
                .assign(**{"stat type": "MWU"})
            )
    if "association x vs y" in d2:
        d2["association x vs y"] = (
            d2["association x vs y"]
            .rename(
                columns={
                    "colx": "variable x",
                    "coly": "variable y",
                },
                errors="raise",
            )
            .assign(**{"stat type": "FE/CHI2"})
        )
    if coff_p is not None:
        for k in d2:
            d2[k] = d2[k].loc[(d2[k]["P"] < coff_p), :]
    ## gather
    df2 = (
        pd.concat(
            d2,
            axis=0,
            names=["comparison type"],
        )
        .reset_index(0)
        .sort_values("P")
        .log.query(expr="`variable x` != `variable y`")
    )
    return df2


def compare_strings(
    l0: list,
    l1: list,
    cutoff: float = 0.50,
) -> pd.DataFrame:
    """
    Compare two lists of strings.

    Parameters:
        l0 (list): list of strings.
        l1 (list): list of strings to compare with.
        cutoff (float): threshold to filter the comparisons.

    Returns:
        table with the similarity scores.

    TODOs:
        1. Add option for semantic similarity.
    """
    from roux.lib.set import get_pairs
    from difflib import SequenceMatcher

    return (
        get_pairs(l0, l1)
        .add_prefix("string")
        .assign(
            **{
                "similarity": lambda df: df.apply(
                    lambda x: SequenceMatcher(None, x["string1"], x["string2"]).ratio(),
                    axis=1,
                )
            }
        )
        .log.query(expr=f"`similarity` > {cutoff}")
        .sort_values("similarity", ascending=False)
    )

"""For paired stats."""

import logging
import numpy as np
import pandas as pd
import roux.lib.dfs as rd  # noqa


# difference between values
def get_ratio_sorted(
    a: float,
    b: float,
    increase=True,
) -> float:
    """Get ratio sorted.

    Args:
        a (float): value #1.
        b (float): value #2.
        increase (bool, optional): check for increase. Defaults to True.

    Returns:
        float: output.
    """
    l = sorted([a, b])
    if not increase:
        l = l[::-1]
    if l[0] != 0 and l[1] != 0:
        return l[1] / l[0]


def diff(a: float, b: float, absolute=True) -> float:
    """Get difference

    Args:
        a (float): value #1.
        b (float): value #2.
        absolute (bool, optional): get absolute difference. Defaults to True.

    Returns:
        float: output.
    """
    diff = a - b
    if absolute:
        return abs(diff)
    else:
        return diff


def get_diff_sorted(a: float, b: float) -> float:
    """Difference sorted/absolute.

    Args:
        a (float): value #1.
        b (float): value #2.

    Returns:
        float: output.
    """
    return diff(a, b, absolute=True)


def balance(a: float, b: float, absolute=True) -> float:
    """Balance.

    Args:
        a (float): value #1.
        b (float): value #2.
        absolute (bool, optional): absolute difference. Defaults to True.

    Returns:
        float: output.
    """
    sum_ = a + b
    if sum_ != 0:
        return 1 - (diff(a, b, absolute=absolute) / (sum_))
    else:
        return np.nan


def get_paired_sets_stats(l1: list, l2: list, test: bool = False) -> list:
    """Paired stats comparing two sets.

    Args:
        l1 (list): set #1.
        l2 (list): set #2.
        test (bool): test mode. Defaults to False.

    Returns:
        list: tuple (overlap, intersection, union, ratio).
    """
    from roux.lib.set import jaccard_index

    if all([isinstance(l, list) for l in [l1, l2]]):
        if test:
            print(l1, l2)
        l = list(jaccard_index(l1, l2))
        # from roux.stat.diff import get_ratio_sorted
        l.append(get_ratio_sorted(len(l1), len(l2)))
        return l


def get_stats_paired(
    df1: pd.DataFrame,
    cols: list,
    input_logscale: bool,
    prefix: str = None,
    drop_cols: bool = False,
    unidirectional_stats: list = ["min", "max"],
    fast: bool = False,
) -> pd.DataFrame:
    """Paired stats, row-wise.

    Args:
        df1 (pd.DataFrame): input data.
        cols (list): columns.
        input_logscale (bool): if the input data is log-scaled.
        prefix (str, optional): prefix of the output column/s. Defaults to None.
        drop_cols (bool, optional): drop these columns. Defaults to False.
        unidirectional_stats (list, optional): column-wise status. Defaults to ['min','max'].
        fast (bool, optional): parallel processing. Defaults to False.

    Returns:
        pd.DataFrame: output dataframe.
    """
    assert len(cols) == 2
    if prefix is None:
        from roux.lib.str import get_fix

        prefix = get_fix(*cols, common=True, clean=True)
        logging.info(prefix)
    # from roux.stat.diff import get_ratio_sorted,get_diff_sorted
    df1[f"{prefix} {'ratio' if not input_logscale else 'diff'}"] = getattr(
        df1, "parallel_apply" if fast else "apply"
    )(
        lambda x: (get_ratio_sorted if not input_logscale else get_diff_sorted)(
            x[cols[0]], x[cols[1]]
        ),
        axis=1,
    )
    assert not any(df1[f"{prefix} {'ratio' if not input_logscale else 'diff'}"] < 0)
    for k in unidirectional_stats:
        df1[f"{prefix} {k}"] = getattr(df1.loc[:, cols], k)(axis=1)
    if drop_cols:
        df1 = df1.log.drop(labels=cols, axis=1)
    return df1


def get_stats_paired_agg(
    x: np.array, y: np.array, ignore: bool = False, verb: bool = True
) -> pd.Series:
    """Paired stats aggregated, for example, to classify 2D distributions.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.
        ignore (bool, optional): suppress warnings. Defaults to False.
        verb (bool, optional): verbose. Defaults to True.

    Returns:
        pd.Series: output.
    """
    import scipy as sc

    ## escape duplicate value inputs
    if not ignore:
        if len(x) != len(y):
            if verb:
                logging.error("len(x)!=len(y)")
            return
        if len(x) < 5:
            if verb:
                logging.error("at least 5 data points needed.")
            return
        if ((x.nunique() / len(x)) < 0.5) or ((y.nunique() / len(y)) < 0.5):
            if verb:
                logging.error("half or more duplicate values.")
            return
    d = {}
    d["n"] = len(x)
    from roux.stat.corr import get_spearmanr

    d["$r_s$"], d["P ($r_s$)"] = get_spearmanr(x, y)
    (
        d["linear regression slope"],
        d["linear regression y-intercept"],
        d["$r_p$"],
        d["P ($r_p$)"],
        d["linear regression stderr"],
    ) = sc.stats.linregress(x, y)
    #     shape of distributions
    for k, a in zip(["x", "y"], [x, y]):
        from roux.stat.cluster import cluster_1d

        d2 = cluster_1d(
            ds=a,
            n_clusters=2,
            returns=["clf"],
            clf_type="gmm",
            random_state=88,
            #                    test=True,
        )
        d[f"{k} peak1 weight"], d[f"{k} peak2 weight"] = d2["clf"].weights_.flatten()
        d[f"{k} peak1 mean"], d[f"{k} peak2 mean"] = d2["clf"].means_.flatten()
        d[f"{k} peak1 std"], d[f"{k} peak2 std"] = (
            np.sqrt(d2["clf"].covariances_).ravel().reshape(2, 1).flatten()
        )
    return pd.Series(d)


def classify_sharing(
    df1: pd.DataFrame,
    column_value: str,
    bins: list = [0, 25, 75, 100],
    labels: list = ["low", "medium", "high"],
    prefix: str = "",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Classify sharing % calculated from Jaccard index.

    Parameters:
        df1 (pd.DataFrame): input table.
        column_value (str): column with values.
        bins (list): bins. Defaults to [0,25,75,100].
        labels (list): bin labels. Defaults to ['low','medium','high'],
        prefix (str): prefix of the columns.
        verbose (bool): verbose. Defaults to False.

    """
    df1 = df1.assign(
        **{
            f"{prefix}shared (>=1)": lambda df: df[column_value] != 0,
            f"{prefix}shared (>=25%)": lambda df: (df[column_value] >= 25),
            f"{prefix}shared (>=50%)": lambda df: (df[column_value] >= 50),
            f"{prefix}shared (==100%)": lambda df: (df[column_value] == 100),
            f"{prefix}sharing type": lambda df: df[column_value].apply(
                lambda x: "same" if x == 100 else "distinct" if x == 0 else "shared"
            ),
            f"{prefix}sharing bin": lambda df: (
                pd.cut(
                    df[column_value],
                    include_lowest=True,
                    bins=bins,
                    labels=labels,
                )
                # .apply(lambda x: pd.Interval(left=int(round(x.left)), right=int(round(x.right))))
            ),
        }
    )
    if verbose:
        logging.info(df1[f"{prefix}shared (>=1)"].value_counts())
        logging.info(df1[f"{prefix}shared (>=50%)"].value_counts())
        logging.info(df1[f"{prefix}shared (==100%)"].value_counts())
        logging.info(df1[f"{prefix}sharing type"].value_counts())
        logging.info(df1[f"{prefix}sharing bin"].value_counts())
    return df1


## apply on columns

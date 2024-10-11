"""For transformations."""

import pandas as pd
import numpy as np
import logging


def plog(x, p: float, base: int):
    """Psudo-log.

    Args:
        x (float|np.array): input.
        p (float): pseudo-count.
        base (int): base of the log.

    Returns:
        output.
    """
    if base is not None:
        return np.log(x + p) / np.log(base)
    else:
        return np.log(x + p)


def anti_plog(x, p: float, base: int):
    """Anti-psudo-log.

    Args:
        x (float|np.array): input.
        p (float): pseudo-count.
        base (int): base of the log.

    Returns:
        output.
    """
    return (base**x) - p


def log_pval(
    x,
    errors: str = "raise",
    replace_zero_with: float = None,
    p_min: float = None,
):
    """Transform p-values to Log10.

    Paramters:
        x: input.
        errors (str): Defaults to 'raise' else replace (in case of visualization only).
        p_min (float): Replace zeros with this value. Note: to be used for visualization only.

    Returns:
        output.
    """
    if isinstance(x, pd.Series):
        if any(x == 0):
            if errors == "raise" and p_min is None:
                raise ValueError(f"{sum(x==0)} zeros found in x")
            else:
                logging.info(f"{sum(x==0)} zeros will be replaced")
                ## for visualisation purpose e.g. volcano plot.
                if replace_zero_with is None:
                    if p_min is None:
                        p_min = x.replace(0, np.nan).min()
                    for replace_zero_with in [0.01, 0.001, 0.0001, p_min]:
                        if p_min > replace_zero_with:
                            break
                x = x.replace(0, replace_zero_with)
                logging.warning(f"zeros found, replaced with min {replace_zero_with}")
    return -1 * (np.log10(x))


def get_q(
    ds1: pd.Series,
    col: str = None,
    verb: bool = True,
    test_coff: float = 0.1,
):
    """
    To FDR corrected P-value.
    """
    if col is not None:
        df1 = ds1.copy()
        ds1 = ds1[col]
    ds2 = ds1.dropna()
    from statsmodels.stats.multitest import fdrcorrection

    ds3 = fdrcorrection(pvals=ds2, alpha=0.05, method="indep", is_sorted=False)[1]
    ds4 = ds1.map(
        pd.DataFrame({"P": ds2, "Q": ds3}).drop_duplicates().set_index("P")["Q"]
    )
    if verb:
        from roux.stat.io import perc_label  # noqa

        logging.info(f"significant at Q<{test_coff}: {perc_label(ds4<test_coff)}")
    if col is None:
        return ds4
    else:
        df1["Q"] = ds4
        return df1


def glog(x: float, l=2):
    """Generalised logarithm.

    Args:
        x (float): input.
        l (int, optional): psudo-count. Defaults to 2.

    Returns:
        float: output.
    """
    return np.log((x + np.sqrt(x**2 + l**2)) / 2) / np.log(l)


def rescale(a: np.array, range1: tuple = None, range2: tuple = [0, 1]) -> np.array:
    """Rescale within a new range.

    Args:
        a (np.array): input vector.
        range1 (tuple, optional): existing range. Defaults to None.
        range2 (tuple, optional): new range. Defaults to [0,1].

    Returns:
        np.array: output.
    """
    if not isinstance(a, pd.Series):
        if not isinstance(a, np.ndarray):
            ## float or list
            a = np.array(a)
    if range1 is None:
        if not isinstance(a, pd.Series):
            range1 = [np.min(a), np.max(a)]
        else:
            ## faster
            range1 = [a.min(), a.max()]

    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (a - range1[0]) / delta1) + range2[0]


def rescale_divergent(
    df1: pd.DataFrame,
    col: str,
    col_sign: str = None,
    # rank=True,
) -> pd.DataFrame:
    """Rescale divergently i.e. two-sided.

    Args:
        df1 (pd.DataFrame): _description_
        col (str): column.

    Returns:
        pd.DataFrame: column.

    Notes:
        Under development.
    """

    def apply_(
        df2,
    ):
        sign = df2.name
        df2[f"{col} rescaled"] = rescale(
            df2[col], range2=[1, 0] if sign == "+" else [0, -1]
        )
        df2[f"{col} rank"] = df2[col].rank(ascending=True if sign == "+" else False) * (
            1 if sign == "+" else -1
        )
        return df2

    if col_sign is None:
        col_sign = f"{col} sign"
    return (
        df1.assign(
            **{
                col_sign: lambda df: df[col].apply(
                    lambda x: "+" if x > 0 else "-" if x < 0 else np.nan
                )
            }
        )
        .groupby([col_sign])
        .apply(lambda df: apply_(df))
    )

"""For difference related stats."""

## logging
import logging
from argparse import ArgumentError

## data
import pandas as pd
import numpy as np

## stats
import scipy as sc
import itertools

## viz
## internal
from roux.lib.set import nunique


def compare_classes(
    x,
    y,
    method=None,
):
    """
    Compare classes
    """
    if len(x) != 0 and len(y) != 0:  # and (nunique(x+y)!=1):
        df1 = pd.crosstab(x, y)
    else:
        return np.nan, np.nan
    if len(df1) == 0:
        return np.nan, np.nan
    if df1.shape != (2, 2) or method == "chi2":
        stat, pval, _, _ = sc.stats.chi2_contingency(df1)
        if method is None:
            logging.info("method=chi2_contingency")
    elif df1.shape == (2, 2):
        stat, pval = sc.stats.fisher_exact(df1)
        if method is None:
            logging.info("method=fisher_exact")
    else:
        raise ValueError(df1)
    return stat, pval


def compare_classes_many(
    df1: pd.DataFrame,
    cols_y: list,
    cols_x: list,
) -> pd.DataFrame:
    df0 = pd.DataFrame(
        itertools.product(
            cols_y,
            cols_x,
        )
    ).rename(columns={0: "colx", 1: "coly"}, errors="raise")
    # df0.head(1)
    return df0.join(
        df0.apply(lambda x: compare_classes(df1[x["colx"]], df1[x["coly"]]), axis=1)
        .apply(pd.Series)
        .rename(columns={0: "stat", 1: "P"}, errors="raise")
    )


def get_pval(
    df: pd.DataFrame,
    colvalue="value",
    colsubset="subset",
    colvalue_bool=False,
    colindex=None,
    subsets=None,
    test=False,
    func=None,
) -> tuple:
    """Get p-value.

    Args:
        df (DataFrame): input dataframe.
        colvalue (str, optional): column with values. Defaults to 'value'.
        colsubset (str, optional): column with subsets. Defaults to 'subset'.
        colvalue_bool (bool, optional): column with boolean values. Defaults to False.
        colindex (str, optional): column with the index. Defaults to None.
        subsets (list, optional): subset types. Defaults to None.
        test (bool, optional): test. Defaults to False.
        func (function, optional): function. Defaults to None.

    Raises:
        ArgumentError: colvalue or colsubset not found in df.
        ValueError: need only 2 subsets.

    Returns:
        tuple: stat,p-value
    """
    if not ((colvalue in df) and (colsubset in df)):
        raise ArgumentError(
            f"colvalue or colsubset not found in df: {colvalue} or {colsubset}"
        )
    if subsets is None:
        subsets = sorted(df[colsubset].unique())
    if len(subsets) != 2:
        raise ValueError("need only 2 subsets")
        return
    else:
        df = df.loc[df[colsubset].isin(subsets), :]
    if colvalue_bool and not df[colvalue].dtype == bool:
        logging.warning(f"colvalue_bool {colvalue} is not bool")
        return
    if not colvalue_bool:
        #         try:
        x, y = (
            df.loc[(df[colsubset] == subsets[0]), colvalue].tolist(),
            df.loc[(df[colsubset] == subsets[1]), colvalue].tolist(),
        )
        if len(x) != 0 and len(y) != 0 and (nunique(x + y) != 1):
            if func is None:
                if test:
                    logging.warning("mannwhitneyu used")
                return sc.stats.mannwhitneyu(
                    x,
                    y,
                    alternative="two-sided",
                )
            else:
                if test:
                    logging.warning(f"custom function used: {str(func)}")
                return func(
                    # df.loc[(df[colsubset] == subsets[0]), colvalue],
                    # df.loc[(df[colsubset] == subsets[1]), colvalue],
                    x,
                    y,
                )
        else:
            # if empty list: RuntimeWarning: divide by zero encountered in double_scalars  z = (bigu - meanrank) / sd
            return np.nan, np.nan
    else:
        assert colindex is not None
        df1 = df.pivot(index=colindex, columns=colsubset, values=colvalue)
        return compare_classes(df1[subsets[0]], df1[subsets[1]], method=None)
        # ct=pd.crosstab(df1[subsets[0]],df1[subsets[1]])
        # if ct.shape==(2,2):
        #     ct=ct.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False)
        #     if test:
        #         print(ct)
        #     return sc.stats.fisher_exact(ct)
        # else:
        #     return np.nan,np.nan


def get_stat(
    df1: pd.DataFrame,
    colsubset: str,
    colvalue: str,
    colindex: str,
    subsets=None,
    cols_subsets=["subset1", "subset2"],
    df2=None,
    stats=["mean", "median", "var", "size"],
    coff_samples_min=None,
    verb=False,
    func=None,
    **kws,
) -> pd.DataFrame:
    """Get statistics.

    Args:
        df1 (DataFrame): input dataframe.
        colvalue (str, optional): column with values. Defaults to 'value'.
        colsubset (str, optional): column with subsets. Defaults to 'subset'.
        colindex (str, optional): column with the index. Defaults to None.
        subsets (list, optional): subset types. Defaults to None.
        cols_subsets (list, optional): columns with subsets. Defaults to ['subset1', 'subset2'].
        df2 (DataFrame, optional): second dataframe. Defaults to None.
        stats (list, optional): summary statistics. Defaults to [np.mean,np.median,np.var]+[len].
        coff_samples_min (int, optional): minimum sample size required. Defaults to None.
        verb (bool, optional): verbose. Defaults to False.

    Keyword Arguments:
        kws: parameters provided to `get_pval` function.

    Raises:
        ArgumentError: colvalue or colsubset not found in df.
        ValueError: len(subsets)<2

    Returns:
        DataFrame: output dataframe.

    TODOs:
        1. Rename to more specific `get_diff`, also other `get_stat*`/`get_pval*` functions.
    """
    if not ((colvalue in df1) and (colsubset in df1)):
        raise ArgumentError(
            f"colvalue or colsubset not found in df: {colvalue} or {colsubset}"
        )
        return
    if subsets is None:
        subsets = sorted(df1[colsubset].unique())
    if len(subsets) < 2:
        raise ValueError("len(subsets)<2")
        return
    colvalue_bool = df1[colvalue].dtype == bool
    if df2 is None:
        import itertools

        df2 = pd.DataFrame(
            [t for t in list(itertools.permutations(subsets, 2))]
            if len(subsets) > 2
            else [subsets]
        )
        df2.columns = cols_subsets
    df2 = (
        df2.groupby(cols_subsets)
        .apply(
            lambda df: get_pval(
                df1,
                colvalue=colvalue,
                colsubset=colsubset,
                colindex=colindex,
                subsets=df.name,
                colvalue_bool=colvalue_bool,
                func=func,
                **kws,
            )
        )
        .apply(pd.Series)
    )
    df2 = df2.rename(
        columns={
            0: "stat"+(f" ({'MWU' if not colvalue_bool else 'FE'} test)" if func is None else ""),
            1: "P"+(f" ({'MWU' if not colvalue_bool else 'FE'} test)" if func is None else ""),
        },
    ).reset_index()
    from roux.lib.dfs import merge_paired
    from roux.lib.str import get_prefix, get_suffix

    colsubset_ = get_prefix(*cols_subsets, common=True, clean=True)
    df_ = (
        df1.groupby([colsubset])[colvalue]
        .agg(stats if not colvalue_bool else ["sum", "len"])
        .reset_index()
        # TODOs rename to subset1 subset2
        .rename(columns={colsubset: colsubset_}, errors="raise")
    )
    df3 = merge_paired(
        df1=df2,
        df2=df_,
        left_ons=cols_subsets,
        right_on=colsubset_,
        common=[],
        right_ons_common=[],
        how="inner",
        validates=["m:1", "m:1"],
        suffixes=get_suffix(*cols_subsets, common=False, clean=False),
        test=False,
        verb=False,
        # **kws,
    )
    df3 = df3.rename(
        columns={
            f"{c}{i}": f"{c} {colsubset_}{i}"
            for c in df_
            for i in [1, 2]
            if c != colsubset_
        },
        errors="raise",
    )
    ## minimum samples
    if coff_samples_min is not None:
        # logging.info("coff_samples_min applied")
        df3.loc[
            (
                (df3[f"len {cols_subsets[0]}"] < coff_samples_min)
                | (df3[f"len {cols_subsets[1]}"] < coff_samples_min)
            ),
            df3.filter(like="P (").columns.tolist(),
        ] = np.nan
    return df3


def get_stats(
    df1: pd.DataFrame,
    colsubset: str,
    cols_value: list,
    colindex: str,
    subsets=None,
    df2=None,
    cols_subsets=["subset1", "subset2"],
    stats=["mean", "median", "var", "size"],
    axis=0,  # concat
    test=False,
    **kws,
) -> pd.DataFrame:
    """Get statistics by iterating over columns wuth values.

    Args:
        df1 (DataFrame): input dataframe.
        colsubset (str, optional): column with subsets.
        cols_value (list): list of columns with values.
        colindex (str, optional): column with the index.
        subsets (list, optional): subset types. Defaults to None.
        df2 (DataFrame, optional): second dataframe, e.g. `pd.DataFrame({"subset1":['test'],"subset2":['reference']})`. Defaults to None.
        cols_subsets (list, optional): columns with subsets. Defaults to ['subset1', 'subset2'].
        stats (list, optional): summary statistics. Defaults to [np.mean,np.median,np.var]+[len].
        axis (int, optional): 1 if different tests else use 0. Defaults to 0.

    Keyword Arguments:
        kws: parameters provided to `get_pval` function.

    Raises:
        ArgumentError: colvalue or colsubset not found in df.
        ValueError: len(subsets)<2

    Returns:
        DataFrame: output dataframe.

    TODOs:
        1. No column prefix if `len(cols_value)==1`.

    """
    dn2df = {}
    for colvalue in cols_value:
        df1_ = df1.dropna(subset=[colsubset, colvalue])
        if subsets is None:
            subsets = sorted(df1[colsubset].unique())
        if len(df1_[colsubset].unique()) > 1:
            dn2df[colvalue] = get_stat(
                df1_,
                colsubset=colsubset,
                colvalue=colvalue,
                colindex=colindex,
                subsets=subsets,
                cols_subsets=cols_subsets,
                df2=df2,
                stats=stats,
                **kws,
            ).set_index(cols_subsets)
        else:
            if test:
                logging.warning(
                    f"not processed: {colvalue}; probably because of dropna"
                )
    if len(dn2df.keys()) == 0:
        return
    import pandas as pd  # remove?

    df3 = pd.concat(
        dn2df,
        ignore_index=False,
        axis=axis,
        verify_integrity=True,
        names=None if axis == 1 else ["variable"],
    )
    if axis == 1:
        df3 = df3.reset_index().rd.flatten_columns()
    return df3


def get_significant_changes(
    df1: pd.DataFrame,
    coff_p=0.025,
    coff_q=0.1,
    alpha=None,
    change_type=["diff", "ratio"],
    changeby="mean",
    # fdr=True,
    value_aggs=["mean", "median"],
) -> pd.DataFrame:
    """Get significant changes.

    Args:
        df1 (DataFrame): input dataframe.
        coff_p (float, optional): cutoff on p-value. Defaults to 0.025.
        coff_q (float, optional): cutoff on q-value. Defaults to 0.1.
        alpha (float, optional): alias for `coff_p`. Defaults to None.
        changeby (str, optional): "" if check for change by both mean and median. Defaults to "".
        value_aggs (list, optional): values to aggregate. Defaults to ['mean','median'].

    Returns:
        DataFrame: output dataframe.
    """
    if coff_p is None and alpha is not None:
        coff_p = alpha
    logging.info(changeby)
    if df1.filter(regex="|".join([f"{s} subset(1|2)" for s in value_aggs])).shape[1]:
        for s in value_aggs:
            if "diff" in change_type:
                df1[f"difference between {s} (subset1-subset2)"] = (
                    df1[f"{s} subset1"] - df1[f"{s} subset2"]
                )
                df1.loc[
                    (df1[f"difference between {changeby} (subset1-subset2)"] > 0),
                    "change",
                ] = "increase"
                df1.loc[
                    (df1[f"difference between {changeby} (subset1-subset2)"] < 0),
                    "change",
                ] = "decrease"
            if "ratio" in change_type:
                df1[f"ratio between {s} (subset1-subset2)"] = (
                    df1[f"{s} subset1"] / df1[f"{s} subset2"]
                )
                # df1.loc[(df1[f'ratio between {changeby} (subset1-subset2)']>1),'change']='increase'
                # df1.loc[(df1[f'ratio between {changeby} (subset1-subset2)']<1),'change']='decrease'
        df1["change"] = df1["change"].fillna("ns")
    stat_suffixs=[c.split('P')[1] if c.startswith('P ') else "" for c in df1.filter(regex="^P.*").columns]
    # for stat in ["MWU", "FE"]:
    #     stat_suffix=f" ({stat} test)"
    for stat_suffix in stat_suffixs:
        if "P"+stat_suffix not in df1:
            continue
        # without fdr
        df1[f"change is significant, P{stat_suffix} < {coff_p}"] = (
            df1["P"+stat_suffix] < coff_p
        )
        if coff_q is not None:
            from roux.stat.transform import get_q

            df1["Q"+stat_suffix] = get_q(df1["P"+stat_suffix])
            # df1[f'change is significant, Q{stat_suffix} < {coff_q}']=df1[f'Q{stat_suffix}']<coff_q
            #     info(f"corrected alpha alphacSidak={alphacSidak},alphacBonf={alphacBonf}")
            # if test!='FE':
            df1[f"significant change, Q{stat_suffix} < {coff_q}"] = df1.apply(
                lambda x: x["change"] if x["Q"+stat_suffix] < coff_q else "ns",
                axis=1,
            )
            # df1.loc[df1[f'change is significant, Q{stat_suffix} < {coff_q}'],f"significant change, Q{stat_suffix} < {coff_q}"]=df1.loc[df1[f"change is significant, Q{stat_suffix} < {coff_q}"],'change']
            # df1[f"significant change, Q{stat_suffix} < {coff_q}"]=df1[f"significant change, Q{stat_suffix} < {coff_q}"].fillna('ns')
    return df1


def apply_get_significant_changes(
    df1: pd.DataFrame,
    cols_value: list,
    cols_groupby: list,  # e.g. genes id
    cols_grouped: list,  # e.g. tissue
    fast=False,
    **kws,
) -> pd.DataFrame:
    """Apply on dataframe to get significant changes.

    Args:
        df1 (DataFrame): input dataframe.
        cols_value (list): columns with values.
        cols_groupby (list): columns with groups.

    Returns:
        DataFrame: output dataframe.
    """
    d1 = {}
    from tqdm import tqdm

    for c in tqdm(cols_value):
        df1_ = (
            df1.set_index(cols_groupby)
            .filter(regex=f"^{c} .*")
            .filter(
                regex="^(?!(" + " |".join([s for s in cols_value if c != s]) + ")).*"
            )
        )
        df1_ = df1_.rd.renameby_replace({f"{c} ": ""}).reset_index()
        d1[c] = getattr(
            df1_.groupby(cols_groupby), "apply" if not fast else "parallel_apply"
        )(lambda df: get_significant_changes(df, **kws))
    d1 = {k: d1[k].set_index(cols_groupby) for k in d1}
    ## to attach the info
    d1["grouped"] = df1.set_index(cols_groupby).loc[:, cols_grouped]
    df2 = pd.concat(
        d1,
        ignore_index=False,
        axis=1,
        verify_integrity=True,
    )
    df2 = df2.rd.flatten_columns().reset_index()
    assert not df2.columns.duplicated().any()
    return df2


def get_stats_groupby(
    df1: pd.DataFrame,
    cols_group: list,
    coff_p: float = 0.05,
    coff_q: float = 0.1,
    alpha=None,
    fast=False,
    **kws,
) -> pd.DataFrame:
    """Iterate over groups, to get the differences.

    Args:
        df1 (DataFrame): input dataframe.
        cols_group (list): columns to interate over.
        coff_p (float, optional): cutoff on p-value. Defaults to 0.025.
        coff_q (float, optional): cutoff on q-value. Defaults to 0.1.
        alpha (float, optional): alias for `coff_p`. Defaults to None.
        fast (bool, optional): parallel processing. Defaults to False.

    Returns:
        DataFrame: output dataframe.
    """
    df2 = (
        getattr(
            df1.groupby(cols_group),
            f"{'progress_' if not fast and hasattr(df1.groupby(cols_group),'progress_apply') else '' if not fast else 'parallel_'}apply",
        )(
            lambda df: get_stats(
                df1=df,
                **kws,
            )
        )
        .reset_index()
        .rd.clean()
    )
    return get_significant_changes(
        df1=df2,
        alpha=alpha,
        coff_p=coff_p,
        coff_q=coff_q,
    )


def get_diff(
    df1: pd.DataFrame,
    cols_x: list,
    cols_y: list,
    cols_index: list,
    cols_group: list,
    coff_p: float = None,
    test: bool = False,
    func=None,
    **kws,
) -> pd.DataFrame:
    """
    Wrapper around the `get_stats_groupby`

    Keyword parameters:
          cols=['variable x','variable y'],
          coff_p=0.05,
          coff_q=0.01,
          colindex=['id'],
    """
    ## melt the table to make it linear
    d_ = {}
    for colx in cols_x:
        assert (
            df1[colx].nunique() == 2
        ), f"df1[{colx}].nunique() = {df1[colx].nunique()}"
        d_[colx] = df1.melt(
            id_vars=cols_index + [colx] + cols_group,
            value_vars=cols_y,
            var_name="variable y",
            value_name="value y",
        ).rename(columns={colx: "value x"}, errors="raise")
    df2 = (
        pd.concat(
            d_,
            names=["variable x"],
        )
        .reset_index()
        .rd.clean()
        .log.dropna()
    )
    if test:
        logging.info(df2.iloc[0, :])
    ## calculate the differences
    df3 = get_stats_groupby(
        df1=df2,
        colsubset="value x",
        cols_value=["value y"],
        colindex=cols_index,
        cols_group=cols_group,
        func=func,
        **kws,
    )
    if coff_p is not None:
        df3 = df3.loc[(df3["P"+(" (MWU test)" if func is None else "")] < coff_p), :]
    else:
        logging.warning("not filtered by P-value cutoff")
    return df3.sort_values("P"+(" (MWU test)" if func is None else ""))


def binby_pvalue_coffs(
    df1: pd.DataFrame,
    coffs=[0.01, 0.05, 0.1],
    color=False,
    testn="MWU test, FDR corrected",
    colindex="genes id",
    colgroup="tissue",
    preffix="",
    colns=None,  # plot as ns, not counted
    palette=None,  # ['#f55f5f','#ababab','#046C9A',],
) -> tuple:
    """Bin data by pvalue cutoffs.

    Args:
        df1 (DataFrame): input dataframe.
        coffs (list, optional): cut-offs. Defaults to [0.01,0.05,0.25].
        color (bool, optional): color asignment. Defaults to False.
        testn (str, optional): test number. Defaults to 'MWU test, FDR corrected'.
        colindex (str, optional): column with index. Defaults to 'genes id'.
        colgroup (str, optional): column with the groups. Defaults to 'tissue'.
        preffix (str, optional): prefix. Defaults to ''.
        colns (_type_, optional): columns number. Defaults to None.
        notcountedpalette (_type_, optional): _description_. Defaults to None.

    Returns:
        tuple: output.

    Notes:
        1. To be deprecated in the favor of the functions used for enrichment analysis for example.
    """
    assert len(df1) != 0
    if palette is None:
        from roux.viz.colors import get_colors_default

        palette = get_colors_default()[:3]
    assert len(palette) == 3
    coffs = np.array(sorted(coffs))
    # df1[f'{preffix}P (MWU test, FDR corrected) bin']=pd.cut(x=df1[f'{preffix}P (MWU test, FDR corrected)'],
    #       bins=[0]+coffs+[1],
    #        labels=coffs+[1],
    #        right=False,
    #       ).fillna(1)
    from roux.viz.colors import saturate_color

    d1 = {}
    for i, coff in enumerate(coffs[::-1]):
        col = f"{preffix}significant change, P ({testn}) < {coff}"
        df1[col] = df1.apply(
            lambda x: "increase"
            if (
                (x[f"{preffix}P ({testn})"] < coff)
                and (x[f"{preffix}difference between mean (subset1-subset2)"] > 0)
            )
            else "decrease"
            if (
                (x[f"{preffix}P ({testn})"] < coff)
                and (x[f"{preffix}difference between mean (subset1-subset2)"] < 0)
            )
            else "ns",
            axis=1,
        )
        if color:
            if i == 0:
                df1.loc[(df1[col] == "ns"), "c"] = palette[1]
            saturate = 1 - ((len(coffs) - (i + 1)) / len(coffs))
            d2 = {}
            d2["increase"] = saturate_color(palette[0], saturate)
            d2["decrease"] = saturate_color(palette[2], saturate)
            d1[coff] = d2
            df1["c"] = df1.apply(
                lambda x: d2[x[col]] if x[col] in d2 else x["c"], axis=1
            )
            assert df1["c"].isnull().sum() == 0
    if color:
        import itertools
        from roux.stat.transform import rescale, log_pval

        d3 = {}
        for i, (k, coff) in enumerate(
            list(itertools.product(["increase", "decrease"], coffs))
        ):
            col = f"{preffix}significant change, P ({testn}) < {coff}"
            d4 = {}
            d4["y alpha"] = rescale(
                1 - (list(coffs).index(coff)) / len(coffs), [0, 1], [0.5, 1]
            )
            d4["y"] = log_pval(coff)
            d4["y text"] = f" P < {coff}"
            d4["x"] = (
                df1.loc[
                    (df1[col] == k),
                    f"{preffix}difference between mean (subset1-subset2)",
                ].min()
                if k == "increase"
                else df1.loc[
                    (df1[col] == k),
                    f"{preffix}difference between mean (subset1-subset2)",
                ].max()
            )
            d4["change"] = k
            d4["text"] = (
                f"{df1.loc[(df1[col]==k),colindex].nunique()}/{df1.loc[(df1[col]==k),colgroup].nunique()}"
            )
            d4["color"] = d1[coff][k]
            d3[i] = d4
        df2 = pd.DataFrame(d3).T
    if colns is not None:
        df1.loc[df1[colns], "c"] = palette[1]
    #     info(df1.shape,df1.shape)
    return df1, df2

"""For processing multiple pandas DataFrames/Series"""

import logging

import numpy as np
import pandas as pd

import roux.lib.df as rd #noqa
from roux.lib import to_rd

## ids
@to_rd
def make_ids(
    df: pd.DataFrame,
    cols: list,
    ids_have_equal_length: bool,
    sep: str = "--",
    sort: bool = False,
) -> pd.Series:
    """Make ids by joining string ids in more than one columns.

    Parameters:
        df (DataFrame): input dataframe.
        cols (list): columns.
        ids_have_equal_length (bool): ids have equal length, if True faster processing.
        sep (str): separator between the ids ('--').
        sort (bool): sort the ids before joining (False).

    Returns:
        ds (Series): output series.
    """

    def get_ids(x):
        return "--".join(x)

    def get_ids_sorted(x):
        return "--".join(sorted(x))

    if ids_have_equal_length:
        logging.debug(
            "the ids should be of equal character length and should not contain non-alphanumeric characters e.g. '.'"
        )
        return np.apply_along_axis(
            get_ids if not sort else get_ids_sorted, 1, df.loc[:, cols].values
        )
    else:
        return df.loc[:, cols].agg(
            lambda x: sep.join(x if not sort else sorted(x)), axis=1
        )


@to_rd
def make_ids_sorted(
    df: pd.DataFrame,
    cols: list,
    ids_have_equal_length: bool,
    sep: str = "--",
    sort: bool = False,
) -> pd.Series:
    """Make sorted ids by joining string ids in more than one columns.

    Parameters:
        df (DataFrame): input dataframe.
        cols (list): columns.
        ids_have_equal_length (bool): ids have equal length, if True faster processing.
        sep (str): separator between the ids ('--').

    Returns:
        ds (Series): output series.
    """
    return make_ids(df, cols, ids_have_equal_length, sep=sep, sort=True)


def get_alt_id(
    s1: str,
    s2: str,
    sep: str = "--",
):
    """Get alternate/partner id from a paired id.

    Parameters:
        s1 (str): joined id.
        s2 (str): query id.

    Returns:
        s (str): partner id.
    """
    return [s for s in s1.split(sep) if s != s2][0]


@to_rd
def split_ids(df1, col, sep="--", prefix=None):
    """Split joined ids to individual ones.

    Parameters:
        df1 (DataFrame): input dataframe.
        col (str): column containing the joined ids.
        sep (str): separator within the joined ids ('--').
        prefix (str): prefix of the individual ids (None).

    Return:
        df1 (DataFrame): output dataframe.
    """
    # assert not df1._is_view, "input series should be a copy not a view"
    df = df1[col].str.split(sep, expand=True)
    for i in range(len(df.columns)):
        df1[f"{col} {i+1}"] = df[i].copy()
    if prefix is not None:
        df1 = df1.rd.renameby_replace(replaces={f"{col} ": prefix})
    return df1

def filter_dfs(
    dfs: list,
    cols: list,
    how: str = "inner",
) -> pd.DataFrame:
    """Filter dataframes based items in the common columns.

    Parameters:
        dfs (list): list of dataframes.
        cols (list): list of columns.
        how (str): how to filter ('inner')

    Returns
        dfs (list): list of dataframes.
    """

    def apply_(dfs, col, how):
        from roux.lib.set import list2intersection, list2union

        if how == "inner":
            l = list(list2intersection([df[col].tolist() for df in dfs]))  # noqa
        elif how == "outer":
            l = list(list2union([df[col].tolist() for df in dfs]))  # noqa
        else:
            raise ValueError("how")
        logging.info(f"len({col})={len(l)}")
        return [df.loc[(df[col].isin(l)), :] for df in dfs]

    if isinstance(cols, str):
        cols = [cols]
    # sort columns by nunique
    cols = dfs[0].loc[:, cols].nunique().sort_values().index.tolist()
    for c in cols:
        dfs = apply_(dfs=dfs, col=c, how=how)
    return dfs


def merge_with_many_columns(
    df1: pd.DataFrame,
    right: str,
    left_on: str,
    right_ons: list,
    right_id: str,
    how: str = "inner",
    validate: str = "1:1",
    test: bool = False,
    verbose: bool = False,
    **kws_merge,
) -> pd.DataFrame:
    """
    Merge with many columns.
    For example, if ids in the left table can map to ids located in multiple columns of the right table.

    Parameters:
        df1 (pd.DataFrame): left table.
        right (pd.DataFrame): right table.
        left_on (str): column in the left table to merge on.
        right_ons (list): columns in the right table to merge on.
        right_id (str): column in the right dataframe with for example the ids to be merged.

    Keyword parameters:
        kws_merge: to be supplied to `pandas.DataFrame.merge`.

    Returns:
        Merged table.
    """
    if test:
        verbose = True
    ## melt the right-side df
    df2 = right.melt(
        id_vars=right_id,
        value_vars=right_ons,
        value_name=left_on,
    ).log.dropna()
    if verbose:
        logging.info(df2.head(1))

    def log_overlap(df2):
        ## overlap per column
        df3 = df2.assign(
            **{
                "overlap": lambda df: df.groupby("variable")[left_on].transform(
                    lambda x: len(set(x.tolist()) & set(df1[left_on].tolist()))
                ),
            }
        ).sort_values("overlap", ascending=False)
        logging.info(
            df3.loc[:, ["variable", "overlap"]]
            .drop_duplicates()
            .set_index("variable")["overlap"]
        )

    log_overlap(df2)

    ## unique ids
    df3 = df2.log.drop_duplicates(subset=left_on, keep="first")
    if verbose:
        logging.info(df3.head(1))
    log_overlap(df3)
    return df1.log.merge(
        right=df3.drop(["variable"], axis=1),
        how=how,
        on=left_on,
        validate=validate,
        **kws_merge,
    )


@to_rd
def merge_paired(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    left_ons: list,  # suffixed
    right_on: list,  # to be suffixed
    common: list = [],  # not suffixed
    right_ons_common: list = [],  # not to be suffixed
    how: str = "inner",
    validates: list = ["1:1", "1:1"],
    suffixes: list = None,
    test: bool = False,
    verb: bool = True,
    **kws,
) -> pd.DataFrame:
    """Merge uppaired dataframes to a paired dataframe.

    Parameters:
        df1 (DataFrame): paired dataframe.
        df2 (DataFrame): unpaired dataframe.
        left_ons (list): columns of the `df1` (suffixed).
        right_on (str|list): column/s of the `df2` (to be suffixed).
        common (str|list): common column/s between `df1` and `df2` (not suffixed).
        right_ons_common (str|list): common column/s between `df2` to be used for merging (not to be suffixed).
        how (str): method of merging ('inner').
        validates (list): validate mappings for the 1st mapping between `df1` and `df2` and 2nd one between `df1+df2` and `df2` (['1:1','1:1']).
        suffixes (list): suffixes to be used (None).
        test (bool): testing (False).
        verb (bool): verbose (True).

    Keyword Parameters:
        kws (dict): parameters provided to `merge`.

    Returns:
        df (DataFrame): output dataframe.

    Examples:
        Parameters:
            how='inner',
            left_ons=['gene id gene1','gene id gene2'], # suffixed
            common='sample id', # not suffixed
            right_on='gene id', # to be suffixed
            right_ons_common=[], # not to be suffixed
    """
    if isinstance(right_on, str):
        right_on = [right_on]
    if isinstance(right_ons_common, str):
        right_ons_common = [right_ons_common]
    if isinstance(common, str):
        common = [common]
    if how != "inner":
        logging.warning(
            f"how!='inner' ('{how}'), other types of merging are under development mode."
        )

    if isinstance(left_ons[0], list):
        logging.error(
            "Merge `on` lists not supported. Suggestion: Use groupby on one of the `on` columns of the left dataframe."
        )
        return
    if suffixes is None:
        from roux.lib.str import get_suffix

        suffixes = get_suffix(*left_ons, common=False, clean=True)
        suffixes = [f" {s}" for s in suffixes]
    d1 = {}
    d1["from"] = df1.shape

    def apply_(df2, cols_on, suffix, test=False):
        if len(cols_on) != 0:
            df2 = df2.set_index(cols_on)
        df2 = df2.add_suffix(suffix)
        if len(cols_on) != 0:
            df2 = df2.reset_index()
        if test:
            print(df2.columns)
        return df2

    df3 = df1.copy()
    for i, (suffix, validate) in enumerate(zip(suffixes, validates)):
        if test:
            print(df3.columns.tolist())
            print([f"{s}{suffix}" for s in right_on] + right_ons_common)
        df3 = df3.merge(
            right=apply_(df2, common + right_ons_common, suffix, test=test),
            on=[f"{s}{suffix}" for s in right_on]
            + common
            + (right_ons_common if i == 1 else []),
            how=how,
            validate=validate,
            **kws,
        )
    d1["to"] = df3.shape
    from roux.lib.df import log_shape_change

    if verb:
        log_shape_change(d1)
    return df3


## append


## merge dfs
def merge_dfs(
    dfs,
    force_suffixes=False,
    **kws,
) -> pd.DataFrame:
    """Merge dataframes from left to right.

    Parameters:
        dfs (list): list of dataframes.

    Keyword Parameters:
        kws (dict): parameters provided to `merge`.

    Returns:
        df (DataFrame): output dataframe.

    Notes:
        For example, reduce(lambda x, y: x.merge(y), [1, 2, 3, 4, 5]) merges ((((1.merge(2)).merge(3)).merge(4)).merge(5)).
    """
    if kws["how"] != "outer":
        logging.warning(
            "Drop-outs may occur if on!='outer'. Make sure that the dataframes are ordered properly from left to right."
        )

    if isinstance(dfs,list):
        dfs=[(f" {i+1}",df) for i,df in enumerate(dfs)]
    elif isinstance(dfs,dict):
        dfs=[(f" {k}",v) for k,v in dfs.items()]
        
    logging.info(
        f"dfs shape={[df.shape for k,df in dfs]}"
    )
    for i,(k2,df2) in enumerate(dfs):
        if i==0:
            df3=df2.copy()
            if force_suffixes:
                df3=(
                    df3
                    .set_index(kws['on'])
                    .add_suffix(k2)
                    .reset_index()
                ) 
        else:
            df3=(
                df3
                .log.merge(
                    right=df2,
                    **{
                        **kws,
                        **dict(suffixes=[k1,k2]),
                      },
                )
            )
        k1=k2        
    return df3


def compare_rows(
    df1,
    df2,
    test=False,
    **kws,
):
    cols = list(set(df1.columns.tolist()) & set(df1.columns.tolist()))
    if test:
        logging.info(cols)
    cols_sort = list(set(df1.select_dtypes(object).columns.tolist()) & set(cols))
    if test:
        logging.info(cols_sort)
    if len(df1) == len(df2):
        return (
            df1.loc[:, cols]
            .sort_values(cols_sort)
            .reset_index(drop=True)
            .compare(
                df2.loc[:, cols].sort_values(cols_sort).reset_index(drop=True),
                keep_equal=True,
                **kws,
            )
        )  # .rd.assert_no_na()
    else:
        logging.warning(f"unequal lengths: {len(df1)}!={len(df2)}")
        df_ = df1.loc[:, cols].merge(
            right=df1.loc[:, cols], on=cols_sort, how="outer", indicator=True
        )
        logging.info(df_["_merge"].value_counts())
        return df_.loc[(df_["_merge"] != "both"), :]

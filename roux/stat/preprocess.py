"""For classification."""

## logging
import logging

## data
import numpy as np
import pandas as pd

## internal
from roux.stat.corr import check_collinearity
from roux.stat.variance import get_variance_inflation

## attach functions as attributes of dataframes
from roux.lib import to_rd


# matrix data
@to_rd
def dropna_matrix(
    df1,
    coff_cols_min_perc_na=5,
    coff_rows_min_perc_na=5,
    test=False,
    verbose=False,
):
    if test:
        verbose = True
    assert df1.columns.name is not None
    assert df1.index.name is not None

    d1 = {
        df1.columns.name: df1.rd.check_na(perc=True).sort_values(ascending=False),
        df1.index.name: df1.T.rd.check_na(perc=True).sort_values(ascending=False),
    }

    if test:
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=[5, 2.5])
        for k, ax in zip(d1, axs):
            d1[k].hist(ax=ax).set(title=k)

    d2 = {k: len(v) for k, v in d1.items()}
    if verbose:
        logging.info(d2)
    l1 = sorted(d2, key=d2.get, reverse=False)
    if verbose:
        logging.info(l1)

    ## dropna axis
    d3 = {
        df1.columns.name: 1,
        df1.index.name: 0,
    }
    ## cutoffs
    d4 = {
        df1.columns.name: coff_cols_min_perc_na,
        df1.index.name: coff_rows_min_perc_na,
    }
    if verbose:
        logging.info(d3)

    if verbose:
        for k in l1:
            from roux.stat.binary import perc

            logging.info(f"min. percent of '{k}'s to be dropped={perc(d1[k]>=d4[k])}")

    return (
        df1.log.drop(
            labels=d1[l1[0]][d1[l1[0]] >= d4[l1[0]]].index.tolist(), axis=d3[l1[0]]
        )
        .log.drop(
            labels=d1[l1[1]][d1[l1[1]] >= d4[l1[1]]].index.tolist(), axis=d3[l1[1]]
        )
        .log.dropna(axis=d3[l1[0]])
        .log.dropna(axis=d3[l1[1]])
        .rd.assert_no_na()
    )


# curate data
def drop_low_complexity(
    df1: pd.DataFrame,
    min_nunique: int,
    max_inflation: int,
    max_nunique: int = None,
    cols: list = None,
    cols_keep: list = [],
    test: bool = False,
    verbose: bool = False,
) -> pd.DataFrame:
    """Remove low-complexity columns from the data.

    Args:
        df1 (pd.DataFrame): input data.
        min_nunique (int): minimum unique values.
        max_inflation (int): maximum over-representation of the values.
        cols (list, optional): columns. Defaults to None.
        cols_keep (list, optional): columns to keep. Defaults to [].
        test (bool, optional): test mode. Defaults to False.

    Returns:
        pd.DataFrame: output data.
    """

    if cols is None:
        cols = df1.columns.tolist()
    if len(cols) < 2:
        logging.warning("skipped `drop_low_complexity` because len(cols)<2.")
        return df1
    df_ = pd.concat(
        [df1.rd.check_nunique(cols), df1.rd.check_inflation(cols)],
        axis=1,
    )
    df_.columns = ["nunique", "% inflation"]
    if verbose:
        logging.info(
            df_.sort_values(
                ["nunique", "% inflation"],
                ascending=[True, False],
            )
        )
    df_ = df_.sort_values(df_.columns.tolist(), ascending=False)
    df1_ = df_.loc[
        ((df_["nunique"] < min_nunique) | (df_["% inflation"] >= max_inflation)), :
    ]
    l1 = df1_.index.tolist()
    if len(l1) != 0:
        logging.info(df1_)
    if max_nunique is not None:
        df2_ = df_.loc[(df_["nunique"] > max_nunique), :]
        l1 += df2_.index.tolist()
        logging.info(df2_)

    #     def apply_(x,df1,min_nunique,max_inflation):
    #         ds1=x.value_counts()
    #         return (len(ds1)<=min_nunique) or ((ds1.values[0]/len(df1))>=max_inflation)
    #     l1=df1.loc[:,cols].apply(lambda x: apply_(x,df1,min_nunique=min_nunique,max_inflation=max_inflation)).loc[lambda x: x].index.tolist()
    logging.info(
        f"{len(l1)}(/{len(cols)}) columns {'could be ' if test else ''}dropped:"
    )
    if l1 == 0:
        return df1
    else:
        if len(cols_keep) != 0:
            assert all([c in df1 for c in cols_keep]), [
                c for c in cols_keep if c not in df1
            ]
            cols_kept = [c for c in l1 if c in cols_keep]
            logging.info(cols_kept)
            l1 = [c for c in l1 if c not in cols_keep]

        return df1.log.drop(labels=l1, axis=1)


def get_cols_x_for_comparison(
    df1: pd.DataFrame,
    cols_y: list,
    cols_index: list,
    ## drop columns
    cols_drop: list = [],
    cols_dropby_patterns: list = [],
    ## complexity
    dropby_low_complexity: bool = True,
    min_nunique: int = 5,
    max_inflation: int = 50,
    ## collinearity
    dropby_collinearity: bool = True,
    coff_rs: float = 0.7,
    dropby_variance_inflation: bool = True,
    verbose: bool = False,
    test: bool = False,
) -> dict:
    """
    Identify X columns.

    Parameters:
        df1 (pd.DataFrame): input table.
        cols_y (list): y columns.

    """
    ## drop columns
    df1 = (
        df1.drop(cols_drop, axis=1)
        .rd.dropby_patterns(cols_dropby_patterns)
        .log(label="all na")
        .log.dropna(how="all", axis=1)
    )
    if dropby_low_complexity:
        df1 = (
            df1.log(label="constants")
            ## drop single value columns
            .rd.drop_constants()
        )
    ## columns to drop after identifying them
    cols_drop = []
    ## make the dictionary with column names
    columns = dict(
        cols_y={
            "cont": df1.loc[:, cols_y].select_dtypes((int, float)).columns.tolist(),
        },
        cols_index=cols_index,
    )
    columns["cols_y"]["desc"] = list(set(cols_y) - set(columns["cols_y"]["cont"]))
    columns["cols_x"] = {}

    ## get continuous cols_x
    if dropby_low_complexity:
        logging.info("[1] Checking complexity..")
        df1 = drop_low_complexity(
            df1=df1,
            min_nunique=min_nunique,
            max_inflation=max_inflation,
            cols=list(
                set(df1.select_dtypes((int, float)).columns.tolist())
                - set(columns["cols_y"]["cont"])
            ),
            cols_keep=columns["cols_y"]["cont"],
            verbose=verbose,
        )
    # except:
    #     logging.warning('skipped `drop_low_complexity`, possibly because of a single x variable')
    #     df_=df1.copy()

    columns["cols_x"]["cont"] = (
        df1.drop(columns["cols_y"]["cont"], axis=1)
        .select_dtypes((int, float))
        .columns.tolist()
    )
    if dropby_collinearity:
        if len(columns["cols_x"]["cont"]) > 1:
            ## check collinearity
            logging.info("[2] Checking collinearity..")
            ds1_ = check_collinearity(
                df1=df1.loc[:, columns["cols_x"]["cont"]],
                threshold=coff_rs,
                # colvalue='$r_s$',
                cols_variable=["variable1", "variable2"],
                coff_pval=0.05,
            )
            if ds1_ is not None:
                if verbose:
                    logging.info(
                        f"Minimum correlation among group of variables: {ds1_}"
                    )
                from roux.lib.set import flatten, unique

                cols_drop += unique(
                    flatten([s.split("--") for s in ds1_.index.tolist()])
                )
                if verbose:
                    logging.info(f"Columns to be dropped: {cols_drop}")

    if dropby_variance_inflation:
        logging.info("[3] Checking variance inflation..")
        vifs = {}
        for coly in columns["cols_y"]["cont"]:
            vifs[coly] = get_variance_inflation(
                df1,
                coly=coly,
            )
        if len(vifs) != 0:
            ds2_ = (
                pd.concat(vifs, axis=0)
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .sort_values("VIF", ascending=False)
            )
            if verbose:
                logging.info(ds2_)

    df1 = df1.log.drop(labels=cols_drop, axis=1)
    columns["cols_x"]["cont"] = list(
        sorted(
            set(columns["cols_x"]["cont"]) - set(cols_drop) - set(columns["cols_index"])
        )
    )

    ## get descrete x columns
    ## bools
    ds2_ = df1.nunique().sort_values()
    l1 = ds2_.loc[lambda x: (x == 2)].index.tolist()
    if test:
        print("l1", l1)

    ds2_ = df1.select_dtypes((int, float)).nunique().sort_values()
    l2 = ds2_.loc[lambda x: (x == 2)].index.tolist()
    if test:
        print("l2", l2)

    columns["cols_x"]["desc"] = sorted(
        list(set(l1 + l2) - set(columns["cols_y"]["desc"]) - set(columns["cols_index"]))
    )
    return columns


def to_preprocessed_data(
    df1: pd.DataFrame,
    columns: dict,
    fill_missing_desc_value: bool = False,
    fill_missing_cont_value: bool = False,
    normby_zscore: bool = False,
    verbose: bool = False,
    test: bool = False,
) -> pd.DataFrame:
    """
    Preprocess data.
    """
    ## Filtering by the provided columns
    ### collecting all the columns
    cols = []
    for d1 in columns.values():
        if isinstance(d1, dict):
            cols += list(d1.values())
        elif isinstance(d1, list):
            cols += list(d1)
    from roux.lib.set import flatten, unique

    df1 = df1.loc[:, unique(flatten(cols))]

    if normby_zscore:
        ## Normalise continuous variables by calculating Z-score
        if len(columns["cols_x"]["cont"]) != 0:
            import scipy as sc

            for c in columns["cols_x"]["cont"]:
                df1[c] = sc.stats.zscore(df1[c], nan_policy="omit")
            if verbose:
                logging.info(
                    df1.loc[:, columns["cols_x"]["cont"]]
                    .describe()
                    .loc[["mean", "std"], :]
                )

    ## Fill missing values
    if fill_missing_cont_value != False:  # noqa
        for c in columns["cols_x"]["cont"]:
            if df1[c].isnull().any():
                if verbose:
                    logging.info(df1[c].isnull().sum())
                df1[c] = df1[c].fillna(fill_missing_cont_value)

    if fill_missing_desc_value != False:  # noqa
        for c in columns["cols_x"]["desc"]:
            if df1[c].isnull().any():
                if verbose:
                    logging.info(df1[c].isnull().sum())
                df1[c] = df1[c].fillna(fill_missing_desc_value)
    return df1


## filter
def to_filteredby_samples(
    df1: pd.DataFrame,
    colindex: str,
    colsample: str,
    coff_samples_min: int,
    colsubset: str,
    coff_subsets_min: int = 2,
) -> pd.DataFrame:
    """Filter table before calculating differences.
    (1) Retain minimum number of samples per item representing a subset and
    (2) Retain minimum number of subsets per item.

    Parameters:
        df1 (pd.DataFrame): input table.
        colindex (str): column containing items.
        colsample (str): column containing samples.
        coff_samples_min (int): minimum number of samples.
        colsubset (str): column containing subsets.
        coff_subsets_min (int): minimum number of subsets. Defaults to 2.

    Returns:
        pd.DataFrame

    Examples:
        Parameters:
            colindex='genes id',
            colsample='sample id',
            coff_samples_min=3,
            colsubset= 'pLOF or WT'
            coff_subsets_min=2,
    """

    df1 = df1.reset_index(drop=True)
    df1 = df1.loc[
        (
            df1.groupby([colindex, colsubset])[colsample].transform("nunique")
            >= coff_samples_min
        ),
        :,
    ].log(colindex)
    df1 = df1.loc[
        (df1.groupby(colindex)[colsubset].transform("nunique") == coff_subsets_min), :
    ].log(colindex)
    assert (
        df1.groupby([colindex, colsubset])[colsample].nunique().min()
        >= coff_samples_min
    )
    return df1


def get_cvsplits(
    X: np.array,
    y: np.array = None,
    cv: int = 5,
    random_state: int = None,
    outtest: bool = True,
) -> dict:
    """Get cross-validation splits.
    A friendly wrapper around `sklearn.model_selection.KFold`.

    Args:
        X (np.array): X matrix.
        y (np.array): y vector.
        cv (int, optional): cross validations. Defaults to 5.
        random_state (int, optional): random state. Defaults to None.
        outtest (bool, optional): output test data. Defaults to True.

    Returns:
        dict: output.
    """
    if random_state is None:
        logging.warning("random_state is None")
    X.index = range(len(X))
    if y is not None:
        y.index = range(len(y))

    from sklearn.model_selection import KFold

    cv = KFold(n_splits=cv, random_state=random_state, shuffle=True)
    cv2Xy = {}
    for i, (train, test) in enumerate(cv.split(X.index)):
        dtype2index = dict(zip(("train", "test"), (train, test)))
        cv2Xy[i] = {}
        if outtest:
            for dtype in dtype2index:
                cv2Xy[i][dtype] = {}
                cv2Xy[i][dtype]["X" if isinstance(X, pd.DataFrame) else "x"] = (
                    X.iloc[dtype2index[dtype], :]
                    if isinstance(X, pd.DataFrame)
                    else X[dtype2index[dtype]]
                )
                if y is not None:
                    cv2Xy[i][dtype]["y"] = y[dtype2index[dtype]]
        else:
            cv2Xy[i]["X" if isinstance(X, pd.DataFrame) else "x"] = (
                X.iloc[dtype2index["train"], :]
                if isinstance(X, pd.DataFrame)
                else X[dtype2index["train"]]
            )
            if y is not None:
                cv2Xy[i]["y"] = y[dtype2index["train"]]
    return cv2Xy

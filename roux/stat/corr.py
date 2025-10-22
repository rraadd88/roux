"""For correlation stats."""

import logging

import numpy as np
from scipy import stats, spatial

import pandas as pd

# attributes
import roux.lib.dfs as rd  # noqa

def _pre(
    x: str,
    y: str,
    df: pd.DataFrame = None,
    
    covar=None,
    x_covar=None,
    y_covar=None,
    
    n_min: int = 10,
    drop_same_value=None,  # e.g. 0
    verbose: bool = False,
    test: bool = False,
) -> pd.DataFrame:
    """
    Preprocess correlation inputs.

    Args:
        x (str): x column name or a vector.
        y (str): y column name or a vector.
        df (pd.DataFrame): input table.
        n_min (int): minimum sample size required.
        verbose (bool): verbose.

    Returns:
        df: preprocessed table.
    """
    if test:
        print(x.name, y.name)
    if not (isinstance(x, str) and isinstance(y, str) and df is not None):
        ## get columns
        df = pd.DataFrame({'x': x, 'y': y})
        x, y = 'x', 'y'
    else:
        df = df.rename(columns={x: 'x', y: 'y'}, errors="raise")
    if drop_same_value is not None:
        _before = len(df)
        df = df.query(f"~(`x` == {drop_same_value} & `y` == {drop_same_value})")
        if verbose:
            logging.info(
                f"drop_same_value={drop_same_value}; {len(df)-_before} samples dropped"
            )
    assert pd.api.types.is_numeric_dtype(df['x']), df['x'].dtype
    assert pd.api.types.is_numeric_dtype(df['y']), df['y'].dtype
    # clean
    cols_covar=[]
    if covar or x_covar or y_covar:
        if covar is None:
            covar=[]
        if isinstance(covar,str):
            covar=[covar]
        cols_covar=[c for c in covar+[x_covar,y_covar] if c is not None]
        cols_covar=list(set(cols_covar))
    
    df = (
        df
            .replace([np.inf, -np.inf], np.nan)
            .dropna(
                subset=['x','y']+cols_covar
            )
        )
    
    if len(df) < n_min:
        if verbose:
            logging.error("low sample size")
        return
    return df


def resampled(
    x: np.array,
    y: np.array,
    method_fun: object,
    method_kws: dict = {},
    ci_type: str = "max",
    cv: int = 5,
    random_state: int = 1,
    verbose: bool = False,
) -> tuple:
    """Get correlations after resampling.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.
        method_fun (str, optional): method function.
        ci_type (str, optional): confidence interval type. Defaults to 'max'.
        cv (int, optional): number of resamples. Defaults to 5.
        random_state (int, optional): random state. Defaults to 1.
        verbose (bool): verbose.

    Returns:
        dict: results containing mean correlation coefficient, CI and CI type.
    """
    from roux.stat.preprocess import get_cvsplits
    from roux.stat.variance import get_ci

    cv2xy = get_cvsplits(x, y, cv=cv, outtest=False, random_state=random_state)
    rs = [_post(method_fun(*cv2xy[k].values(), **method_kws))["r"] for k in cv2xy]
    if verbose:
        logging.info(f"resampling: cv={cv},ci_type={ci_type}")
    return {"rr": np.mean(rs), "ci": get_ci(rs, ci_type=ci_type), "ci_type": ci_type}


## post-process
def _post(
    res,
    method: str = None,
    n: int = None,
) -> dict:
    """
    Post-process correlation results.

    Args:
        res: output provided by scipy function or resampling function.
        method (str): method name.
        n (int): sample size.

    Returns:
        res: dictionary containing the results.
    """
    if isinstance(res, (float, int)):  # int if 0
        res = {"r": res, "n": n}
    elif isinstance(res, tuple):
        if len(res) == 2:
            res = {"r": res[0], "P": res[1], "n": n}
        else:
            raise ValueError(res)
    elif isinstance(res, dict):
        ## resampled
        res["n"] = n
    elif isinstance(res, object):
        if hasattr(res, "correlation") and hasattr(res, "pvalue"):
            res = {"r": res.correlation, "P": res.pvalue, "n": n}
        else:
            raise ValueError((res, type(res)))
    else:
        raise ValueError((res, type(res)))
    res["method"] = method
    return res


def get_corr(
    x: str,
    y: str,
    method: str,
    df: pd.DataFrame = None,
    
    method_kws: dict = {},
    covar=None,
    x_covar=None,
    y_covar=None,
        
    preprocess: bool = True,
    n_min=10,
    preprocess_kws: dict = {},

    resample: bool = False,
    cv=5,
    resample_kws: dict = {},

    ## out
    pval: bool = True,
    
    verbose: bool = False,
    test: bool = False,
) -> dict:
    """Correlation between vectors.
    A unifying wrapper around `scipy`'s functions to calculate correlations and distances. Allows application of resampling on those functions.

    Usage:
        1. Linear table with paired values. For a matrix, use `pd.DataFrame.corr` instead.

    Args:
        x (str): x column name or a vector.
        y (str): y column name or a vector.
        method (str): method name.
        df (pd.DataFrame): input table.
        pval (bool): calculate p-value.
        resample (bool, optional): resampling. Defaults to False.
        preprocess (bool): preprocess the input
        preprocess_kws (dict) : parameters provided to the pre-processing function i.e. `_pre`.
        resample (bool): resampling.
        resample_kws (dict): parameters provided to the resampling function i.e. `resample`.
        verbose (bool): verbose.

    Returns:
        res (dict): a dictionary containing results.

    Notes:
        `res` directory contains following values:

            method : method name
            r : correlation coefficient or distance
            p : pvalue of the correlation.
            n : sample size
            rr: resampled average 'r'
            ci: CI
            ci_type: CI type
    """
    ## check inputs
    if resample:
        assert (
            n_min > cv
        ), f"n_min={n_min} !> cv={cv} (set resample=False to turn off resampling)"

    if verbose:
        preprocess_kws["verbose"] = True
        resample_kws["verbose"] = True
    if preprocess:
        df = _pre(
            x,
            y,
            df,
            n_min=n_min,
            
            covar=covar,
            x_covar=x_covar,
            y_covar=y_covar,
            
            test=test,
            **preprocess_kws,
         )
        if df is None:
            return {}
        x, y, n = df['x'], df['y'], len(df)
        
    if not callable(method):
        if hasattr(stats, method + "r"):
            method_fun = getattr(stats, method + "r")
        elif hasattr(spatial.distance, method):
            ## no-pvalue
            method_fun = getattr(spatial.distance, method)
        elif hasattr(stats, method):
            method_fun = getattr(stats, method)
        else:
            raise ValueError(method)
    else:
        method_fun = method
        method = method.__name__

    if pval:
        if covar or x_covar or y_covar:
            method_kws={
                **method_kws,
                **dict(
                    covar=covar,
                    x_covar=x_covar,
                    y_covar=y_covar,
                    method=method,    
                )
            }
            
            from pingouin import partial_corr
            res = (
                partial_corr(
                    data=df,
                    x='x',
                    y='y',
                    **method_kws,
                )
                .rename(
                    columns={
                        'p-val':'P',
                    }
                )
                .astype({'n':int})
                .iloc[0,:].to_dict()
            )
            res={
                **res,
                **method_kws,
            }
        else:
            res = _post(method_fun(x, y, **method_kws), method, n)
    else:
        res = {}
    if resample:
        res_ = resampled(
            x=x,
            y=y,
            method_fun=method_fun,
            method_kws=method_kws,
            cv=cv,
            **resample_kws,
        )
        res_ = _post(res_, method, n)
        res = {**res_, **res}
        if verbose:
            logging.info(
                f"r={res['rr']} (resampled), r={res['r']} and P={res['P']} (collective)"
            )
    return res


def _to_string(
    res: dict,
    show_n: bool = True,
    show_p: bool = True,
    show_n_prefix="$n$=",
    fmt: dict = "<",
    method_suffix=True,
    **kws_pval2annot,
) -> str:
    """Correlation results to string.

    Args:
        res (dict): dictionary containing the results of the correlation.
        show_n (str): show sample size.
        fmt (str, optional): format of the p-value. Defaults to '<'.

    Keyword arguments:
        kws_pval2annot: Keyword arguments provided to the `pval2annot` function.

    Returns:
        str: string with the correation stats.
    """
    from roux.viz.annot import pval2annot
    from roux.lib.str import num2str

    method = res["method"]
    if method_suffix:
        if res.get('covar',None):
            method_suffix="($\\tilde{x},\\tilde{y}$)"
        elif res.get('x_covar',None):
            method_suffix="($\\tilde{x},y$)"
        elif res.get('y_covar',None):
            method_suffix="($x,\\tilde{y}$)"
        else:
            method_suffix=''
        
    s0 = (
        (f"$r_{method[0]}$" if "tau" not in method else "$\\tau$")
        + (method_suffix if method_suffix else '')
        + (f"={res['rr' if 'rr' in res else 'r']:.2f}")  ##prefer the resampled r value
    )
    if "ci" in res:
        s0 += f"$\pm${res['ci']:.2f}{res['ci_type'] if res['ci_type']!='max' else ''}"
    if show_p:
        s0 += f"\n{pval2annot(res['P'],fmt='<',linebreak=False, alpha=0.05)}"
    if show_n:
        s0 += f"\n({show_n_prefix}{num2str(num=res['n'],magnitude=False)})"
    return s0


## many correlations
def get_corrs(
    data: pd.DataFrame,
    method: str,
    cols: list = None,
    cols_with: list = None,
    coff_inflation_min: float = None, # fast
    get_pairs_kws={},

    cpus: bool = 1,
    out_q=True, # False, # fast
    
    test: bool = False,
    verbose: bool = False,
    kws_chunks={},
    **kws_get_corr,
) -> pd.DataFrame:
    """
    Correlate many columns of a dataframes.

    Parameters:
        df1 (DataFrame): input dataframe.
        method (str): method of correlation `spearman` or `pearson`.
        cols (str): columns.
        cols_with (str): columns to correlate with i.e. variable2.
        cpus (int): cpus to use.

    Keyword arguments:
        kws_get_corr: parameters provided to `get_corr` function.

    Returns:
        DataFrame: output dataframe.
    """
    # check inflation/over-representations
    if coff_inflation_min is not None:
        from roux.lib.df import check_inflation

        ds_ = check_inflation(
            data,
            subset=cols + cols_with,
        )
        ds_ = ds_.loc[lambda x: x >= coff_inflation_min]
        if len(ds_) != 0:
            logging.info(f"columns dropped because of inflation: {ds_.index.tolist()}")
        # remove inflated
        cols = [c for c in cols if c not in ds_.index.tolist()]
        cols_with = [c for c in cols_with if c not in ds_.index.tolist()]

    ## pair columns
    from roux.lib.set import get_pairs
    from roux.stat.transform import get_q

    df0 = (
        get_pairs(
            items=cols if cols is not None else data.columns.tolist(),
            items_with=cols_with,
            **get_pairs_kws,
        )
        .add_prefix("variable")
        .log.dropna()
    )
    if verbose:
        logging.info(df0.shape)
    # debug
    # return df0, data
    ## get correlations
    if cpus==1:
        df1 = getattr(
            df0,
            "apply" if not hasattr(df0, "progress_apply") else "progress_apply",
        )(
            lambda x: pd.Series(
                {
                    **{"variable1": x["variable1"], "variable2": x["variable2"]},
                    **get_corr(
                        data[x["variable1"]],
                        data[x["variable2"]],
                        method=method,
                        **kws_get_corr,
                    ),
                }
            ),
            axis=1,
        )
    else:
        ## paralel
        import roux.lib.df_apply as rd  # noqa
        
        # return df0,data
        # logging.error('rd.apply_async not implemented.')
        if 'out_df' in kws_chunks:
            assert not kws_chunks['out_df'], kws_chunks
            
        df1=(
            df0
            .reset_index(drop=True) # for join later
            .assign(
                res=lambda df: df.rd.apply(
                    func=lambda x: get_corr(
                        data[x['variable1']],
                        data[x['variable2']],
                        method=method,
                        **kws_get_corr,
                    ),
                    axis=1,
                    cpus=cpus,
                    kws_chunks=kws_chunks,
                )
            )
            .rd.explode(
                'res'
            )
        )        

    if out_q and "P" in df1:
        ## FDR
        return df1.assign(
            **{
                "Q": lambda df: get_q(df["P"]),
            },
        ).sort_values("Q", ascending=[True])
    else:
        return df1


def check_collinearity(
    df1: pd.DataFrame,
    threshold: float = 0.7,
    colvalue: str = "r",
    cols_variable: list = ["variable1", "variable2"],
    coff_pval: float = 0.05,
    method: str = "spearman",
    coff_inflation_min: int = 50,
) -> pd.Series:
    """Check collinearity.

    Args:
        df1 (DataFrame): input dataframe.
        threshold (float): minimum threshold for the colinearity.

    Returns:
        DataFrame: output dataframe with minimum correlation among correlated subnetwork of columns.
    """
    cols = df1.columns.tolist()
    df2 = get_corrs(
        df1,
        method=method,
        cols=cols,
        cols_with=cols,
        coff_inflation_min=coff_inflation_min,
    )
    df2 = df2.loc[(df2["P"] < coff_pval), :]
    df2["is collinear"] = df2[colvalue].abs().apply(lambda x: x >= threshold)
    perc = (df2["is collinear"].sum() / len(df2)) * 100
    if df2["is collinear"].sum() == 0:
        logging.info(f"max corr={df2[colvalue].max()}")
        return
    df2 = df2.loc[(df2["is collinear"]), :]
    logging.info(
        f"collinear vars: {perc:.1f}% ({df2['variable1'].nunique()}/{len(df1.columns)})"
    )

    ## find subgraphs (communities) variables with correalted values
    from roux.stat.network import get_subgraphs

    df3 = get_subgraphs(
        df2.loc[df2["is collinear"], :], cols_variable[0], cols_variable[1]
    )
    df3 = df3.groupby("subnetwork name").agg({"node name": list}).reset_index()
    return (
        df3.groupby("subnetwork name")
        .apply(
            lambda df: df2.apply(
                lambda x: x[colvalue]
                if len(
                    set([x[cols_variable[0]], x[cols_variable[1]]])
                    - set(df["node name"].tolist()[0])
                )
                == 0
                else np.nan,
                axis=1,
            ).min()
        )
        .sort_values(ascending=False)
    )


def pairwise_chi2(df1: pd.DataFrame, cols_values: list) -> pd.DataFrame:
    """Pairwise chi2 test.

    Args:
        df1 (DataFrame): pd.DataFrame
        cols_values (list): list of columns.

    Returns:
        DataFrame: output dataframe.

    TODOs:
        0. use `lib.set.get_pairs` to get the combinations.
    """
    import itertools

    d1 = {}
    for cols in list(itertools.combinations(cols_values, 2)):
        _, d1[cols], _, _ = stats.chi2_contingency(
            pd.crosstab(df1[cols[0]], df1[cols[1]])
        )

    df2 = pd.Series(d1).to_frame("P")
    df2.index.names = ["value1", "value2"]
    df2 = df2.reset_index()
    return df2


# def par_pcorr(df, x, y, covars, method='spearman'):
#     """
#     Calculates the Spearman partial correlation using a robust regression method on ranks.

#     Parameters
#     ----------
#     df : pd.DataFrame
#         The input DataFrame.
#     x : str
#         The name of the x variable.
#     y : str
#         The name of the y variable.
#     covars : list of str
#         The name of the covariate(s).

#     Returns
#     -------
#     float
#         The Spearman partial correlation coefficient.
#     Notes:
#     ------
#     not as robust as pg.partial_corr
#     """
#     assert method=='spearman', method
#     # Spearman is a Pearson correlation on the ranks of the data.
#     df_ranked = df.rank(method='average')

#     # All variables to be included in the regression models
#     all_vars = [x, y] + covars

#     # Regress x on all other variables
#     x_covariates_df = df_ranked[[v for v in all_vars if v != x]]
#     x_residuals = regress_out_pinv(
#         df_ranked[x].values,
#         x_covariates_df.values
#     )

#     # Regress y on all other variables
#     y_covariates_df = df_ranked[[v for v in all_vars if v != y]]
#     y_residuals = regress_out_pinv(
#         df_ranked[y].values,
#         y_covariates_df.values
#     )

#     # The partial correlation is the Pearson correlation between the residuals
#     correlation, _ = pearsonr(x_residuals, y_residuals)

#     return correlation

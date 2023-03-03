"""For correlation stats."""
import pandas as pd
import numpy as np

from scipy import stats,spatial

import logging
from roux.lib.sys import info

def _pre(
    x:str,
    y:str,
    df:pd.DataFrame=None,
    n_min: int=10,
    verbose: bool=False,
    )-> pd.DataFrame:
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
    if not (isinstance(x,str) and isinstance(y,str) and not df is None):
        ## get columns
        df=pd.DataFrame({'x':x,'y':y})
        x,y='x','y'
    else:
        df=df.rename(columns={x:'x',y:'y'},errors='raise')
    if len(df)<n_min:
        if verbose:
            logging.error("low sample size")
        return
    assert df['x'].dtype in [int,float], df['x'].dtype
    assert df['y'].dtype in [int,float], df['y'].dtype
    # clean
    df=df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def resampled(
    x: np.array,
    y: np.array,
    method_fun: object,
    method_kws: dict={},
    ci_type: str='max',
    cv:int=5,
    random_state: int=1,
    verbose: bool=False,
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
    from roux.stat.classify import get_cvsplits
    from roux.stat.variance import get_ci
    cv2xy=get_cvsplits(x,y,cv=cv,outtest=False,random_state=random_state)
    rs=[_post(method_fun(*cv2xy[k].values(),**method_kws))['r'] for k in cv2xy]
    if verbose:
        logging.info(f"resampling: cv={cv},ci_type={ci_type}")
    return {'rr':np.mean(rs), 'ci':get_ci(rs,ci_type=ci_type),'ci_type':ci_type}

## post-process
def _post(
    res,
    method: str=None,
    n: int=None,
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
    if isinstance(res,float):
        res={'r':res,'n':n}
    elif isinstance(res,tuple):
        if len(res)==2:
            res={'r':res[0],'P':res[1],'n':n}
        else:
            raise ValueError(res)
    elif isinstance(res,dict):
        ## resampled
        res['n']=n
    elif isinstance(res,object):
        if hasattr(res,'correlation') and hasattr(res,'pvalue'):
            res={'r':res.correlation,'P':res.pvalue,'n':n}                
        else:
            raise ValueError(res)
    else:
        raise ValueError(res)
    res['method']=method
    return res

def get_corr(
    x: str,
    y: str,
    method: str,
    df: pd.DataFrame=None,
    method_kws: dict={},
    pval: bool=True,
    preprocess: bool=True,
    preprocess_kws: dict={},
    resample: bool=False,
    resample_kws: dict={},
    verbose: bool=False,
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
    if verbose:
        preprocess_kws['verbose']=True
        resample_kws['verbose']=True
    if preprocess:
        df=_pre(x,y,df)
        if df is None: 
            return
        x,y,n=df['x'],df['y'],len(df)
    if hasattr(stats,method+'r'):
        get_corr=getattr(stats,method+'r')
    elif hasattr(spatial.distance,method):
        ## no-pvalue
        get_corr=getattr(spatial.distance,method)
    else:
        raise ValueError(method)
    if pval:
        res=_post(get_corr(x,y,**method_kws),method,n)
    else:
        res={}
    if resample:
        res_=resampled(
            x=x,y=y,
            method_fun=get_corr,
            method_kws=method_kws,
            **resample_kws,
        )
        res_=_post(res_,method,n)
        res={**res_,**res}
        if verbose:
            logging.info(f"r={res['rr']} (resampled), r={res['r']} and P={res['P']} (collective)")
    return res
        
def _to_string(
    res: dict,
    show_n: bool=True,
    fmt: dict='<',
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
    method=res['method']
    s0=(f"$r_{method[0]}$" if not 'tau' in method else '$\\tau$')+f"={res['rr' if 'rr' in res else 'r']:.2f}" ##prefer the resampled r value
    if 'ci' in res:
        s0+=f"$\pm${res['ci']:.2f}{res['ci_type'] if res['ci_type']!='max' else ''}"
    s0+=f"\n{pval2annot(res['P'],fmt='<',linebreak=False, alpha=0.05)}"
    if show_n:
        s0+=f"\n({num2str(num=res['n'],magnitude=False)})"
    return s0

## many correlations
def get_corrs(
    data: pd.DataFrame,
    method: str,
    cols: list=None,
    cols_with: list = None,
    inflation_min: float=None,
    get_pairs_kws={},
    fast: bool=False,
    test: bool=False,
    verbose: bool=False,    
    **kws_get_corr,
    ) -> pd.DataFrame:
    """
    Correlate many columns of a dataframes.

    Parameters:
        df1 (DataFrame): input dataframe.
        method (str): method of correlation `spearman` or `pearson`.        
        cols (str): columns.
        cols_with (str): columns to correlate with i.e. variable2.
        fast (bool): use parallel-processing if True.

    Keyword arguments:
        kws_get_corr: parameters provided to `get_corr` function.
        
    Returns:
        DataFrame: output dataframe.
    """
    # check inflation/over-representations
    if not inflation_min is None:
        from roux.lib.df import check_inflation
        ds_=check_inflation(df1,subset=cols+cols_with)
        ds_=ds_.loc[lambda x: x>=coff_inflation_min]
        info(ds_)
        # remove inflated
        cols=[c for c in cols if not c in ds_.index.tolist()]
        cols_with=[c for c in cols_with if not c in ds_.index.tolist()]
        
    ## pair columns
    from roux.lib.set import get_pairs
    from roux.stat.transform import get_q
    df0=(
        get_pairs(
            items=cols if not cols is None else data.columns.tolist(),
            items_with= cols_with,
            **get_pairs_kws,
            ).add_prefix('variable')
    )
    ## get correlations
    df1=(getattr(df0,'apply' if not fast else 'parallel_apply')
        (lambda x: pd.Series({**{"variable1":x["variable1"],"variable2":x["variable2"]},
                                  **get_corr(data[x["variable1"]],data[x["variable2"]],
                                  method=method,**kws_get_corr)}),
       axis=1,
          )
        )
    if 'P' in df1:
        ## FDR
        return (df1
            .assign(
                **{'Q':lambda df: get_q(df['P']),
                  },
            ) 
            .sort_values('Q',ascending=[True])
            )
    else:
        return df1

def check_collinearity(
    df1: pd.DataFrame,
    threshold: float=0.7,
    colvalue: str='$r_s$',
    cols_variable: list=['variable1','variable2'],
    coff_pval: float=0.05,
    method: str='spearman',
    coff_inflation_min: int=50,
    )-> pd.Series:
    """Check collinearity.

    Args:
        df1 (DataFrame): input dataframe.
        threshold (float): minimum threshold for the colinearity.

    Returns:
        DataFrame: output dataframe.
    
    TODOs:
        1. Calculate variance inflation factor (VIF).
    """
    cols=df1.columns.tolist()
    df2=get_corrs(df1=df1,
        method=method,
        cols=cols,
        cols_with=cols,
        coff_inflation_min=coff_inflation_min,
        )
    df2=df2.loc[(df2['P']<coff_pval),:]

    df2['is collinear']=df2[colvalue].abs().apply(lambda x: x>=threshold)
    perc=(df2['is collinear'].sum()/len(df2))*100
    logging.info(f"collinear vars: {perc:.1f}% ({df2['is collinear'].sum()}/{len(df1.columns)})")
    if df2['is collinear'].sum()==0:
        logging.info(f"max corr={df2[colvalue].max()}")
        return
    df2=df2.loc[(df2['is collinear']),:]
    from roux.stat.network import get_subgraphs
    df3=get_subgraphs(df2.loc[df2['is collinear'],:],cols_variable[0],cols_variable[1])
    df3=df3.groupby('subnetwork name').agg({'node name':list}).reset_index()
    return (df3
            .groupby('subnetwork name')
            .progress_apply(lambda df: df2.apply(lambda x: \
                                                 x[colvalue] if len(set([x[cols_variable[0]],x[cols_variable[1]]]) - set(df['node name'].tolist()[0]))==0 else \
                                                 np.nan,
                                                 axis=1).min())
            .sort_values(ascending=False)
           )    

def pairwise_chi2(
    df1: pd.DataFrame,
    cols_values: list
    ) -> pd.DataFrame:
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
    d1={}
    for cols in list(itertools.combinations(cols_values,2)):
        _,d1[cols],_,_=stats.chi2_contingency(pd.crosstab(df1[cols[0]],df1[cols[1]]))

    df2=pd.Series(d1).to_frame('P')
    df2.index.names=['value1','value2']
    df2=df2.reset_index()
    return df2

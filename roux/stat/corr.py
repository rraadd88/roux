"""For correlation stats."""
import pandas as pd
import numpy as np

import scipy as sc

import logging
from roux.lib.sys import info

def _pre(x,y,
         df=None,
         sample_size_min=10,
         verbose=False):
    if not (isinstance(x,str) and isinstance(y,str) and not df is None):
        ## get columns
        df=pd.DataFrame({'x':x,'y':y})
        x,y='x','y'
    else:
        df=df.rename(columns={x:'x',y:'y'},errors='raise')
    if len(df)<sample_size_min:
        if verbose:
            logging.error("low sample size")
        return
    assert df['x'].dtype in [int,float], df['x'].dtype
    assert df['y'].dtype in [int,float], df['y'].dtype
    # clean
    df=df.dropna()
    return df

def resampled(
    x: np.array,
    y: np.array,
    method_fun,
    method_kws={},
    ci_type='max',
    cv:int=5,
    random_state=1,
    verbose=False,
    ) -> tuple:
    """Get correlations after resampling.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.
        method (str, optional): method name. Defaults to 'spearman'.
        ci_type (str, optional): confidence interval type. Defaults to 'max'.
        cv (int, optional): number of resamples. Defaults to 5.
        random_state (int, optional): random state. Defaults to 1.

    Returns:
        tuple: mean correlation coefficient, confidence interval
    """
    from roux.stat.classify import get_cvsplits
    from roux.stat.variance import get_ci
    cv2xy=get_cvsplits(x,y,cv=cv,outtest=False,random_state=random_state)
    rs=[method_fun(*cv2xy[k].values(),**method_kws)[0] for k in cv2xy]
    if verbose:
        logging.info(f"resampling: cv={cv},ci_type={ci_type}")
    return {'rr':np.mean(rs), 'ci':get_ci(rs,ci_type=ci_type),'ci_type':ci_type}

## post-process
# @staticmethod
def _post(res,method,n):
    """
    uniform output
    get r,p,ci,n
    """
    if isinstance(res,float):
        return {'r':res,'n':n}
    elif isinstance(res,tuple):
        if len(res)==2:
            return {'r':res[0],'p':res[1],'n':n}
        else:
            raise ValueError(res)
    elif isinstance(res,dict):
        ## resampled
        res['n']=n
        res['method']=method
        return res
    elif isinstance(res,object):
        if hasattr(res,'correlation') and hasattr(res,'pvalue'):
            return {'r':res.correlation,'p':res.pvalue,'n':n}                
        else:
            raise ValueError(res)
    else:
        raise ValueError(res)

def corr(
    x,y,
    df=None,
    pval=True,
    method=None,
    method_kws={},
    preprocess=True,
    preprocess_kws={},
    resample=False,
    resample_kws={},
    # out_str=False,
    # out_str_kws={},
    verbose=False,
    ):
    """Correlation between vectors (wrapper).

    Usage:
        1. Linear table with paired values. For a matrix, use `pd.DataFrame.corr` instead.

    Args:
        x (np.array): x.
        y (np.array): y.
        method (str, optional): method name. Defaults to 'spearman'.
        resample (bool, optional): resampling. Defaults to False.
        ci_type (str, optional): confidence interval type. Defaults to 'max'.
        magnitide (bool, optional): show magnitude. Defaults to True.
        outstr (bool, optional): output as string. Defaults to False.

    Keyword arguments:
        kws: parameters provided to `get_corr_resampled` function.

    """
    if verbose:
        preprocess_kws['verbose']=True
        resample_kws['verbose']=True
    if preprocess:
        df=_pre(x,y,df)
        if df is None: 
            return
        x,y,n=df['x'],df['y'],len(df)
    if hasattr(sc.stats,method+'r'):
        get_corr=getattr(sc.stats,method+'r')
    elif hasattr(sc.spatial.distance,method):
        ## no-pvalue
        get_corr=getattr(sc.spatial.distance,method)
    else:
        raise ValueError(method)
    if pval:
        res=get_corr(x,y,**method_kws)
        res=_post(res,method,n)
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
            logging.info(f"r={res['rr']} (resampled), r={res['r']} and P={res['p']} (collective)")
    return res
        
def to_string(
    res,
    # method: str,
    show_n: bool=True,
    show_n_prefix: str='',    
    fmt='<',
    magnitide=True,
    ) -> str:
    """Correlation to string

    Args:
        method (str): method name.
        r (float): correlation coefficient.
        p (float): p-value
        fmt (str, optional): format of the p-value. Defaults to '<'.
        n (bool, optional): sample size. Defaults to True.
        ci (_type_, optional): confidence interval. Defaults to None.
        ci_type (_type_, optional): confidence interval type. Defaults to None.
        magnitide (bool, optional): show magnitude of the sample size. Defaults to True.

    Returns:
        str: string with the correation stats. 
    """
    from roux.viz.annot import pval2annot
    from roux.lib.str import num2str
    method=res['method']
    s0=(f"$r_{method[0]}$" if not 'tau' in method else '$\\tau$')+f"={res['rr' if 'rr' in res else 'r']:.2f}" ##prefer the resampled r value
    if 'ci' in res:
        s0+=f"$\pm${res['ci']:.2f}{res['ci_type'] if res['ci_type']!='max' else ''}"
    s0+=f"\n{pval2annot(res['p'],fmt='<',linebreak=False, alpha=0.05)}"
    if show_n:
        s0+=f"\n({num2str(num=res['n'],magnitude=False)})"
    return s0

## to be deprecated in favor of the new `corr`
# def get_corr_resampled(
#     x: np.array,
#     y: np.array,
#     method='spearman',
#     ci_type='max',
#     cv:int=5,
#     random_state=1,
#     verbose=False,
#     ) -> tuple:
#     """Get correlations after resampling.

#     Args:
#         x (np.array): x vector.
#         y (np.array): y vector.
#         method (str, optional): method name. Defaults to 'spearman'.
#         ci_type (str, optional): confidence interval type. Defaults to 'max'.
#         cv (int, optional): number of resamples. Defaults to 5.
#         random_state (int, optional): random state. Defaults to 1.

#     Returns:
#         tuple: mean correlation coefficient, confidence interval
#     """
#     from roux.stat.classify import get_cvsplits
#     from roux.stat.variance import get_ci
#     cv2xy=get_cvsplits(x,y,cv=cv,outtest=False,random_state=random_state)
#     rs=[globals()[f"get_{method}r"](**cv2xy[k])[0] for k in cv2xy]
#     if verbose: info(cv,ci_type)
#     return np.mean(rs), get_ci(rs,ci_type=ci_type)

# def corr_to_str(
#     method: str,
#     r: float,
#     p: float,
#     show_n: bool=True,
#     n:int=None,
#     show_n_prefix: str='',    
#     fmt='<',
#     ci=None,
#     ci_type=None, 
#     magnitide=True,
#     ) -> str:
#     """Correlation to string

#     Args:
#         method (str): method name.
#         r (float): correlation coefficient.
#         p (float): p-value
#         fmt (str, optional): format of the p-value. Defaults to '<'.
#         n (bool, optional): sample size. Defaults to True.
#         ci (_type_, optional): confidence interval. Defaults to None.
#         ci_type (_type_, optional): confidence interval type. Defaults to None.
#         magnitide (bool, optional): show magnitude of the sample size. Defaults to True.

#     Returns:
#         str: string with the correation stats. 
#     """
#     from roux.viz.annot import pval2annot
#     from roux.lib.str import num2str
#     s0=(f"$r_{method[0]}$" if not 'tau' in method else '$\\tau$')+f"={r:.2f}"
#     if not ci is None:
#         s0+=f"$\pm${ci:.2f}{ci_type if ci_type!='max' else ''}"
#     s0+=f"\n{pval2annot(p,fmt='<',linebreak=False, alpha=0.05)}"
#     if show_n:
#         assert not n is None, n
#         s0+=f"\n({num2str(num=n,magnitude=False)})"
#     return s0

# ## TODOs: combine different methods of correlation, with a common preprocessing
# def get_spearmanr(
#     x: np.array,
#     y: np.array,
#     ) -> tuple:
#     """Get Spearman correlation coefficient.

#     Args:
#         x (np.array): x vector.
#         y (np.array): y vector.

#     Returns:
#         tuple: rs, p-value
#     """
#     assert x.dtype in [int,float]
#     assert y.dtype in [int,float]
    
#     t=sc.stats.spearmanr(x,y,nan_policy='omit')
#     return t.correlation,float(t.pvalue)

# def get_pearsonr(
#     x: np.array,
#     y: np.array,
#     ) -> tuple:
#     """Get Pearson correlation coefficient.

#     Args:
#         x (np.array): x vector.
#         y (np.array): y vector.

#     Returns:
#         tuple: rs, p-value
#     """
#     return sc.stats.pearsonr(x,y)

# def get_kendalltaur(
#     x: np.array,
#     y: np.array,
#     ) -> tuple:
#     """Get Kendall rank correlation coefficient.

#     Args:
#         x (np.array): x vector.
#         y (np.array): y vector.

#     Returns:
#         tuple: rs, p-value
#     """
#     assert x.dtype in [int,float]
#     assert y.dtype in [int,float]
    
#     return sc.stats.kendalltau(x,y)

# def get_cosine(
#     x: np.array,
#     y: np.array,
#     ) -> tuple:
#     """Get cosine distance.

#     Args:
#         x (np.array): x vector.
#         y (np.array): y vector.

#     Returns:
#         tuple: rs, np.nan
        
#     Notes:
#         1. No p-value for the distances.
#         2. distance can be greater than 1 if the dot product is negative (anticorrelation).
#     """
#     assert x.dtype in [int,float]
#     assert y.dtype in [int,float]
    
#     return sc.spatial.distance.cosine(x,y),np.nan

# ## Wrapper aroung the correlations
# def get_corr(
#     x: np.array,
#     y: np.array,
#     method='spearman',
#     resample=False,
#     ci_type='max',
#     sample_size_min=10,
#     magnitide=True,
#     outstr=False,
#     kws_to_str={},
#     verbose: bool= False,
#     **kws_boots,
#     ):
#     """Correlation between vectors (wrapper).
    
#     Usage:
#         1. Linear table with paired values. For a matrix, use `pd.DataFrame.corr` instead.

#     Args:
#         x (np.array): x.
#         y (np.array): y.
#         method (str, optional): method name. Defaults to 'spearman'.
#         resample (bool, optional): resampling. Defaults to False.
#         ci_type (str, optional): confidence interval type. Defaults to 'max'.
#         magnitide (bool, optional): show magnitude. Defaults to True.
#         outstr (bool, optional): output as string. Defaults to False.
    
#     Keyword arguments:
#         kws: parameters provided to `get_corr_resampled` function.

#     """
#     n=len(x)
#     if n<sample_size_min:
#         if verbose:
#             logging.error("low sample size")
#         return
#     if resample:
#         r,ci=get_corr_resampled(x,y,method=method,ci_type=ci_type,**kws_boots)
#         _,p=globals()[f"get_{method}r"](x, y)
#         if verbose:
#             logging.info(f"r={r} (resampled), P={p} (collective)")
#         if not outstr:
#             return r,p,ci,n
#         else:
#             if 'show_n' in kws_to_str:
#                 if kws_to_str['show_n']==True:
#                     kws_to_str['show_n']=n
#             return corr_to_str(method,r,p,n=n,
#                                ci=ci,ci_type=ci_type, 
#                                magnitide=magnitide,
#                                **kws_to_str),r
#     else:
#         r,p=globals()[f"get_{method}r"](x, y)
#         if verbose:
#             logging.info(f"r={r},P={p}")
#         if not outstr:
#             return r,p,n
#         else:
#             if 'show_n' in kws_to_str:
#                 if kws_to_str['show_n']==True:
#                     kws_to_str['show_n']=n
#             return corr_to_str(method,r,p,n=n,
#                                ci=None,ci_type=None, 
#                                magnitide=magnitide,
#                                **kws_to_str),r
## to be deprecated

def get_corrs(
    df1: pd.DataFrame,
    method: str,
    cols: list=None,
    cols_with: list=None,
    pairs: list=None,
    coff_inflation_min: float=None,
    fast: bool=False,
    test: bool=False,
    verbose: bool=False,
    **kws
    ):
    """Correlate columns of a dataframes.

    Args:
        df1 (DataFrame): input dataframe.
        method (str): method of correlation `spearman` or `pearson`.        
        cols (str): columns.
        cols_with (str): columns to correlate with i.e. variable2.
        pairs (list): list of tuples of column (variable) pairs.
        fast (bool): use parallel-processing if True.
        
    Keyword arguments:
        kws: parameters provided to `get_corr` function.

    Returns:
        DataFrame: output dataframe.
        
    TODOs:
        0. Use `lib.set.get_pairs` to get the combinations.
        1. Provide 2D array to `scipy.stats.spearmanr`?
        2. Compare with `Pingouin`'s equivalent function.
    """
    if cols is None:
        if pairs is None:
            cols=df1.columns.tolist()
        else:
            cols=list(set(np.array(pairs).flatten()))
    # cols=list(set(df1.columns.tolist()) & set(cols))
    if cols_with is None:
        cols_with=[]
    import itertools
    from roux.stat.transform import get_q
    
    # check inflation/over-representations
    from roux.lib.df import check_inflation
    ds_=check_inflation(df1,subset=cols+cols_with)
    if not coff_inflation_min is None:
        ds_=ds_.loc[lambda x: x>=coff_inflation_min]
        info(ds_)
        # remove inflated
        cols=[c for c in cols if not c in ds_.index.tolist()]
        cols_with=[c for c in cols_with if not c in ds_.index.tolist()]
    # remove inf
    df1=df1.loc[:,np.unique(cols+cols_with)].replace([np.inf, -np.inf], np.nan)
    assert len(np.unique(cols+cols_with))!=0, "len(np.unique(cols+cols_with))==0" 
    
    ## get pairs
    if pairs is None:
        ## get pairs
        if len(cols_with)==0:
            pairs=itertools.combinations(cols,2) # cols->cols
        else:
            pairs=itertools.product(cols,cols_with) # cols->cols_with
    df0=pd.DataFrame(
        pairs,
        columns=['variable1','variable2'],
        )
    if test:
        info(df0)
    df0=df0.loc[(df0['variable1']!=df0['variable2']),:]
    if test:
        info(df0)
     
    ## correlations
    def pre(df1,cols):
        ## drop missing values if any, in a pairwise manner
        df_=df1.dropna(subset=cols)
        return df_[cols[0]],df_[cols[1]]
    
    from roux.lib.df import get_name
    df2=(getattr(df0
        .groupby(['variable1','variable2']),
               'progress_apply' if not fast else 'parallel_apply'
               )(lambda df: get_corr(
            *pre(df1,
            cols=[get_name(df, cols='variable1'),
            get_name(df, cols='variable2'),
                ],
            ),
            method=method,
            verbose=False,
            **kws))
        .apply(pd.Series)
        )
    try:
        df2.columns=[f"$r_{method[0]}$",'P','n']
    except:
        print(df2.head(1))
        print(df2.shape)
        
    ## FDR
    df2=(df2
        .reset_index()
        .log.dropna(subset=['P'])
        .groupby(['variable1']+(['variable2'] if len(cols_with)==0 else []),
                 as_index=False,
                ).apply(lambda df: get_q(df,'P',verb=verbose)).reset_index(drop=True)
        .sort_values(['Q',f"$r_{method[0]}$"],ascending=[True,False])
        )
    return df2

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
        _,d1[cols],_,_=sc.stats.chi2_contingency(pd.crosstab(df1[cols[0]],df1[cols[1]]))

    df2=pd.Series(d1).to_frame('P')
    df2.index.names=['value1','value2']
    df2=df2.reset_index()
    return df2

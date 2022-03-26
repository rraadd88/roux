import pandas as pd
import numpy as np
import scipy as sc
import logging

from scipy.stats import spearmanr,pearsonr
def get_spearmanr(x: np.array,y: np.array) -> tuple:
    """Get Spearman correlation coefficient.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.

    Returns:
        tuple: rs, p-value
    """
    t=sc.stats.spearmanr(x,y,nan_policy='omit')
    return t.correlation,float(t.pvalue)

def get_pearsonr(x: np.array,y: np.array) -> tuple:
    """Get Pearson correlation coefficient.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.

    Returns:
        tuple: rs, p-value
    """
    return sc.stats.pearsonr(x,y)

def get_corr_bootstrapped(x: np.array,y: np.array,
                          method='spearman',ci_type='max',random_state=1) -> tuple:
    """Get correlations after bootstraping.

    Args:
        x (np.array): x vector.
        y (np.array): y vector.
        method (str, optional): method name. Defaults to 'spearman'.
        ci_type (str, optional): confidence interval type. Defaults to 'max'.
        random_state (int, optional): random state. Defaults to 1.

    Returns:
        tuple: mean correlation coefficient, confidence interval
    """
    from roux.lib.stat.ml import get_cvsplits
    from roux.stat.variance import get_ci
    cv2xy=get_cvsplits(x,y,cv=5,outtest=False,random_state=1)
    rs=[globals()[f"get_{method}r"](**cv2xy[k])[0] for k in cv2xy]
    return np.mean(rs), get_ci(rs,ci_type=ci_type)

def corr_to_str(method: str,
                r: float,p: float,
                fmt='<',n=True, ci=None,ci_type=None, magnitide=True) -> str:
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
    s0=f"$r_{method[0]}$={r:.2f}"
    if not ci is None:
        s0+=f"$\pm${ci:.2f}{ci_type if ci_type!='max' else ''}"
    s0+=f"\n{pval2annot(p,fmt='<',linebreak=False, alpha=0.05)}"+('' if not n else f"\nn="+num2str(num=n,magnitude=False))
    return s0

def get_corr(x: np.array,y: np.array,
            method='spearman',bootstrapped=False,ci_type='max',magnitide=True,
            outstr=False,
            **kws):
    """Correlation between vectors (wrapper).

    Args:
        x (np.array): x.
        y (np.array): y.
        method (str, optional): method name. Defaults to 'spearman'.
        bootstrapped (bool, optional): bootstraping. Defaults to False.
        ci_type (str, optional): confidence interval type. Defaults to 'max'.
        magnitide (bool, optional): show magnitude. Defaults to True.
        outstr (bool, optional): output as string. Defaults to False.
    
    Keyword arguments:
        kws: parameters provided to `get_corr_bootstrapped` function.

    """
    n=len(x)
    if bootstrapped:
        r,ci=get_corr_bootstrapped(x,y,method=method,ci_type=ci_type,**kws)
        _,p=globals()[f"get_{method}r"](x, y)
        if not outstr:
            return r,p,ci,n
        else:
            return corr_to_str(method,r,p,n=n, ci=ci,ci_type=ci_type, magnitide=magnitide)
    else:
        r,p=globals()[f"get_{method}r"](x, y)
        if not outstr:
            return r,p,n
        else:
            return corr_to_str(method,r,p,n=n, ci=None,ci_type=None, magnitide=magnitide)

def corr_within(df: pd.DataFrame,
         method='spearman',
         ) -> pd.DataFrame:
    """Correlation within a dataframe.

    Args:
        df (DataFrame): input dataframe.
        method (str, optional): method name. Defaults to 'spearman'.

    Returns:
        DataFrame: output dataframe.
    """
    
    df1=df.corr(method=method).rd.fill_diagonal(filler=np.nan)
    df2=df1.melt(ignore_index=False)
    col=df2.index.name
    df2=df2.rename(columns={col:col+'2'}).reset_index().rename(columns={col:col+'1','value':f'$r_{method[0]}$'})
    return df2


def corrdf(df1: pd.DataFrame,
           colindex: str,
           colsample: str,
           colvalue: str,
           colgroupby=None,
           min_samples=1,
           fast=False,
           drop_diagonal=True,
           drop_duplicates=True,
           **kws) -> pd.DataFrame:
    """Correlate within a dataframe.

    Args:
        df1 (DataFrame): input dataframe.
        colindex (str): index column.
        colsample (str): column with samples.
        colvalue (str): column with the values.
        colgroupby (str, optional): column with the groups. Defaults to None.
        min_samples (int, optional): minimum allowed sample size. Defaults to 1.
        fast (bool, optional): use parallel processing. Defaults to False.
        drop_diagonal (bool, optional): drop values at the diagonal of the output. Defaults to True.
        drop_duplicates (bool, optional): drop duplicate values. Defaults to True.

    Keyword arguments:
        kws: parameters provided to `corr_within` function.

    Returns:
        DataFrame: output dataframe.
    """
    
    if df1[colsample].nunique()<min_samples:
        logging.info(f"not enough samples in {colsample} ({df1[colsample].nunique()})")
        return     
    if colgroupby is None:
        df2=corr_within(
            df1.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )
    else:
        df2=getattr(df1.groupby(colgroupby),'progress_apply' if not fast else 'parallel_apply')(lambda df: corr_within(
            df.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )).rd.clean().reset_index(0)
    if drop_diagonal:
        df2=df2.loc[(df2['{colsample}1']!=df2['{colsample}2']),:]
    if drop_duplicates:
        df2=df2.drop_duplicates(subset=colgroupby)
    return df2


def corr_between(df1: pd.DataFrame,df2: pd.DataFrame,method: str) -> pd.DataFrame:
    """Correlate between dataframes.

    Args:
        df1 (DataFrame): pd.DataFrame #1. Its columns are mapped to the columns of the output matrix.
        df2 (DataFrame): pd.DataFrame #2. Its columns are mapped to the rows of the output matrix.
        method (methodname): method name

    Returns:
        DataFrame: correlation matrix.
    """
    from roux.lib.set import list2intersection
    index_common=list2intersection([df1.index.tolist(),df2.index.tolist()])
    # get the common indices with loc. it sorts the dfs too.
    df1=df1.loc[index_common,:]
    df2=df2.loc[index_common,:]
    
    dcorr=pd.DataFrame(columns=df1.columns,index=df2.columns)
    dpval=pd.DataFrame(columns=df1.columns,index=df2.columns)
    from tqdm import tqdm
    for c1 in tqdm(df1.columns):
        for c2 in df2:
            if method=='spearman':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=get_spearmanr(df1[c1],df2[c2],)
            elif method=='pearson':
                dcorr.loc[c2,c1],dpval.loc[c2,c1]=get_pearsonr(df1[c1],df2[c2],)                
    dn2df={f'$r_{method[0]}$':dcorr,
          f'P ($r_{method[0]}$)':dpval,}
    dn2df={k:dn2df[k].melt(ignore_index=False) for k in dn2df}
    df3=pd.concat(dn2df,
             axis=0,
             )
    df3=df3.rename(columns={df1.columns.name:df1.columns.name+'2'}
              ).reset_index(1).rename(columns={df1.columns.name:df1.columns.name+'1'})
    df3.index.name='variable correlation'
    return df3.reset_index()


def corrdfs(df1: pd.DataFrame,df2: pd.DataFrame,
           colindex: str,
           colsample: str,
           colvalue: str,
           colgroupby=None,
           min_samples=1,
           **kws):
    """Correlate between dataframes.

    Args:
        df1 (DataFrame): input dataframe.
        colindex (str): index column.
        colsample (str): column with samples.
        colvalue (str): column with the values.
        colgroupby (str, optional): column with the groups. Defaults to None.
        min_samples (int, optional): minimum allowed sample size. Defaults to 1.

    Keyword arguments:
        kws: parameters provided to `corr_between` function.

    Returns:
        DataFrame: output dataframe.
    """
    if len(df1[colsample].unique())<min_samples or len(df2[colsample].unique())<min_samples:
        return
    if colgroupby is None:
        df3=corr_between(
            df1.pivot(index=colindex,columns=colsample,values=colvalue),
            df2.pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )
    else:
        df3=df1.groupby(colgroupby).apply(lambda df: corr_between(
            df.pivot(index=colindex,columns=colsample,values=colvalue),
            df2.loc[(df2[colgroupby]==df.name),:].pivot(index=colindex,columns=colsample,values=colvalue),
            **kws,
            )).reset_index(0)
    return df3

## partial 
def get_partial_corrs(df: pd.DataFrame,xs: list,ys: list, method='spearman',splits=5) -> pd.DataFrame:
    """Get partial correlations.

    Args:
        df (DataFrame): input dataframe.
        xs (list): columns used as x variables.
        ys (list): columns used as y variables.
        method (str, optional): method name. Defaults to 'spearman'.
        splits (int, optional): number of splits. Defaults to 5.

    Returns:
        DataFrame: output dataframe.
    """
    import pingouin as pg
    import itertools
    chunks=np.array_split(df.sample(frac=1,random_state=88),splits)
    dn2df={}
    for chunki,chunk in enumerate(chunks):
        dn2df_={}
        for x,y in list(itertools.product(xs,ys)):
            if (x if isinstance(x,str) else x[0])!=(y if isinstance(y,str) else y[0]): 
                params=dict(
                        x=(x if isinstance(x,str) else x[0]),x_covar=(None if isinstance(x,str) else x[1:]), 
                        y=(y if isinstance(y,str) else y[0]),y_covar=(None if isinstance(y,str) else y[1:]), )
                label=str(params)#+(x if isinstance(x,str) else f"{x[0]} (corrected for {' '.join(x[1:])})")+" versus "+(y if isinstance(y,str) else f"{y[0]} (corrected for {' '.join(y[1:])})")
                dn2df_[label]=pg.partial_corr(data=chunk, 
                                        tail='two-sided', method=method,
                                         **params)
#                 print(dn2df_[label])
#                 print(params)
                for k in params:
                    dn2df_[label].loc[method,k]=params[k] if isinstance(params[k],str) else str(params[k])
        dn2df[chunki]=pd.concat(dn2df_,axis=0)
    df1=pd.concat(dn2df,axis=0)
    df1.index.names=['chunk #','correlation name','correlation method']
    for c in ['x','y']:
        if f"{c}_covar" in df1:
            df1[f"{c}_covar"]=df1[f"{c}_covar"].apply(eval)
    return df1.reset_index()


def check_collinearity(df3: pd.DataFrame,threshold: float) -> pd.DataFrame:
    """Check collinearity.

    Args:
        df3 (DataFrame): input dataframe.
        threshold (float): minimum threshold for the colinearity.

    Returns:
        DataFrame: output dataframe.
    
    TODOs:
        1. Calculate variance inflation factor (VIF).
    """
    df4=df3.corr(method='spearman')
    # df4=df4.applymap(abs)
    from roux.lib.dfs import get_offdiagonal_values
    df5=get_offdiagonal_values(df4.copy())
    df6=df5.melt(ignore_index=False).dropna().reset_index()
    df6['value']=df6['value'].apply(abs)
    df6['is collinear']=df6['value'].apply(lambda x: x>=threshold)
    perc=(df6['is collinear'].sum()/len(df6))*100
    logging.info(f"% collinear vars: {perc} ({df6['is collinear'].sum()}/{len(df3.columns)})")
    if perc==0:
        logging.info(f"max corr={df6['value'].max()}")
        return
    df6=df6.loc[(df6['is collinear']),:]
    from roux.stat.network import get_subgraphs
    df7=get_subgraphs(df6.loc[df6['is collinear'],:],'index','variable')
    df7=df7.groupby('subnetwork name').agg({'node name':list}).reset_index()
    return df7.groupby('subnetwork name').progress_apply(lambda df: df6.apply(lambda x: x['value'] if len(set([x['index'],x['variable']]) - set(df['node name'].tolist()[0]))==0 else np.nan,axis=1).min()).sort_values(ascending=False)

def pairwise_chi2(df1: pd.DataFrame,cols_values: list) -> pd.DataFrame:
    """Pairwise chi2 test.

    Args:
        df1 (DataFrame): pd.DataFrame
        cols_values (list): list of columns.

    Returns:
        DataFrame: output dataframe.
    """
    import itertools
    d1={}
    for cols in list(itertools.combinations(cols_values,2)):
        _,d1[cols],_,_=sc.stats.chi2_contingency(pd.crosstab(df1[cols[0]],df1[cols[1]]))

    df2=pd.Series(d1).to_frame('P')
    df2.index.names=['value1','value2']
    df2=df2.reset_index()
    return df2

import pandas as pd
import numpy as np
import scipy as sc
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
    if not base is None:
        return np.log(x+p)/np.log(base)
    else:
        return np.log(x+p)

def anti_plog(x, p: float, base: int):
    """Anti-psudo-log.

    Args:
        x (float|np.array): input.
        p (float): pseudo-count.
        base (int): base of the log.

    Returns:
        output.
    """
    return (base**x)-p
    
def log_pval(x):
    """Transform p-values to Log10.

    Paramters: 
        x: input.

    Returns:
        output.
    """ 
    if not isinstance(x,(int,float)): 
        if any(x==0):
            x=x.replace(0,x.replace(0,np.nan).min())
            logging.warning('zeros found, replaced with min pval')
    return -1*(np.log10(x))

def glog(x: float,l = 2):
    """Generalised logarithm.

    Args:
        x (float): input.
        l (int, optional): psudo-count. Defaults to 2.

    Returns:
        float: output.
    """
    return np.log((x+np.sqrt(x**2+l**2))/2)/np.log(l)

def rescale(a: np.array, range1: tuple=None, range2: tuple=[0,1]) -> np.array:
    """Rescale within a new range.

    Args:
        a (np.array): input vector.
        range1 (tuple, optional): existing range. Defaults to None.
        range2 (tuple, optional): new range. Defaults to [0,1].

    Returns:
        np.array: output.
    """
    if range1 is None:
        range1=[np.min(a),np.max(a)]
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (a - range1[0]) / delta1) + range2[0]

def rescale_divergent(df1: pd.DataFrame,col: str,
#                       rank=True,
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
    def apply_(df2,
#                sign=None,
              ):
        sign=df2.name
#         from roux.stat.transform import rescale
        df2[f'{col} rescaled']=rescale(df2[col],range2=[0, 1] if sign=='+' else [-1,0])
        df2[f'{col} rank']=df2[col].rank(ascending=True if sign=='+' else False)*(1 if sign=='+' else -1)
        return df2
    assert(not any(df1[col]==0))
    df1.loc[df1[col]<0,f'{col} sign']='-'
    df1.loc[df1[col]>0,f'{col} sign']='+'
#     return pd.concat({k:apply_(df1.loc[(df1[f'{col} sign']==k),:],k) for k in df1[f'{col} sign'].unique()},
#                     axis=0)
    return df1.groupby([f'{col} sign']).apply(lambda df: apply_(df))
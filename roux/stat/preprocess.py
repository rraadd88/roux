"""For classification."""
## logging
import logging
## data
import numpy as np
import pandas as pd
## internal
import roux.lib.dfs as rd

# curate data 
def drop_low_complexity(
    df1: pd.DataFrame,
    min_nunique: int,
    max_inflation: int,
    max_nunique:int=None,
    cols: list=None,
    cols_keep: list=[],
    test: bool=False,
    verbose: bool=False,
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
        cols=df1.columns.tolist()
    if len(cols)<2:
        logging.warning('skipped `drop_low_complexity` because len(cols)<2.')
        return df1
    df_=pd.concat([df1.rd.check_nunique(cols),df1.rd.check_inflation(cols)],axis=1,)
    df_.columns=['nunique','% inflation']
    if verbose:
        logging.info(df_)
    df_=df_.sort_values(df_.columns.tolist(),ascending=False)
    df1_=df_.loc[((df_['nunique']<=min_nunique) | (df_['% inflation']>=max_inflation)),:]
    l1=df1_.index.tolist()
    logging.info(df1_)
    if not max_nunique is None:
        df2_=df_.loc[(df_['nunique']>max_nunique),:]
        l1+=df2_.index.tolist()
        logging.info(df2_)
    
#     def apply_(x,df1,min_nunique,max_inflation):
#         ds1=x.value_counts()
#         return (len(ds1)<=min_nunique) or ((ds1.values[0]/len(df1))>=max_inflation)
#     l1=df1.loc[:,cols].apply(lambda x: apply_(x,df1,min_nunique=min_nunique,max_inflation=max_inflation)).loc[lambda x: x].index.tolist()
    logging.info(f"{len(l1)}(/{len(cols)}) columns {'could be ' if test else ''}dropped:")
    if len(cols_keep)!=0:
        assert all([c in df1 for c in cols_keep]), ([c for c in cols_keep if not c in df1])
        cols_kept=[c for c in l1 if c in cols_keep]
        logging.info(cols_kept)
        l1=[c for c in l1 if not c in cols_keep]
        
    return df1.log.drop(labels=l1,axis=1)

def get_cvsplits(
    X: np.array,
    y: np.array,
    cv: int=5,
    random_state: int=None,
    outtest: bool=True
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
    if random_state is None: logging.warning(f"random_state is None")
    X.index=range(len(X))
    y.index=range(len(y))
    
    from sklearn.model_selection import KFold
    cv = KFold(n_splits=cv,random_state=random_state,shuffle=True)
    cv2Xy={}
    for i, (train ,test) in enumerate(cv.split(X.index)):
        dtype2index=dict(zip(('train' ,'test'),(train ,test)))
        cv2Xy[i]={}
        if outtest:
            for dtype in dtype2index:
                cv2Xy[i][dtype]={}
                cv2Xy[i][dtype]['X' if isinstance(X,pd.DataFrame) else 'x']=X.iloc[dtype2index[dtype],:] if isinstance(X,pd.DataFrame) else X[dtype2index[dtype]]
                cv2Xy[i][dtype]['y']=y[dtype2index[dtype]]
        else:
            cv2Xy[i]['X' if isinstance(X,pd.DataFrame) else 'x']=X.iloc[dtype2index['train'],:] if isinstance(X,pd.DataFrame) else X[dtype2index['train']]
            cv2Xy[i]['y']=y[dtype2index['train']]                
    return cv2Xy
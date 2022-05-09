"""For processing multiple pandas DataFrames/Series"""
from roux.lib.df import *
from roux.lib import to_rd
        
def filter_dfs(dfs,cols,how='inner'):
    """Filter dataframes based items in the common columns.

    Parameters:
        dfs (list): list of dataframes.
        cols (list): list of columns.
        how (str): how to filter ('inner')
    
    Returns
        dfs (list): list of dataframes.        
    """
    def apply_(dfs,col,how):
        from roux.lib.set import list2intersection,list2union
        if how=='inner':
            l=list(list2intersection([df[col].tolist() for df in dfs]))
        elif how=='outer':
            l=list(list2union([df[col].tolist() for df in dfs]))
        else:
            raise ValueError("how")
        logging.info(f"len({col})={len(l)}")
        return [df.loc[(df[col].isin(l)),:] for df in dfs]
    if isinstance(cols,str):
        cols=[cols]
    # sort columns by nunique
    cols=dfs[0].loc[:,cols].nunique().sort_values().index.tolist()
    for c in cols:
        dfs=apply_(dfs=dfs,col=c,how=how)
    return dfs

@to_rd
def merge_paired(
    df1,df2,
    left_ons, # suffixed
    right_on, # to be suffixed
    common=[], # not suffixed
    right_ons_common=[], # not to be suffixed
    how='inner',
    validates=['1:1','1:1'],
    suffixes=None,
    test=False,
    verb=True,
    **kws,
    ):
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
    if isinstance(right_on,str):
        right_on=[right_on]
    if isinstance(right_ons_common,str):
        right_ons_common=[right_ons_common]
    if isinstance(common,str):
        common=[common]
        
    if isinstance(left_ons[0],list):
        logging.erorr(f'groupby on common index instead')
        return 
    if suffixes is None:
        from roux.lib.str import get_suffix
        suffixes=get_suffix(*left_ons,common=False, clean=True)
        suffixes=[f" {s}" for s in suffixes]
    d1={}
    d1['from']=df1.shape
    
    def apply_(df2,cols_on,suffix,test=False):
        if len(cols_on)!=0:
            df2=df2.set_index(cols_on)
        df2=df2.add_suffix(suffix)
        if len(cols_on)!=0:
            df2=df2.reset_index()
        if test: print(df2.columns)
        return df2
    df3=df1.copy()
    for i,(suffix,validate) in enumerate(zip(suffixes,validates)):
        if test:
            print(df3.columns.tolist())
            print([f"{s}{suffix}" for s in right_on]+right_ons_common)
        df3=df3.merge(
        right=apply_(df2,common+right_ons_common,suffix,test=test),
        on=[f"{s}{suffix}" for s in right_on]+common+(right_ons_common if i==1 else []),
        how=how,
        validate=validate,
        **kws,
        )
    d1['to']=df3.shape        
    if verb:log_shape_change(d1)        
    return df3

## append

## merge dfs
def merge_dfs(dfs,
             **kws):
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
    if kws['on']!='outer': logging.warning("Drop-outs may occur if on!='outer'. Make sure that the dataframes are ordered properly from left to right.")
    from functools import reduce
    logging.info(f"merge_dfs: shape changed from : dfs shape={[df.shape for df in dfs]}")
    df3=reduce(lambda df1,df2: pd.merge(df1,df2,**kws), dfs)
    logging.info(f"merge_dfs: shape changed to   : {df3.shape}")
    return df3

def compare_rows(df1,df2,
                 test=False,
                 **kws,):
    cols=list(set(df1.columns.tolist()) & set(df1.columns.tolist()))
    if test: info(cols)
    cols_sort=list(set(df1.select_dtypes(object).columns.tolist()) & set(cols))
    if test: info(cols_sort)
    if len(df1)==len(df2):
        return df1.loc[:,cols].sort_values(cols_sort).reset_index(drop=True).compare(
        df2.loc[:,cols].sort_values(cols_sort).reset_index(drop=True),
        keep_equal=True,
            **kws
        )#.rd.assert_no_na()
    else:
        logging.warning(f'unequal lengths: {len(df1)}!={len(df2)}')
        df_=df1.loc[:,cols].merge(right=df1.loc[:,cols],
                                    on=cols_sort,
                                    how='outer',
                                    indicator=True)
        logging.info(df_['_merge'].value_counts())
        return df_.loc[(df_['_merge']!='both'),:]
"""
io_df -> io_dfs -> io_files

dtypes
'b'       boolean
'i'       (signed) integer
'u'       unsigned integer
'f'       floating-point
'c'       complex-floating point
'O'       (Python) objects
'S', 'a'  (byte-)string
'U'       Unicode
'V'       raw data (void)
"""
from roux.lib.df import *
from roux.lib import to_rd
        
def filter_dfs(dfs,cols,how='inner'):
    """
    
    """
    def apply_(dfs,col,how):
        from roux.lib.set import list2intersection,list2union
        if how=='inner':
            l=list(list2intersection([df[col].tolist() for df in dfs]))
        elif how=='outer':
            l=list(list2union([df[col].tolist() for df in dfs]))
        else:
            raise ValueError("")
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
def merge_paired(df1,df2,
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
    """
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
             **params_merge):
    from functools import reduce
    logging.info(f"merge_dfs: shape changed from : dfs shape={[df.shape for df in dfs]}")
    df3=reduce(lambda df1,df2: pd.merge(df1,df2,**params_merge), dfs)
    logging.info(f"merge_dfs: shape changed to   : {df3.shape}")
    return df3

def merge_dfs_auto(dfs,how='left',suffixes=['','_'],
              test=False,fast=False,drop_duplicates=True,
              sort=True,
              **params_merge):
    """
    
    """
    from roux.lib.set import list2intersection,flatten
    if isinstance(dfs,dict):
        dfs=list(dfs.values())
    if all([isinstance(df,str) for df in dfs]):
        dfs=[read_table(p) for p in dfs]
    if not 'on' in params_merge:
        params_merge['on']=list(list2intersection([df.columns for df in dfs]))
        if len(params_merge['on'])==0:
            logging.error('no common columns found for infer params_merge[on]')
            return
    else:
        if isinstance(params_merge['on'],str):
            params_merge['on']=[params_merge['on']]
    params_merge['how']=how
    params_merge['suffixes']=suffixes
    # sort largest first
    if test:
        logging.info(params_merge)
        d={dfi:[len(df)] for dfi,df in enumerate(dfs)}
        logging.info(f'size: {d}')
    dfi2cols_value={dfi:df.select_dtypes([int,float]).columns.tolist() for dfi,df in enumerate(dfs)}
    cols_common=list(np.unique(params_merge['on']+list(list2intersection(dfi2cols_value.values()))))
    dfi2cols_value={k:list(set(dfi2cols_value[k]).difference(cols_common)) for k in dfi2cols_value}
    dfis_duplicates=[dfi for dfi in dfi2cols_value if len(dfs[dfi])!=len(dfs[dfi].loc[:,cols_common].drop_duplicates())]
    if test:
        logging.info(f'cols_common: {cols_common}',)
        logging.info(f'dfi2cols_value: {dfi2cols_value}',)
        logging.info(f'duplicates in dfs: {dfis_duplicates}',)
    for dfi in dfi2cols_value:
        if (dfi in dfis_duplicates) and drop_duplicates:
            dfs[dfi]=drop_duplicates_by_agg(dfs[dfi],cols_common,dfi2cols_value[dfi],fast=fast)
    if sort:
        d={dfi:[len(df)] for dfi,df in enumerate(dfs)}
        logging.info(f"size agg: {d}")
        from roux.lib.dict import sort_dict
        sorted_indices_by_size=sort_dict({dfi:[len(df.drop_duplicates(subset=params_merge['on']))] for dfi,df in enumerate(dfs)},0)
        logging.info(f'size dedup: {sorted_indices_by_size}')
        sorted_indices_by_size=list(sorted_indices_by_size.keys())#[::-1]
        dfs=[dfs[i] for i in sorted_indices_by_size]
#     from functools import reduce
#     df1=reduce(lambda df1,df2: pd.merge(df1,df2,**params_merge), dfs)
    df1=merge_dfs(dfs,**params_merge)
    cols_std=[f"{c} var" for c in flatten(list(dfi2cols_value.values())) if f"{c} var" in df1]
    cols_del=[c for c in cols_std if df1[c].isnull().all()]
    df1=df1.drop(cols_del,axis=1)
    return df1

def merge_subset(df,colsubset,subset,cols_value,
                          on,how='left',suffixes=['','.1'],
                          **kws_merge):
    """
    merge a subset from a linear df, sideways
    """
    if isinstance(on,str): on=[on]
    return df.loc[(df[colsubset]!=subset),:].merge(
                                            df.loc[(df[colsubset]==subset),on+cols_value],
                                          on=on,
                                          how=how, 
                                        suffixes=suffixes,
                                          **kws_merge,
                                            )



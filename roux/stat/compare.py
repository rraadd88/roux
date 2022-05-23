from roux.lib.df import *
import logging

def get_cols_x_for_comparison(
    df1: pd.DataFrame,
    cols_y: list,
    cols_index: list,
    cols_drop=[],
    cols_dropby_patterns=[],
    ) -> dict:
    """
    Identify X columns.
    
    """
    ## drop columns
    df1=(df1
    .drop(cols_drop,axis=1)
    .rd.dropby_patterns(cols_dropby_patterns)
    .log.dropna(how='all',axis=1)
    )

    ## drop single value columns
    drop_cols=df1.rd.check_nunique().loc[lambda x: x==1].index.tolist()#+df1.filter(like='coexp').columns.tolist()
    df1=df1.drop(drop_cols,axis=1)
    
    ## make the dictionary with column names
    d0=dict(cols_y={
            'cont':df1.loc[:,cols_y].select_dtypes((int,float)).columns.tolist(),
            },
            cols_index=cols_index,
        )
    d0['cols_y']['desc']=list(set(cols_y) - set(d0['cols_y']['cont']))
    d0['cols_x']={}
    
    ## get continuous cols_x
    from roux.lib.stat.classify import drop_low_complexity
    df_=drop_low_complexity(df1=df1,
                        min_nunique=5,
                        max_inflation=50,
                        cols=df1.select_dtypes((int,float)).columns.tolist(),
                        cols_keep=d0['cols_y']['cont'],
                       )
    d0['cols_x']['cont']=df_.drop(d0['cols_y']['cont'],axis=1).select_dtypes((int,float)).columns.tolist()
    
    ## get non-colinear ones 
    from roux.lib.stat.corr import check_collinearity
    check_collinearity(
        df1=df1.loc[:,d0['cols_x']['cont']],
        threshold=0.7,
        colvalue='$r_s$',
        cols_variable=['variable1','variable2'],
        coff_pval=0.05,)
    
    ## get descrete x columns
    ds_=df1.rd.check_nunique().sort_values()
    l1=ds_.loc[lambda x: (x==2)].index.tolist()
    logging.info(l1)
    ds_=df1.select_dtypes((int,float)).nunique().sort_values()
    l2=ds_.loc[lambda x: (x==2)].index.tolist()
    logging.info(l2)
    d0['cols_x']['desc']=sorted(list(set(l1+l2) - set(d0['cols_y']['desc'])))
    return d0

def get_comparison(
    df1: pd.DataFrame,
    cols_y: list,
    cols_index,
    coff_p: float=0.05,
    **kws,
    ):
    """
    Compare the x and y columns.
    
    Notes:
        Column types:
            cols_x: decrete and continuous
            cols_y: decrete and continuous

        Comparison types:
            1. continuous vs continuous -> correlation
            2. decrete vs continuous -> difference
            3. decrete vs decrete -> FE or chi square
    """
    ## 
    d1=get_cols_x_for_comparison(
    df1,
    cols_y,
    cols_index,
    **kws
    )
    from roux.lib.set import get_alt
    
    ## gather stats in a dictionary
    d2={}
    
    ## 1. correlations 
    from roux.stat.corr import get_corrs
    d2['correlation x vs y']=get_corrs(df1=df1,#.drop(['loci distance'],axis=1),
    method='spearman',
    cols=d1['cols_y']['cont'],
    cols_with=d1['cols_x']['cont'],#list(set(df1.columns.tolist())-set(['protein abundance difference (DELTA-WT)','responsiveness'])),
    # cols_with=['loci distance'],
                  coff_inflation_min=50,
             )
    if isinstance(coff_p,float):
        print(d2.keys())
        d2['correlation x vs y']=d2['correlation x vs y'].loc[(d2['correlation x vs y']['P']<0.05),:]
    
    ## 2. difference 
    from roux.stat.diff import get_diff
    for k in ['x','y']:
        d2[f"difference {k} vs {get_alt(['x','y'],k)}"]=get_diff(df1,
             cols_x=d1[f"cols_{k}"]['desc'],
             cols_y=d1[f"cols_{get_alt(['x','y'],k)}"]['cont'],
             cols_index=d1['cols_index'],
            )
        if isinstance(coff_p,float):
            d2[f"difference {k} vs {get_alt(['x','y'],k)}"]=(d2[f"difference {k} vs {get_alt(['x','y'],k)}"]
                                                            .loc[(d2[f"difference {k} vs {get_alt(['x','y'],k)}"]['P (MWU test)']<coff_p),:]
                                                            .sort_values('P (MWU test)')
                                                            )
    
    ## 3. association
    # df0=pd.DataFrame(itertools.product(d1['cols_y']['desc'],d1['cols_x']['desc'],)).rename(columns={0:'colx',1:'coly'},errors='raise')
    # df0.head(1)
    # from roux.lib.stat.diff import compare_classes
    # d2['association x vs y']=df0.join(df0.apply(lambda x: compare_classes(df1[x['colx']], df1[x['coly']]),axis=1).apply(pd.Series).rename(columns={0:'stat',1:'P'},errors='raise'))
    from roux.lib.stat.diff import compare_classes_many
    d2['association x vs y']=compare_classes_many(
        df1=df1,
        cols_y=d1['cols_y']['desc'],
        cols_x=d1['cols_x']['desc'],
        )
    if isinstance(coff_p,float):
        d2['association x vs y']=d2['association x vs y'].loc[(d2['association x vs y']['P']<coff_p),:].sort_values('P')
        
    ## rename to uniform column names
    d2['correlation x vs y']=(d2['correlation x vs y']
                            .rename(columns={'variable1':'variable x',
                                'variable2':'variable y',
                                '$r_s$':'stat',},
                                errors='raise')
                             .assign(**{'stat type':'$r_s$'})
                             )
    for k in ['x vs y','y vs x']:
        d2[f'difference {k}']=(d2[f'difference {k}']
                                .rename(columns={
                                    'P (MWU test)':'P',
                                    'stat (MWU test)':'stat',},
                                    errors='raise')
                                 .assign(**{'stat type':'MWU'})
                                 )
    d2['association x vs y']=(d2['association x vs y']
                            .rename(columns={'colx':'variable x',
                                'coly':'variable y',
                                            },
                                errors='raise')
                             .assign(**{'stat type':'FE'})
                             )
    ## gather
    return pd.concat(d2,axis=0,names=['comparison type'],
             ).reset_index(0)    

from roux.lib.df import *
import logging

def get_cols_x_for_comparison(
    df1: pd.DataFrame,
    cols_y: list,
    cols_index: list,
    cols_drop=[],
    cols_dropby_patterns=[],
    test=False,
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
    from roux.stat.classify import drop_low_complexity
    df_=drop_low_complexity(df1=df1,
                        min_nunique=5,
                        max_inflation=50,
                        cols=df1.select_dtypes((int,float)).columns.tolist(),
                        cols_keep=d0['cols_y']['cont'],
                       )
    d0['cols_x']['cont']=df_.drop(d0['cols_y']['cont'],axis=1).select_dtypes((int,float)).columns.tolist()
    
    ## get non-colinear ones 
    from roux.stat.corr import check_collinearity
    check_collinearity(
        df1=df1.loc[:,d0['cols_x']['cont']],
        threshold=0.7,
        colvalue='$r_s$',
        cols_variable=['variable1','variable2'],
        coff_pval=0.05,)
    
    ## get descrete x columns
    ds_=df1.rd.check_nunique().sort_values()
    l1=ds_.loc[lambda x: (x==2)].index.tolist()
    if test: print('l1',l1)
    
    ds_=df1.select_dtypes((int,float)).nunique().sort_values()
    l2=ds_.loc[lambda x: (x==2)].index.tolist()
    if test: print('l2',l2)
    
    d0['cols_x']['desc']=sorted(list(set(l1+l2) - set(d0['cols_y']['desc'])))
    return d0

def get_comparison(
    df1: pd.DataFrame,
    d1: dict=None,
    coff_p: float=0.05,
    between_ys: bool=False,
    **kws,
    ):
    """
    Compare the x and y columns.
    
    Parameters:
        df1 (pd.DataFrame): input table.
        d1 (dict): columns dict, output of `get_cols_x_for_comparison`.  
        between_ys (bool): compare y's
    Notes:
        Column types:
            cols_x: decrete and continuous
            cols_y: decrete and continuous

        Comparison types:
            1. continuous vs continuous -> correlation
            2. decrete vs continuous -> difference
            3. decrete vs decrete -> FE or chi square
    """
    # ## 
    if d1 is None:
        d1=get_cols_x_for_comparison(
            df1,
            cols_y,
            cols_index,
            **kws,
            )
    
    ## gather stats in a dictionary
    if between_ys:
        for dtype in ['desc','cont']:
            d1['cols_x'][dtype]=list(np.unique(d1['cols_x'][dtype]+d1['cols_y'][dtype]))
        
    from roux.lib.set import get_alt
    d2={}
    ## 1. correlations 
    if len(d1['cols_y']['cont'])!=0 and len(d1['cols_x']['cont'])!=0:
        from roux.stat.corr import get_corrs
        d2['correlation x vs y']=get_corrs(df1=df1,
            method='spearman',
            cols=d1['cols_y']['cont'],
            cols_with=d1['cols_x']['cont'],
            coff_inflation_min=50,
                 )
    
    ## 2. difference 
    from roux.stat.diff import get_diff
    for k in ['x','y']:
        if len(d1[f"cols_{k}"]['desc'])!=0 and len(d1[f"cols_{get_alt(['x','y'],k)}"]['cont'])!=0:
            d2[f"difference {k} vs {get_alt(['x','y'],k)}"]=get_diff(df1,
                                                                     cols_x=d1[f"cols_{k}"]['desc'],
                                                                     cols_y=d1[f"cols_{get_alt(['x','y'],k)}"]['cont'],
                                                                     cols_index=d1['cols_index'],
                                                                     cols=['variable x','variable y'],
                                                                     coff_p=coff_p,
                                                                )
        else:
            logging.warning(f"not len(d1[f'cols_{k}']['desc'])!=0 and len(d1[f'cols_{get_alt(['x','y'],k)}']['cont'])!=0")
    ## 3. association
    if len(d1["cols_x"]['desc'])!=0 and len(d1["cols_y"]['desc'])!=0:
        from roux.stat.diff import compare_classes_many
        d2['association x vs y']=compare_classes_many(
            df1=df1,
            cols_y=d1['cols_y']['desc'],
            cols_x=d1['cols_x']['desc'],
            )
        
    ## rename to uniform column names
    if 'correlation x vs y' in d2:
        d2['correlation x vs y']=(d2['correlation x vs y']
                                .rename(columns={'variable1':'variable x',
                                    'variable2':'variable y',
                                    '$r_s$':'stat',},
                                    errors='raise')
                                 .assign(**{'stat type':'$r_s$'})
                                 )
    for k in ['x vs y','y vs x']:
        if f'difference {k}' in d2:
            d2[f'difference {k}']=(d2[f'difference {k}']
                                    .rename(columns={
                                        'P (MWU test)':'P',
                                        'stat (MWU test)':'stat',},
                                        errors='raise')
                                     .assign(**{'stat type':'MWU'})
                                     )
    if 'association x vs y' in d2:
        d2['association x vs y']=(d2['association x vs y']
                                .rename(columns={'colx':'variable x',
                                    'coly':'variable y',
                                                },
                                    errors='raise')
                                 .assign(**{'stat type':'FE'})
                                 )
    if not coff_p is None:
        for k in d2:
            d2[k]=(d2[k]
                    .loc[(d2[k]['P']<0.05),:]
                  )    
    ## gather
    df2=(pd.concat(
                d2,
                axis=0,
                names=['comparison type'],
         )
         .reset_index(0)
         .sort_values('P')
         .log.query(expr="`variable x` != `variable y`")
        )
    return df2

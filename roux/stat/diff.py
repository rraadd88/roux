import pandas as pd
import numpy as np
import scipy as sc
import logging
from icecream import ic as info
from roux.lib.set import *

def get_subset2metrics(df,colvalue,colsubset,colindex,outstr=False,subset_control=None):
    if subset_control is None:
        subset_control=df[colsubset].unique().tolist()[-1]
    from scipy.stats import mannwhitneyu    
    df1=df.merge(df.loc[(df[colsubset]==subset_control),:],on=colindex, 
                 how='left',
                 suffixes=['',' reference'],
                )
    subset2metrics=df1.groupby(colsubset).apply(lambda df : mannwhitneyu(df[colvalue],df[f"{colvalue} reference"],
                                                     alternative='two-sided')).apply(pd.Series)[1].to_dict()
    if subset2metrics[subset_control]<0.9:
        logging.warning(f"pval for reference condition vs reference condition = {subset2metrics[subset_control]}. shouldn't be. check colindex")
    subset2metrics={k:subset2metrics[k] for k in subset2metrics if k!=subset_control}
    if outstr:
        from roux.viz.annot import pval2annot
        subset2metrics={k: pval2annot(subset2metrics[k],
                                      fmt='<',
                                      alternative='two-sided',
                                      linebreak=False) for k in subset2metrics}
    return subset2metrics

def get_ratio_sorted(a,b,increase=True):
    l=sorted([a,b])
    if not increase:
        l=l[::-1]
    if l[0]!=0 and l[1]!=0:
        return l[1]/l[0]

def diff(a,b,absolute=True): 
    diff=a-b
    if absolute:
        return abs(diff)
    else:
        return diff
def get_diff_sorted(a,b): return diff(a,b,absolute=True)

def balance(a,b,absolute=True):
    sum_=a+b
    if sum_!=0:
        return 1-(diff(a,b,absolute=absolute)/(sum_))
    else:
        return np.nan
    
def get_col2metrics(df,colxs,coly,method='mannwhitneyu',alternative='two-sided'):
    """
    coly: two values
    """
    from scipy import stats
    class1,class2=df[coly].unique()
    d={}
    for colx in colxs:
        _,d[colx]=getattr(stats,method)(df.loc[(df[coly]==class1),colx],
                                       df.loc[(df[coly]==class2),colx],
                                       alternative=alternative)
    return d    

## for linear dfs
def get_demo_data():
    subsets=list('abcd')
    np.random.seed(88)
    df1=pd.concat({s:pd.Series([np.random.uniform(0,si+1) for _ in range((si+1)*100)]) for si,s in enumerate(subsets)},
             axis=0).reset_index(0).rename(columns={'level_0':'subset',0:'value'})
    df1['bool']=df1['value']>df1['value'].quantile(0.5)
    return df1

def get_pval(df,
             colvalue='value',
             colsubset='subset',
             colvalue_bool=False,
             colindex=None,
             subsets=None,
            test=False):
    """
    either colsubset or subsets are needed 
    :param colvalue: dtype=bool (dtype=object not supported)
    """
    if not ((colvalue in df) and (colsubset in df)):
        logging.error(f"colvalue or colsubset not found in df: {colvalue} or {colsubset}")
    if subsets is None:
        subsets=sorted(df[colsubset].unique())
    if len(subsets)!=2:
        logging.error('need only 2 subsets')
        return
    else:
        df=df.loc[df[colsubset].isin(subsets),:]
    if colvalue_bool and not df[colvalue].dtype==bool:        
        logging.warning(f"colvalue_bool {colvalue} is not bool")
        return
    if not colvalue_bool:      
#         try:
        x,y=df.loc[(df[colsubset]==subsets[0]),colvalue].tolist(),df.loc[(df[colsubset]==subsets[1]),colvalue].tolist()
        if len(x)!=0 and len(y)!=0 and (nunique(x+y)!=1):
            return sc.stats.mannwhitneyu(x,y,alternative='two-sided')
        else:
            #if empty list: RuntimeWarning: divide by zero encountered in double_scalars  z = (bigu - meanrank) / sd
            return np.nan,np.nan
    else:
        assert(not colindex is None)
        df1=df.pivot(index=colindex,columns=colsubset,values=colvalue)
        ct=pd.crosstab(df1[subsets[0]],df1[subsets[1]])
        if ct.shape==(2,2):
            ct=ct.sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False)
            if test:
                print(ct)
            return sc.stats.fisher_exact(ct)
        else:
            return np.nan,np.nan      
        
def get_stat(df1,
              colsubset,
              colvalue,
              colindex,
              subsets=None,
              cols_subsets=['subset1', 'subset2'],
              df2=None,
              stats=[np.mean,np.median,np.var]+[len],
             coff_samples_min=None,
#               debug=False,
             verb=False,
             **kws,
             ):
    """
    Either MWU or FE
    :param df2: pairs of comparisons between subsets
    either colsubset or subsets are needed 
    """        
    if not ((colvalue in df1) and (colsubset in df1)):
        logging.error(f"colvalue or colsubset not found in df: {colvalue} or {colsubset}")
        return
    if subsets is None:
        subsets=sorted(df1[colsubset].unique())
    if len(subsets)<2:
        logging.error("len(subsets)<2")
        return
    colvalue_bool=df1[colvalue].dtype==bool
    if df2 is None:
        import itertools
        df2=pd.DataFrame([t for t in list(itertools.permutations(subsets,2))] if len(subsets)>2 else [subsets])
        df2.columns=cols_subsets
    df2=df2.groupby(cols_subsets).apply(lambda df: get_pval(df1,colvalue=colvalue,
                                                            colsubset=colsubset,
                                                            colindex=colindex,
                                                            subsets=df.name,
                                                            colvalue_bool=colvalue_bool,
                                                            **kws,
                                                           )).apply(pd.Series)
    df2=df2.rename(columns={0:f"stat ({'MWU' if not colvalue_bool else 'FE'} test)",
                            1:f"P ({'MWU' if not colvalue_bool else 'FE'} test)",
                            }).reset_index()
    from roux.lib.dfs import merge_paired
    from roux.lib.str import get_prefix,get_suffix
    colsubset_=get_prefix(*cols_subsets,common=True, clean=True)
    df_=(df1
        .groupby([colsubset])[colvalue]
        .agg(stats if not colvalue_bool else [sum,len])
        .reset_index()
             # TODOs rename to subset1 subset2
        .rename(columns={colsubset:colsubset_},errors='raise'))        
    df3=merge_paired(
        df1=df2,
        df2=df_,
        left_ons=cols_subsets,
        right_on=colsubset_,
        common=[],
        right_ons_common=[],
        how='inner',
        validates=['m:1', 'm:1'],
        suffixes=get_suffix(*cols_subsets,common=False, clean=False),
        test=False,
        verb=False,
        # **kws,
    )
    df3=df3.rename(columns={f"{c}{i}":f"{c} {colsubset_}{i}" for c in df_ for i in [1,2] if c!=colsubset_},errors='raise')
    ## minimum samples
    if not coff_samples_min is None:
        # logging.info("coff_samples_min applied")
        df3.loc[((df3[f'len {cols_subsets[0]}']<coff_samples_min) \
                 | (df3[f'len {cols_subsets[1]}']<coff_samples_min)),
                df3.filter(like="P (").columns.tolist()]=np.nan
    return df3

def get_stats(df1,
              colsubset,
              cols_value,
              colindex,
              subsets=None,
              df2=None,
              cols_subsets=['subset1', 'subset2'],
              stats=[np.mean,np.median,np.var,len],
              axis=0, # concat 
              test=False,
              **kws):
    """
    Iterate over cols_value,
    :param axis: 1 if different tests else use 0.
    :param df2: pd.DataFrame({"subset1":['test'],
                              "subset2":['reference']})
                              
    TODOS:
    no col prefix if len(cols_value)==1
    """
#     from roux.lib.dfs import to_table
#     to_table(df1,'test/get_stats.tsv')
#     stats=[s for s in stats if not s in [sum,len]]
#     from tqdm import tqdm
    dn2df={}
    if subsets is None:
        subsets=sorted(df1[colsubset].unique())
    from roux.stat.diff import get_stat ## remove?
    for colvalue in cols_value:
        df1_=df1.dropna(subset=[colsubset,colvalue])
        if len(df1_[colsubset].unique())>1:
            dn2df[colvalue]=get_stat(df1_,
                          colsubset=colsubset,
                          colvalue=colvalue,
                          colindex=colindex,
                          subsets=subsets,
                          cols_subsets=cols_subsets,
                          df2=df2,
                          stats=stats,
                          **kws,
                         ).set_index(cols_subsets)
        else:
            if test:
                logging.warning(f"not processed: {colvalue}; probably because of dropna")
    if len(dn2df.keys())==0:
        return 
    import pandas as pd # remove?
    df3=pd.concat(dn2df,
                  ignore_index=False,
                  axis=axis,
                  verify_integrity=True,
                  names=None if axis==1 else ['variable'],
                 )
    if axis==1:
        df3=df3.reset_index().rd.flatten_columns()
    return df3

def get_q(ds1,col=None,verb=True,test_coff=0.1):
    if not col is None:
        df1=ds1.copy()
        ds1=ds1[col]
    ds2=ds1.dropna()
    from statsmodels.stats.multitest import fdrcorrection
    ds3=fdrcorrection(pvals=ds2, alpha=0.05, method='indep', is_sorted=False)[1]
    ds4=ds1.map(pd.DataFrame({'P':ds2,'Q':ds3}).drop_duplicates().rd.to_dict(['P','Q']))
    if verb:
        from roux.viz.annot import perc_label        
        logging.info(f"significant at Q<{test_coff}: {perc_label(ds4<test_coff)}")
    if col is None:
        return ds4
    else:
        df1['Q']=ds4
        return df1
def get_significant_changes(df1,alpha,
                            changeby="",
                            fdr=True,
                            value_aggs=['mean','median'],
                           ):
    """
    groupby to get the comparable groups 
    # if both mean and median are high
    :param changeby: "" if check for change by both mean and median
    """    
    if df1.filter(regex='|'.join([f"{s} subset(1|2)" for s in value_aggs])).shape[1]:
        for s in value_aggs:
            df1[f'difference between {s} (subset1-subset2)']=df1[f'{s} subset1']-df1[f'{s} subset2']
        ## call change if both mean and median are changed
        df1.loc[((df1.filter(like=f'difference between {changeby}')>0).T.sum()==2),'change']='increase'
        df1.loc[((df1.filter(like=f'difference between {changeby}')<0).T.sum()==2),'change']='decrease'
        df1['change']=df1['change'].fillna('ns')
    from statsmodels.stats.multitest import multipletests
    for test in ['MWU','FE']:
        if not f'P ({test} test)' in df1:
            continue
        # without fdr
        df1[f'change is significant, P ({test} test) < {alpha}']=df1[f'P ({test} test)']<alpha
        df1[f'Q ({test} test)']=get_q(df1[f'P ({test} test)'])
        df1[f'change is significant ({test} test)']=df1[f'P ({test} test)']<alpha
        #     info(f"corrected alpha alphacSidak={alphacSidak},alphacBonf={alphacBonf}")
        # if test!='FE':
        #     df1.loc[df1[f"change is significant ({test} test)"],f"significant change ({test} test)"]=df1.loc[df1[f"change is significant ({test} test)"],'change']
        #     df1[f"significant change ({test} test)"]=df1[f"significant change ({test} test)"].fillna('ns')
    return df1

def apply_get_significant_changes(df1,cols_value,
                                    cols_groupby, # e.g. genes id
                                    cols_grouped, # e.g. tissue
                                    fast=False,
                                    **kws,
                                    ):
    d1={}
    from tqdm import tqdm
    for c in tqdm(cols_value):
        df1_=df1.set_index(cols_groupby).filter(regex=f"^{c} .*").filter(regex="^(?!("+' |'.join([s for s in cols_value if c!=s])+")).*")
        df1_=df1_.rd.renameby_replace({f'{c} ':''}).reset_index()
        d1[c]=getattr(df1_.groupby(cols_groupby),'apply' if not fast else 'parallel_apply')(lambda df: get_significant_changes(df,**kws))
    d1={k:d1[k].set_index(cols_groupby) for k in d1}
    ## to attach the info
    d1['grouped']=df1.set_index(cols_groupby).loc[:,cols_grouped]
    df2=pd.concat(d1,
                  ignore_index=False,
                  axis=1,
                  verify_integrity=True,
                 )    
    df2=df2.rd.flatten_columns().reset_index()
    assert(not df2.columns.duplicated().any())
    return df2

def get_stats_groupby(df1,cols,
                      alpha,
                      fast=False,
                      **kws):
    """
    Iterate over groups
    """
    df2=getattr(df1.groupby(cols),f"{'progress' if not fast else 'parallel'}_apply")(lambda df: get_stats(df1=df,**kws)).reset_index().rd.clean()
    return get_significant_changes(df1=df2,alpha=alpha)
    
def binby_pvalue_coffs(df1,coffs=[0.01,0.05,0.25],
                      color=False,
                       testn='MWU test, FDR corrected',
                       colindex='genes id',
                       colgroup='tissue',
                      preffix='',
                      colns=None, # plot as ns, not counted
                      palette=None,#['#f55f5f','#ababab','#046C9A',],
                      ):
    assert(len(df1)!=0)
    if palette is None:
        from roux.viz.colors import get_colors_default
        palette=get_colors_default()[:3]
    assert(len( palette)==3)
    coffs=np.array(sorted(coffs))
    # df1[f'{preffix}P (MWU test, FDR corrected) bin']=pd.cut(x=df1[f'{preffix}P (MWU test, FDR corrected)'],
    #       bins=[0]+coffs+[1],
    #        labels=coffs+[1],
    #        right=False,
    #       ).fillna(1)
    from roux.viz.colors import saturate_color
    d1={}
    for i,coff in enumerate(coffs[::-1]):
        col=f"{preffix}significant change, P ({testn}) < {coff}"
        df1[col]=df1.apply(lambda x: 'increase' if ((x[f'{preffix}P ({testn})']<coff) \
                                                    and (x[f'{preffix}difference between mean (subset1-subset2)']>0))\
                                else 'decrease' if ((x[f'{preffix}P ({testn})']<coff) \
                                                    and (x[f'{preffix}difference between mean (subset1-subset2)']<0))\
                                else 'ns', axis=1)
        if color:
            if i==0:
                df1.loc[(df1[col]=='ns'),'c']=palette[1]
            saturate=1-((len(coffs)-(i+1))/len(coffs))
            d2={}
            d2['increase']=saturate_color(palette[0],saturate)
            d2['decrease']=saturate_color(palette[2],saturate)
            d1[coff]=d2
            df1['c']=df1.apply(lambda x: d2[x[col]] if x[col] in d2 else x['c'],axis=1)
            assert(df1['c'].isnull().sum()==0)
    if color:
        import itertools
        from roux.stat.transform import rescale,log_pval
        d3={}
        for i,(k,coff) in enumerate(list(itertools.product(['increase','decrease'],coffs))):
            col=f"{preffix}significant change, P ({testn}) < {coff}"
            d4={}
            d4['y alpha']=rescale(1-(list(coffs).index(coff))/len(coffs),[0,1],[0.5,1])
            d4['y']=log_pval(coff)
            d4['y text']=f" P < {coff}"
            d4['x']=df1.loc[(df1[col]==k),f'{preffix}difference between mean (subset1-subset2)'].min() if k=='increase' \
                                        else df1.loc[(df1[col]==k),f'{preffix}difference between mean (subset1-subset2)'].max()
            d4['change']=k
            d4['text']=f"{df1.loc[(df1[col]==k),colindex].nunique()}/{df1.loc[(df1[col]==k),colgroup].nunique()}"
            d4['color']=d1[coff][k]
            d3[i]=d4
        df2=pd.DataFrame(d3).T
    if not colns is None:
        df1.loc[df1[colns],'c']=palette[1]
#     info(df1.shape,df1.shape)
    return df1,df2

# from roux.viz.diff import plot_stats_diff
## confounding effects
def get_stats_regression(df_,
                        d0={},
                        variable=None,
                        covariates=None,
                        converged_only=False,
                         out='df',
                        verb=False,
                        test=False,
                        **kws,
                        ):
    """
    :params variable: to get params of e.g. 'C(variable)[T.True]'
    :params covariates: variables
    :params d0: model name to base equation e.g. 'y ~ x'
    """
    def to_df(res):
        if isinstance(res.summary().tables[1],pd.DataFrame):
            df1=res.summary().tables[1]
        elif hasattr(res.summary().tables[1],'as_html'):
            df1=pd.read_html(res.summary().tables[1].as_html(), header=0, index_col=0)[0]
        else:
            logging.error('dataframe not found')
            return
        df1.columns.name='stat'
        df1.index.name='variable'
        return df1.melt(ignore_index=False).reset_index()
    def get_stats(res,variable):
        return pd.Series([res.pvalues[variable], 
                          res.params[variable], 
                        ], 
                        index=['P', 'coefficient',
                        ]).to_frame('value')
    if not (verb or test):
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', UserWarning)
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from numpy.linalg import LinAlgError
    if not covariates is None:
        #formats
        d1=df_.dtypes.to_dict()
        formula_covariates=' + '+' + '.join([k if ((d1[k]==int) or (d1[k]==float)) else f"C({k})" for k in covariates if k in d1])
    else:
        formula_covariates=''
    # info(formula_covariates)
    d1={}
    for k,formula_base in d0.items():
        if isinstance(k,str):
            model=getattr(smf,k)
        elif isinstance(k,object) and hasattr(k,'from_formula'):
            model=k.from_formula
        elif isinstance(k,object):
            model=k
        else:
            logging.error(model)
            return
        # print(str(model))
        formula=formula_base+formula_covariates
        if test: info(formula)
        modeln=str(model).split('.')[-1].split("'")[0]
        if not 'groups' in kws:
            try:
                d1[modeln]=model(data=df_,formula=formula).fit(disp=False)
            except (PerfectSeparationError,LinAlgError) as e:
                if verb or test: logging.error('PerfectSeparationError/LinAlgError')
        else:
            try:
                d1[modeln]=model(data=df_,formula=formula,
                                groups=df_[kws["groups"]],
                                ).fit(disp=False)
            except (PerfectSeparationError,LinAlgError) as e:
                if verb or test: logging.error('PerfectSeparationError/LinAlgError')
    if out=='model':
        return d1
    elif out=='df':
        d1={k:to_df(v) for k,v in d1.items() if ((hasattr(v,'converged') and (v.converged)) or (not converged_only))}
        if len(d1)!=0:
            if not variable is None:
                return pd.concat({k:get_stats(v,variable=variable) for k,v in d1.items()},
                                 axis=0,names=['model type','variable']).reset_index()
            else:
                return pd.concat(d1,axis=0,names=['model type']).reset_index(0)
    
def filter_regressions(df1,variable,colindex='gene id',coff_q=0.1,
                      by_covariates=True,coff_p_covariates=0.05,
                      test=False):
    pval='P>|t|' if 'P>|t|' in df1['stat'].tolist() else 'P>|z|' if 'P>|z|' in df1['stat'].tolist() else ValueError() 
    df1['stat']=df1['stat'].apply(lambda x: 'P' if x==pval else x )
    df2=df1.loc[((df1['variable']==variable) & (df1['stat'].isin(['coef','P']))),:]
#     print(df1['stat'].unique())
    df3=df2.pivot(index=colindex,columns='stat',values='value').reset_index()
#     print(df3.columns)
    df3=df3.rename(columns={'coef':'score'},
                   errors='raise')
    df3=df3.log.dropna(subset=['P'])
    if test:
        df3['P'].hist()
    from statsmodels.stats.multitest import fdrcorrection
    df3['Q']=fdrcorrection(pvals=df3['P'], alpha=0.05, method='indep', is_sorted=False)[1]
    if test:
        df3['Q'].hist()
    info(sum(df3['P']<coff_q))
    info(sum(df3['Q']<coff_q))
    df3=df3.loc[(df3['Q']<coff_q),:]
    info(df3[colindex].nunique())
    if by_covariates:
        ## #2
        ## all covariates (potential confounding effects) are non-sinificant
        ## non-standardised regression 'P>|t|' or standardised 'P>|z|'
        df5=df1.loc[(
                (df1['variable']!='Intercept') \
              & (df1['variable']!=variable) \
              & (df1['stat']=='P')
                ),:]
        df6=df5.groupby(colindex).filter(lambda df: (df['value']>=coff_p_covariates).all()).loc[:,[colindex]]
        df3=df3.log().loc[df3[colindex].isin(df6[colindex]),:].log()
    else:
        logging.warning("not filtered by_covariates")
    info(df3[colindex].nunique())
    return df3
    
from argparse import ArgumentError
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import logging
from icecream import ic as info
from roux.lib.set import *

## Correcting confounding effects
def to_columns_renamed_for_regression(
    df1:pd.DataFrame,
    columns:dict,
    ) -> pd.DataFrame:
    """
    """
    import re
    from roux.lib.str import replace_many
    ## Rename columns to be comparible with the formula
    columns['rename']={}
    rename_columns=[]
    for var_type in ['cols_x','cols_y']:
        columns['rename'][var_type]={}
        for dtype in columns[var_type]:
            # columns['rename'][var_type][dtype]={c:re.sub('[^0-9a-zA-Z]+', '_', replace_many(c,[':','(',')','=','%'],'_',ignore=True)) for c in columns[var_type][dtype]}
            columns['rename'][var_type][dtype]={c:re.sub('[^0-9a-zA-Z%]+', '_', c.replace('%','perc')) for c in columns[var_type][dtype]}
            ## for renaming dataframe
            rename_columns.append(columns['rename'][var_type][dtype])
            ## desc values to integers
            for c in columns[var_type][dtype]:
                if var_type=='cols_y' and dtype=='desc':
                    if df1[c].dtype!=float:
                        df1=df1.assign(
                            **{c:lambda df: (df[c]==columns['desc_test_values'][c]).astype(int)}
                        )
                        info(df1[c].value_counts())
                        
    from roux.lib.dict import merge_dicts
    rename_columns=merge_dicts(rename_columns)                       
    assert len(rename_columns.keys())==len(rename_columns.values())                    
    df2=(df1
    .rename(
        columns=rename_columns,
        errors='raise',
        )
    )
    return df2,columns

def to_input_data_for_regression(
    df1: pd.DataFrame,
    cols_y: list,
    cols_index: list,
    desc_test_values:dict,
    verbose: bool=False,
    test: bool=False,
    **kws,
    ) -> tuple:
    """
    Input data for the regression.
    
    Parameters:
        df1 (pd.DataFrame): input data.
        cols_y (list): y columns.
        cols_index (list): index columns.
        
    Returns:
        Output table.
    """
    ## get columns dictionary
    from roux.stat.compare import get_cols_x_for_comparison,to_preprocessed_data
    columns=get_cols_x_for_comparison(
        df1=df1,
        cols_y=cols_y,
        cols_index=cols_index,
        verbose=verbose,
        test=test,        
        **kws,
    )
    
    ## pre-process data
    df2=to_preprocessed_data(
        df1=df1,
        columns=columns,
        fill_missing_desc_value='-',
        fill_missing_cont_value=0,
        normby_zscore=True,
        verbose=verbose,
        test=test,    
    )

    columns['desc_test_values']=desc_test_values
    
    ## rename columns
    return to_columns_renamed_for_regression(
        df1=df2,
        columns=columns,
        )

def get_stats_regression(
    data: pd.DataFrame,
    formulas:dict={},
    variable:str=None,
    covariates:list=None,
    converged_only=False,
    out='df',
    verb=False,
    test=False,
    **kws_model,
    ) -> pd.DataFrame:
    """Get stats from regression models.

    Args:
        data (DataFrame): input dataframe.
        formulas (dict, optional): base formula e.g. 'y ~ x' to model name map. Defaults to {}.
        variable (str, optional): variable name e.g. 'C(variable)[T.True]', used to retrieve the stats for. Defaults to None.
        covariates (list, optional): variables. Defaults to None.
        converged_only (bool, optional): get the stats from the converged models only. Defaults to False.
        out (str, optional): output format. Defaults to 'df'.
        verb (bool, optional): verbose. Defaults to False.
        test (bool, optional): test. Defaults to False.

    Returns:
        DataFrame: output.
    """
    if not '~' in list(formulas.keys())[0]:
        ## back-compatibility warning
        formulas=flip_dict(formulas)
        logging.warning('parameter `formulas` should contain formulas as keys (Opposite to the previous version where formulas were the values).')
        
    if test and hasattr(data,'name'):
        info(data.name)
    ## functions
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
    ## set verbose
    if not (verb or test):
        import warnings
        from statsmodels.tools.sm_exceptions import ConvergenceWarning
        warnings.simplefilter('ignore', ConvergenceWarning)
        warnings.simplefilter('ignore', RuntimeWarning)
        warnings.simplefilter('ignore', UserWarning)
            
    ## add covariates to the equation
    if not covariates is None:
        #formats
        covariate_types=data.dtypes.to_dict()
        formula_covariates=' + '+' + '.join([k if ((covariate_types[k]==int) or (covariate_types[k]==float)) else f"C({k})" for k in covariates if k in covariate_types])
    else:
        formula_covariates=''
    
    ## set additional parameters
    if 'groups' in kws_model:
        ## get the data from the list of columns
        kws_model['groups']=data[kws_model["groups"]]
    
    ## iterate over the models and equations
    ### import required modules
    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError
    from numpy.linalg import LinAlgError
    
    fitted_models={} ## collects the stats
    for formula_base,k in formulas.items():
        
        ## get the model
        if isinstance(k,str):
            model=getattr(smf,k)
        elif isinstance(k,object) and hasattr(k,'from_formula'):
            model=k.from_formula
        elif isinstance(k,object):
            model=k
        else:
            logging.error(model)
            return
        if verb or test: info(str(model))
        ### label for the model
        modeln=str(model).split('.')[-1].split("'")[0]
        
        ## construct full formula
        formula=formula_base+formula_covariates
        if verb or test: info(formula)
        try:
            fitted_models[(modeln,formula)]=model(
                data=data,
                formula=formula,
                **kws_model,
                ).fit(disp=False)
        except (PerfectSeparationError,LinAlgError) as e:
            if verb or test: logging.error('PerfectSeparationError/LinAlgError')
    
    ## output
    if out=='model':
        return fitted_models
    elif out=='df':
        fitted_models={k:to_df(v) for k,v in fitted_models.items() if ((hasattr(v,'converged') and (v.converged)) or (not converged_only))}
        if len(fitted_models)!=0:
            if not variable is None:
                ## return the stats for the selected variable
                return pd.concat({k:get_stats(v,variable=variable) for k,v in fitted_models.items()},
                                 axis=0,
                                 names=['model type','formula','variable']
                                ).reset_index()
            else:
                return pd.concat(fitted_models,
                                 axis=0,
                                 names=['model type','formula'],
                                ).reset_index([0,1])
    
    
def to_filteredby_variable(
    df1: pd.DataFrame,
    variable: str,
    colindex: str,
    coff_q : float=0.1,
    coff_p_covariates: float=0.05,    
    test: bool=False,
    # pval: str='P',
    ) -> pd.DataFrame:
    """Filter regression statistics.

    Args:
        df1 (DataFrame): input dataframe.
        variable (str): variable name to filter by.
        colindex (str): columns with index.
        coff_q (float, optional): cut-off on the q-value. Defaults to 0.1.
        by_covariates (bool, optional): filter by these covaliates. Defaults to True.
        coff_p_covariates (float, optional): cut-off on the p-value for the covariates. Defaults to 0.05.
        test (bool, optional): test. Defaults to False.

    Raises:
        ValueError: pval.

    Returns:
        DataFrame: output.
    
    Notes:
        Filtering steps:
            1. By variable of interest.
            2. By statistical significance.
            3. By statistical significance of co-variates.
    """
    ## filter by variable of interest
    if not 'score' in df1:
        ## non-standardised regression 'P>|t|' or standardised 'P>|z|'
        pval='P>|t|' if 'P>|t|' in df1['stat'].tolist() else 'P>|z|' if 'P>|z|' in df1['stat'].tolist() else None
        if pval is None:
            raise ValueError(pval)
        score='coef' if 'coef' in df1['stat'].tolist() else 'Coef.' if 'Coef.' in df1['stat'].tolist() else None
        if score is None:
            raise ValueError(score)
        df1['stat']=df1['stat'].apply(lambda x: 'P' if x==pval else x ).apply(lambda x: 'score' if x==score else x )

        df2=df1.loc[((df1['variable']==variable) & (df1['stat'].isin(['score','P']))),:]
        df3=df2.pivot(index=colindex,columns='stat',values='value').reset_index()
        # info(df3.columns.tolist())
    else:
        logging.warning("not filtered by variable of interest")
      
    ## calculate q value
    df3=(df3
        .log.dropna(subset=['P'])
        .astype({'P':float}) ## LMM specific
        )
    if test:
        df3['P'].hist()
    # from statsmodels.stats.multitest import fdrcorrection
    # df3['Q']=fdrcorrection(pvals=df3['P'], alpha=0.05, method='indep', is_sorted=False)[1]
    from roux.stat.transform import get_q
    df3['Q']=get_q(df3['P'])
    if test:
        df3['Q'].hist()
        
    if not coff_q is None:
        df3[f"Q<{coff_q}"]=df3['Q']<coff_q
        info(sum(df3['P']<coff_q),sum(df3['Q']<coff_q)) 
    else:
        logging.warning(f"coff_q={coff_q}")
    if not coff_p_covariates is None:
        ##
        ### get the p-values of the covariates
        df5=(df1
            .log.query(expr="variable not in [variable,'Intercept','Group Var']")
            .log.query(expr="stat == 'P'")
            )
        
        ### all covariates are ns
        df6=(df5
            .astype({'value':float}) ## LMM specific
            .groupby(colindex)
            .filter(lambda df: (df['value']>=coff_p_covariates).all())
            .loc[:,[colindex]]
            )
        df3[f'no covariate significant (P<{coff_p_covariates})']=df3[colindex].isin(df6[colindex])
        info(sum(df3[f'no covariate significant (P<{coff_p_covariates})']))
    else:
        logging.warning(f"coff_p_covariates={coff_p_covariates}")
    return df3

# def to_filteredby_stats(
#     df3: pd.DataFrame,
#     coff_q : float=0.1,
#     by_covariates: bool=True,
#     coff_p_covariates: float=0.05,
#     test: bool=False,
#     # pval: str='P',
#     ) -> pd.DataFrame:
#     """Filter regression statistics.

#     Args:
#         df1 (DataFrame): input dataframe.
#         variable (str): variable name to filter by.
#         colindex (str): columns with index.
#         coff_q (float, optional): cut-off on the q-value. Defaults to 0.1.
#         by_covariates (bool, optional): filter by these covaliates. Defaults to True.
#         coff_p_covariates (float, optional): cut-off on the p-value for the covariates. Defaults to 0.05.
#         test (bool, optional): test. Defaults to False.

#     Raises:
#         ValueError: pval.

#     Returns:
#         DataFrame: output.
    
#     Notes:
#         Filtering steps:
#             1. By variable of interest.
#             2. By statistical significance.
#             3. By statistical significance of co-variates.
#     """
#     ## filter by statistical significance
#     if not coff_q is None:
#         info(sum(df3['P']<coff_q),sum(df3['Q']<coff_q)) 
#         df3=df3.log.query(expr=f"Q < {coff_q}") #[(df3['Q']<coff_q),:]
#     else:
#         logging.warning("not filtered by statistical significance")
    
#     ## filter by statistical significance of co-variates
#     if by_covariates:
        
#         df3=df3.log.query(expr=f"`any covariate significant (P<{coff_p_covariates})` == True")
#     else:
#         logging.warning("not filtered by_covariates")
#     return df3

## model comparisons
def run_lr_test(
    data: pd.DataFrame,
    formula: str,
    covariate: str,
    col_group: str,
    params_model: dict ={'reml':False}
    ) -> tuple:
    """Run LR test.

    Args:
        data (pd.DataFrame): input data.
        formula (str): formula.
        covariate (str): covariate.
        col_group (str): column with the group.
        params_model (dict, optional): parameters of the model. Defaults to {'reml':False}.

    Returns:
        tuple: output tupe (stat, pval,dres).
    """
    import statsmodels.formula.api as smf

    sc.stats.chisqprob = lambda chisq, df: sc.stats.chi2.sf(chisq, df)
    def get_lrtest(llmin, llmax):
        stat = 2 * (llmax - llmin)
        pval = sc.stats.chisqprob(stat, 1)
        return stat, pval        
    data=data.dropna()
    # without covariate
    model = smf.mixedlm(
        formula, 
        data,
        groups=data[col_group],
        )
    modelf = model.fit(**params_model)
    llf = modelf.llf

    # with covariate
    model_covariate = smf.mixedlm(
        f"{formula}+ {covariate}",
        data,
        groups=data[col_group],
        )
    modelf_covariate = model_covariate.fit(**params_model)
    llf_covariate = modelf_covariate.llf

    # compare
    stat, pval = get_lrtest(llf, llf_covariate)
    print(f'stat {stat:.2e} pval {pval:.2e}')
    
    # results
    dres=delunnamedcol(pd.concat({
        False:get_model_summary(modelf),
        True:get_model_summary(modelf_covariate)},
        axis=0,
        names=['covariate included','Unnamed'],
        ).reset_index())
    return stat, pval,dres

## model QCs
def plot_residuals_versus_fitted(
    model: object,
    ) -> plt.Axes:
    """plot Residuals Versus Fitted (RVF).

    Args:
        model (object): model.

    Returns:
        plt.Axes: output.
    """
    fig = plt.figure(figsize = (5, 3))
    ax = sns.scatterplot(y = model.resid, x = model.fittedvalues,alpha=0.2)
    ax.set_xlabel("fitted")
    ax.set_ylabel("residuals")
    import statsmodels.api as sm
    l = sm.stats.diagnostic.het_white(model.resid, model.model.exog)
    ax.set_title("LM test "+pval2annot(l[1],alpha=0.05,fmt='<',linebreak=False)+", FE test "+pval2annot(l[3],alpha=0.05,fmt='<',linebreak=False))    
    return ax

def plot_residuals_versus_groups(
    model: object,
    ) -> plt.Axes:
    """plot Residuals Versus groups.

    Args:
        model (object): model.

    Returns:
        plt.Axes: output.
    """
    fig = plt.figure(figsize = (5, 3))
    ax = sns.pointplot(x = model.model.groups, 
                       y = model.resid,
                      ci='sd',
                      join=False)
    ax.set_ylabel("residuals")
    ax.set_xlabel("groups")
    return ax

def plot_model_qcs(
    model: object,
    ):
    """Plot Quality Checks.

    Args:
        model (object): model.
    """
    from roux.viz.scatter import plot_qq 
    from roux.viz.dist import plot_normal 
    plot_normal(x=model.resid)
    plot_qq(x=model.resid)
    plot_residuals_versus_fitted(model)
    plot_residuals_versus_groups(model)

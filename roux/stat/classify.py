from roux.global_imports import *

# curate data 
def drop_low_complexity(df1: pd.DataFrame,
                        min_nunique: int,
                        max_inflation: int,
                        cols: list=None,
                        cols_keep: list=[],
                        test: bool=False) -> pd.DataFrame:
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
    df_=pd.concat([df1.rd.check_nunique(cols),df1.rd.check_inflation(cols)],axis=1,)
    df_.columns=['nunique','% inflation']
    df_=df_.sort_values(df_.columns.tolist(),ascending=False)
    df_=df_.loc[((df_['nunique']<=min_nunique) | (df_['% inflation']>=max_inflation)),:]
    l1=df_.index.tolist()
#     def apply_(x,df1,min_nunique,max_inflation):
#         ds1=x.value_counts()
#         return (len(ds1)<=min_nunique) or ((ds1.values[0]/len(df1))>=max_inflation)
#     l1=df1.loc[:,cols].apply(lambda x: apply_(x,df1,min_nunique=min_nunique,max_inflation=max_inflation)).loc[lambda x: x].index.tolist()
    logging.info(f"{len(l1)}(/{len(cols)}) low complexity columns {'could be ' if test else ''}dropped:")
    info(df_)
    if len(cols_keep)!=0:
        assert all([c in df1 for c in cols_keep]), ([c for c in cols_keep if not c in df1])
        cols_kept=[c for c in l1 if c in cols_keep]
        info(cols_kept)
        l1=[c for c in l1 if not c in cols_keep]
    if not test:
        return df1.log.drop(labels=l1,axis=1)
    else:
        return df1

def get_Xy_for_classification(df1: pd.DataFrame,coly: str,qcut: float=None,
                              # low_complexity filters
                              drop_xs_low_complexity: bool=False,
                              min_nunique: int=5,
                              max_inflation: float=0.5,
                              **kws,
                             ) -> dict:
    """Get X matrix and y vector. 
    
    Args:
        df1 (pd.DataFrame): input data, should be indexed.
        coly (str): column with y values, bool if qcut is None else float/int
        qcut (float, optional): quantile cut-off. Defaults to None.
        drop_xs_low_complexity (bool, optional): to drop columns with <5 unique values. Defaults to False.
        min_nunique (int, optional): minimum unique values in the column. Defaults to 5.
        max_inflation (float, optional): maximum inflation. Defaults to 0.5.

    Keyword arguments:
        kws: parameters provided to `drop_low_complexity`.

    Returns:
        dict: output.
    """
    df1=df1.rd.clean(drop_constants=True)
    cols_X=[c for c in df1 if c!=coly]
    if not qcut is None:
        if qcut>0.5:
            logging.error('qcut should be <=0.5')
            return 
        lims=[df1[coly].quantile(1-qcut),df1[coly].quantile(qcut)]
        df1[coly]=df1.progress_apply(lambda x: True if x[coly]>=lims[0] else False if x[coly]<lims[1] else np.nan,axis=1)
        df1=df1.log.dropna()
    df1[coly]=df1[coly].apply(bool)
    info(df1[coly].value_counts())
    y=df1[coly]
    X=df1.loc[:,cols_X]
    # remove low complexity features
    X=X.rd.clean(drop_constants=True)
    X=drop_low_complexity(X,cols=None,
                          min_nunique=min_nunique,
                          max_inflation=max_inflation,
                          test=False if drop_xs_low_complexity else True,
                          **kws,
                         )
    return {'X':X,'y':y}

def get_cvsplits(
    X: np.array,
    y: np.array,
    cv: int=5,
    random_state: int=None,
    outtest: bool=True
    ) -> dict:
    """Get cross-validation splits.

    Args:
        X (np.array): X matrix.
        y (np.array): y vector.
        cv (int, optional): cross validations. Defaults to 5.
        random_state (int, optional): random state. Defaults to None.
        outtest (bool, optional): output testing. Defaults to True.

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

# search estimator
def get_grid_search(modeln: str,
                    X: np.array,
                    y: np.array,
                    param_grid: dict={},
                    cv: int=5,
                    n_jobs: int=6,
                    random_state: int=None,
                    scoring: str='balanced_accuracy',
                    **kws,
                   ) -> object:
    """Grid search.

    Args:
        modeln (str): name of the model.
        X (np.array): X matrix.
        y (np.array): y vector.
        param_grid (dict, optional): parameter grid. Defaults to {}.
        cv (int, optional): cross-validations. Defaults to 5.
        n_jobs (int, optional): number of cores. Defaults to 6.
        random_state (int, optional): random state. Defaults to None.
        scoring (str, optional): scoring system. Defaults to 'balanced_accuracy'.

    Keyword arguments:
        kws: parameters provided to the `GridSearchCV` function.

    Returns:
        object: `grid_search`.

    References: 
        1. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
        2. https://scikit-learn.org/stable/modules/model_evaluation.html
    """
    if random_state is None: logging.warning(f"random_state is None")
    from sklearn.model_selection import GridSearchCV
    from sklearn import ensemble
    estimator = getattr(ensemble,modeln)(random_state=random_state)
    grid_search = GridSearchCV(estimator, 
                               param_grid,
                               cv=cv,
                               n_jobs=n_jobs,
                               scoring=scoring,
                               **kws)
    grid_search.fit(X, y)
    info(modeln,grid_search.best_params_)
    info(modeln,grid_search.best_score_)
    return grid_search

def get_estimatorn2grid_search(estimatorn2param_grid: dict,
                                X: pd.DataFrame,
                                y: pd.Series,
                                **kws
                                ) -> dict:
    """Estimator-wise grid search.

    Args:
        estimatorn2param_grid (dict): estimator name to the grid search map.
        X (pd.DataFrame): X matrix.
        y (pd.Series): y vector.

    Returns:
        dict: output.
    """
    estimatorn2grid_search={}
    for k in tqdm(estimatorn2param_grid.keys()):
        estimatorn2grid_search[k]=get_grid_search(modeln=k,
                        X=X,y=y,
                        param_grid=estimatorn2param_grid[k],
                        cv=5,
                        n_jobs=6,
                        **kws,
                       )
#     info({k:estimatorn2grid_search[k].best_params_ for k in estimatorn2grid_search})
    return estimatorn2grid_search

def get_test_scores(d1: dict) -> pd.DataFrame:
    """Test scores.

    Args:
        d1 (dict): dictionary with objects.

    Returns:
        pd.DataFrame: output.

    TODOs: 
        Get best param index.
    """
    d2={}
    for k1 in d1:
#             info(k1,dict2str(d1[k1].best_params_))
        l1=list(d1[k1].cv_results_.keys())
        l1=[k2 for k2 in l1 if not re.match("^split[0-9]_test_.*",k2) is None]
        d2[k1+"\n("+dict2str(d1[k1].best_params_,sep='\n')+")"]={k2: d1[k1].cv_results_[k2] for k2 in l1}
    df1=pd.DataFrame(d2).applymap(lambda x: x[0] if (len(x)==1) else max(x)).reset_index()
    df1['variable']=df1['index'].str.split('_test_',expand=True)[1].str.replace('_',' ')
    df1['cv #']=df1['index'].str.split('_test_',expand=True)[0].str.replace('split','').apply(int)
    df1=df1.rd.clean()
    return df1.melt(id_vars=['variable','cv #'],
                   value_vars=d2.keys(),
                   var_name='model',
                   )

## evaluate metrics
def plot_metrics(outd: str,plot: bool=False) -> pd.DataFrame:
    """Plot performance metrics.

    Args:
        outd (str): output directory.
        plot (bool, optional): make plots. Defaults to False.

    Returns:
        pd.DataFrame: output data.
    """
    d0=read_dict(f'{outd}/input.json')
    d1=read_pickle(f'{outd}/estimatorn2grid_search.pickle')
    df01=read_table(f'{outd}/input.pqt')
    df2=get_test_scores(d1)
    df2.loc[(df2['variable']=='average precision'),'value reference']=sum(df01[d0['coly']])/len(df01[d0['coly']])
    if plot:
        _,ax=plt.subplots(figsize=[3,3])
        sns.pointplot(data=df2,
        y='variable',
        x='value',
        hue='model',
        join=False,
        dodge=0.2,
        ax=ax)
        ax.axvline(0.5,linestyle=":",
                   color='k',
                  label='reference: accuracy')
        ax.axvline(sum(df01[d0['coly']])/len(df01[d0['coly']]),linestyle=":",
                   color='b',
                  label='reference: precision')
        ax.legend(bbox_to_anchor=[1,1])
        ax.set(xlim=[-0.1,1.1])
    return df2

def get_probability(estimatorn2grid_search: dict,
                    X: np.array,y: np.array,
                    colindex: str,
                    coff: float=0.5,
                   test: bool=False):
    """Classification probability.

    Args:
        estimatorn2grid_search (dict): estimator to the grid search map.
        X (np.array): X matrix.
        y (np.array): y vector.
        colindex (str): index column. 
        coff (float, optional): cut-off. Defaults to 0.5.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        pd.DataFrame: output.
    """
    assert(all(X.index==y.index))
    df0=y.to_frame('actual').reset_index()
    df1=pd.DataFrame({k:estimatorn2grid_search[k].best_estimator_.predict(X) for k in estimatorn2grid_search})#.add_prefix('prediction ')
    df1.index=X.index
    df1=df1.melt(ignore_index=False,
            var_name='estimator',
            value_name='prediction').reset_index()
    df2=pd.DataFrame({k:estimatorn2grid_search[k].best_estimator_.predict_proba(X)[:,1] for k in estimatorn2grid_search})#.add_prefix('prediction probability ')
    df2.index=X.index
    df2=df2.melt(ignore_index=False,
            var_name='estimator',
            value_name='prediction probability').reset_index()

    df3=df1.log.merge(right=df2,
                  on=['estimator',colindex],
                 how='inner',
                 validate="1:1")\
           .log.merge(right=df0,
                  on=[colindex],
                 how='inner',
    #              validate="1:1",
                )
    ## predicted correctly
    df3['TP']=df3.loc[:,['prediction','actual']].all(axis=1)
    if test:
        def plot_(df5):
            assert len(df5)==4, df5
            df6=df5.pivot(index='prediction',columns='actual',values='count').sort_index(axis=0,ascending=False).sort_index(axis=1,ascending=False)
            from roux.viz.heatmap import plot_crosstab
            ax=plot_crosstab(df6,annot_pval=None,
                            confusion=True)
            ax.set_title(df5.name,loc='left')
        df4=df3.groupby('estimator').apply(lambda df: pd.crosstab(df['prediction'],df['actual']).melt(ignore_index=False,value_name='count')).reset_index()
        df4.groupby('estimator').apply(plot_)
    return df3
#     df1=dellevelcol(pd.concat({k:pd.DataFrame({'sample name':X.index,
#                                               'true':y,
#                                               'probability':estimatorn2grid_search[k].best_estimator_.predict_proba(X)[:,1],}) for k in estimatorn2grid_search,
#                                               'prediction':estimatorn2grid_search[k].best_estimator_.predict(X) for k in estimatorn2grid_search,
#                               },
#                              axis=0,names=['estimator name'],
#                              ).reset_index())
#     info(df1.shape)
#     df1.loc[:,'correct by truth']=df1.apply(lambda x: ((x['true'] and x['probability']>coff) or (not x['true'] and x['probability']<1-coff)),axis=1)
#     info(df1.loc[:,'correct by truth'].sum())

#     df1['probability per class']=df1.apply(lambda x: np.nan if not x['correct by truth'] else 1-x['probability'] if x['probability']<0.5 else x['probability'],axis=1)
#     if test:
#         plt.figure(figsize=[4,4])
#         ax=plt.subplot()
#         df1.groupby('estimator name').apply(lambda df: df['probability'].hist(bins=50,label=df.name,histtype='step'))
#         ax.axvline(coff,label='cut off')
#         ax.set(xlim=[0.5,1])
#         ax.legend(loc=2)
#         _=ax.set(xlabel='probability',ylabel='count')

#     df1=df1.merge(df1.groupby(['sample name']).agg({'probability per class': lambda x: all([i>coff or i<1-coff for i in x])}).rename(columns={'probability per class':'correct by estimators'}).reset_index(),
#              on='sample name',how='left')

#     info('total samples\t',len(df1))
#     info(df1.groupby(['sample name']).agg({c:lambda x: any(x) for c in df1.filter(regex='^correct ')}).sum())
#     return df1

def run_grid_search(df: pd.DataFrame,
    colindex: str,
    coly: str,
    n_estimators: int,
    qcut: float=None,
    evaluations: list=['prediction','feature importances',
    'partial dependence',
    ],
    estimatorn2param_grid: dict=None,
    drop_xs_low_complexity: bool=False,
    min_nunique: int=5,
    max_inflation: float=0.5,      
    cols_keep: list=[],
    outp: str=None,
    test: bool=False,
    **kws, ## grid search
    ) -> dict:
    """Run grid search.

    Args:
        df (pd.DataFrame): input data.
        colindex (str): column with the index.
        coly (str): column with y values. Data type bool if qcut is None else float/int.
        n_estimators (int): number of estimators.
        qcut (float, optional): quantile cut-off. Defaults to None.
        evaluations (list, optional): evaluations types. Defaults to ['prediction','feature importances', 'partial dependence', ].
        estimatorn2param_grid (dict, optional): estimator to the parameter grid map. Defaults to None.
        drop_xs_low_complexity (bool, optional): drop the low complexity columns. Defaults to False.
        min_nunique (int, optional): minimum unique values allowed. Defaults to 5.
        max_inflation (float, optional): maximum inflation allowed. Defaults to 0.5.
        cols_keep (list, optional): columns to keep. Defaults to [].
        outp (str, optional): output path. Defaults to None.
        test (bool, optional): test mode. Defaults to False.

    Keyword arguments:
        kws: parameters provided to `get_estimatorn2grid_search`.

    Returns:
        dict: estimator to grid search map.
    """
    assert('random_state' in kws)
    if kws['random_state'] is None: logging.warning(f"random_state is None")
    
    if estimatorn2param_grid is None: 
        from sklearn import ensemble
        estimatorn2param_grid={k:getattr(ensemble,k)().get_params() for k in estimatorn2param_grid}
        if test=='estimatorn2param_grid':
            return estimatorn2param_grid
    #     info(estimatorn2param_grid)
        for k in estimatorn2param_grid:
            if 'n_estimators' not in estimatorn2param_grid[k]:
                estimatorn2param_grid[k]['n_estimators']=[n_estimators]
        if test: info(estimatorn2param_grid)
        d={}
        for k1 in estimatorn2param_grid:
            d[k1]={}
            for k2 in estimatorn2param_grid[k1]:
                if isinstance(estimatorn2param_grid[k1][k2],list):
                    d[k1][k2]=estimatorn2param_grid[k1][k2]
        estimatorn2param_grid=d
    if test: info(estimatorn2param_grid)
    params=get_Xy_for_classification(df.set_index(colindex),coly=coly,
                                    qcut=qcut,drop_xs_low_complexity=drop_xs_low_complexity,
                                    min_nunique=min_nunique,
                                    max_inflation=max_inflation,
                                    cols_keep=cols_keep,
                                    )
    dn2df={}
    dn2df['input']=params['X'].join(params['y'])
    estimatorn2grid_search=get_estimatorn2grid_search(estimatorn2param_grid,
                                                      X=params['X'],y=params['y'],
                                                      **kws)
#     to_dict({k:estimatorn2grid_search[k].cv_results_ for k in estimatorn2grid_search},
#            f'{outp}/estimatorn2grid_search_results.json')
    if not outp is None:
        to_dict(estimatorn2grid_search,f'{outp}/estimatorn2grid_search.pickle')
        to_dict(estimatorn2grid_search,f'{outp}/estimatorn2grid_search.joblib')
        d1={} # cols
        d1['colindex']=colindex
        d1['coly']=coly
        d1['cols_x']=dn2df['input'].filter(regex=f"^((?!({d1['colindex']}|{d1['coly']})).)*$").columns.tolist()
        d1['estimatorns']=list(estimatorn2param_grid.keys())
        d1['evaluations']=evaluations
        to_dict(d1,f'{outp}/input.json')
#     return estimatorn2grid_search
    ## interpret
    kws2={'random_state':kws['random_state']}
    if 'prediction' in evaluations:
        dn2df['prediction']=get_probability(estimatorn2grid_search,
                                            X=params['X'],y=params['y'],
                                            colindex=colindex,
#                                             coly=coly,
                                            test=True,
#                                             **kws2,
                                           )

    if 'feature importances' in evaluations:
        dn2df['feature importances']=get_feature_importances(estimatorn2grid_search,
                                X=params['X'],y=params['y'],
                                test=test,**kws2)
    if 'partial dependence' in evaluations:
        dn2df['partial dependence']=get_partial_dependence(estimatorn2grid_search,
                                X=params['X'],y=params['y'],
#                                                            **kws2,
                                                          )
    ## save data
    if not outp is None:
        for k in dn2df:
            if isinstance(dn2df[k],dict):
                dn2df[k]=pd.concat(dn2df[k],axis=0,names=['estimator name']).reset_index(0)
            if 'permutation #' in dn2df[k]:
                dn2df[k]['permutation #']=dn2df[k]['permutation #'].astype(int)
            to_table(dn2df[k],f'{outp}/{k}.pqt')
        df_=plot_metrics(outd=outp,plot=True)
        to_table(df_,f'{outp}/metrics.pqt')
    else:
        return estimatorn2grid_search

# interpret 
def plot_feature_predictive_power(df3: pd.DataFrame,
                                  ax: plt.Axes=None,
                                  figsize: list=[3,3],
                                  **kws) -> plt.Axes:
    """Plot feature-wise predictive power.

    Args:
        df3 (pd.DataFrame): input data.
        ax (plt.Axes, optional): axes object. Defaults to None.
        figsize (list, optional): figure size. Defaults to [3,3].

    Returns:
        plt.Axes: output.
    """

    df4=df3.rd.filter_rows({'variable':'ROC AUC'}).rd.groupby_sort_values(col_groupby='feature',
                     col_sortby='value',
                     ascending=False)
    if ax is None:
        _,ax=plt.subplots(figsize=figsize)
    sns.pointplot(data=df3,
                 y='feature',
                 x='value',
                 hue='variable',
                  order=df4['feature'].unique(),
                 join=False,
                  ax=ax,
                 )
    ax.legend(bbox_to_anchor=[1,1])
    ax.axvline(0.5, linestyle=':', color='lightgray')
    return ax

def get_feature_predictive_power(d0: dict,df01: pd.DataFrame,
                                n_splits: int=5, 
                                n_repeats: int=10,
                                random_state: int=None,
                                 plot: bool=False,
                                 drop_na: bool=False,
                               **kws) -> pd.DataFrame:
    """get_feature_predictive_power _summary_

    Notes: 
        x-values should be scale and sign agnostic.

    Args:
        d0 (dict): input dictionary.
        df01 (pd.DataFrame): input data, 
        n_splits (int, optional): number of splits. Defaults to 5.
        n_repeats (int, optional): number of repeats. Defaults to 10.
        random_state (int, optional): random state. Defaults to None.
        plot (bool, optional): plot. Defaults to False.
        drop_na (bool, optional): drop missing values. Defaults to False.

    Returns:
        pd.DataFrame: output data.
    """
    if random_state is None: logging.warning(f"random_state is None")
    from sklearn.metrics import average_precision_score,roc_auc_score
    from sklearn.model_selection import StratifiedKFold,RepeatedStratifiedKFold

    d2={}
    for colx in tqdm(d0['cols_x']):
        df1=df01.loc[:,[d0['coly'],colx]]
        if drop_na:
            df1=df1.dropna() 
        if df1[d0['coly']].nunique()==1: continue
        if sum(df1[d0['coly']]==True)<5: continue
        if sum(df1[d0['coly']]==False)<5: continue        
        # if perc(df1[d0['coly']])>90: continue
        # if perc(df1[d0['coly']])<10: continue
        if df1[colx].nunique()==1: continue
        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state,**kws)
        d1={i: ids for i,(_, ids) in enumerate(cv.split(df1[colx], df1[d0['coly']]))}
        df2=pd.DataFrame({'cv #':range(cv.get_n_splits())})
        if roc_auc_score(df1[d0['coly']], df1[colx])<roc_auc_score(df1[d0['coly']], -df1[colx]):
#             df1[d0['coly']]=~df1[d0['coly']]
            df1[colx]=-df1[colx]
        try:
            df2['ROC AUC']=df2['cv #'].apply(lambda x: roc_auc_score(df1.iloc[d1[x],:][d0['coly']],
                                                                     df1.iloc[d1[x],:][colx]))
            df2['average precision']=df2['cv #'].apply(lambda x: average_precision_score(df1.iloc[d1[x],:][d0['coly']],
                                                                                         df1.iloc[d1[x],:][colx]))
        except:
            print(df1)
        d2[colx]=df2.melt(id_vars='cv #',value_vars=['ROC AUC','average precision'])

    df3=pd.concat(d2,axis=0,names=['feature']).reset_index(0)
    if plot: plot_feature_predictive_power(df3)
    return df3

def get_feature_importances(estimatorn2grid_search: dict,
                            X: pd.DataFrame,y: pd.Series,
                            scoring: str='roc_auc',
                            n_repeats: int=20,
                            n_jobs: int=6,
                            random_state: int=None,
                            plot: bool=False,
                            test: bool=False,
                           **kws) -> pd.DataFrame:
    """Feature importances.

    Args:
        estimatorn2grid_search (dict): map between estimator name and grid search object. 
        X (pd.DataFrame): X matrix.
        y (pd.Series): y vector.
        scoring (str, optional): scoring type. Defaults to 'roc_auc'.
        n_repeats (int, optional): number of repeats. Defaults to 20.
        n_jobs (int, optional): number of cores. Defaults to 6.
        random_state (int, optional): random state. Defaults to None.
        plot (bool, optional): plot. Defaults to False.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        pd.DataFrame: output data.
    """
    if random_state is None: logging.warning(f"random_state is None")    
    def plot_(df,ax=None):
        if ax is None:
            fig,ax=plt.subplots(figsize=[4,(df['estimator name'].nunique()*0.5)+2])
        dplot=groupby_sort_values(df,
             col_groupby=['estimator name','feature'],
             col_sortby='importance rescaled',
             func='mean', ascending=False
            )
        dplot=dplot.loc[(dplot['importance']!=0),:]

        sns.pointplot(data=dplot,
              x='importance rescaled',
              y='feature',
              hue='estimator name',
             linestyles=' ',
              markers='o',
              alpha=0.1,
              dodge=True,
              scatter_kws = {'facecolors':'none'},
              ax=ax
             )
        return ax
    
    dn2df={}
    for k in tqdm(estimatorn2grid_search.keys()):
        from sklearn.inspection import permutation_importance
        r = permutation_importance(estimator=estimatorn2grid_search[k].best_estimator_, 
                                   X=X, y=y,
                                   scoring=scoring,
                                   n_repeats=n_repeats,
                                   n_jobs=n_jobs,
                                   random_state=random_state,
                                   **kws,
                                  )
        df=pd.DataFrame(r.importances)
        df['feature']=X.columns
        dn2df[k]=df.melt(id_vars=['feature'],value_vars=range(n_repeats),
            var_name='permutation #',
            value_name='importance',
           )
    df2=pd.concat(dn2df,axis=0,names=['estimator name']).reset_index(0)
    from roux.stat.transform import rescale
    def apply_(df):
        df['importance rescaled']=rescale(df['importance'])
        df['importance rank']=len(df)-df['importance'].rank()
        return df#.sort_values('importance rescaled',ascending=False)
    df3=df2.groupby(['estimator name','permutation #']).apply(apply_)
    if plot:
        plot_(df3)
    return df3

def get_partial_dependence(estimatorn2grid_search: dict,
                            X: pd.DataFrame,y: pd.Series,
                            ) -> pd.DataFrame:
    """Partial dependence.

    Args:
        estimatorn2grid_search (dict): map between estimator name and grid search object.
        X (pd.DataFrame): X matrix.
        y (pd.Series): y vector.

    Returns:
        pd.DataFrame: output data.
    """
    df3=pd.DataFrame({'feature #':range(len(X.columns)),
                     'feature name':X.columns})

    def apply_(featuren,featurei,estimatorn2grid_search):
        from sklearn.inspection import partial_dependence
        dn2df={}
        for k in estimatorn2grid_search:
            t=partial_dependence(estimator=estimatorn2grid_search[k].best_estimator_,
                                 X=X,
                                 features=[featurei],
                                 response_method='predict_proba',
                                 method='brute',
                                 percentiles=[0,1],
                                 grid_resolution=100,
                                 )
            dn2df[k]=pd.DataFrame({'probability':t[0][0],
                                    'feature value':t[1][0]})
        df1=pd.concat(dn2df,axis=0,names=['estimator name']).reset_index()
        df1['feature name']=featuren
        return df1.rd.clean()
    df4=df3.groupby('feature #',as_index=False).progress_apply(lambda df:apply_(featuren=df.iloc[0,:]['feature name'],
                                                                                featurei=df.iloc[0,:]['feature #'],
                                                                                estimatorn2grid_search=estimatorn2grid_search))
    
    return df4




from roux.lib.dfs import *
import matplotlib.pyplot as plt
    
# compare    
def get_ddist(df,window_size_max=10,corr=False):
    print(df.shape)
    print(f"window=",end='')
    method2ddists={}
    for window in range(1,window_size_max,1):
        print(window,end=' ')
        method2ddists[f'DTW (window={window:02d})']=get_ddist_dtw(df,window)
    if corr:
        method2ddists['1-spearman']=dmap2lin((1-df.T.corr(method='spearman')),colvalue_name='distance').set_index(['index','column'])
        method2ddists['1-pearson']=dmap2lin((1-df.T.corr(method='pearson')),colvalue_name='distance').set_index(['index','column'])

    ddist=pd.concat(method2ddists,axis=1,)
    ddist.columns=coltuples2str(ddist.columns)
    ddist=ddist.reset_index()
    ddist=ddist.loc[(ddist['index']!=ddist['column']),:]
    ddist['interaction id']=ddist.apply(lambda x : '--'.join(list(sorted([x['index'],x['column']]))),axis=1)
    print(ddist.shape,end='')
    ddist=ddist.drop_duplicates(subset=['interaction id'])
    print(ddist.shape)   
    return ddist

# scikit learn below       
def check_clusters(df):
    return (df.groupby(['cluster #']).agg({'silhouette value':np.max})['silhouette value']>=df['silhouette value'].mean()).all()
def get_clusters(X,n_clusters,random_state=88,
                 params={},
                 test=False):
    from sklearn import cluster,metrics
    kmeans = cluster.MiniBatchKMeans(n_clusters=n_clusters,
                             random_state=random_state,
                             **params,
                            ).fit(X)
    clusters=kmeans.predict(X)
    ds=pd.Series(dict(zip(X.index,clusters)))
    ds.name='cluster #'
    df=pd.DataFrame(ds)
    df.index.name=X.index.name
    df['cluster #'].value_counts()
    # Compute the silhouette scores for each sample
    df['silhouette value'] = metrics.silhouette_samples(X, clusters)
#     if test:
#         print(f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: {is_optimum}, random state {random_state}")
    if not check_clusters:
        logging.warning(f"{n_clusters} cluster : silhouette average score {df['silhouette value'].mean():1.2f}, ok?: {is_optimum}, random state {random_state}")
    dn2df={'clusters':df.reset_index(),
           'inertia':kmeans.inertia_,
           'centers':pd.DataFrame(kmeans.cluster_centers_,index=range(n_clusters),columns=X.columns).rename_axis(index='cluster #')#.stack().reset_index().rename(columns={'level_1':'variable',0:'value'}),
          }
    return dn2df

def get_n_clusters_optimum(df5,test=False):
    from kneed import KneeLocator
    kn = KneeLocator(x=df5['total clusters'], y=df5['inertia'], curve='convex', direction='decreasing')
    if test:
        import matplotlib.pyplot as plt
        kn.plot_knee()
        plt.title(f"knee point={kn.knee}")
    return kn.knee
        
def plot_silhouette(df,n_clusters_optimum=None,ax=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ax=plt.subplot() if ax is None else ax
    ax=sns.violinplot(data=df.groupby(['total clusters','cluster #']).agg({'silhouette value':np.mean}).reset_index(),
                     y='silhouette value',x='total clusters',
                     color='salmon',alpha=0.7,
                     ax=ax)
    ax=sns.pointplot(data=df.groupby('total clusters').agg({'silhouette value':np.mean}).reset_index(),
                     y='silhouette value',x='total clusters',
                     color='k',
                     ax=ax)
    if not n_clusters_optimum is None:
        ax.annotate('optimum', 
                xy=(n_clusters_optimum-int(ax.get_xticklabels()[0].get_text()),
                    ax.get_ylim()[0]),  
                xycoords='data',
                    xytext=(+50, +50), textcoords='offset points',
                    arrowprops=dict(arrowstyle="->",ec='k',
                                    connectionstyle="angle3,angleA=0,angleB=-90"),
                    )
    ax.set_xlabel('clusters')  
    return ax

def get_clusters_optimum(X,n_clusters=range(2,11),
                         params_clustering={},
                         test=False,
                        ):
    """
    :param X: samples to cluster in indexed 
    
    cluster center intertia
    """
    dn2d={}
    for n in n_clusters:
        dn2d[n]=get_clusters(X=X,n_clusters=n,test=test,params=params_clustering)
    df1=pd.DataFrame(pd.Series({k:dn2d[k]['inertia'] for k in dn2d}),columns=['inertia']).rename_axis(index='total clusters').reset_index()
    # TODO identify saturation point in the intertia plot for n_clusters_optimum
    n_clusters_optimum=get_n_clusters_optimum(df1,test=test)
    #
    dn2df={dn:pd.concat({k:dn2d[k][dn] for k in dn2d},axis=0,names=['total clusters']).reset_index() for dn in ['clusters','centers']}
    if not check_clusters(dn2df['clusters']):
        logging.warning('low silhoutte scores')
        return
    if test:
        import matplotlib.pyplot as plt
        plt.figure()
        plot_silhouette(df=dn2df['clusters'],n_clusters_optimum=None)
    # make output
    dn2df={dn:dn2df[dn].loc[(dn2df[dn]['total clusters']==n_clusters_optimum),:].drop(['total clusters'],axis=1) for dn in dn2df}
    return dn2df

def cluster_1d(ds,n_clusters,clf_type='gmm',
               random_state=88,
                 test=False,
               returns=['df','coff','ax','model'],
#                 return_coff=False,
#                   return_ax=False,
               ax=None,
               bins=50,
              **kws_clf):
    x=ds.to_numpy()
    X=x.reshape(-1,1)
    if clf_type.lower()=='gmm':
        from sklearn.mixture import GaussianMixture
        model = GaussianMixture(n_components=n_clusters,)
    elif clf_type.lower()=='kmeans':
        from sklearn.cluster import KMeans
        model=KMeans(n_clusters=n_clusters,**kws_clf).fit(X,)
    else:
        raise ValueError(clf_type)
    ## fit and predic
    labels =model.fit_predict(X)
    if not model.converged_:
        logging.warning('not converged')
    df=pd.DataFrame({'value':x,
    'label':labels==1})
    
    if test:
        if ax is None:
            plt.figure(figsize=[2.5,2.5])
            ax=plt.subplot()
        df['value'].hist(bins=bins,density=True,
                         histtype='step',
                         ax=ax)
        if clf_type=='gmm':
            from roux.viz.dist import plot_gaussianmixture    
            ax,coff=plot_gaussianmixture(g=model,x=x,
                                         n_clusters=n_clusters,
                                         ax=ax,
                                        )
        else:
            coffs=df.groupby('label')['value'].agg(min).values
            for c in coffs:
                if  c > df['value'].quantile(0.05) and c < df['value'].quantile(0.95):
                    coff=c
                    break
            logging.info(f"coff:{c}; selected from {coffs}")
            ax.axvline(coff,color='k')
            ax.text(coff,ax.get_ylim()[1],f"{coff:.1f}",ha='center',va='bottom')            
    d={'df':df,}
#     if clf_type=='gmm':    
#         weights = clf.weights_
#         means = clf.means_
#         covars = clf.covariances_
#         stds=np.sqrt(covars).ravel().reshape(2,1)
    for k in returns:
        d[k]=locals()[k]
#     if 'coff' in returns:
#         d['coff']=coff
#     if 'ax' in returns:
#         d['ax']=coff
    return d

## umap
def get_pos_umap(df1,spread=100,
                 test=False,k='',
                 **kws):
    import umap
    reducer = umap.UMAP(spread=spread,*kws)
    embedding = reducer.fit_transform(df1)
    if test:
        plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=['r' if k in s else 'k' for s in df1.index.get_level_values(0)],
        )
        plt.gca().set_aspect('equal', 'datalim')
    df2=pd.DataFrame(embedding,
                    columns=['x','y'])
    df2.index=df1.index
    return df2.reset_index()
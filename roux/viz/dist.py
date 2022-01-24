from roux.global_imports import *
from roux.viz.colors import *
from roux.viz.annot import *
from roux.viz.ax_ import *

## single 
def hist_annot(dplot,colx,
               colssubsets=[],
               bins=100,
                subset_unclassified=True,cmap='Reds_r',
               ylimoff=1.2,
               ywithinoff=1.2,
                annotaslegend=True,
                annotn=True,
                params_scatter={'zorder':2,'alpha':0.1,'marker':'|'},
               xlim=None,
                ax=None):
    from roux.viz.ax_ import reset_legend_colors
    if not xlim is None:
        logging.warning('colx adjusted to xlim')
        dplot.loc[(dplot[colx]<xlim[0]),colx]=xlim[0]
        dplot.loc[(dplot[colx]>xlim[1]),colx]=xlim[1]
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=bins,ax=ax,color='gray',zorder=1)
    ax.set_xlabel(colx)
    ax.set_ylabel('count')
    if not xlim is None:
        ax.set_xlim(xlim)
    ax.set_ylim(0,ax.get_ylim()[1]*ylimoff)        
    from roux.viz.colors import get_ncolors
    colors=get_ncolors(len(colssubsets),cmap=cmap)
    for colsubsetsi,(colsubsets,color) in enumerate(zip(colssubsets,colors)):
        subsets=[s for s in dropna(dplot[colsubsets].unique()) if not (subset_unclassified and s=='unclassified')]
        for subseti,subset in enumerate(subsets):
            y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-(subseti*ywithinoff+colsubsetsi))/10-0.05)+ax.set_ylim()[0]
            X=dplot.loc[(dplot[colsubsets]==subset),colx]
            Y=[y for i in X]
            ax.scatter(X,Y,
                       color=color,**params_scatter)
            ax.text(max(X) if not annotaslegend else ax.get_xlim()[1],
                    max(Y),
                    f" {subset}\n(n={len(X)})" if annotn else f" {subset}",
                    ha='left',va='center')
    #     break
#     ax=reset_legend_colors(ax)
#     ax.legend(bbox_to_anchor=[1,1])
    return ax

def plot_gaussianmixture(g,x,
                         n_clusters=2,
                         ax=None,
                         test=False,
                        ):
    from roux.stat.solve import get_intersection_locations
    weights = g.weights_
    means = g.means_
    covars = g.covariances_
    stds=np.sqrt(covars).ravel().reshape(n_clusters,1)
    
    f = x.reshape(-1,1)
    x.sort()
#     plt.hist(f, bins=100, histtype='bar', density=True, ec='red', alpha=0.5)
    two_pdfs = sc.stats.norm.pdf(np.array([x,x]), means, stds)
    mix_pdf = np.matmul(weights.reshape(1,n_clusters), two_pdfs)
    ax.plot(x,mix_pdf.ravel(), c='lightgray')
    _=[ax.plot(x,two_pdfs[i]*weights[i], c='gray') for i in range(n_clusters)]
#     ax.plot(x,two_pdfs[1]*weights[1], c='gray')
    logging.info(f'weights {weights}')
    if n_clusters!=2:
        coff=None
        return ax,coff
    idxs=get_intersection_locations(y1=two_pdfs[0]*weights[0],
                                    y2=two_pdfs[1]*weights[1],
                                    test=False,x=x)
    x_intersections=x[idxs]
#     x_intersections=get_intersection_of_gaussians(means[0][0],stds[0][0],
#                                                   means[1][0],stds[1][0],)
    if test: logging.info(f'intersections {x_intersections}')
    ms=sorted([means[0][0],means[1][0]])
#     print(ms)
#     print(x_intersections)    
    if len(x_intersections)>1:
        coff=[i for i in x_intersections if i>ms[0] and i<ms[1]][0]
    else:
        coff=x_intersections[0]
    ax.axvline(coff,color='k')
    ax.text(coff,ax.get_ylim()[1],f"{coff:.1f}",ha='center',va='bottom')
    return ax,coff

def plot_normal(x):
    import statsmodels.api as sm
    fig = plt.figure(figsize = [3, 3])
    ax = sns.distplot(x, hist = True, 
                      kde_kws = {"shade" : True, "lw": 1, }, 
                      fit = sc.stats.norm,
                      label='residuals',
                     )
    ax.set_title("SW test "+pval2annot(sc.stats.shapiro(x)[1],alpha=0.05,fmt='<',linebreak=False))
    ax.legend()
    return ax

## pair 
def plot_dists(df1,x,y,
               colindex,
               hue=None,
               order=None,
               hue_order=None,
               kind='box',
               show_p=True,
               show_n=True,
               show_n_prefix='',
               offx_n=0,
               xlim=None,
               offx_pval=0.05,
               ax=None,
               **kws):
    """
    TODOs:
    1. show n 
    2. show pval
    3. sort
    """
    if order is None:
        order=df1[y].unique().tolist()
    if not hue is None and hue_order is None:
        hue_order=df1[hue].unique().tolist()
    if (hue is None) and (isinstance(show_p,bool) and show_p):
        from roux.stat.diff import get_stats
        df2=get_stats(df1,
                      colindex=colindex,
                          colsubset=y,
                          cols_value=[x],
                          subsets=order,
        #                   alpha=0.05
                        axis=0,
                         ).reset_index()
        # df1=df1.rd.renameby_replace({f"{} ":''})
        df2=df2.loc[(df2['subset1']==order[0]),:]
        d1=df2.rd.to_dict(['subset2','P (MWU test)'])
    elif (not hue is None) and (isinstance(show_p,bool) and show_p):
        from roux.stat.diff import get_stats_groupby
        df2=get_stats_groupby(df1,cols=[y],
                          colsubset=hue,
                          cols_value=[x],
                              colindex=colindex,
                          alpha=0.05,
                         axis=0,
                         ).reset_index()
        # df1=df1.rd.renameby_replace({f"{} ":''})
        df2=df2.loc[(df2['subset1']==hue_order[0]),:]
        d1=df2.rd.to_dict([y,'P (MWU test)'])

    if ax is None:
        _,ax=plt.subplots(figsize=[2,2])
    if isinstance(kind,str):
        kind={kind:{}}
    elif isinstance(kind,list):
        kind={k:{} for k in kind}
    for k in kind:
        # print(kws['palette'],kind)
        if 'palette' in kws and k in ['swarm','strip']:
            from roux.viz.colors import saturate_color
            kws['palette']=[saturate_color(color=c, alpha=1.5) for c in kws['palette']]
            # print(kws['palette'])
        getattr(sns,k+"plot")(data=df1,
                    x=x,y=y,
                    hue=hue,
                    order=order,
                    hue_order=hue_order,
                    **kind[k],
                    **kws,
                     ax=ax)
    ax.set(xlabel=x)
    d2=get_ticklabel2position(ax,'y')
    ax.set(
          ylabel=None if hue is None else y,
          xlim=xlim,
          )
    d3=get_axlims(ax)
    if isinstance(show_p,(bool,dict)):
        if isinstance(show_p,bool) and show_p:
            d1={k:pval2annot(d1[k],fmt='<',linebreak=False) for k in d1}
        else:
            d1=show_p
        if isinstance(d1,dict):
            for k,s in d1.items():
                ax.text(d3['x']['max']+(d3['x']['len']*offx_pval),d2[k],s,va='center')
    if show_n:
        df1_=df1.groupby(y).apply(lambda df: df.groupby(colindex).ngroups).to_frame('n').reset_index()
        df1_['y']=df1_[y].map(d2)
        import matplotlib.transforms as transforms
        
        df1_.apply(lambda x: ax.text(x=1.15+offx_n,y=x['y'],
                                   s=show_n_prefix+str(x['n']),va='center',ha='right',
                                     transform=transforms.blended_transform_factory(ax.transAxes,ax.transData),
                                   ),axis=1)
    ax.tick_params(axis='y', colors='k')
    if not hue is None:
        o1=ax.legend(
                  loc='upper left', 
                  bbox_to_anchor=(1, 0),
            frameon=True,
            title=hue,
            )
        o1.get_frame().set_edgecolor((0.95,0.95,0.95))
    return ax

def pointplot_join_hues(df1,x,y,hue,hues,
                        order,hue_order,
                        dodge,
                        cmap='Reds',
                        ax=None,
                        **kws_pointplot):
    if ax is None:_,ax=plt.subplots(figsize=[3,3])
    df1.groupby([hues]).apply(lambda df2: sns.pointplot(data=df2,
                                                        x=x,y=y,hue=hue,hues=hues,
                                                        order=order,hue_order=hue_order,
                                                        dodge=dodge,
                                                      **kws_pointplot,
                                                       zorder=5,
                                                      ax=ax,
                                                     ))
    # ax.legend()
    from roux.viz.ax_ import get_ticklabel2position,sort_legends
    df1['y']=df1[y].map(get_ticklabel2position(ax,axis='y'))
    df1['hue']=df1[hue].map(dict(zip(hue_order,[-1,1])))*dodge*0.5
    df1['y hue']=df1['y']+df1['hue']

    df2=df1.pivot(index=[y,hues],
                columns=[hue,],
                values=[x,'y hue','y','hue'],
                ).reset_index()#.rd.flatten_columns()
    from roux.viz.colors import get_val2color
    df2['color'],_=get_val2color(df2[hues],vmin=-0.2,cmap=cmap)
    df2['label']=df2[hues].apply(lambda x: f"{hues}{x:.1f}")
    # x=df2.iloc[0,:]
#     return df2
    _=df2.groupby([hues,'color']).apply(lambda df2: df2.apply(lambda x1: ax.plot(x1[x],x1['y hue'],
                                                           color=df2.name[1],
                                                           label=x1['label'].tolist()[0] if x1[y].tolist()[0]==order[0] else None,
                                                           zorder=1,
                                                           ),axis=1))
    sort_legends(ax, sort_order=hue_order+sorted(df2['label'].unique()),
                bbox_to_anchor=[1,1])
    return ax


def pointplot_groupbyedgecolor(data,ax=None,**kws_pointplot):
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=data,
                     ax=ax,
                     **kws_pointplot)
    plt.setp(ax.collections, sizes=[100])   
    for c in ax.collections:
        if c.get_label().startswith(kws_pointplot['hue_order'][0].split(' ')[0]):
            c.set_linewidth(2)
            c.set_edgecolor('k')
        else:
            c.set_linewidth(2)        
            c.set_edgecolor('w')
    ax.legend(bbox_to_anchor=[1,1])
    return ax


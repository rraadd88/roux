from roux.global_imports import *
from roux.viz.colors import *
from roux.viz.annot import *
from roux.viz.ax_ import *

## single distributions.
def hist_annot(
    dplot: pd.DataFrame,
    colx: str,
    colssubsets: list=[],
    bins: int=100,
    subset_unclassified: bool=True,
    cmap: str='hsv',
    ylimoff: float=1.2,
    ywithinoff: float=1.2,
    annotaslegend: bool=True,
    annotn: bool=True,
    params_scatter: dict={'zorder':2,'alpha':0.1,'marker':'|'},
    xlim: tuple=None,
    ax: plt.Axes = None,
    **kws,
    ) -> plt.Axes:
    """Annoted histogram.

    Args:
        dplot (pd.DataFrame): input dataframe.
        colx (str): x column.
        colssubsets (list, optional): columns indicating subsets. Defaults to [].
        bins (int, optional): bins. Defaults to 100.
        subset_unclassified (bool, optional): call non-annotated subset as 'unclassified'. Defaults to True.
        cmap (str, optional): colormap. Defaults to 'Reds_r'.
        ylimoff (float, optional): y-offset for y-axis limit . Defaults to 1.2.
        ywithinoff (float, optional): y-offset for the distance within labels. Defaults to 1.2.
        annotaslegend (bool, optional): convert labels to legends. Defaults to True.
        annotn (bool, optional): annotate sample sizes. Defaults to True.
        params_scatter (_type_, optional): parameters of the scatter plot. Defaults to {'zorder':2,'alpha':0.1,'marker':'|'}.
        xlim (tuple, optional): x-axis limits. Defaults to None.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `hist` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.viz.ax_ import reset_legend_colors
    if not xlim is None:
        logging.warning('colx adjusted to xlim')
        dplot.loc[(dplot[colx]<xlim[0]),colx]=xlim[0]
        dplot.loc[(dplot[colx]>xlim[1]),colx]=xlim[1]
    if ax is None:ax=plt.subplot(111)
    ax=dplot[colx].hist(bins=bins,ax=ax,zorder=1,**kws,)
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
        
def plot_gmm(
    x: pd.Series,
    coff: float=None,
    mix_pdf: object=None,
    two_pdfs: tuple=None,
    weights: tuple=None,
    n_clusters: int=2,
    bins: int=20,
    test: bool=False,
    ax: plt.Axes = None,
    **kws,
    ) -> plt.Axes:
    """Plot Gaussian mixture Models (GMMs).

    Args:
        x (pd.Series): input vector.
        coff (float, optional): intersection between two fitted distributions. Defaults to None.
        mix_pdf (object, optional): Probability density function of the mixed distribution. Defaults to None.
        two_pdfs (tuple, optional): Probability density functions of the separate distributions. Defaults to None.
        weights (tuple, optional): weights of the individual distributions. Defaults to None.
        n_clusters (int, optional): number of distributions. Defaults to 2.
        bins (int, optional): bins. Defaults to 50.
        test (bool, optional): test mode. Defaults to False.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        
    Keyword Args:
        kws: parameters provided to the `hist` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if mix_pdf is None:
        from roux.stat.cluster import cluster_1d
        d_=cluster_1d(x,
                n_clusters=n_clusters,
                clf_type='gmm',
                random_state=88,
                test=False,
                returns=['coff','mix_pdf','two_pdfs','weights'],
                )
        coff,mix_pdf,two_pdfs,weights=d_['coff'],d_['mix_pdf'],d_['two_pdfs'],d_['weights']
    if ax is None:
        plt.figure(figsize=[2.5,2.5])
        ax=plt.subplot()
    # plot histogram
    pd.Series(x).hist(density=True,
                      histtype='step',
                      bins=bins,
                      ax=ax,
                      **kws)
    # plot fitted distributions
    ax.plot(x,mix_pdf.ravel(), c='lightgray')
    _=[ax.plot(x,two_pdfs[i]*weights[i], c='gray') for i in range(n_clusters)]
#     ax.plot(x,two_pdfs[1]*weights[1], c='gray')
    if n_clusters==2:
        ax.axvline(coff,color='k')
        ax.text(coff,ax.get_ylim()[1],f"{coff:.1f}",ha='center',va='bottom')
    return ax

def plot_normal(
    x: pd.Series,
    ax: plt.Axes = None,
    ) -> plt.Axes:
    """Plot normal distribution.

    Args:
        x (pd.Series): input vector.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if not ax is None:
        fig,ax = plt.subplots(figsize = [3, 3])
    import statsmodels.api as sm
    ax = sns.distplot(x, hist = True, 
                      kde_kws = {"shade" : True, "lw": 1, }, 
                      fit = sc.stats.norm,
                      label='residuals',
                     )
    ax.set_title("SW test "+pval2annot(sc.stats.shapiro(x)[1],alpha=0.05,fmt='<',linebreak=False))
    ax.legend()
    return ax

## paired distributions.
def plot_dists(
    df1: pd.DataFrame,
    x: str,
    y: str,
    colindex: str,
    hue: str=None,
    order: list=None,
    hue_order: list=None,
    kind: str='box',
    show_p: bool=True,
    show_n: bool=True,
    show_n_prefix: str='',
    offx_n: float=0,
    xlim: tuple=None,
    offx_pval: float=0.05,
    offy_pval: float=None,
    saturate_color_alpha: float=1.5,
    ax: plt.Axes = None,
    kws_stats: dict={},
    **kws
    ) -> plt.Axes:
    """Plot distributions.

    Args:
        df1 (pd.DataFrame): input data.
        x (str): x column.
        y (str): y column.
        colindex (str): index column.
        hue (str, optional): column with values to be encoded as hues. Defaults to None.
        order (list, optional): order of categorical values. Defaults to None.
        hue_order (list, optional): order of values to be encoded as hues. Defaults to None.
        kind (str, optional): kind of distribution. Defaults to 'box'.
        show_p (bool, optional): show p-values. Defaults to True.
        show_n (bool, optional): show sample sizes. Defaults to True.
        show_n_prefix (str, optional): show prefix of sample size label i.e. `n=`. Defaults to ''.
        offx_n (float, optional): x-offset for the sample size label. Defaults to 0.
        xlim (tuple, optional): x-axis limits. Defaults to None.
        offx_pval (float, optional): x-offset for the p-value labels. Defaults to 0.05.
        offy_pval (float, optional): y-offset for the p-value labels. Defaults to None.
        saturate_color_alpha (float, optional): saturation of the color. Defaults to 1.5.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        kws_stats (dict, optional): parameters provided to the stat function. Defaults to {}.

    Keyword Args:
        kws: parameters provided to the `seaborn` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
        
    TODOs:
        1. Sort categories.
        2. Change alpha of the boxplot rather than changing saturation of the swarmplot. 

    """
    df1[y]=df1[y].astype(str)
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
                        **kws_stats,
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
                         **kws_stats,
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
        # if 'palette' in kws and any([k_ in kind for k_ in ['swarm','strip']]):
        #     from roux.viz.colors import saturate_color
        #     kws['palette']=[saturate_color(color=c, alpha=saturate_color_alpha-1) for c in kws['palette']]            
        if 'palette' in kws and k in ['swarm','strip']:
            from roux.viz.colors import saturate_color
            kws['palette']=[saturate_color(color=c, alpha=saturate_color_alpha+0.5) for c in kws['palette']]
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
            d1={k:pval2annot(d1[k],alternative='two-sided',fmt='<',linebreak=False) for k in d1}
        else:
            d1=show_p
        if offy_pval is None and hue is None:
            offy_pval=-0.5
        if isinstance(d1,dict):
            for k,s in d1.items():
                ax.text(d3['x']['max']+(d3['x']['len']*offx_pval),d2[k]+offy_pval,s,va='center')
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

def pointplot_groupbyedgecolor(
    data: pd.DataFrame,
    ax: plt.Axes = None,
    **kws
    ) -> plt.Axes:
    """Plot seaborn's `pointplot` grouped by edgecolor of points.

    Args:
        data (pd.DataFrame): input data.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
    
    Keyword Args:
        kws: parameters provided to the `seaborn`'s `pointplot` function. 
        
    Returns:
        plt.Axes: `plt.Axes` object.
    """
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=data,
                     ax=ax,
                     **kws)
    plt.setp(ax.collections, sizes=[100])   
    for c in ax.collections:
        if c.get_label().startswith(kws['hue_order'][0].split(' ')[0]):
            c.set_linewidth(2)
            c.set_edgecolor('k')
        else:
            c.set_linewidth(2)        
            c.set_edgecolor('w')
    ax.legend(bbox_to_anchor=[1,1])
    return ax    
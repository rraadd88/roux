from roux.global_imports import *

# stats
# def annot_stats(dplot: pd.DataFrame,
#             colx: str,
#             coly: str,
#             colz: str,
#             stat_method: list=[],
#             bootstrapped: bool=False,
#             params_set_label: dict={},
#             ax: plt.Axes = None,
#             ) -> plt.Axes:
#     """Annotate stats on a scatter plot.

#     Args:
#         dplot (pd.DataFrame): input dataframe.
#         colx (str): x column.
#         coly (str): y column.
#         colz (str): z column.
#         stat_method (list, optional): names of stat methods to apply. Defaults to [].
#         bootstrapped (bool, optional): to bootstrap the data or not. Defaults to False.
#         params_set_label (dict, optional): parameters provided to the `set_label` function. Defaults to {}.
#         ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

#     Returns:
#         plt.Axes: `plt.Axes` object.
#     """
#     stat_method = [stat_method] if isinstance(stat_method,str) else stat_method
#     from roux.viz.ax_ import set_label
#     if 'mlr' in stat_method:
#         from roux.lib.stat.poly import get_mlr_2_str
#         ax=set_label(ax,label=get_mlr_2_str(dplot,colz,[colx,coly]),
#                     title=True,params={'loc':'left'})
#     if 'spearman' in stat_method or 'pearson' in stat_method:
#         from roux.stat.corr import get_corr
#         ax=set_label(ax,label=get_corr(dplot[colx],dplot[coly],method=stat_method[0],
#                                        bootstrapped=bootstrapped,
#                                        outstr=True,n=True),
#                     **params_set_label)
#     return ax

def plot_trendline(dplot: pd.DataFrame,
                    colx: str,
                    coly: str,
                    params_plot: dict={'color':'r','lw':2},
                    poly: bool=False,
                    lowess: bool=True,
                    linestyle: str= 'solid',
                    params_poly: dict={'deg':1},
                    params_lowess: dict={'frac':0.7,'it':5},
                    ax: plt.Axes = None,
                  **kws
            ) -> plt.Axes:
    """Plot a trendline.

    Args:
        dplot (pd.DataFrame): input dataframe.
        colx (str): x column.
        coly (str): y column.
        params_plot (dict, optional): parameters provided to the plot. Defaults to {'color':'r','linestyle':'solid','lw':2}.
        poly (bool, optional): apply polynomial function. Defaults to False.
        lowess (bool, optional): apply lowess function. Defaults to True.
        params_poly (_type_, optional): parameters provided to the polynomial function. Defaults to {'deg':1}.
        params_lowess (_type_, optional): parameters provided to the lowess function.. Defaults to {'frac':0.7,'it':5}.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `plot` function. 

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs: 
        1. Label with goodness of fit, r (y_hat vs y)
    """
    ax= plt.subplot() if ax is None else ax    
    if poly:
        coef = np.polyfit(dplot[colx], dplot[coly],**params_poly)
        poly1d_fn = np.poly1d(coef)
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        ax.plot(dplot[colx], poly1d_fn(dplot[colx]),linestyle=linestyle, **params_plot,**kws)
    if lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xys_lowess=lowess(dplot[coly], dplot[colx],frac=0.7,it=5)
        ax.plot(xys_lowess[:,0],xys_lowess[:,1],linestyle=linestyle, **params_plot,**kws)
    return ax

# @to_class(rd)
def plot_scatter(
    dplot: pd.DataFrame,
    colx: str,
    coly: str,
    colz: str=None,
    kind: str='scatter',
    trendline_method: str='poly',
    stat_method: str="spearman",
    bootstrapped: bool=False,
    cmap: str='Reds',
    label_colorbar: str=None,
    gridsize: int=25,
    bbox_to_anchor: list=[1,1],
    loc: str='upper left',
    title: str=None,
    params_plot: dict={},
    params_plot_trendline: dict={},
    params_set_label: dict={},
    ax: plt.Axes = None,
    **kws,
    ) -> plt.Axes:
    """Plot scatter.

    Args:
        dplot (pd.DataFrame): input dataframe.
        colx (str): x column.
        coly (str): y column.
        colz (str, optional): z column. Defaults to None.
        kind (str, optional): kind of scatter. Defaults to 'hexbin'.
        trendline_method (str, optional): trendline method ['poly','lowess']. Defaults to 'poly'.
        stat_method (str, optional): method of annoted stats ['mlr',"spearman"]. Defaults to "spearman".
        bootstrapped (bool, optional): bootstrap data. Defaults to False.
        cmap (str, optional): colormap. Defaults to 'Reds'.
        label_colorbar (str, optional): label of the colorbar. Defaults to None.
        gridsize (int, optional): number of grids in the hexbin. Defaults to 25.
        bbox_to_anchor (list, optional): location of the legend. Defaults to [1,1].
        loc (str, optional): location of the legend. Defaults to 'upper left'.
        title (str, optional): title of the plot. Defaults to None.
        params_plot (dict, optional): parameters provided to the `plot` function. Defaults to {}.
        params_plot_trendline (dict, optional): parameters provided to the `plot_trendline` function. Defaults to {}.
        params_set_label (dict, optional): parameters provided to the `set_label` function. Defaults to dict(x=0,y=1).
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `plot` function. 

    Returns:
        plt.Axes: `plt.Axes` object.

    Notes:
        For a rasterized scatter plot set `scatter_kws={'rasterized': True}`
    
    TODOs:
        1. Access the function as an attribute of roux-data i.e. `rd`. 

    """
    ax= plt.subplot() if ax is None else ax
    
    trendline_method = [trendline_method] if isinstance(trendline_method,str) else [] if trendline_method is None else trendline_method
    stat_method = [stat_method] if isinstance(stat_method,str) else [] if stat_method is None else stat_method
    
    dplot=dplot.dropna(subset=[colx,coly]+[] if colz is None else [colz],how='any')
    if kind in ['hexbin']:
        if colz is None: 
            colz='count'
            dplot[colz]=1
        if colz in dplot:
            params_plot['C']=colz
            params_plot['reduce_C_function']=len if colz=='count' else params_plot['reduce_C_function'] if 'reduce_C_function' in params_plot else np.mean        
            params_plot['gridsize']=params_plot['gridsize'] if 'gridsize' in params_plot else gridsize
            params_plot['cmap']=params_plot['cmap'] if 'cmap' in params_plot else cmap
            print(params_plot)
        ax=dplot.plot(kind=kind, x=colx, y=coly, 
    #         C=colz,
            ax=ax,
            **params_plot,
                      **kws,
            )
    else:
        ax=sns.scatterplot(data=dplot,
                            x=colx,
                            y=coly,
                            hue=colz,       
                            palette=cmap,
                            ax=ax,
                           **params_plot,
                           **kws)
        if not colz is None:
            leg=ax.legend(loc=loc,bbox_to_anchor=bbox_to_anchor,title=colz if title is None else title)
            if '\n' in title:
                leg._legend_box.align = "center"
    from roux.viz.ax_ import set_colorbar_label
#     print(colz)
    ax=set_colorbar_label(ax,colz if label_colorbar is None else label_colorbar)
    from roux.viz.ax_ import set_label
    if 'mlr' in stat_method:
        from roux.lib.stat.poly import get_mlr_2_str
        ax=set_label(ax,label=get_mlr_2_str(dplot,colz,[colx,coly]),
                    title=True,params={'loc':'left'})
    if 'spearman' in stat_method or 'pearson' in stat_method:
        from roux.stat.corr import get_corr
        label,r=get_corr(dplot[colx],dplot[coly],method=stat_method[0],
                                       bootstrapped=bootstrapped,
                                       outstr=True,
                                      # n=True
                                     )
        if r>=0:
            params_set_label['loc']=2
        elif r<0:
            params_set_label['loc']=1
        ax=set_label(ax=ax,s=label,**params_set_label)
    from roux.viz.colors import saturate_color
    plot_trendline(dplot,colx,coly,
                    params_plot={'color':saturate_color(params_plot['color'],alpha=1.75) if 'color' in params_plot else None,},
                    poly='poly' in trendline_method,
                    lowess='lowess' in trendline_method,
                   ax=ax, 
                   **params_plot_trendline,
                    )    
    return ax
    
def plot_qq(
    x: pd.Series
    ) -> plt.Axes:
    """plot QQ.

    Args:
        x (pd.Series): input vector.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    import statsmodels.api as sm
    fig = plt.figure(figsize = [3, 3])
    ax = plt.subplot()
    sm.qqplot(x, dist = sc.stats.norm, 
              line = 's', 
              ax=ax)
    ax.set_title("SW test "+pval2annot(sc.stats.shapiro(x)[1],alpha=0.05,fmt='<',linebreak=False))
    from roux.viz.ax_ import set_equallim
    ax=set_equallim(ax)
    return ax

def plot_ranks(
    df1: pd.DataFrame,
    colid: str,
    colx: str,
    coly: str='rank',
    ascending: bool=True,
    # line: bool=False,
    ax=None,
    **kws,
    ) -> plt.Axes:
    """Plot rankings.

    Args:
        dplot (pd.DataFrame): input data.
        colx (str): x column.
        coly (str): y column.
        colid (str): column with unique ids.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `seaborn.scatterplot` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """    
    assert not df1[colid].duplicated().any()
    df1[coly]=df1[colx].rank(ascending=ascending)
    if ax is None:
        fig,ax=plt.subplots(figsize=[2,2])
    ax=sns.scatterplot(data=df1,
                   x=colx,y=coly,
                       **kws,
                   ax=ax)
    # if line:
    ax.set(
        yticks=[int(i) for i in np.linspace(1,len(df1),4)], ## start with 1
            )
    if ascending:
        ax.invert_yaxis()
    return ax
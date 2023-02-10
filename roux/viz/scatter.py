"""For scatter plots."""

import matplotlib.pyplot as plt
import seaborn as sns

from roux.lib.df import *
import pandas as pd
import numpy as np

import logging
from icecream import ic as info
from roux.viz.ax_ import *

## annotations
def annot_outlines(
    data: pd.DataFrame,
    colx: str,
    coly: str,
    column_outlines: str,
    outline_colors: dict,
    style=None,
    legend: bool=True,
    kws_legend:dict={},
    zorder:int=3,
    ax: plt.Axes=None,
    **kws_scatter,
    ) -> plt.Axes:
    """
    Outline points on the scatter plot by categories.
    
    """
    if isinstance(outline_colors,list):
        outline_colors=dict(zip(data[column_outlines].unique(),outline_colors))
        logging.warning(f"Mapping between the categories and the colors of the coutlines: {outline_colors}.")
    for (cat, df_) in data.groupby(column_outlines):
        ax=sns.scatterplot(
            data=df_,
            x=colx,
            y=coly,
            ec=outline_colors[cat],
            linewidth=1,
            s=50,
            fc="none",
            style=style,
            ax=ax,
            legend=False,
            label=f"{df_[column_outlines].unique()[0]} ({len(df_)})" if legend else None,
            zorder=zorder,
            **kws_scatter,
        )
        if legend:
            ax.legend(
                title=column_outlines,
                **kws_legend,
            )  
    return ax

def plot_trendline(
    dplot: pd.DataFrame,
    colx: str,
    coly: str,
    poly: bool=False,
    lowess: bool=True,
    linestyle: str= ':',
    params_poly: dict={'deg':1},
    params_lowess: dict={'frac':0.7,'it':5},
    lw:float=2,
    color:str='k',
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
        ax.plot(dplot[colx], poly1d_fn(dplot[colx]),linestyle=linestyle,**kws)
    if lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xys_lowess=lowess(dplot[coly], dplot[colx],frac=0.7,it=5)
        ax.plot(xys_lowess[:,0],xys_lowess[:,1],linestyle=linestyle,**kws)
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
    resample: bool=False,
    cmap: str='Reds',
    label_colorbar: str=None,
    gridsize: int=25,
    bbox_to_anchor: list=[1,1],
    loc: str='upper left',
    title: str=None,
    show_n:bool=True,
    show_n_prefix:str='',
    # params_plot: dict={},
    params_plot_trendline: dict={},
    params_set_label: dict={},
    verbose: bool=False,
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
        resample (bool, optional): resample data. Defaults to False.
        cmap (str, optional): colormap. Defaults to 'Reds'.
        label_colorbar (str, optional): label of the colorbar. Defaults to None.
        gridsize (int, optional): number of grids in the hexbin. Defaults to 25.
        bbox_to_anchor (list, optional): location of the legend. Defaults to [1,1].
        loc (str, optional): location of the legend. Defaults to 'upper left'.
        title (str, optional): title of the plot. Defaults to None.
        #params_plot (dict, optional): parameters provided to the `plot` function. Defaults to {}.
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
    ## string to list
    stat_method = [stat_method] if isinstance(stat_method,str) else [] if stat_method is None else stat_method
    
    dplot=dplot.dropna(subset=[colx,coly]+[] if colz is None else [colz],how='any')
    if kind in ['hexbin']:
        if colz is None: 
            colz='count'
            dplot[colz]=1
        if colz in dplot:
            kws['C']=colz
            kws['reduce_C_function']=len if colz=='count' else kws['reduce_C_function'] if 'reduce_C_function' in kws else np.mean        
            kws['gridsize']=kws['gridsize'] if 'gridsize' in kws else gridsize
            kws['cmap']=kws['cmap'] if 'cmap' in kws else cmap
            if verbose: print(kws)
        ax=dplot.plot(
            kind=kind,
            x=colx,
            y=coly, 
            ax=ax,
            # **params_plot,
            **kws,
            )
    else:
        ax=sns.scatterplot(data=dplot,
                            x=colx,
                            y=coly,
                            hue=colz,       
                            palette=cmap if not colz is None else None,
                            ax=ax,
                           # **params_plot,
                           **kws)
        if not colz is None:
            leg=ax.legend(loc=loc,bbox_to_anchor=bbox_to_anchor,title=colz if title is None else title)
            if '\n' in title:
                leg._legend_box.align = "center"
    from roux.viz.ax_ import set_colorbar_label
#     print(colz)
    ax=set_colorbar_label(ax,colz if label_colorbar is None else label_colorbar)
    from roux.viz.annot import set_label
    if 'mlr' in stat_method:
        from roux.stat.fit import get_mlr_2_str
        ax=set_label(ax=ax,s=get_mlr_2_str(dplot.loc[:,[colx,coly,colz,]].dropna(),
                                            colz,[colx,coly]),
                     x=0.5,
                     y=1.1,
                     ha='center',
                    # title=True,
                     # params={'loc':'left'}
                    )
    if 'spearman' in stat_method or 'pearson' in stat_method or 'kendalltau' in stat_method:
        from roux.stat.corr import get_corr
        label,r=get_corr(dplot[colx],dplot[coly],method=stat_method[0],
                         resample=resample,
                         outstr=True,
                         kws_to_str=dict(
                             show_n=show_n,
                             show_n_prefix=show_n_prefix,
                         ),
                         verbose=verbose,
                         )
        if not 'loc' in params_set_label:
            if r>=0:
                params_set_label['loc']=2
            elif r<0:
                params_set_label['loc']=1
        ax=set_label(ax=ax,s=label,
                     zorder=5,
                     **params_set_label)
    if not 'color' in params_plot_trendline:
        from roux.viz.colors import saturate_color,get_colors_default
        params_plot_trendline['color']=saturate_color(kws['color'] if 'color' in kws else get_colors_default()[0],
                                    alpha=1.75)
    plot_trendline(dplot,colx,coly,
                    # params_plot=kws,
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

def plot_volcano(
    data: pd.DataFrame,
    colx:str,
    coly:str,
    colindex:str,
    hue:str='x',
    style:str='P=0',
    style_order: list=['o','^'],
    markers: list=['o','^'],    
    show_labels: int=None,
    show_outlines: int=None,
    outline_colors: list=['k'],
    collabel:str=None,
    show_line=True,
    line_pvalue=0.1,
    line_x=0.0,
    show_text: bool=True,
    text_increase: str=None,
    text_decrease: str=None,
    text_diff: str=None,
    legend:bool=False,
    verbose:bool=False,
    p_min:float=None,
    ax:plt.Axes=None,
    outmore:bool=False,
    kws_legend: dict={},
    **kws_scatterplot,
    ) -> plt.Axes:
    """
    [UNDER DEVELOPMENT]Volcano plot.

    Parameters:

    Keyword parameters:

    Returns:
        plt.Axes
    """
    if ax is None:
        fig,ax=plt.subplots(figsize=[4,3])
    if collabel is None:
        collabel=colindex
    assert not data[colindex].duplicated().any()
    from roux.stat.transform import log_pval
    if not coly.lower().startswith('significance'):
        data=data.assign(
            **{style: lambda df: (df[coly]==0).map({True:"^",False:'o'}) },
            # **{style: lambda df: df[coly]==0 },
            )
        logging.warning(f'transforming the coly ("{coly}") values.')
        coly_=f'significance\n(-log10({coly}))'
        data=data.assign(
            **{coly_:lambda df: log_pval(df[coly],p_min=p_min,errors=None)}
            )
        coly=coly_
    elif not style in data:
        data[style]='o'
    data['significance bin']=pd.cut(data[coly],
                        bins=log_pval([0,0.05,0.1,1])[::-1],
                        labels=['ns','q<0.1','q<0.05'],
                        include_lowest=True,
                       )
    assert not data['significance bin'].isnull().any()
    data=(data
          .assign(
        **{'significance direction bin':lambda df: df.apply(lambda x: 'increase' if x[coly]>log_pval(line_pvalue) and x[colx]>=line_x else \
                                                                      'decrease' if x[coly]>log_pval(line_pvalue) and x[colx]<=-1*line_x else \
                                                                      'ns',
                                   axis=1),
                       })
         .sort_values('significance direction bin',ascending=False) # put 'ns' at the background
         )
    assert not data['significance direction bin'].isnull().any()
    if hue=='x':
        hue='significance direction bin'
        kws_scatterplot['hue_order']=['increase','decrease','ns']
        if not 'palette' in kws_scatterplot:
            from roux.viz.colors import get_colors_default
            kws_scatterplot['palette']=[get_colors_default()[2],get_colors_default()[0],get_colors_default()[1]]
    elif hue=='y':
        hue='significance bin'            
    ax=sns.scatterplot(
        data=data,
        x=colx,
        y=coly,
        hue=hue,
        style=style,
        style_order=style_order,
        markers=markers,
        ec=None,
        ax=ax,
        legend=False,
        **kws_scatterplot,
        )
    ## set text
    axlims=get_axlims(ax)
    if show_text:
        if not text_diff is None:
            ax.text(
                x=axlims['x']['min']+(axlims['x']['len']*0.5),
                y=-75,s=text_diff,
                ha='center',va='center',color='gray',
            )
        ax.text(x=ax.get_xlim()[1],
                y=ax.get_ylim()[1],
                s="increase $\\rightarrow$"+(f"\n(n="+str(data.query(expr="`significance direction bin` == 'increase'")[colindex].nunique())+")" if text_increase=='n' else f"\n({text_increase})" if not text_increase is None else ''),
                ha='right',va='bottom',
                color='k' if not 'palette' in kws_scatterplot else kws_scatterplot['palette'][0],
               )
        ax.text(x=ax.get_xlim()[0],
                y=ax.get_ylim()[1],
                s="$\\leftarrow$ decrease"+(f"\n(n="+str(data.query(expr="`significance direction bin` == 'decrease'")[colindex].nunique())+")" if text_increase=='n' else f"\n({text_decrease})" if not text_decrease is None else ''),
                ha='left',va='bottom',
                color='k' if not 'palette' in kws_scatterplot else kws_scatterplot['palette'][1],
               )
    ## set lines
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    for side in [-1,1]:
        ax.plot(
            [xlim[0 if side==-1 else 1],line_x*side,line_x*side],
            [log_pval(line_pvalue),log_pval(line_pvalue),ylim[1]],
            color='gray',linestyle=':',
            )
    ## set labels
    if not show_labels is None: # show_labels overrides show_outlines
        show_outlines=show_labels
    if not show_outlines is None:
        if isinstance(show_outlines,int): 
            ## show_outlines top n
            data1=(data
                .query(expr="`significance direction bin` != 'ns'"))
            data1=(data1
                .sort_values(colx)
                .head(show_outlines)
                .append(
                data1.sort_values(colx)
                .tail(show_outlines)
                ))        
        elif isinstance(show_outlines, dict):
            ## subset
            data1=data.rd.filter_rows(show_outlines)
        elif isinstance(show_outlines, str):
            ## column with categories
            data1=(data
                   .dropna(subset=[show_outlines])
                  )
        if verbose:
            print(data1)
        # plot
        if not isinstance(show_outlines, str):
            # borders 
            ax=sns.scatterplot(
                data=data1,
                x=colx,
                y=coly,
                # hue=show_outlines if isinstance(show_outlines, str) else None,
                ec='k',
                # ec="face",
                lw=4,
                s=50,
                fc="none",
                style=style,
                style_order=style_order,
                markers=markers,
                ax=ax,
                legend=False,
            )
        else:
            ax=annot_outlines(
                data1,
                colx,
                coly,
                column_outlines=show_outlines,
                outline_colors= outline_colors,
                style=style,
                style_order=style_order,
                markers=markers,
                legend=legend,
                kws_legend=kws_legend,
                ax=ax,
                )
    if show_labels: 
        texts=(data1
                .apply(lambda x: ax.text(x=x[colx],
                    y=x[coly],
                    s=x[collabel],
                    ),axis=1)
                .tolist()
            )
        try:
            from adjustText import adjust_text
            adjust_text(texts,
                       arrowprops=dict(arrowstyle='-', color='k'),
                       )
        except:
            logging.error("install adjustText to repel the labels.")
    
    ax.set(
        xlabel='Log$_\mathrm{2}$ Fold Change (LFC)',
        ylabel='Significance\n(-Log$_\mathrm{10}$($q$))',
        xlim=xlim,
        ylim=ylim,
    )
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    if not outmore:
        return ax
    else:
        return ax,data  
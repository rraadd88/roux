from roux.global_imports import *
from roux.viz.ax_ import *

def plot_table(
    df1: pd.DataFrame,
    xlabel: str=None,
    ylabel: str=None,
    annot: bool=True,
    cbar: bool=False,
    linecolor: str='k',
    linewidths: float=1,               
    cmap: str=None,
    sorty: bool=False,
    linebreaky: bool=False,
    scales: tuple=[1,1],
    ax: plt.Axes=None,
    **kws
    ) -> plt.Axes:
    """Plot to show a table.

    Args:
        df1 (pd.DataFrame): input data.
        xlabel (str, optional): x label. Defaults to None.
        ylabel (str, optional): y label. Defaults to None.
        annot (bool, optional): show numbers. Defaults to True.
        cbar (bool, optional): show colorbar. Defaults to False.
        linecolor (str, optional): line color. Defaults to 'k'.
        linewidths (float, optional): line widths. Defaults to 1.
        cmap (str, optional): color map. Defaults to None.
        sorty (bool, optional): sort rows. Defaults to False.
        linebreaky (bool, optional): linebreak for y labels. Defaults to False.
        scales (tuple, optional): scale of the table. Defaults to [1,1].
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `sns.heatmap` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    print(df1.index.name,df1.columns.name)
    if xlabel is None and not df1.index.name is None:
        ylabel=df1.index.name
    if ylabel is None and not df1.columns.name is None:
        xlabel=df1.columns.name
#     print(xlabel,ylabel)
    from roux.viz.colors import make_cmap
    if sorty:
        df1=df1.loc[df1.sum(axis=1).sort_values(ascending=False).index,:]
    if linebreaky:
        df1.index=[linebreaker(s,break_pt=35) for s in df1.index]
    if ax is None:
        fig,ax=plt.subplots(figsize=[(df1.shape[1]*0.6)*scales[0],(df1.shape[0]*0.5)*scales[1]])
    ax=sns.heatmap(df1,
                cmap=make_cmap(['#ffffff','#ffffff']) if cmap is None else cmap,
                annot=annot,cbar=cbar,
                linecolor=linecolor,linewidths=linewidths,
                   ax=ax,
                   **kws,
               )
    ax.xaxis.set_label_position('top') 
    ax.xaxis.set_ticks_position('top')
    for tick in ax.yaxis.get_majorticklabels():
        tick.set_verticalalignment("center")
    ax.patch.set_edgecolor('k')
    ax.patch.set_linewidth(1) 
    set_ylabel(ax=ax,s=df1.index.name if ylabel is None else ylabel,xoff=0.05,yoff=0.01)
    return ax

def plot_crosstab(
    df1: pd.DataFrame,
    cols: list=None,
    alpha: float=0.05,
    method: str=None,#'chi2'|fe
    confusion: bool=False,
    rename_cols: bool=False,
    sort_cols: tuple=(True,True),    
    annot_pval: str='bottom',
    cmap: str='Reds',
    ax: plt.Axes=None,
    **kws,
    ) -> plt.Axes:
    """Plot crosstab table.

    Args:
        df1 (pd.DataFrame): input data
        cols (list, optional): columns. Defaults to None.
        alpha (float, optional): alpha for the stats. Defaults to 0.05.
        method (str, optional): method to check the association ['chi2','FE']. Defaults to None.
        rename_cols (bool, optional): rename the columns. Defaults to True.
        annot_pval (str, optional): annotate p-values. Defaults to 'bottom'.
        cmap (str, optional): colormap. Defaults to 'Reds'.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Raises:
        ValueError: `annot_pval` position should be the allowed one.

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs:
        1. Use `compare_classes` to get the stats.
    """
    if not cols is None:
        dplot=pd.crosstab(df1[cols[0]],df1[cols[1]])
    else:
        dplot=df1.copy()
    dplot=(dplot
        .sort_index(axis=0,ascending=sort_cols[0])
        .sort_index(axis=1,ascending=sort_cols[1])
          )        
    if dplot.shape!=(2,2) or method=='chi2':
        stat,pval,_,_=sc.stats.chi2_contingency(dplot)
        stat_label='${\chi}^2$'
    else:
        stat,pval=sc.stats.fisher_exact(dplot)
        stat_label='OR'
    if dplot.shape==(2,2) and rename_cols:
        dplot=dplot.rename(columns={True:dplot.columns.name,
                                   False:'not'},
                    index={True:dplot.index.name,
                          False:'not'},)
        dplot.columns.name=None
        dplot.index.name=None
        if 'not' in dplot.columns:
            dplot=dplot.loc[:,
                            [s for s in dplot.columns if s!='not']+['not']]
        if 'not' in dplot.index:
            dplot=dplot.loc[[s for s in dplot.index if s!='not']+['not'],
                            :]
    info(stat,pval)
    # dplot=dplot.sort_index(ascending=False,axis=1).sort_index(ascending=False,axis=0)
    ax=plot_table(dplot,
                    cmap=cmap,
                    ax=ax,
                    **kws,
                 )        
    if annot_pval:
        if annot_pval=='bottom':
            kws_set_label=dict(x=0.5,y=-0.2,ha='center',va='center',)
            linebreak=False
        elif annot_pval=='right':
            kws_set_label=dict(x=1,y=0,ha='left',va='bottom',)            
            linebreak=True
        else:
            raise ValueError(annot_pval)
        set_label(s=f"{stat_label}={stat:.1f}"+(', ' if not linebreak else '\n')+pval2annot(pval, alternative='two-sided', alpha=alpha, fmt='<', linebreak=False),
                 **kws_set_label,
                 ax=ax)
    if confusion:    
        ax=annot_confusion_matrix(dplot,ax=ax,
                                      off=0.5)
    return ax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from roux.viz.ax_ import *

def plot_barh(
    df1: pd.DataFrame,
    colx: str,
    coly: str,
    colannnotside: str=None,
    x1: float=None,
    offx: float=0,
    ax: plt.Axes=None,
    **kws
    ) -> plt.Axes:
    """Plot horizontal bar plot with text on them.

    Args:
        df1 (pd.DataFrame): input data.
        colx (str): x column.
        coly (str): y column.
        colannnotside (str): column with annotations to show on the right side of the plot.
        x1 (float): x position of the text.
        offx (float): x-offset of x1, multiplier.
        color (str): color of the bars.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `barh` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """    
    df1['y']=range(len(df1[coly]))
    if ax is None:
        fig,ax=plt.subplots(figsize=[4.45,len(df1)*0.33])
    if x1 is None:
        x1=get_axlims(ax)['x']['min']+(get_axlims(ax)['x']['len']*(0.05*(1-offx)))
    ax=df1.set_index(coly)[colx].plot.barh(width=0.8,ax=ax,**kws)
    df1['yticklabel']=[t.get_text() for t in ax.get_yticklabels()]
    _=df1.apply(lambda x: ax.text(x1,x['y'],x['yticklabel'],va='center'),axis=1)
    if colannnotside is None:
        colannnotside=colx
    _=df1.apply(lambda x: ax.text(ax.get_xlim()[1],x['y'],f" {x[colannnotside]}",va='center'),axis=1)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()
    ax.set(xlabel=colx,)
    return ax

def plot_value_counts(
    df: pd.DataFrame,
    col: str,
    logx: bool=False,
    kws_hist: dict={'bins':10},
    kws_bar: dict={},
    grid: bool=False,
    axes: list=None,
    fig: object=None,
    hist: bool=True
    ):
    """Plot pandas's `value_counts`. 

    Args:
        df (pd.DataFrame): input data `value_counts`.
        col (str): column with counts.
        logx (bool, optional): x-axis on log-scale. Defaults to False.
        kws_hist (_type_, optional): parameters provided to the `hist` function. Defaults to {'bins':10}.
        kws_bar (dict, optional): parameters provided to the `bar` function. Defaults to {}.
        grid (bool, optional): show grids or not. Defaults to False.
        axes (list, optional): list of `plt.axes`. Defaults to None.
        fig (object, optional): figure object. Defaults to None.
        hist (bool, optional): show histgram. Defaults to True.
    """
    dplot=pd.DataFrame(df[col].value_counts()).sort_values(by=col,ascending=True)
    figsize=(4, len(dplot)*0.4)
    if axes is None:
        if hist:
            fig, axes = plt.subplots(2,1, sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 6]},
                                       figsize=figsize,
                                    )
        else:
            fig, axes = plt.subplots(1,1, 
                                    figsize=figsize,
                                    )
    if hist:
        _=dplot.plot.hist(ax=axes[0],legend=False,**kws_hist
        #                   orientation="horizontal"
                         )
        axbar=axes[1]
    else:
        axbar=axes
    _=dplot.plot.barh(ax=axbar,legend=False,**kws_bar)
    axbar.set_xlabel('count')
    from roux.lib.str import linebreaker
    axbar.set_ylabel(col.replace(' ','\n'))
    if logx:
        if hist:
            for ax in axes.flat:
                ax.set_xscale("log")
                if grid:
                    ax.set_axisbelow(False)
        else:
            axes.set_xscale("log")
            if grid:
                axes.set_axisbelow(False)

def plot_barh_stacked_percentage(
    df1: pd.DataFrame,
    coly: str,
    colannot: str,
    color: str=None,
    yoff: float=0,
    ax: plt.Axes = None,
    ) -> plt.Axes:
    """Plot horizontal stacked bar plot with percentages.

    Args:
        df1 (pd.DataFrame): input data. values in rows sum to 100%.
        coly (str): y column. yticklabels, e.g. retained and dropped.
        colannot (str): column with annotations.
        color (str, optional): color. Defaults to None.
        yoff (float, optional): y-offset. Defaults to 0.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """

    from roux.viz.ax_ import get_ticklabel2position
    from roux.viz.colors import get_colors_default
    if color is None:
        color=get_colors_default()[0]
    ax=plt.subplot() if ax is None else ax
    df2=df1.set_index(coly).apply(lambda x: (x/sum(x))*100, axis=1)
    ax=df2.plot.barh(stacked=True,ax=ax)
    ticklabel2position=get_ticklabel2position(ax,'y')
    from roux.viz.colors import saturate_color
    _=df2.reset_index().apply(lambda x: ax.text(1,
                                                  ticklabel2position[x[coly]]-yoff,
                                                  f"{x[colannot]:.1f}%",ha='left',va='center',
                                                 color=saturate_color(color,2),
                                               ),
                                axis=1)
    ax.legend(bbox_to_anchor=[1,1],title=df1.columns.name)
    d1=df1.set_index(coly).T.sum().to_dict()
    ax.set(xlim=[0,100],xlabel='%',
          yticklabels=[f"{t.get_text()}\n(n={d1[t.get_text()]})" for t in ax.get_yticklabels()])
    return ax

def plot_bar_serial(
    d1: dict,
    polygon: bool=False,
    polygon_x2i: float=0,
    labelis: list=[],
    y: float=0,
    ylabel: str=None,
    off_arrowy: float=0.15,
    kws_rectangle=dict(height=0.5,linewidth=1),
    ax: plt.Axes = None,
    ) -> plt.Axes:
    """Barplots with serial increase in resolution.

    Args:
        d1 (dict): dictionary with the data.
        polygon (bool, optional): show polygon. Defaults to False.
        polygon_x2i (float, optional): connect polygon to this subset. Defaults to 0.
        labelis (list, optional): label these subsets. Defaults to [].
        y (float, optional): y position. Defaults to 0.
        ylabel (str, optional): y label. Defaults to None.
        off_arrowy (float, optional): offset for the arrow. Defaults to 0.15.
        kws_rectangle (_type_, optional): parameters provided to the `rectangle` function. Defaults to dict(height=0.5,linewidth=1).
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """

    kws1=dict(
    xs=[(list(d1.values())[i]/sum(list(d1.values())))*100 for i in range(len(d1))],
    cs=['#f55f5f', '#D3DDDC'] if len(d1)==2 else ['#f55f5f','#e49e9d', '#D3DDDC'],
    labels=[],
    size=sum(list(d1.values())),
    xmax=100,
    y=y,
    ax=None,
    )
    
    import matplotlib.patches as patches
    l1=[
        patches.Rectangle((0,kws1['y']), kws1['xmax'],  
                          fc='none', ec='k',alpha=1,zorder=2,**kws_rectangle),
    ]
    x=0
    for i,(x_,c,s) in enumerate(zip(kws1['xs'],kws1['cs'],d1.keys())):
        print(x,x_)
        l1.append(patches.Rectangle((x, kws1['y']), x_, 
                          fc=c, ec='none',alpha=1,**kws_rectangle))
        if i in labelis:
            ax.text(x if i!=len(d1.keys())-1 else x+x_,
                    kws1['y']+kws_rectangle['height']*0.5,
                    f"{s} ({x_:.0f})%",
                    ha='left' if i!=len(d1.keys())-1 else 'right',
                    va='center')
        x+=x_
        if polygon:
            if polygon_x2i==i:
                l1.append(patches.Polygon(xy=[(0,kws1['y']),
                                              (x,kws1['y']),
                                              (100,kws1['y']-1),
                                              (0,kws1['y']-1),], closed=True,fc=[0.95,0.95,0.95],
                                          ec='gray',
                                          lw=0.1,
                                          zorder=-2))        
    #     break
    s=num2str(kws1['size'],magnitude=True,decimals=1)
    ax.annotate(s=s, xy=(0,kws1['y']-off_arrowy), xytext=(50,kws1['y']-off_arrowy), arrowprops=dict(arrowstyle='->',shrinkA=0,shrinkB=0,color='k'),zorder=-1,
               va='center',ha='center')
    ax.annotate(s=' '*(len(s)*2), xy=(100,kws1['y']-off_arrowy), xytext=(50,kws1['y']-off_arrowy), arrowprops=dict(arrowstyle='->',shrinkA=0,shrinkB=0,color='k'),zorder=-1,
               va='center',ha='center')
    _=[ax.add_patch(o) for o in l1]
    if not ylabel is None:
        ax.text(-2.5,kws1['y']+kws_rectangle['height']*0.5,ylabel,ha='right',va='center')
            
    return ax

def plot_barh_stacked_percentage_intersections(
    df0: pd.DataFrame,
    colxbool: str,
    colybool: str,
    colvalue: str,
    colid: str,
    colalt: str,
    colgroupby: str,
    coffgroup: float=0.95,
    ax: plt.Axes = None,
    ) -> plt.Axes:
    """Plot horizontal stacked bar plot with percentages and intesections.

    Args:
        df0 (pd.DataFrame): input data.
        colxbool (str): x column.
        colybool (str): y column.
        colvalue (str): column with the values.
        colid (str): column with ids.
        colalt (str): column with the alternative subset.
        colgroupby (str): column with groups.
        coffgroup (float, optional): cut-off between the groups. Defaults to 0.95.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.

    Examples:
        Parameters:
            colxbool='paralog',
            colybool='essential',
            colvalue='value',
            colid='gene id',
            colalt='singleton',
            coffgroup=0.95,
            colgroupby='tissue',

    """

    ##1 threshold for value by group
    def apply_(df):
        coff=np.quantile(df.loc[df[colybool],colvalue],coffgroup)
        df[colybool]=df[colvalue]<coff
        return df
    df1=df0.groupby(colgroupby).progress_apply(apply_)
    ##2 % 
    df2=df1.groupby([colid,colxbool]).agg({colybool: perc}).reset_index().rename(columns={colybool:f'% {colgroupby}s with {colybool}'},
                                                                                        errors='raise')
    coly=f"% of {colgroupby}s"
    ##3 bin y
    df2[coly]=pd.cut(df2[f'% {colgroupby}s with {colybool}'],bins=pd.interval_range(0,100,4),)
    ##3 % sum
    df3=df2.groupby(coly)[colxbool].agg([sum]).rename(columns={'sum':colxbool})
    dplot=df3.join(df2.groupby(coly).size().to_frame('total'))
    dplot[colalt]=dplot['total']-dplot[colxbool]
    dplot.index=[str(i) for i in dplot.index]
    dplot.index.name=coly
    dplot.columns.name=f"{colid} type"
    dplot=dplot.sort_values(coly,ascending=False)
    dplot=dplot.reset_index()
#     from roux.viz.bar import plot_barh_stacked_percentage
    fig,ax=plt.subplots(figsize=[3,3])
    plot_barh_stacked_percentage(df1=dplot.loc[:,[coly,colxbool,colalt]],
                                coly=coly,
                                colannot=colxbool,
                                ax=ax)
    set_ylabel(ax)
    ax.set(xlabel=f'% of {colid}s',
          ylabel=None)
    return ax

# redirections, to be deprecated in the future
from roux.viz.sets import plot_intersections

## plotly
def to_input_data_sankey(df0,cols_groupby,colid,width=20):    
    df0=df0.dropna(subset=df0.filter(like='group').columns.tolist(),how='all')
    df0['all']='all'
    ## map labels
    df0=df0.applymap(lambda x : linebreaker(d0[x],width,sep='<br>') if x in d0 else x)
    # df0

    df1=pd.concat([
        df0.groupby(list(cols))[colid].nunique() for cols in np.lib.stride_tricks.sliding_window_view(cols_groupby[1:],2)
    ],
             axis=0
             ).to_frame('count').rename_axis(['source','target'],axis=0).reset_index().drop_duplicates()
    df1

    df2=pd.concat([df0.groupby(list(cols))[colid].nunique() for cols in itertools.product(['all'],cols_groupby[1:])
    ],
    axis=0
    ).to_frame('total').rename_axis(['source','target'],axis=0).reset_index().drop_duplicates()
    # df2

    df3=(
    df2
    .merge(df1.groupby('target')['count'].sum().to_frame('substract').reset_index(),
              on=['target'],
             how='left',
             )
    .assign(**{
        'substract':lambda x: x['substract'].fillna(0),    
        'count':lambda x: x['total']-x['substract']})
    .append(df1)
    )
    # map ids
    labels=list(pd.unique(df3['source'].unique().tolist()+df3['target'].unique().tolist()))
    d_={s:i for i,s in enumerate(labels)}
    for c in ['source','target']:
        df3[c+' id']=df3[c].map(d_)
    return df3,labels

def plot_sankey(
    df0,
    cols_groupby,
    colcount,
    node_color=None,
    link_color=None,
    outp=None,
    test=False,
    **kws,
    ):
    df1,labels=to_input_data_sankey(df0,cols_groupby)
    import plotly.io as pio
    pio.renderers.default = 'iframe'
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Sankey(
        arrangement='fixed',
        node = dict(
            pad = 10,
            thickness = 20,
            line = dict(
                color = '#FFFFFF',
                width = 0.1,
                ),
            label =labels,
            color=node_color,
            ),
        link = dict(
            source = df1['source id'],
            target = df1['target id'],
            value = df1['count'],
            color=link_color,
            )
      )])
    fig.update_layout(
                    **kws,
                     )
    # fig.show()
    if not outp is None:
        fig.write_image(outp,format=Path(outp).suffix[1:], engine="kaleido")
    return fig
from roux.global_imports import * 

## set
def set_(
    ax: plt.Axes,
    test: bool=False,
    **kws
    ) -> plt.Axes:
    """Ser many axis parameters.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        test (bool, optional): test mode. Defaults to False.

    Keyword Args:
        kws: parameters provided to the `ax.set` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    kws1={k:v for k,v in kws.items() if not isinstance(v,dict)}
    kws2={k:v for k,v in kws.items() if isinstance(v,dict)}    
    ax.set(**kws1)
    for k,v in kws2.items():
        getattr(ax,f"set_{k}")(**v)
    return ax

def set_ylabel(
    ax: plt.Axes,
    s: str=None,
    x: float=-0.1,
    y: float=1.02,
    xoff: float=0,
    yoff: float=0
    ) -> plt.Axes:
    """Set ylabel horizontal.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        s (str, optional): ylabel. Defaults to None.
        x (float, optional): x position. Defaults to -0.1.
        y (float, optional): y position. Defaults to 1.02.
        xoff (float, optional): x offset. Defaults to 0.
        yoff (float, optional): y offset. Defaults to 0.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if (not s is None):# and (ax.get_ylabel()=='' or ax.get_ylabel() is None):
        ax.set_ylabel(s)
    ax.set_ylabel(ax.get_ylabel(),rotation=0,ha='right',va='center')
    ax.yaxis.set_label_coords(x+xoff,y+yoff) 
    return ax
#     return set_label(x=x,y=y,s=ax.get_ylabel() if s is None else s,
#                                                        ha='right',va='bottom',ax=ax)

def rename_labels(ax,d1):
    ax.set_xlabel(replace_many(ax.get_xlabel(),d1,ignore=True))
    ax.set_ylabel(replace_many(ax.get_ylabel(),d1,ignore=True))
    ax.set_title(replace_many(ax.get_title(),d1,ignore=True))
    return ax

## ticklabels
def rename_ticklabels(
    ax: plt.Axes,
    axis: str,
    rename: dict=None,
    replace: dict=None,
    ignore: bool=False
    ) -> plt.Axes:
    """Rename the ticklabels.

    Args:
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        axis (str): axis (x|y).
        rename (dict, optional): replace strings. Defaults to None.
        replace (dict, optional): replace sub-strings. Defaults to None.
        ignore (bool, optional): ignore warnings. Defaults to False.

    Raises:
        ValueError: either `rename` or `replace` should be provided.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    k=f"{axis}ticklabels"
    if not rename is None:
        _=getattr(ax,f"set_{k}")([rename[t.get_text()] for t in getattr(ax,f"get_{k}")()])
    elif not replace is None:
        _=getattr(ax,f"set_{k}")([replace_many(t.get_text(),replace,ignore=ignore) for t in getattr(ax,f"get_{k}")()])
    else:
        raise ValueError("either `rename` or `replace` should be provided.")
    return ax

def get_ticklabel2position(
    ax: plt.Axes,
    axis: str,
    ) -> plt.Axes:
    """Get positions of the ticklabels.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        axis (str): axis (x|y).

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    return dict(zip([t.get_text() for t in getattr(ax,f'get_{axis}ticklabels')()],
                  getattr(ax,f"{axis}axis").get_ticklocs()))

def set_ticklabels_color(
    ax: plt.Axes,
    ticklabel2color: dict,
    axis: str='y'
    ) -> plt.Axes:
    """Set colors to ticklabels.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        ticklabel2color (dict): colors of the ticklabels.
        axis (str): axis (x|y).

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    for tick in getattr(ax,f'get_{axis}ticklabels')():
        if tick.get_text() in ticklabel2color.keys():
            tick.set_color(ticklabel2color[tick.get_text()])
    return ax 
color_ticklabels=set_ticklabels_color

def format_ticklabels(
    ax: plt.Axes,
    axes: tuple=['x','y'],
    n: int=None,
    fmt: str=None,
    font: str=None,#'DejaVu Sans Mono',#"Monospace"
    ) -> plt.Axes:
    """format_ticklabels _summary_

    Args:
        ax (plt.Axes): `plt.Axes` object.
        axes (tuple, optional): axes. Defaults to ['x','y'].
        n (int, optional): number of ticks. Defaults to None.
        fmt (str, optional): format. Defaults to None.
        font (str, optional): font. Defaults to 'DejaVu Sans Mono'.

    Returns:
        plt.Axes: `plt.Axes` object.
        
    TODOs: 
        1. include color_ticklabels
    """
    if isinstance(n,int):
        n={'x':n,
           'y':n}
    if isinstance(fmt,str):
        fmt={'x':fmt,
           'y':fmt}
    for axis in axes:
        if not n is None:        
            getattr(ax,axis+'axis').set_major_locator(plt.MaxNLocator(n[axis]))
        if not fmt is None:
            getattr(ax,axis+'axis').set_major_formatter(plt.FormatStrFormatter(fmt[axis]))
        if not font is None:
            for tick in getattr(ax,f'get_{axis}ticklabels')():
                tick.set_fontname(font)
    return ax

## lims
def set_equallim(
    ax: plt.Axes,
    diagonal: bool=False,
    difference: float=None,
    **kws_format_ticklabels,
    ) -> plt.Axes:
    """Set equal axis limits.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        diagonal (bool, optional): show diagonal. Defaults to False.
        difference (float, optional): difference from . Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    min_,max_=np.min([ax.get_xlim()[0],ax.get_ylim()[0]]),np.max([ax.get_xlim()[1],ax.get_ylim()[1]])
    if diagonal:
        ax.plot([min_,max_],[min_,max_],':',color='gray',zorder=5)
    if not difference is None:
        off=np.sqrt(difference**2+difference**2)
        ax.plot([min_+off ,max_+off],[min_,max_],':',color='gray',zorder=5)        
        ax.plot([min_-off ,max_-off],[min_,max_],':',color='gray',zorder=5)        
    ax=format_ticklabels(ax,n=len(ax.get_xticklabels()),**kws_format_ticklabels)
    ax.set_xlim(min_,max_)
    ax.set_ylim(min_,max_)
#     ax.set_xticks(ax.get_yticks())
    ax.set_aspect('equal', 'box')
    return ax

def get_axlims(ax: plt.Axes
    ) -> plt.Axes:
    """Get axis limits.

    Args:
        ax (plt.Axes): `plt.Axes` object.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    d1={}
    for axis in ['x','y']:
        d1[axis]={}
        d1[axis]['min'],d1[axis]['max']=getattr(ax,f'get_{axis}lim')()
        if d1[axis]['min'] > d1[axis]['max']:
            d1[axis]['min'],d1[axis]['max']=d1[axis]['max'],d1[axis]['min']
        d1[axis]['len']=abs(d1[axis]['min']-d1[axis]['max'])
    return d1


# axis limits    
def set_axlims(
    ax: plt.Axes,
    off: float,
    axes: list=['x','y']
    ) -> plt.Axes:
    """Set axis limits.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        off (float): offset.
        axes (list, optional): axis name/s. Defaults to ['x','y'].

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    logging.warning("prefer `ax.margins`")
    d1=get_axlims(ax)
    for k in axes:
        off_=(d1[k]['len'])*off
        if not getattr(ax,f"{k}axis").get_inverted():
            getattr(ax,f"set_{k}lim")(d1[k]['min']-off_,d1[k]['max']+off_)
        else:
            getattr(ax,f"set_{k}lim")(d1[k]['max']+off_,d1[k]['min']-off_)            
    return ax

def get_axlimsby_data(
    X: pd.Series,
    Y: pd.Series,
    off: float=0.2,
    equal: bool=False
    ) -> plt.Axes:
    """Infer axis limits from data.

    Args:
        X (pd.Series): x values.
        Y (pd.Series): y values.
        off (float, optional): offsets. Defaults to 0.2.
        equal (bool, optional): equal limits. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    try:
        xmin=np.min(X)
        xmax=np.max(X)
    except:
        print(X)
    xlen=xmax-xmin
    ymin=np.min(Y)
    ymax=np.max(Y)
    ylen=ymax-ymin
    xlim=(xmin-off*xlen,xmax+off*xlen)
    ylim=(ymin-off*ylen,ymax+off*ylen)
    if not equal:
        return xlim,ylim
    else:
        lim=[np.min([xlim[0],ylim[0]]),np.max([xlim[1],ylim[1]])]
        return lim,lim

def split_ticklabels(
    ax: plt.Axes,
    axis='x',
    grouped=False,
    group_x=0.01,
    group_prefix=None,
    group_loc='left',
    # group_pad=0.02,
    group_colors=None,
    group_alpha=0.2,
    show_group_line=True,
    show_group_span=True,
    sep: str='-',
    pad_major=6,
    **kws,
    ) -> plt.Axes:
    """Split ticklabels into major and minor. Two minor ticks are created per major tick. 

    Args:
        ax (plt.Axes): `plt.Axes` object.
        sep (str, optional): separator within the tick labels. Defaults to ' '.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    ticklabels=getattr(ax,f'get_{axis}ticklabels')()
    import pandas as pd
    if not grouped:
        if axis=='y':logging.warning(f'axis={axis} is not tested.')        
        ticklabels_major=pd.unique(['\u2014\n'+s.get_text().split(sep)[0] for s in ticklabels])
        ticklabels_minor=[s.get_text().split(sep)[1] for s in ticklabels]

        ticks_minor=getattr(ax,f'get_{axis}ticks')()
        ticks_major=ticks_minor.reshape(int(len(ticks_minor)/2),2).mean(axis=1)
        _=getattr(ax,f'set_{axis}ticks')(ticks_major, minor=False)
        getattr(ax,f'set_{axis}ticklabels')(ticklabels_major,minor=False,**kws,)
        _=ax.set_xticks( ticks_minor, minor=True )
        getattr(ax,f'set_{axis}ticklabels')(ticklabels_minor,minor=True,**kws,)
        ax.tick_params(axis=axis, which='minor', bottom=True,pad=0)
        ax.tick_params(axis=axis, which='major', bottom=False,pad=pad_major)
    else:
        if axis=='x':logging.warning(f'axis={axis} is not tested.')
        from roux.lib.df import dict2df
        df0_=dict2df(get_ticklabel2position(ax=ax,axis=axis),
                   colkey=axis+'ticklabel',colvalue=axis)
        df0_[axis+'ticklabel major']=df0_[axis+'ticklabel'].str.split(sep,1,expand=True)[0]
        df0_[axis+'ticklabel minor']=df0_[axis+'ticklabel'].str.split(sep,1,expand=True)[1]
        df_=(df0_
        .groupby(axis+'ticklabel major')
        .agg({axis:[min,max,len],})
        .rd.flatten_columns()
        )
        axlims=get_axlims(ax)
        # if group_loc=='left':
        #     group_x=group_x
        #     group_x=axlims[axis]['min']-(axlims[axis]['len']*group_pad)
        #     group_xlabel=axlims[axis]['min']-(axlims[axis]['len']*group_pad-0.1)
        # elif group_loc=='right':
        #     group_x=axlims[axis]['max']+(axlims[axis]['len']*group_pad)
        #     group_xlabel=axlims[axis]['max']+(axlims[axis]['len']*group_pad+0.1)
        # print(axlims[axis]['min']-(group_pad*5.5))
        if show_group_span:
            # print(axlims[axis]['min']-(group_pad*5.5))
            # print(group_x)
            df_.apply(lambda x: ax.axhspan(
            # xmin=(group_x/axlims[axis]['len'])*0.5,
            # xmin=axlims[axis]['min']-(group_pad*5.5),
            xmin=group_x,
            xmax=axlims[axis]['min'], 
            ymin=x[axis+' min']-0.5,
            ymax=x[axis+' max']+0.5, 
            transform="axes",
            clip_on=False,
            zorder=0,
            color=None if group_colors is None else group_colors[x.name],
            edgecolor='none',
            alpha=group_alpha,
            ),
            axis=1)
        if show_group_line:
            df_.apply(lambda x: ax.plot(
                [group_x,group_x],
                [x[axis+' min']-0.2,x[axis+' max']+0.2],
                clip_on=False,
                lw=0.5,
                color='k',
                # transform="axes",
                ),
                axis=1)
        from roux.lib.set import get_alt
        import matplotlib.transforms as transforms
        df_.apply(lambda x: ax.text(
                                    x=group_x,#label,
                                    y=np.mean([x[axis+' min'],x[axis+' max']]),
                                    s=(group_prefix+'\n' if not group_prefix is None else '')+f"{x.name}".replace(' ','\n')+'\n'+f"(n={int(x[axis+' len'])})",
                                    color='k',
                                    # ha=get_alt(['left','right'],group_loc),
                                    ha=group_loc,
                                    va='center',
                                    transform=transforms.blended_transform_factory(ax.transAxes,ax.transData),
                                    # transform=ax.transAxes,
                                   ),axis=1)
        getattr(ax,f'set_{axis}ticklabels')([s.get_text().split(sep,1)[1] for s in ticklabels],
                                            **kws,)        
    return ax

def set_grids(
    ax: plt.Axes,
    axis: str=None
    ) -> plt.Axes:
    """Show grids.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        axis (str, optional): axis name. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    w,h=ax.figure.get_size_inches()
    if w/h>=1.1 or axis=='y' or axis=='both':
        ax.set_axisbelow(True)
        ax.yaxis.grid(color='gray', linestyle='dashed')
    if w/h<=0.9 or axis=='x' or axis=='both':
        ax.set_axisbelow(True)
        ax.xaxis.grid(color='gray', linestyle='dashed')
    return ax

## legends
def rename_legends(
    ax: plt.Axes,
    replaces: dict,
    **kws_legend
    ) -> plt.Axes:
    """Rename legends.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        replaces (dict): _description_

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    handles, labels = ax.get_legend_handles_labels()
    labels=[str(s) for s in labels]
    if len(set(labels) - set(replaces.keys()))==0:
        labels=[replaces[s] for s in labels]
    else:
        labels=[replacemany(s,replaces) for s in labels]
    return ax.legend(handles=handles,labels=labels,
                     title=ax.get_legend().get_title().get_text(),
                     **kws_legend)

def append_legends(
    ax: plt.Axes,
    labels: list,
    handles: list,
    **kws
    ) -> plt.Axes:
    """Append to legends.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        labels (list): labels. 
        handles (list): handles.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    h1, l1 = ax.get_legend_handles_labels()
    print(l1)
    ax.legend(handles=h1+handles,
              labels=l1+labels,
              **kws)
    return ax

## legend related stuff: also includes colormaps

def sort_legends(
    ax: plt.Axes,
    sort_order: list=None,
    **kws
    ) -> plt.Axes:
    """Sort or filter legends.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        sort_order (list, optional): order of legends. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
        
    Notes:
        1. Filter the legends by providing the indices of the legends to keep.
    """
    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    if sort_order is None:
        handles,labels = zip(*sorted(zip(handles,labels), key=lambda t: t[1]))
    else:
        if all([isinstance(i,str) for i in sort_order]):
            sort_order=[labels.index(s) for s in sort_order]
        if not all([isinstance(i,int) for i in sort_order]):
            raise ValueError("sort_order should contain all integers")
        handles,labels =[handles[idx] for idx in sort_order],[labels[idx] for idx in sort_order]
        # print(handles,labels)
    return ax.legend(handles, labels,**kws)

def drop_duplicate_legend(ax,**kws): return sort_legends(ax=ax,sort_order=None,**kws)

def reset_legend_colors(ax):
    """Reset legend colors.

    Args:
        ax (plt.Axes): `plt.Axes` object.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    leg=plt.legend()
    for lh in leg.legendHandles: 
        lh.set_alpha(1)
#         lh._legmarker.set_alpha(1)
    return ax

def set_legends_merged(axs):
    """Reset legend colors.

    Args:
        axs (list): list of `plt.Axes` objects.

    Returns:
        plt.Axes: first `plt.Axes` object in the list.
    """
    df_=pd.concat([pd.DataFrame(ax.get_legend_handles_labels()[::-1]).T for ax in axs],
             axis=0)
    df_['fc']=df_[1].apply(lambda x: x.get_fc())
    df_=df_.log.drop_duplicates(subset=[0,'fc'])
    if df_[0].duplicated().any(): logging.error("duplicate legend labels")
    return axs[1].legend(handles=df_[1].tolist(), labels=df_[0].tolist(),
                       bbox_to_anchor=[-0.2,0],loc=2,frameon=True).get_frame().set_edgecolor((0.95,0.95,0.95))

def set_legend_custom(
    ax: plt.Axes,
    legend2param: dict,
    param: str='color',
    lw: float=1,
    marker: str='o',
    markerfacecolor: bool=True,
    size: float=10,
    color: str='k',
    linestyle: str='',
    title_ha: str='center',
    frameon: bool=True,
    **kws
    ) -> plt.Axes:
    """Set custom legends.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        legend2param (dict): legend name to parameter to change e.g. name of the color. 
        param (str, optional): parameter to change. Defaults to 'color'.
        lw (float, optional): line width. Defaults to 1.
        marker (str, optional): marker type. Defaults to 'o'.
        markerfacecolor (bool, optional): marker face color. Defaults to True.
        size (float, optional): size of the markers. Defaults to 10.
        color (str, optional): color of the markers. Defaults to 'k'.
        linestyle (str, optional): line style. Defaults to ''.
        title_ha (str, optional): title horizontal alignment. Defaults to 'center'.
        frameon (bool, optional): show frame. Defaults to True.

    Returns:
        plt.Axes: `plt.Axes` object.
        
    TODOs:
        1. differnet number of points for eachh entry
        
            from matplotlib.legend_handler import HandlerTuple
            l1, = plt.plot(-1, -1, lw=0, marker="o",
                        markerfacecolor='k', markeredgecolor='k')
            l2, = plt.plot(-0.5, -1, lw=0, marker="o",
                        markerfacecolor="none", markeredgecolor='k')
            plt.legend([(l1,), (l1, l2)], ["test 1", "test 2"], 
                    handler_map={tuple: HandlerTuple(2)}
                    )
    
    References: 
        https://matplotlib.org/stable/api/markers_api.html
        http://www.cis.jhu.edu/~shanest/mpt/js/mathjax/mathjax-dev/fonts/Tables/STIX/STIX/All/All.html
    """
    from matplotlib.lines import Line2D
    legend_elements=[Line2D([0], [0],
                       marker=marker,
                       color=color if param!='color' else legend2param[k],
                       markeredgecolor=(color if param!='color' else legend2param[k]), 
                       markerfacecolor=(color if param!='color' else legend2param[k]) if not markerfacecolor is None else 'none',
                       markersize=(size if param!='size' else legend2param[k]),
                       label=k,
                       lw=(lw if param!='lw' else legend2param[k]),
                       linestyle=linestyle if param!='lw' else '-',
                      ) for k in legend2param]   
    o1=ax.legend(handles=legend_elements,frameon=frameon,
              **kws)
    o1._legend_box.align=title_ha
    o1.get_frame().set_edgecolor((0.95,0.95,0.95))
    return ax

## line round
def get_line_cap_length(
    ax: plt.Axes,
    linewidth: float
    ) -> plt.Axes:
    """Get the line cap length.

    Args:
        ax (plt.Axes): `plt.Axes` object
        linewidth (float): width of the line.

    Returns:
        plt.Axes: `plt.Axes` object
    """
    radius = linewidth / 2
    ppd = 72. / ax.figure.dpi  # points per dot
    trans = ax.transData.inverted().transform
    x_radius = ((trans((radius / ppd, 0)) - trans((0, 0))))[0]
    y_radius = ((trans((0, radius / ppd)) - trans((0, 0))))[1]
    return x_radius,y_radius


# shape
def get_subplot_dimentions(ax=None):
    """Calculate the aspect ratio of `plt.Axes`.

    Args:
        ax (plt.Axes): `plt.Axes` object

    Returns:
        plt.Axes: `plt.Axes` object

    References: 
        https://github.com/matplotlib/matplotlib/issues/8013#issuecomment-285472404    
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    ll, ur = ax.get_position() * fig.get_size_inches()
    width, height = ur - ll
    return width, height,height / width

# colorbar
def set_colorbar(
    fig: object,
    ax: plt.Axes,
    ax_pc: plt.Axes,
    label: str,
    bbox_to_anchor: tuple=(0.05, 0.5, 1, 0.45),
    orientation: str="vertical",
    ):
    """Set colorbar.

    Args:
        fig (object): figure object.
        ax (plt.Axes): `plt.Axes` object.
        ax_pc (plt.Axes): `plt.Axes` object for the colorbar.
        label (str): label
        bbox_to_anchor (tuple, optional): location. Defaults to (0.05, 0.5, 1, 0.45).
        orientation (str, optional): orientation. Defaults to "vertical".

    Returns:
        figure object.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    if orientation=="vertical":
        width,height="5%","50%"
    else:
        width,height="50%","5%"        
    axins = inset_axes(ax,
                       width=width,  # width = 5% of parent_bbox width
                       height=height,  # height : 50%
                       loc=2,
                       bbox_to_anchor=bbox_to_anchor,
                       bbox_transform=ax.transAxes,
                       borderpad=0,
                       )
    fig.colorbar(ax_pc, cax=axins,
                 label=label,orientation=orientation,)
    return fig

def set_colorbar_label(
    ax: plt.Axes,
    label: str
    ) -> plt.Axes:
    """Find colorbar and set label for it.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        label (str): label.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    for a in ax.figure.get_axes()[::-1]:
        if a.properties()['label']=='<colorbar>':
            if hasattr(a,'set_ylabel'):
                a.set_ylabel(label)
                break
    return ax
    
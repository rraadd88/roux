"""For setting up figures."""

import logging

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
## types
from matplotlib.axes import Axes
from typing import Tuple

def to_pos_in_ax2(
    ax: Axes,
    ax2: Axes,
    pt: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Converts a point from ax's data coordinates to ax2's data coordinates.

    Args:
        ax: The source Axes object.
        ax2: The target Axes object.
        pt: A tuple (x, y) in ax's data coordinates.

    Returns:
        A tuple (x_prime, y_prime) representing the point in 
        ax2's data coordinates.
    """
    # Convert points from their respective data spaces to a common figure space (pixels)
    # g: 1. Convert source point (ax data) to display (pixel) coordinates
    pt_pixels = ax.transData.transform(pt)

    # g: 2. Convert display (pixel) coordinates to target point (ax2 data)
    return ax2.transData.inverted().transform(pt_pixels)
    
def fig_grid(
    data,
    plot_func=None,  ## takes data, ax and **kws
    kws_plot={},
    **kws_fig,    
    ):
    """
    Examples:
        func=lambda data,ax,**kws_plot: plot_image(
                data['path'].tolist()[0],
                ax=ax,
                **kws_plot,
            )
    """
    def read_plot_(data,ax,**kws_plot):
        from roux.viz.io import read_plot
        return read_plot(
            data['path'].tolist()[0],
            ax=ax,
            **kws_plot,
            )
    if plot_func is None and 'path' in data:
        plot_func=read_plot_
        
    import seaborn as sns
    g = sns.FacetGrid(
        data,
        **{
            **dict(
                height=2.5,
                aspect=1,
                margin_titles=True,
            ),        
            **kws_fig,
        },
    )
    g.set_titles(
        row_template="{row_name}",
        col_template="{col_name}",
        )
    def _map_plot(*args, **kwargs):
        ## extract the inputs
        ax=plt.gca()
        kwargs['plot_func'](
            kwargs['data'],
            ax=ax,
            **kwargs['kws_plot'],
        )
        return ax
        
    if plot_func is None:
        logging.warning("plot_func is None")
        return g
        
    # print(data.shape)
    
    g.map_dataframe(
        func=_map_plot,
        plot_func=plot_func,
        kws_plot=kws_plot,
        )
    # for (row, col), ax in g.axes_dict.items():
    #     ax.set(
    #         xlim=[xlims.loc[col,'min'],xlims.loc[col,'max']]
    #     )

    # --- 4. Rotate the Right Margin Titles (by finding Text objects) ---
    return g

def set_fig(figsize):
    # 1. Get the current figure and axes
    fig = plt.gcf()
    if list(fig.get_size_inches())!=figsize:
        fig.set_size_inches(figsize[0], figsize[1], forward=True)
    return fig

## subplots in relaation to figs
def set_ax(
    cols_max=4,
    figsize=[8,2],
    ):
    """
    Dynamically adds a subplot, creating new rows after cols_max is reached.

    Args:
        cols_max (int): The maximum number of subplots per row. Defaults to 2.

    Returns:
        matplotlib.axes.Axes: The newly created subplot axes.
    """
    fig=set_fig(figsize)

    existing_axes = fig.axes
    n_existing = len(existing_axes)
    n_new_total = n_existing + 1

    # 2. Calculate the new grid dimensions
    # Use cols_max unless there are fewer total plots than cols_max
    num_cols = min(n_new_total, cols_max)
    # Calculate rows needed for the new total number of plots
    num_rows = (n_new_total + num_cols - 1) // num_cols

    # 3. Create a new GridSpec for the entire figure
    gs = GridSpec(num_rows, num_cols, figure=fig)

    # 4. Reposition all existing subplots within the new GridSpec
    for i, ax in enumerate(existing_axes):
        # Calculate the 2D position for the existing subplot
        row_idx, col_idx = divmod(i, num_cols)
        ax.set_subplotspec(gs[row_idx, col_idx])

    # 5. Add the new subplot at the next available position
    row_idx, col_idx = divmod(n_existing, num_cols)
    new_ax = fig.add_subplot(gs[row_idx, col_idx])

    return new_ax

def get_ax(
    ax=None,
    figsize=[8,2],
    **kws
):    
    if isinstance(ax,plt.Axes): 
        return ax
    elif ax in ['same','.',None]:
        fig=set_fig(figsize)
        return plt.gca()
    elif isinstance(ax,dict):
        ## recurse
        ## e.g. ax=dict(ax='gca',cols_max=1)
        return get_ax(
            **{
                **dict(figsize=figsize),
                **ax
            },
        )
    elif ax=='new':
        return plt.subplots(
            1,1,
            figsize=figsize,
            **kws                
        )[1]
    elif ax in ['+','append','gca']:
        return set_ax(
            figsize=figsize,
            **kws    
        )
    else:
        raise ValueError(ax)
        
def get_children(fig):
    """
    Get all the individual objects included in the figure.
    """
    # from tqdm import tqdm
    from roux.lib.set import flatten

    ## figure
    l1 = []
    # for ax in tqdm(fig.get_children()):
    for ax in fig.get_children():
        if isinstance(ax, plt.Subplot):
            l1 += ax.get_children()
        else:
            l1 += [ax]
    l1 = flatten(l1)
    ## subfigure
    l2 = []
    # for ax in tqdm(l1):
    for ax in l1:
        if isinstance(ax, plt.Subplot):
            l2 += ax.get_children()
        else:
            l2 += [ax]
    return flatten(l2)


def get_text(
    text,
    all_children=None,
    fig=None,
    ax=None
):
    """
    Get text object.
    """
    if fig is None:
        fig=plt.gcf()
    if all_children is None:
        if ax is not None:
            all_children = get_children(fig=fig)
        else:
            all_children = ax.get_children()

    outs = []
    for c in all_children:
        if isinstance(c, plt.Text):
            if c.get_text() == text:
                outs.append(c)
            
    assert len(outs)!=0, (text, all_children)    
    return outs

def align_texts(
    fig,
    texts: list,
    align: str,
    test=False,
):
    """
    Align text objects.
    """
    all_children = get_children(fig=fig)
    x_px_set, y_px_set = None, None
    for i, search_name in enumerate(texts):
        text = get_text(
            text=search_name,
            all_children=all_children,
        )[0]
        extent = text.get_window_extent(renderer=fig.canvas.get_renderer())
        x_px, y_px = np.array(extent)[0][0], np.array(extent)[1][1]
        if test:
            print(x_px, y_px)
        if i == 0 and align == "v":
            y_px_set = y_px
            continue
        elif i == 0 and align == "h":
            x_px_set = x_px
            continue
        else:
            ## set
            if x_px_set is not None:
                x_px = x_px_set
            if y_px_set is not None:
                y_px = y_px_set
            ax_ = text.axes
            x_data, y_data = ax_.transData.inverted().transform((x_px, y_px))
            if test:
                print(x_data, y_data)
            # Add text to the subplot using data coordinates
            _ = ax_.text(
                x_data,
                y_data,
                s=text.get_text(),
                fontsize=text.get_fontsize(),
                ha="left",  # text.get_ha(),
                va="top",  # text.get_va(),
                color=text.get_color(),  ##TODO transfer all the properties
            )
            if not test:
                text.remove()


def labelplots(
    axes: list = None,
    fig=None,
    labels: list = None,
    xoff: float = 0,
    yoff: float = 0,
    auto: bool = False,
    xoffs: dict = {},
    yoffs: dict = {},
    va: str = "center",
    ha: str = "left",
    verbose: bool = True,
    test: bool = False,
    # transform='ax',
    **kws_text,
):
    """Label (sub)plots.

    Args:
        fig : `plt.figure` object.
        axes (_type_): list of `plt.Axes` objects.
        xoff (int, optional): x offset. Defaults to 0.
        yoff (int, optional): y offset. Defaults to 0.
        params_alignment (dict, optional): alignment parameters. Defaults to {}.
        params_text (dict, optional): parameters provided to `plt.text`. Defaults to {'size':20,'va':'bottom', 'ha':'right' }.
        test (bool, optional): test mode. Defaults to False.

    Todos:
        1. Get the x coordinate of the ylabel.
    """
    if axes is None:
        axes = fig.axes
    if labels is None:
        import string

        labels = string.ascii_uppercase[: len(axes)]
    else:
        assert len(axes) == len(labels)
    label2ax = dict(zip(labels, axes))
    axi2xy = {}
    for axi, label in enumerate(label2ax.keys()):
        ax = label2ax[label]
    if auto:
        fig.draw_without_rendering()  ## get positions after drawing, applicable to ylabel's x.
        for label, ax in label2ax.items():
            x, y = (
                ax.yaxis.get_label().get_position()[0],
                (
                    ax.transAxes.transform(ax.title.get_position())[1]
                    if hasattr(ax, "title")
                    else 1.0
                ),
            )
            if verbose:
                logging.info(f"x,y={x},{y}")
            ax.annotate(label, xy=(x, y), xycoords="figure pixels")

    else:
        ## manual
        # if len(xoffs)!=0 or len(xoffs)!=0:
        for label, ax in label2ax.items():
            x, y = 0, (ax.title.get_position()[1] if hasattr(ax, "title") else 1.0)
            ax.text(
                s=label,
                x=x + xoff + (xoffs[label] if label in xoffs else 0),
                y=y + 0.045 + yoff + (yoffs[label] if label in yoffs else 0),
                **kws_text,
                transform=ax.transAxes,
            )

def annot_axs(
    data, # contains the x and y coord.s in ax1 and 2
    ax1,
    ax2,
    cols,
    kind='straight',
    verbose=False,
    **kws_line,
    ):
    ## inferred
    fig=ax1.get_figure()
    
    cols_inferred={}
    for k,col in cols.items():
        if col not in data:
            if isinstance(col,float):
                logging.info(f"{k}:{col}")
                off=col
                if col>0.5:
                    col=f'{k[-1]}max'
                # elif col<0.5:
                else:
                    col=f'{k[-1]}min'
            else:
                off=None
            if col in ['xmin','xmax','ymin','ymax']:
                if k.startswith('ax1'):
                    ax=ax1
                elif k.startswith('ax2'):
                    ax=ax2
                from roux.viz.ax_ import get_axlims
                lims=get_axlims(ax)
                pos=lims[col[0]][col[1:]]
                if off is not None:
                    if col[1:]=='max':
                        pos+=(off-1)*lims[col[0]]['len']
                    else:
                        pos+=(off)*lims[col[0]]['len']                
                data=data.assign(
                    **{
                        col:pos,
                    }
                )
                logging.info(f"col={col}")
            else:
                raise ValueError(col)
            cols_inferred[k]=col
            
    if verbose:
        print(data)
        print(cols_inferred)
    cols_inferred={
        **cols,
        **cols_inferred,
    }
    if kind=='straight':
        from matplotlib.patches import ConnectionPatch
        _=data.apply(lambda x:  fig.add_artist(
            ConnectionPatch(
                xyA=[x[cols_inferred['ax1x']],x[cols_inferred['ax1y']]], 
                xyB=[x[cols_inferred['ax2x']],x[cols_inferred['ax2y']]],
                coordsA=ax1.transData, 
                coordsB=ax2.transData,
                **{
                    **dict(
                        clip_on=False,
                        color='gray',
                    ),
                    **kws_line
                }
            ),
            ),
            axis=1)
    else:
        from roux.viz.line import plot_bezier
        [ax.set_facecolor('none') for ax in fig.axes]
        # This is the block that needs to be updated for compatibility with the new plot_bezier function
        # The new plot_bezier function takes two separate axes objects for coordinate transformation
        _=data.apply(lambda x:  plot_bezier(
                pt1=[x[cols_inferred['ax1x']],x[cols_inferred['ax1y']]],
                pt2=[x[cols_inferred['ax2x']],x[cols_inferred['ax2y']]],
                ax=ax1, # The axes where the line is plotted
                ax2=ax2, # The axes for the second point's coordinates
                # direction='v', # inferred
                off_guide=0.5,
                zorder=1,
                **{
                    **dict(
                        clip_on=False, # clip_on is added for consistency and to fix the clipping issue
                        color='gray',
                    ),
                    **kws_line
                }
            ),
            axis=1
        )

    return data
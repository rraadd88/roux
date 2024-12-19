"""For setting up figures."""

import numpy as np

import matplotlib.pyplot as plt
import logging


def get_children(fig):
    """
    Get all the individual objects included in the figure.
    """
    from tqdm import tqdm
    from roux.lib.set import flatten

    ## figure
    l1 = []
    for ax in tqdm(fig.get_children()):
        if isinstance(ax, plt.Subplot):
            l1 += ax.get_children()
        else:
            l1 += [ax]
    l1 = flatten(l1)
    ## subfigure
    l2 = []
    for ax in tqdm(l1):
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
    **kws_line,
    ):
    ## inferred
    fig=ax1.get_figure()
    
    for k,col in cols.items():
        if col not in data:
            if col in ['xmin','xmax','ymin','ymax']:
                if k.startswith('ax1'):
                    ax=ax1
                elif k.startswith('ax2'):
                    ax=ax2
                from roux.viz.ax_ import get_axlims
                lims=get_axlims(ax)
                data=data.assign(
                    **{
                        col:lims[col[0]][col[1:]]
                    }
                )
                logging.info(f"col={col}")
            else:
                raise ValueError(col)
    
    from matplotlib.patches import ConnectionPatch
    _=data.apply(lambda x:  fig.add_artist(
        ConnectionPatch(
            xyA=[x[cols['ax1x']],x[cols['ax1y']]], 
            xyB=[x[cols['ax2x']],x[cols['ax2y']]],
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
    return data
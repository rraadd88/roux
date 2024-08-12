"""For line plots."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from roux.viz.ax_ import get_axlims, logging


def plot_range(
    df00: pd.DataFrame,
    colvalue: str,
    colindex: str,
    k: str,
    headsize: int = 15,
    headcolor: str = "lightgray",
    ax: plt.Axes = None,
    **kws_area,
) -> plt.Axes:
    """Plot range/intervals e.g. genome coordinates as lines.

    Args:
        df00 (pd.DataFrame): input data.
        colvalue (str): column with values.
        colindex (str): column with ids.
        k (str): subset name.
        headsize (int, optional): margin at top. Defaults to 15.
        headcolor (str, optional): color of the margin. Defaults to 'lightgray'.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword args:
        kws: keyword parameters provided to `area` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    df00["rank"] = df00[colvalue].rank()
    x, y = (
        df00.rd.filter_rows({colindex: k}).iloc[0, :]["rank"],
        df00.rd.filter_rows({colindex: k}).iloc[0, :][colvalue],
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=[1, 1])
    ax = df00.set_index("rank").sort_index(0)[colvalue].plot.area(ax=ax, **kws_area)
    ax.annotate(
        "",
        xy=(x, y),
        xycoords="data",
        xytext=(x, ax.get_ylim()[1]),
        textcoords="data",
        arrowprops=dict(
            facecolor=headcolor,
            shrink=0,
            width=0,
            ec="none",
            headwidth=headsize,
            headlength=headsize,
        ),
        horizontalalignment="right",
        verticalalignment="top",
    )
    d_ = get_axlims(ax)
    ax.text(
        x,
        y + (d_["y"]["len"]) * 0.25,
        int(y),  # f"{y:.1f}",
        # transform=ax.transAxes,
        va="bottom",
        ha="center",
    )
    ax.text(
        0.5,
        0,
        colvalue,
        transform=ax.transAxes,
        va="top",
        ha="center",
    )
    ax.axis(False)
    return ax


def plot_bezier(
    pt1,
    pt2,
    pt1_guide=None,
    pt2_guide=None,
    direction="h",
    off_guide=0.25,
    ax=None,
    test=False,
    **kws_line,
):
    from matplotlib.path import Path
    import matplotlib.patches as patches

    if direction == "h":
        assert pt1[0] < pt2[0], (pt1[0], pt2[0])
        off = abs(pt1[0] - pt2[0]) * off_guide
        if pt1_guide is None:
            pt1_guide = [pt1[0] + off, pt1[1]]
        if pt2_guide is None:
            pt2_guide = [pt2[0] - off, pt2[1]]
    elif direction == "v":
        assert pt1[1] < pt2[1], (pt1[1], pt2[1])
        off = abs(pt1[1] - pt2[1]) * off_guide
        if pt1_guide is None:
            pt1_guide = [pt1[0], pt1[1] + off]
        if pt2_guide is None:
            pt2_guide = [pt2[0], pt2[1] - off]
    else:
        raise ValueError(direction)
    # Create the Path object using the control points
    vertices = [pt1, pt1_guide, pt2_guide, pt2]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]

    path = Path(vertices, codes)

    # Create a patch from the path
    patch = patches.PathPatch(
        path,
        # facecolor='none',
        # edgecolor='blue',
        # lw=2,
        fc="none",
        **kws_line,
    )

    # Plot the path
    if ax is None:
        ax = plt.gca()
    ax.add_patch(patch)
    if test:
        # Plot control points
        control_points = np.array([pt1, pt1_guide, pt2_guide, pt2])
        ax.plot(
            control_points[:, 0],
            control_points[:, 1],
            "--",
            label=f'direction={direction}; off_guide={off_guide}'
        )
    return ax


# def plot_connections(
#     dplot: pd.DataFrame,
#     label2xy: dict,
#     colval: str='$r_{s}$',
#     line_scale: int=40,
#     legend_title: str='similarity',
#     label2rename: dict=None,
#     element2color: dict=None,
#     xoff: float=0,
#     yoff: float=0,
#     rectangle: dict={'width':0.2,'height':0.32},
#     params_text: dict={'ha':'center','va':'center'},
#     params_legend: dict={'bbox_to_anchor':(1.1, 0.5),
#     'ncol':1,
#     'frameon':False},
#     legend_elements: list=[],
#     params_line: dict={'alpha':1},
#     ax: plt.Axes=None,
#     test: bool=False
#     ) -> plt.Axes:
#     """Plot connections between points with annotations.

#     Args:
#         dplot (pd.DataFrame): input data.
#         label2xy (dict): label to position.
#         colval (str, optional): column with values. Defaults to '{s}$'.
#         line_scale (int, optional): line_scale. Defaults to 40.
#         legend_title (str, optional): legend_title. Defaults to 'similarity'.
#         label2rename (dict, optional): label2rename. Defaults to None.
#         element2color (dict, optional): element2color. Defaults to None.
#         xoff (float, optional): xoff. Defaults to 0.
#         yoff (float, optional): yoff. Defaults to 0.
#         rectangle (_type_, optional): rectangle. Defaults to {'width':0.2,'height':0.32}.
#         params_text (_type_, optional): params_text. Defaults to {'ha':'center','va':'center'}.
#         params_legend (_type_, optional): params_legend. Defaults to {'bbox_to_anchor':(1.1, 0.5), 'ncol':1, 'frameon':False}.
#         legend_elements (list, optional): legend_elements. Defaults to [].
#         params_line (_type_, optional): params_line. Defaults to {'alpha':1}.
#         ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
#         test (bool, optional): test mode. Defaults to False.

#     Returns:
#         plt.Axes: `plt.Axes` object.
#     """
#     import matplotlib.patches as mpatches
#     label2xy={k:[label2xy[k][0]+xoff,label2xy[k][1]+yoff] for k in label2xy}
#     dplot['index xy']=dplot['index'].map(label2xy)
#     dplot['column xy']=dplot['column'].map(label2xy)

#     ax=plt.subplot() if ax is None else ax
#     # from roux.viz.ax_ import set_logos,get_subplot_dimentions
#     patches=[]
#     label2xys_rectangle_centers={}
#     for label in label2xy:
#         xy=label2xy[label]
#         rect = mpatches.Rectangle(xy, **rectangle, fill=False,fc="none",lw=2,
#                                   ec=element2color[label] if element2color is not None else 'k',
#                                  zorder=0)

#         patches.append(rect)
#         line_xys=[np.transpose(np.array(rect.get_bbox()))[0],np.transpose(np.array(rect.get_bbox()))[1][::-1]]
#         label2xys_rectangle_centers[label]=[np.mean(line_xys[0]),np.mean(line_xys[1])]
#         inset_width=0.2
#         inset_height=inset_width/get_subplot_dimentions(ax)[2]
#         axin=ax.inset_axes([*[l-(off*0.5) for l,off in zip(label2xys_rectangle_centers[label],[inset_width,inset_height])],
#                             inset_width,inset_height])
#         # if not test:
#             # axin=set_logos(label=label,element2color=element2color,ax=axin,test=test)
#         axin.text(np.mean(axin.get_xlim()),np.mean(axin.get_ylim()),
#                  label2rename[label] if label2rename is not None else label,
#                   **params_text,
#                  )
#     dplot.apply(lambda x: ax.plot(*[[label2xys_rectangle_centers[x[k]][0] for k in ['index','column']],
#                                   [label2xys_rectangle_centers[x[k]][1] for k in ['index','column']]],
#                                   lw=(x[colval]-0.49)*line_scale,
#                                   linestyle=params_line['linestyle'],
#                                   color='k',zorder=-1,
#                                   alpha=params_line['alpha'],
#                                 ),axis=1)
#     if params_line['annot']:
#         def set_text_position(ax,x):
#             xs,ys=[[label2xys_rectangle_centers[x[k]][i] for k in ['index','column']] for i in [0,1]]
#             xy=[np.mean(xs),np.mean(ys)]
#             if np.subtract(*xs)==0 or np.subtract(*ys)==0:
#                 ha,va='center','center'
#                 rotation=0
#             else:
#                 if np.subtract(*xs)<0:
#                     ha,va='right','bottom'
#                     xy[1]=xy[1]+0.025
#                     rotation=-45
#                 else:
#                     ha,va='right','top'
#                     xy[1]=xy[1]-0.025
#                     rotation=45
#             ax.text(xy[0],xy[1],f"{x[colval]:.2f}",
#                     ha=ha,va=va,
#                     color='k',rotation=rotation,
#                    bbox=dict(boxstyle="round",
#                    fc='lightgray',ec=None,)
#                    )
#             return ax

#         dplot.apply(lambda x: set_text_position(ax,x),axis=1)
#     from matplotlib.lines import Line2D
#     legend_elements=legend_elements+[Line2D([0], [0], color='k', linestyle='solid', lw=(i-0.49)*line_scale,
#                                 alpha=params_line['alpha'],
#                                 label=f' {colval}={i:1.1f}') for i in [1.0,0.8,0.6]]
#     ax.legend(handles=legend_elements,
#               title=legend_title,**params_legend)
#     ax.set(**{'xlim':[0,1],'ylim':[0,1]})
#     if not test:
#         ax.set_axis_off()
#     return ax


def plot_kinetics(
    df1: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    cmap: str = "Reds_r",
    ax: plt.Axes = None,
    test: bool = False,
    kws_legend: dict = {},
    **kws_set,
) -> plt.Axes:
    """Plot time-dependent kinetic data.

    Args:
        df1 (pd.DataFrame): input data.
        x (str): x column.
        y (str): y column.
        hue (str): hue column.
        cmap (str, optional): colormap. Defaults to 'Reds_r'.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        test (bool, optional): test mode. Defaults to False.
        kws_legend (dict, optional): legend parameters. Defaults to {}.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.viz.ax_ import rename_legends
    from roux.viz.colors import get_ncolors

    df1 = df1.sort_values(hue, ascending=False)
    logging.info(df1[hue].unique())
    if ax is None:
        fig, ax = plt.subplots(figsize=[2.5, 2.5])
    label2color = dict(
        zip(
            df1[hue].unique(),
            get_ncolors(
                df1[hue].nunique(),
                ceil=False,
                cmap=cmap,
            ),
        )
    )
    df2 = (
        df1.groupby([hue, x], sort=False)
        .agg({c: [np.mean, np.std] for c in [y]})
        .rd.flatten_columns()
        .reset_index()
    )
    d1 = (
        df1.groupby([hue, x], sort=False, as_index=False)
        .size()
        .groupby(hue)["size"]
        .agg([min, max])
        .T.to_dict()
    )
    d2 = {
        str(k): str(k)
        + "\n"
        + (
            f"(n={d1[k]['min']})"
            if d1[k]["min"] == d1[k]["max"]
            else f"(n={d1[k]['min']}-{d1[k]['max']})"
        )
        for k in d1
    }
    if test:
        logging.info(d2)
    df2.groupby(hue, sort=False).apply(
        lambda df: df.sort_values(x).plot(
            x=x,
            y=f"{y} mean",
            yerr=f"{y} std",
            elinewidth=0.3,
            label=df.name,
            color=label2color[df.name],
            lw=2,
            ax=ax,
        )
    )
    ax = rename_legends(ax, replaces=d2, title=hue, **kws_legend)
    ax.set(**kws_set)
    return ax


## plot data shape changes
def plot_steps(
    df1: pd.DataFrame,
    col_step_name: str,
    col_step_size: str,
    ax: plt.Axes = None,
    test: bool = False,
) -> plt.Axes:
    """
    Plot step-wise changes in numbers, e.g. for a filtering process.

    Args:
        df1 (pd.DataFrame): input data.
        col_step_size (str): column containing the numbers.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    df1["% change"] = df1[col_step_size].pct_change() * 100
    df1["y"] = range(len(df1))
    if ax is None:
        fig, ax = plt.subplots(figsize=[4, len(df1)])
    kws_line = dict(marker="o", mfc="w", color="gray", ms=17)
    df1.iloc[:-1, :].apply(
        lambda x: ax.plot([0, 0], [x["y"], x["y"] + 1], **kws_line), axis=1
    )
    df1.apply(
        lambda x: ax.text(0.005, x["y"], s=x[col_step_name], ha="left", va="center"),
        axis=1,
    )
    df1.apply(
        lambda x: ax.text(0, x["y"], s=f"{x['y']:.0f}", ha="center", va="center"),
        axis=1,
    )
    df1.apply(
        lambda x: ax.text(
            0.005,
            x["y"] + 0.33,
            s=f"n={x[col_step_size]:.0f}",
            ha="left",
            va="center",
            alpha=0.75,
        ),
        axis=1,
    )
    from roux.viz.colors import saturate_color

    df1.apply(
        lambda x: ax.text(
            0.005,
            x["y"] + 0.66,
            s=f"{'' if x['% change']<0 else '+'}{x['% change']:.1f}%"
            if not pd.isnull(x["% change"])
            else "",
            ha="left",
            va="center",
            color=saturate_color("#FF0000", 0.5 + (x["% change"] / -100))
            if x["% change"] < 0
            else "g",
            alpha=0.75,
        ),
        axis=1,
    )
    ax.set(xlim=[-0.005, 0.1], ylim=[len(df1) + 0.5, -1.5])
    if not test:
        ax.axis("off")
    return ax

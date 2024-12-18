"""For annotations."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from roux.lib.str import linebreaker
from roux.viz.ax_ import set_label

# redirects
from roux.stat.io import pval2annot


def annot_side(
    ax: plt.Axes,
    df1: pd.DataFrame,
    colx: str,
    coly: str,
    cols: str = None,
    hue: str = None,
    loc: str = "right",
    scatter=False,
    scatter_marker="|",
    scatter_alpha=0.75,
    lines=True,
    offx3: float = 0.15,
    offymin: float = 0.1,
    offymax: float = 0.9,
    length_axhline: float = 3,
    text=True,
    text_offx: float = 0,
    text_offy: float = 0,
    invert_xaxis: bool = False,
    break_pt: int = 25,
    va: str = "bottom",
    zorder: int = 2,
    color: str = "gray",
    kws_line: dict = {},
    kws_scatter: dict = {},  #'zorder':2,'alpha':0.75,'marker':'|','s':100},
    **kws_text,
) -> plt.Axes:
    """Annot elements of the plots on the of the side plot.

    Args:
        df1 (pd.DataFrame): input data
        colx (str): column with x values.
        coly (str): column with y values.
        cols (str): column with labels.
        hue (str): column with colors of the labels.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        loc (str, optional): location. Defaults to 'right'.
        invert_xaxis (bool, optional): invert xaxis. Defaults to False.
        offx3 (float, optional): x-offset for bend position of the arrow. Defaults to 0.15.
        offymin (float, optional): x-offset minimum. Defaults to 0.1.
        offymax (float, optional): x-offset maximum. Defaults to 0.9.
        break_pt (int, optional): break point of the labels. Defaults to 25.
        length_axhline (float, optional): length of the horizontal line i.e. the "underline". Defaults to 3.
        zorder (int, optional): z-order. Defaults to 1.
        color (str, optional): color of the line. Defaults to 'gray'.
        kws_line (dict, optional): parameters for formatting the line. Defaults to {}.

    Keyword Args:
        kws: parameters provided to the `ax.text` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if len(df1) == 0:
        logging.warning("annot_side: no data found")
        return
    assert colx != "x", colx
    assert coly != "y", coly
    assert cols != "s", cols
    if isinstance(colx, float):
        df1["colx"] = colx
        colx = "colx"
    if isinstance(coly, float):
        df1["coly"] = coly
        coly = "coly"
    # assert not 'y' in df1, 'table should not contain a column named `y`'
    df1 = df1.sort_values(coly if loc != "top" else colx, ascending=True)
    from roux.viz.ax_ import get_axlims

    d1 = get_axlims(ax)
    # if loc=='top', annotations x is y and y is x
    df1["y"] = np.linspace(
        d1["y" if loc != "top" else "x"]["min"]
        + ((d1["y" if loc != "top" else "x"]["len"]) * offymin),
        d1["y" if loc != "top" else "x"]["max"] * offymax,
        len(df1),
    )
    x2 = (
        d1["x" if loc != "top" else "y"]["min" if not invert_xaxis else "max"]
        if loc == "left"
        else d1["x" if loc != "top" else "y"]["max" if not invert_xaxis else "min"]
    )
    x3 = (
        d1["x"]["min"] - (d1["x"]["len"] * offx3)
        if (loc == "left" and not invert_xaxis)
        else d1["y"]["max"] + (d1["y"]["len"] * offx3)
        if loc == "top"
        else d1["x"]["max"] + (d1["x"]["len"] * offx3)
    )

    # line#1
    if lines:
        df1.apply(
            lambda x: ax.plot(
                [x[colx], x2] if loc != "top" else [x[colx], x["y"]],
                [x[coly], x["y"]] if loc != "top" else [x[coly], x2],
                color=color,
                zorder=zorder - 1,
                clip_on=False,
                **kws_line,
            ),
            axis=1,
        )
    if scatter:
        df1.plot.scatter(
            x=colx,
            y=coly,
            ax=ax,
            marker=scatter_marker,
            alpha=scatter_alpha,
            zorder=zorder,
            **kws_scatter,
        )
    ## text
    if text:
        if "ha" not in kws_text:
            kws_text["ha"] = (
                "right" if loc == "left" else "center" if loc == "top" else "left"
            )
        if "rotation" not in kws_text:
            kws_text["rotation"] = 0 if loc != "top" else 90
        else:
            kws_text["ha"] = "left"
        df1.apply(
            lambda x: ax.text(
                (x3 if loc != "top" else x["y"])
                + text_offx * (1 if loc == "right" else -1),
                (x["y"] if loc != "top" else x3) + text_offy,
                x[cols]
                if break_pt is None
                else linebreaker(
                    x[cols],
                    break_pt=break_pt,
                ),
                va=va,
                color="k"
                if hue is None
                else x[hue]
                if hue in x
                else hue,  ## prefer if column is present
                # **{k:v for k,v in kws_text.items() if not (k==color and not hue is None)},
                **kws_text,
                zorder=zorder + 1,
            ),
            axis=1,
        )
    # line #2
    if lines:
        if loc != "top":
            df1.apply(
                lambda x: ax.axhline(
                    y=x["y"],
                    xmin=0 if loc == "left" else 1,
                    xmax=0 - (length_axhline - 1) - offx3
                    if loc == "left"
                    else length_axhline + offx3,
                    clip_on=False,
                    color=color,
                    zorder=zorder - 1,
                    **kws_line,
                ),
                axis=1,
            )
        else:
            df1.apply(
                lambda x: ax.axvline(
                    x=x["y"],
                    ymin=0 if loc == "left" else 1,
                    ymax=0 - (length_axhline - 1) - offx3
                    if loc == "left"
                    else length_axhline + offx3,
                    clip_on=False,
                    color=color,
                    zorder=zorder - 1,
                    **kws_line,
                ),
                axis=1,
            )
    if loc == "left":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.set(
        xlim=[d1["x"]["min"], d1["x"]["max"]],
        ylim=[d1["y"]["min"], d1["y"]["max"]],
    )
    return ax


def annot_side_curved(
    data,
    colx: str,
    coly: str,
    col_label: str,
    off: float=0.5,
    lim: tuple=None,
    limf: tuple=None, ## limits as fractions
    loc: str='right',
    # x: float=None, ## todo: deprecate
    # ylim: tuple=None, ## todo: deprecate
    ax=None,
    test: bool = False,
    kws_text={},
    **kws_line,
):
    """Annot elements of the plots on the of the side plot using bezier lines.
    
    Usage: 
        1. Allows m:1 mappings between points and labels
    """
    if ax is None:
        ax = plt.gca()        
    
    ## for resettings at the end
    lims = dict(
        xlim=ax.get_xlim(),
        ylim=ax.get_ylim(),
    )
    # print(lims)
    if limf is not None:
        ## fraction to limits
        from roux.stat.transform import rescale
        lim=rescale(
            limf,
            range1=[0,1],
            range2=lims['ylim'] if loc=='right' else lims['xlim'],
        )        
    if lim is None:
        lim=lims['ylim'] if loc=='right' else lims['xlim']

    from roux.stat.paired import get_diff_sorted
    lim_=(lims['xlim'] if loc=='right' else lims['ylim'])
    size=get_diff_sorted(*lim_)
    off=max(lim_)+(size*off)
    # print(off)
    
    if loc=='right':
        kws_text_loc=dict(
            va="center",
            ha="right",    
        )
    else:
        ## top
        kws_text_loc=dict(
            ha="center",
            va="top",
            rotation=90,
        )
    ## sorted labels
    data1 = (
        data
        .sort_values(
            coly if loc=='right' else colx,
            ascending=lims['ylim'][0]<lims['ylim'][1] if loc=='right' else lims['xlim'][0]<lims['xlim'][1],
            # ascending=True,
        )
        .loc[:, [col_label]]
        .drop_duplicates()
        .assign(
            **{
                'x' if loc=='right' else 'y':lambda df: np.repeat(off, len(df)),
                'y' if loc=='right' else 'x':lambda df: np.linspace(
                    *lim,
                    len(df),
                ),
                ('x' if loc=='right' else 'y')+'_text':lambda df: df.apply(
                    lambda x: getattr(
                        (
                            ax.text(
                                x["x"],
                                x["y"],
                                s=x[col_label],
                                **{
                                    **kws_text_loc,
                                    **kws_text, ## override by the inputs
                                },
                            )
                            .get_window_extent(renderer=plt.gcf().canvas.get_renderer())
                            .transformed(ax.transData.inverted())
                        ),
                        'xmin' if  loc=='right' else 'ymin'
                    ),
                    axis=1,
                ),
            }
        )
    )
    # print(data1)
    # return data1
    ## lines
    data2 = data.merge(
        right=data1,
        how="inner",
        on=col_label,
        # validate="1:1"
    )
    from roux.viz.line import plot_bezier

    data2.apply(
        lambda x: plot_bezier(
            [x[colx], x[coly]],
            [x["x_text"], x["y"]] if loc=='right' else [x["x"], x["y_text"]],
            direction='h' if loc=='right' else 'v',
            ax=ax,
            **{
                **dict(
                    clip_on=False,
                    color="darkgray",
                ),
                **kws_line,
            },
        ),
        axis=1,
    )
    ## resets ax lims
    ax.set(**lims)
    if test:
        print(data2)
    return ax


# scatters
## annotations
def show_outlines(
    data: pd.DataFrame,
    colx: str,
    coly: str,
    column_outlines: str,
    outline_colors: dict = None,
    cmap=None,
    style=None,
    legend: bool = True,
    kws_legend: dict = {},
    zorder: int = 3,
    ax: plt.Axes = None,
    **kws_scatter,
) -> plt.Axes:
    """
    Outline points on the scatter plot by categories.

    """
    if outline_colors is None:
        from roux.viz.colors import get_ncolors
        outline_colors=get_ncolors(  
            data[column_outlines].nunique(),
            cmap=cmap,
        )
    # print(outline_colors)
    if isinstance(outline_colors, list):
        outline_colors = dict(zip(data[column_outlines].dropna().unique(), outline_colors))
        logging.info(
            f"Mapping between the categories and the colors of the outlines: {outline_colors}."
        )
    # print(outline_colors)
    for cat, df_ in data.groupby(column_outlines):

        ax = sns.scatterplot(
            data=df_,
            x=colx,
            y=coly,
            edgecolor=outline_colors[cat],
            facecolor="none",
            linewidth=1,
            s=50,
            style=style,
            ax=ax,
            legend=False,
            label=f"{df_[column_outlines].unique()[0]} ({len(df_)})"
            if legend
            else None,
            zorder=zorder,
            **kws_scatter,
        )
        if legend:
            ax.legend(
                title=column_outlines,
                **kws_legend,
            )
    return ax


## variance
def show_confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters:
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse

    References
    ----------
    https://matplotlib.org/3.5.0/gallery/statistics/confidence_ellipse.html
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# heatmaps
def show_box(
    ax: plt.Axes,
    xy: tuple,
    width: float,
    height: float,
    fill: str = None,
    alpha: float = 1,
    lw: float = 1.1,
    edgecolor: str = "k",
    clip_on: bool = False,
    scale_width: float = 1,
    scale_height: float = 1,
    xoff: float = 0,
    yoff: float = 0,
    **kws,
) -> plt.Axes:
    """Highlight sections of a plot e.g. heatmap by drawing boxes.

    Args:
        xy (tuple): position of left, bottom corner of the box.
        width (float): width.
        height (float): height.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        fill (str, optional): fill the box with color. Defaults to None.
        alpha (float, optional): alpha of color. Defaults to 1.
        lw (float, optional): line width. Defaults to 1.1.
        edgecolor (str, optional): edge color. Defaults to 'k'.
        clip_on (bool, optional): clip the boxes by the axis limit. Defaults to False.
        scale_width (float, optional): scale width. Defaults to 1.
        scale_height (float, optional): scale height. Defaults to 1.
        xoff (float, optional): x-offset. Defaults to 0.
        yoff (float, optional): y-offset. Defaults to 0.

    Keyword Args:
        kws: parameters provided to the `Rectangle` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from matplotlib.patches import Rectangle

    return ax.add_patch(
        Rectangle(
            xy=[xy[0] + xoff, xy[1] + yoff],
            width=width * scale_width,
            height=height * scale_height,
            fill=fill,
            alpha=alpha,
            lw=lw,
            edgecolor=edgecolor,
            clip_on=clip_on,
            **kws,
        )
    )


# color
def color_ax(ax: plt.Axes, c: str, linewidth: float = None) -> plt.Axes:
    """Color border of `plt.Axes`.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        c (str): color.
        linewidth (float, optional): line width. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    plt.setp(ax.spines.values(), color=c)
    if linewidth is not None:
        plt.setp(ax.spines.values(), linewidth=linewidth)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=c)
    return ax


# stats
def show_n_legend(
    ax,
    df1: pd.DataFrame,
    colid: str,
    colgroup: str,
    **kws,
):
    from roux.viz.ax_ import rename_legends

    replaces = {
        str(k): str(k) + "\n" + f"(n={v})"
        for k, v in df1.groupby(colgroup)[colid].nunique().items()
    }
    return rename_legends(ax, replaces=replaces, **kws)


def show_scatter_stats(
    ax: plt.Axes,
    data: pd.DataFrame,
    x,
    y,
    z,
    method: str,
    resample: bool = False,
    show_n: bool = True,
    show_n_prefix: str = "",
    prefix: str = "",
    loc=None,
    zorder: int = 5,
    # kws_stat={},
    verbose: bool = True,
    kws_stat={},
    **kws_set_label,
):
    """
    resample (bool, optional): resample data. Defaults to False.
    """
    label = prefix
    if "spearman" in method or "pearson" in method or "kendalltau" in method:
        # label,r=get_corr(data[x],data[y],
        # method=method[0],
        # outstr=True,
        # resample=resample,
        # kws_to_str=dict(
        #     show_n=show_n,
        #     show_n_prefix=show_n_prefix,
        # ),
        # verbose=verbose,
        # **kws_stat,
        # )
        from roux.stat.corr import get_corr, _to_string

        res = get_corr(
            data[x],
            data[y],
            method=method,
            resample=resample,
            verbose=verbose,
        )
        if res is not None and len(res) != 0:
            label += _to_string(
                res,
                show_n=show_n,
                show_n_prefix=show_n_prefix,
            )
            if loc is None:
                ## infer
                if res["r"] >= 0:
                    loc = 2
                elif res["r"] < 0:
                    loc = 3
    if "mlr" in method:
        from roux.stat.fit import get_mlr_2_str

        label += get_mlr_2_str(
            data.loc[
                :,
                [
                    x,
                    y,
                    z,
                ],
            ].dropna(),
            z,
            [x, y],
            **kws_stat,
        )
    _ = set_label(
        ax=ax,
        s=label,
        zorder=zorder,
        loc=loc,
        **kws_set_label,
    )
    return ax


def show_crosstab_stats(
    data: pd.DataFrame,
    cols: list,
    method: str = None,
    alpha: float = 0.05,
    loc: str = None,
    xoff: float = 0,
    yoff: float = 0,
    linebreak: bool = False,
    ax: plt.Axes = None,
    **kws_set_label,
) -> plt.Axes:
    """Annotate a confusion matrix.

    Args:
        data (pd.DataFrame): input data.
        cols (list): list of columns with the categories.
        method (str, optional): method used to calculate the statistical significance.
        alpha (float, optional): alpha for the stats. Defaults to 0.05.
        loc (str, optional): location. Over-rides kws_set_label. Defaults to None.
        xoff (float, optional): x offset. Defaults to 0.
        yoff (float, optional): y offset. Defaults to 0.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws_set_label: keyword parameters provided to `set_label`.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.stat.diff import compare_classes

    stat, pval = compare_classes(data[cols[0]], data[cols[1]], method=method)
    logging.info(f"stat={stat},pval={pval}")

    ## get the label for the stat method
    data_ = pd.crosstab(data[cols[0]], data[cols[1]])
    if data_.shape != (2, 2) or method == "chi2":
        stat_label = "${\chi}^2$"
    else:
        stat_label = "OR"

    if loc == "bottom":
        kws_set_label = dict(
            x=0.5 + xoff,
            y=-0.2 + yoff,
            ha="center",
            va="center",
        )
    elif loc == "right":
        kws_set_label = dict(
            x=1 + xoff,
            y=0 + yoff,
            ha="left",
            va="bottom",
        )
    elif loc == "center":
        kws_set_label = dict(
            x=0.5 + xoff,
            y=0.5 + yoff,
            ha="center",
            va="center",
        )

    from roux.viz.ax_ import set_label

    set_label(
        s=f"{stat_label}={stat:.1f}"
        + (", " if not linebreak else "\n")
        + pval2annot(
            pval, alternative="two-sided", alpha=alpha, fmt="<", linebreak=False
        ),
        ax=ax,
        **kws_set_label,
    )
    return ax


def show_confusion_matrix_stats(
    df_: pd.DataFrame, ax: plt.Axes = None, off: float = 0.5
) -> plt.Axes:
    """Annotate a confusion matrix.

    Args:
        df_ (pd.DataFrame): input data.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        off (float, optional): offset. Defaults to 0.5.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.stat.binary import get_stats_confusion_matrix

    df1 = get_stats_confusion_matrix(df_)
    df2 = pd.DataFrame(
        {
            "TP": [0, 0],
            "TN": [1, 1],
            "FP": [0, 1],
            "FN": [1, 0],
            "TPR": [0, 2],
            "TNR": [1, 2],
            "PPV": [2, 0],
            "NPV": [2, 1],
            "FPR": [1, 3],
            "FNR": [0, 3],
            "FDR": [3, 0],
            "ACC": [2, 2],
        },
        index=["x", "y"],
    ).T
    df2.index.name = "variable"
    df2 = df2.reset_index()
    df3 = df1.merge(df2, on="variable", how="inner", validate="1:1")

    _ = df3.loc[(df3["variable"].isin(["TP", "TN", "FP", "FN"])), :].apply(
        lambda x: ax.text(
            x["x"] + off,
            x["y"] + (off * 2),
            #                               f"{x['variable']}\n{x['value']:.0f}",
            x["variable"],
            #                               f"({x['T|F']+x['P|N']})",
            ha="center",
            va="bottom",
        ),
        axis=1,
    )
    _ = df3.loc[~(df3["variable"].isin(["TP", "TN", "FP", "FN"])), :].apply(
        lambda x: ax.text(
            x["x"] + off,
            x["y"] + (off * 2),
            f"{x['variable']}\n{x['value']:.2f}",
            #                               f"({x['T|F']+x['P|N']})",
            ha="center",
            va="bottom",
        ),
        axis=1,
    )
    return ax


# # logo
# def get_logo_ax(
#     ax: plt.Axes,
#     size: float=0.5,
#     bbox_to_anchor: list=None,
#     loc: str=1,
#     axes_kwargs: dict={'zorder':-1},
#     ) -> plt.Axes:
#     """Get `plt.Axes` for placing the logo.

#     Args:
#         ax (plt.Axes): `plt.Axes` object.
#         size (float, optional): size of the subplot. Defaults to 0.5.
#         bbox_to_anchor (list, optional): location. Defaults to None.
#         loc (str, optional): location. Defaults to 1.
#         axes_kwargs (_type_, optional): parameters provided to `inset_axes`. Defaults to {'zorder':-1}.

#     Returns:
#         plt.Axes: `plt.Axes` object.
#     """
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
#     width, height,aspect_ratio=get_subplot_dimentions(ax)
#     axins = inset_axes(ax,
#                        width=size, height=size,
#                        bbox_to_anchor=[1,1,0,size/(height)] if bbox_to_anchor is None else bbox_to_anchor,
#                        bbox_transform=ax.transAxes,
#                        loc=loc,
#                        borderpad=0,
#                       axes_kwargs=axes_kwargs)
#     return axins

# def set_logo(
#     imp: str,
#     ax: plt.Axes,
#     size: float=0.5,
#     bbox_to_anchor: list=None,
#     loc: str=1,
#     axes_kwargs: dict={'zorder':-1},
#     params_imshow: dict={'aspect':'auto','alpha':1,
#     #                             'zorder':1,
#     'interpolation':'catrom'},
#     height=1,
#     width=1,
#     aspect_ratio=1,
#     test: bool=False,
#     force: bool=False
#     ) -> plt.Axes:
#     """Set logo.

#     Args:
#         imp (str): path to the logo file.
#         ax (plt.Axes): `plt.Axes` object.
#         size (float, optional): size of the subplot. Defaults to 0.5.
#         bbox_to_anchor (list, optional): location. Defaults to None.
#         loc (str, optional): location. Defaults to 1.
#         axes_kwargs (_type_, optional): parameters provided to `inset_axes`. Defaults to {'zorder':-1}.
#         params_imshow (_type_, optional): parameters provided to the `imshow` function. Defaults to {'aspect':'auto','alpha':1, 'interpolation':'catrom'}.
#         test (bool, optional): test mode. Defaults to False.
#         force (bool, optional): overwrite file. Defaults to False.

#     Returns:
#         plt.Axes: `plt.Axes` object.
#     """
#     from pathlib import Path
#     from roux.lib.figs.convert import vector2raster
#     if isinstance(imp,str):
#         if imp.splitext(' ')[1]=='.svg':
#             pngp=vector2raster(imp,force=force)
#         else:
#             pngp=imp
#         if not Path(pngp).exists():
#             logging.error(f'{pngp} not found')
#             return
#         im = plt.imread(pngp)
#     elif isinstance(imp,np.ndarray):
#         im = imp
#     else:
#         logging.warning('imp should be path or image')
#         return
#     axins=get_logo_ax(ax,size=size,bbox_to_anchor=bbox_to_anchor,loc=loc,
#              axes_kwargs=axes_kwargs,)
#     axins.imshow(im, **params_imshow)
#     if not test:
#         axins.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
#         axins.margins(0)
#         axins.axis('off')
#         axins.set_axis_off()
#     else:
#         print(width, height,aspect_ratio,size/(height*2))
#     return axins


# multiple subplots
def set_suptitle(
    axs,
    title,
    offy=0,
    **kws_text,
):
    """
    Combined title for a list of subplots.

    """
    a1 = np.vstack((np.array(ax.get_position()) for ax in axs))
    return plt.text(
        x=np.mean([np.min(a1[:, 0]), np.max(a1[:, 0])]),
        # y=np.mean([np.min(a1[:,1]),np.max(a1[:,1])]),
        y=np.max(a1[:, 1]) + offy,
        s=title,
        ha="center",
        # fontdict=dict(fontsize=15),
        transform=plt.gcf().transFigure,
        **kws_text,
    )

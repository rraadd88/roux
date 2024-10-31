"""For scatter plots."""

import logging

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import scipy as sc

import roux.lib.df as rd  # noqa
from roux.stat.io import pval2annot
from roux.viz.ax_ import get_axlims


def plot_scatter_agg(
    dplot: pd.DataFrame,
    x: str = None,
    y: str = None,
    z: str = None,
    kws_legend=dict(
        bbox_to_anchor=[1, 1],
        loc="upper left",
    ),
    title=None,
    label_colorbar=None,
    ax=None,
    kind=None,
    verbose=False,
    cmap="Blues",
    gridsize=10,
    **kws,
):
    """UNDER DEV."""
    ## with more options compared to the seaborn one.
    ### to be updated
    dplot = dplot.dropna(subset=[x, y, z], how="any")
    if z is None:
        z = "count"
        dplot[z] = 1
    if z in dplot:
        kws["C"] = z
        kws["reduce_C_function"] = (
            len
            if z == "count"
            else kws["reduce_C_function"]
            if "reduce_C_function" in kws
            else np.mean
        )
        kws["gridsize"] = kws["gridsize"] if "gridsize" in kws else gridsize
        kws["cmap"] = kws["cmap"] if "cmap" in kws else cmap
        if verbose:
            print(kws)
    ax = dplot.plot(
        kind=kind,
        x=x,
        y=y,
        ax=ax,
        # **params_plot,
        **kws,
    )
    from roux.viz.ax_ import set_colorbar_label

    ax = set_colorbar_label(ax, z if label_colorbar is None else label_colorbar)

    leg = ax.legend(title=z if title is None else title, **kws_legend)
    if "\n" in title:
        leg._legend_box.align = "center"
    return ax


# @to_class(rd)
def plot_scatter(
    data: pd.DataFrame,
    x: str = None,
    y: str = None,
    z: str = None,
    ## type
    kind: str = "scatter",
    scatter_kws={},
    ## trendline
    line_kws={},
    ## stats
    stat_method: str = "spearman",
    stat_kws={},
    # stats_annot_kws={},
    ## aes
    hollow: bool = False,
    ## set
    ax: plt.Axes = None,
    verbose: bool = True,
    **kws,
) -> plt.Axes:
    """Plot scatter with multiple layers and stats.

    Args:
        data (pd.DataFrame): input dataframe.
        x (str): x column.
        y (str): y column.
        z (str, optional): z column. Defaults to None.
        kind (str, optional): kind of scatter. Defaults to 'hexbin'.
        trendline_method (str, optional): trendline method ['poly','lowess']. Defaults to 'poly'.
        stat_method (str, optional): method of annoted stats ['mlr',"spearman"]. Defaults to "spearman".
        cmap (str, optional): colormap. Defaults to 'Reds'.
        label_colorbar (str, optional): label of the colorbar. Defaults to None.
        gridsize (int, optional): number of grids in the hexbin. Defaults to 25.
        bbox_to_anchor (list, optional): location of the legend. Defaults to [1,1].
        loc (str, optional): location of the legend. Defaults to 'upper left'.
        title (str, optional): title of the plot. Defaults to None.
        #params_plot (dict, optional): parameters provided to the `plot` function. Defaults to {}.
        line_kws (dict, optional): parameters provided to the `plot_trendline` function. Defaults to {}.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `plot` function.

    Returns:
        plt.Axes: `plt.Axes` object.

    Notes:
        1. For a rasterized scatter plot set `scatter_kws={'rasterized': True}`
        2. This function does not apply multiple colors, similar to `sns.regplot`.
    """
    ## axis
    ax = plt.subplot() if ax is None else ax

    ## string to list
    stat_method = (
        [stat_method]
        if isinstance(stat_method, str)
        else []
        if stat_method is None
        else stat_method
    )

    ## data
    data = data.log.dropna(
        subset=[x, y], how="any"
    )  # to show the number of rows with missing values. seaborn applies 'dropna' anyways.
    ## set
    ## background
    if "hexbin" in kind:
        plot_scatter_agg(data, x, y, z, **kws)
    ## points
    if "scatter" in kind:
        from roux.viz.colors import saturate_color, get_colors_default

        ## shape
        if hollow:
            ## short-cut for making the points hollow
            scatter_kws = {
                **dict(
                    ec=scatter_kws["ec"]
                    if "ec" in scatter_kws
                    else scatter_kws["color"]
                    if "color" in kws
                    else get_colors_default()[0],
                    fc="none",
                    linewidth=1,
                ),
                **scatter_kws,
            }
        ### color
        if "color" not in line_kws:
            line_kws["color"] = saturate_color(
                kws["color"] if "color" in kws else get_colors_default()[0], alpha=1.5
            )
        if "fit_reg" in kws and "seed" not in kws:
            kws["seed"] = 0
        if verbose:
            ## methods
            logging.info(
                "sns.regplot:"
                + (
                    "; ".join(
                        [
                            f"{k}={kws[k]}"
                            for k in [
                                "ci",
                                "n_boot",
                                "order",
                                "logistic",
                                "lowess",
                                "robust",
                                "logx",
                                "x_partial",
                                "y_partial",
                                "units",
                                "seed",
                            ]
                            if k in kws
                        ]
                    )
                )
            )
        ax = sns.regplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            scatter_kws=scatter_kws,
            line_kws=line_kws,
            **kws,
        )
    ## stats
    from roux.viz.annot import show_scatter_stats

    show_scatter_stats(
        ax,
        data=data,
        x=x,
        y=y,
        z=z,
        method=stat_method[0],
        zorder=5,
        **stat_kws,
    )
    return ax


def plot_qq(x: pd.Series) -> plt.Axes:
    """plot QQ.

    Args:
        x (pd.Series): input vector.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    import statsmodels.api as sm

    fig = plt.figure(figsize=[3, 3])
    ax = plt.subplot()
    sm.qqplot(x, dist=sc.stats.norm, line="s", ax=ax)
    ax.set_title(
        "SW test "
        + pval2annot(sc.stats.shapiro(x)[1], alpha=0.05, fmt="<", linebreak=False)
    )
    from roux.viz.ax_ import set_equallim

    ax = set_equallim(ax)
    return ax


def plot_ranks(
    df1: pd.DataFrame,
    col: str,
    colid: str,
    ranks_on: str = "y",
    ascending: bool = True,
    col_rank: str = None,
    line: bool = True,
    kws_line={},
    show_topn: int = None,
    show_ids: list = None,
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

    Usage:
        Combined with annotations using `annot_side`.
    """
    assert not df1[colid].duplicated().any()

    if ranks_on == "x":
        cols = dict(
            x="rank",
            y=col,
        )
    elif ranks_on == "y":
        cols = dict(
            y="rank",
            x=col,
        )

    df2 = df1.assign(
        **{
            "rank": lambda df: df[col].rank(ascending=ascending),
        }
    ).sort_values("rank", ascending=ascending)

    if ax is None:
        fig, ax = plt.subplots(figsize=[2, 2])

    if line:
        ## background
        df2.plot(
            **cols,
            **{
                **dict(
                    color="lightgray",
                ),
                **kws_line,
            },
            ax=ax,
            label="_nolegend_",
            zorder=0,
        )
    if show_ids is not None or isinstance(show_topn, int):
        if isinstance(show_topn, int):
            df3 = df2.sort_values("rank", ascending=True).head(show_topn)
        elif show_ids is not None:
            df3 = df2.query(expr=f"`{colid}`=={show_ids}")
        ax = sns.scatterplot(
            data=df3,
            ax=ax,
            **cols,
            **{
                **dict(
                    fc="k",
                    alpha=0.5,
                    clip_on=False,
                ),
                **kws,
            },
            zorder=1,
            ec="none",
        )
    from roux.viz.ax_ import format_ticklabels

    format_ticklabels(
        ax=ax,
        axes=[ranks_on],
        fmt="counts",
    )
    if ranks_on == "y":
        ax.invert_yaxis()
    return ax, df2


def plot_volcano(
    data: pd.DataFrame,
    colx: str,
    coly: str,
    colindex: str,
    hue: str = "x",
    style: str = "marker_style",
    style_order: list = [
        "o",
        "^", #P=0
        "<", #-inf
        ">", #+inf
        ],
    markers: list = [
        "o",
        "^", #P=0
        "<", #-inf
        ">", #+inf
    ],
    show_labels: int = None,
    labels_layout: str = None,
    labels_kws: dict = {},
    show_outlines: int = None,
    outline_colors: list = ["k"],
    collabel: str = None,
    show_line=True,
    line_pvalue=0.1,
    line_x: float = 0.0,
    line_x_min: float = None,
    show_text: bool = True,
    text_increase: str = None,
    text_decrease: str = None,
    text_diff: str = None,
    legend: bool = False,
    verbose: bool = False,
    p_min: float = None,
    ax: plt.Axes = None,
    outmore: bool = False,
    kws_legend: dict = {},
    **kws_scatterplot,
) -> plt.Axes:
    """
    Volcano plot.

    Parameters:

    Keyword parameters:

    Returns:
        plt.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=[4, 3])
    if collabel is None:
        collabel = colindex
    assert not data[colindex].duplicated().any()
    from roux.stat.transform import log_pval

    # to avoid insert index error of seaborn
    data=data.reset_index(drop=True)
    
    if not coly.lower().startswith("significance"):
        
        xlim_data={
            'max':data.loc[~np.isinf(data[colx]),colx].max(),
            'min':data.loc[~np.isinf(data[colx]),colx].min(),
        }
        
        data = data.assign(
            **{
                style: lambda df: df.apply(
                    lambda x:                     
                    (
                        "^" if x[coly]==0 else   
                        ">" if x[colx]==np.inf else 
                        "<" if x[colx]==-np.inf else 
                        "o"
                    ),
                    axis=1,
                ),
                colx: lambda df: df[colx].apply(
                    lambda x: 
                    (
                        xlim_data['max'] if x==np.inf else 
                        xlim_data['min'] if x==-np.inf else 
                        x
                    )
                )
            },
        )
        logging.warning(f'transforming the coly ("{coly}") values.')
        coly_ = f"significance\n(-log10({coly}))"
        data = data.assign(
            **{coly_: lambda df: log_pval(df[coly], p_min=p_min, errors=None)}
        )
        coly = coly_
    elif style not in data:
        data[style] = "o"
    data["significance bin"] = pd.cut(
        data[coly],
        bins=log_pval([0, 0.05, 0.1, 1])[::-1],
        labels=["ns", "q<0.1", "q<0.05"],
        include_lowest=True,
    )
    assert not data["significance bin"].isnull().any()
    data = (
        data.assign(
            **{
                "significance direction bin": lambda df: df.apply(
                    lambda x: "increase"
                    if x[coly] > log_pval(line_pvalue)
                    and (
                        x[colx] > line_x
                        if line_x_min is not None
                        else x[colx] >= line_x
                    )
                    else "decrease"
                    if x[coly] > log_pval(line_pvalue)
                    and (
                        x[colx] < line_x_min
                        if line_x_min is not None
                        else x[colx] <= -1 * line_x
                    )
                    else "ns",
                    axis=1,
                ),
            }
        ).sort_values(
            "significance direction bin", ascending=False
        )  # put 'ns' at the background
    )
    assert not data["significance direction bin"].isnull().any()
    if hue == "x":
        hue = "significance direction bin"
        kws_scatterplot["hue_order"] = ["increase", "decrease", "ns"]
        if "palette" not in kws_scatterplot:
            from roux.viz.colors import get_colors_default

            kws_scatterplot["palette"] = [
                get_colors_default()[2],
                get_colors_default()[0],
                get_colors_default()[1],
            ]
    elif hue == "y":
        hue = "significance bin"
    ax = sns.scatterplot(
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
    axlims = get_axlims(ax)
    if show_text:
        if text_diff is not None:
            ax.text(
                x=axlims["x"]["min"] + (axlims["x"]["len"] * 0.5),
                y=-75,
                s=text_diff,
                ha="center",
                va="center",
                color="gray",
            )
        ax.text(
            x=ax.get_xlim()[1],
            y=ax.get_ylim()[1],
            s="increase $\\rightarrow$"
            + (
                "\n(n="
                + str(
                    data.query(expr="`significance direction bin` == 'increase'")[
                        colindex
                    ].nunique()
                )
                + ")"
                if text_increase == "n"
                else f"\n({text_increase})"
                if text_increase is not None
                else ""
            ),
            ha="right",
            va="bottom",
            color="k"
            if "palette" not in kws_scatterplot
            else kws_scatterplot["palette"][0],
        )
        ax.text(
            x=ax.get_xlim()[0],
            y=ax.get_ylim()[1],
            s="$\\leftarrow$ decrease"
            + (
                "\n(n="
                + str(
                    data.query(expr="`significance direction bin` == 'decrease'")[
                        colindex
                    ].nunique()
                )
                + ")"
                if text_increase == "n"
                else f"\n({text_decrease})"
                if text_decrease is not None
                else ""
            ),
            ha="left",
            va="bottom",
            color="k"
            if "palette" not in kws_scatterplot
            else kws_scatterplot["palette"][1],
        )
    ## set lines
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for side in [-1, 1]:
        print(
            [xlim[0 if side == -1 else 1], line_x * side, line_x * side],
            [log_pval(line_pvalue), log_pval(line_pvalue), ylim[1]],
        )
        ax.plot(
            [
                xlim[0 if side == -1 else 1],
                (line_x_min if line_x_min is not None else line_x * side),
                (line_x_min if line_x_min is not None else line_x * side),
            ],
            [log_pval(line_pvalue), log_pval(line_pvalue), ylim[1]],
            color="gray",
            linestyle=":",
        )
    ## set labels
    if show_labels is not None:  # show_labels overrides show_outlines
        show_outlines = show_labels
    if show_outlines is not None:
        if isinstance(show_outlines, int):
            ## show_outlines top n
            data1 = data.query(expr="`significance direction bin` != 'ns'").sort_values(
                colx
            )
            ## sort the data
            data1 = pd.concat(
                [
                    data1.head(show_outlines),  # left
                    data1.tail(show_outlines),  # right
                ],
                axis=0,
            ).drop_duplicates(subset=[colindex])
        elif isinstance(show_outlines, dict):
            ## subset
            data1 = data.rd.filter_rows(show_outlines)
        elif isinstance(show_outlines, str):
            ## column with categories
            data1 = data.dropna(subset=[show_outlines])
        if verbose:
            print(data1)
        # plot
        if not isinstance(show_outlines, str):
            # borders
            ax = sns.scatterplot(
                data=data1,
                x=colx,
                y=coly,
                # hue=show_outlines if isinstance(show_outlines, str) else None,
                ec="k",
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
            column_outlines = show_outlines
            from roux.viz.annot import show_outlines

            ax = show_outlines(
                data1,
                colx,
                coly,
                column_outlines=column_outlines,
                outline_colors=outline_colors,
                style=style,
                style_order=style_order,
                markers=markers,
                legend=legend,
                kws_legend=kws_legend,
                ax=ax,
            )
    ## setting ylim before setting the labels
    ax.set(
        xlabel="Log$_\mathrm{2}$ Fold Change (LFC)",
        ylabel="Significance\n(-Log$_\mathrm{10}$($q$))",
        xlim=xlim,
        ylim=ylim,
    )
    if show_labels:
        if labels_layout == "side":
            from roux.viz.annot import annot_side

            ax = annot_side(
                ax=ax,
                df1=data1,
                colx=colx,
                coly=coly,
                cols=collabel,
                **labels_kws,
            )
        else:
            texts = data1.apply(
                lambda x: ax.text(
                    x=x[colx],
                    y=x[coly],
                    s=x[collabel],
                ),
                axis=1,
            ).tolist()
            try:
                from adjustText import adjust_text

                adjust_text(
                    texts,
                    arrowprops=dict(arrowstyle="-", color="k"),
                    **labels_kws,
                )
            except:
                logging.error("install adjustText to repel the labels.")
    ## formatting
    ax.spines.top.set(visible=False)
    ax.spines.right.set(visible=False)
    if not outmore:
        return ax
    else:
        return ax, data

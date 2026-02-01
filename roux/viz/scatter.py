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
    show_p=True,
    stat_method: str = "spearman",
    stat_resample: bool = False,
    stat_kws={
        # 'n_min':10
    },
    # stats_annot_kws={},
    ## aes
    hollow: bool = False,
    ## axes
    diagonal=False,
    ## set
    ax: plt.Axes = None,
    verbose: bool = True,
    annot_kws={},
    # kws_set={},
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
    from roux.viz.figure import get_ax
    ax = get_ax(ax)

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
    
    if diagonal:
        from roux.viz.ax_ import set_equallim
        set_equallim(ax=ax,diagonal=True,**(diagonal if isinstance(diagonal,dict) else {}))

    ## stats
    from roux.viz.annot import show_scatter_stats
    
    ## par.corr.
    kws_parcorr={}
    if kws.get('x_partial',None) or kws.get('y_partial',None):
        ax.set(
            xlabel="$\\tilde{"+x+"}$",
            ylabel="$\\tilde{"+y+"}$",
        )
        if kws.get('x_partial',None) and kws.get('y_partial',None):
            kws_parcorr['covar']=[kws['x_partial'],kws['y_partial']]
        elif kws.get('x_partial',None):
            kws_parcorr['x_covar']=kws['x_partial']
        elif kws.get('y_partial',None):
            kws_parcorr['y_covar']=kws['y_partial']

    kws_={
            **{
                **dict(
                    method=stat_method[0],
                    resample=stat_resample,
                ),
                **stat_kws,
                **kws_parcorr,
            },
            **annot_kws,
            **dict(
                show_p=show_p,
            )
        }
    logging.info(kws_)        
    ax=show_scatter_stats(
        ax,
        data=data,
        x=x,
        y=y,
        z=z,
        zorder=5,
        **kws_,
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


def _prepare_volcano_data(
    data: pd.DataFrame,
    colx: str,
    coly: str,
    style: str,
    line_pvalue: float,
    line_x: float,
    line_x_min: float,
    p_min: float,
    hue: str,
    colindex: str,
    errors: str,    
    kws_scatterplot: dict,
    
    xlim=None,
):
    """Prepares the DataFrame for plot_volcano."""
    if errors == 'raise':
        assert not data[colindex].duplicated().any(), 'set errors=None if this is un-necessary ..'
    
    from roux.stat.transform import log_pval
    data = data.reset_index(drop=True)

    if not coly.lower().startswith("significance"):
        if xlim is None:
            xlim = [
                data.loc[~np.isinf(data[colx]), colx].min(),
                data.loc[~np.isinf(data[colx]), colx].max(),
            ]
        data = (
            data
                .assign(
                    **{    
                        ## clipped to inf
                        colx:lambda df: df[colx].apply(
                            lambda x:
                            (
                                np.inf if x > xlim[1]  else
                                -np.inf if x < xlim[0] else
                                x
                            )
                        ),
                        style: lambda df: df.apply(
                            lambda x:
                            (
                                "^" if x[coly] == 0 else
                                ">" if x[colx] == np.inf else
                                "<" if x[colx] == -np.inf else
                                "o"
                            ),
                            axis=1,
                        ),
                    }
                    )
                .assign(
                    **{                         
                        ## clipped to lim
                        colx: lambda df: df[colx].apply(
                            lambda x:
                            (
                                xlim[1] if x == np.inf else
                                xlim[0] if x == -np.inf else
                                x
                            )
                        ),
                    },
                )
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
        )
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
    
    return data, coly, hue, kws_scatterplot


def plot_volcano(
    data: pd.DataFrame,
    colx: str,
    coly: str,
    colindex: str,
    hue: str = "x",

    xlim=None,
    p_min: float = None, # ymax 
    ## thresholds
    line_pvalue=0.1,

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
    labels_layout: str = 'side',
    labels_kws: dict = {},
    show_outlines: int = None,
    outline_colors: list = ["k"],
    collabel: str = None,
    
    ## guide line
    line_x: float = 0.0,  
    ### dotted line
    show_lines: bool = False,
    line_x_min: float = None,
    
    show_text: bool = True,
    show_n=True,
    text_increase: str = 'n',
    text_decrease: str = 'n',
    text_diff: str = None,
    legend: bool = False,
    legend_kws=dict(
        bbox_to_anchor=[1,-0.2],
        loc='lower left',
    ),    
    verbose: bool = False,
    ax: plt.Axes = None,
    outmore: bool = False,
    kws_legend: dict = {},
    errors='raise',
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
        if isinstance(collabel,list):
            assert len(collabel)==1, "if show_labels, collabel (colindex) should be one"
            collabel=collabel[0]

    data, coly, hue, kws_scatterplot = _prepare_volcano_data(        
        data=data, 
        colx=colx, 
        coly=coly, 
        xlim=xlim,
        style=style, 
        line_pvalue=line_pvalue, 
        line_x=line_x, 
        line_x_min=line_x_min, 
        p_min=p_min, 
        hue=hue, 
        colindex=colindex, 
        errors=errors, 
        kws_scatterplot=kws_scatterplot
    )

    ax = sns.scatterplot(
        data=data,
        **{
            **dict(
                x=colx,
                y=coly,
                hue=hue,
                style=style,
                style_order=style_order,
                markers=markers,
                ec=None,
                legend=False,
            ),
            **kws_scatterplot,
        },
        ax=ax,
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
        
    if show_n:
        to_ns=data.groupby('significance direction bin')[colindex].nunique().to_dict()
        kws_text_show_n=dict(
            va="center" if (text_increase == "n" and text_decrease == "n") else 'bottom',
            color="k" if ((text_increase == "n" and text_decrease == "n") or ("palette" not in kws_scatterplot)) else kws_scatterplot["palette"][0],            
        )
        ax.text(
            x=ax.get_xlim()[1],
            y=ax.get_ylim()[1],
            s=(
                to_ns.get('increase','')
                    if text_increase == "n" else
                (
                    "increase $\\rightarrow$"
                    + "\n(n="
                    + str(to_ns.get('increase',0))
                    + ")"
                )
                    if text_increase == "dir(n)" else
                f"\n({text_increase})"
                    if text_increase is not None else
                ""
            ),
            ha="right",
            **kws_text_show_n
        )
        ax.text(
            x=ax.get_xlim()[0],
            y=ax.get_ylim()[1],
            s=(
                to_ns.get('decrease','')
                    if text_increase == "n" else
                (
                    "$\\leftarrow$ decrease"
                    + "\n(n="
                    + str(to_ns.get('decrease',0))
                    + ")"
                )
                    if text_increase == "dir(n)" else
                f"\n({text_decrease})"
                    if text_decrease is not None else
                ""
            ),
            ha="left",
            **kws_text_show_n
        )
    ## get from plot (i.e. data_plot)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    ## set lines
    if show_lines:
        from roux.stat.transform import log_pval
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
    # if show_labels is not None:  # show_labels overrides show_outlines
    #     show_outlines = show_labels
    if show_labels is not None:
        query_expr=show_labels
    elif show_outlines is not None:
        query_expr=show_outlines
    else:
        query_expr=None

    ## filtering to data1 to highlight
    if query_expr is not None:
        if isinstance(query_expr, int):
            ## query_expr top n
            data1 = data.query(
                expr="`significance direction bin` != 'ns'").sort_values(
                colx
            )
            ## sort the data
            data1 = pd.concat(
                [
                    data1.head(query_expr),  # left
                    data1.tail(query_expr),  # right
                ],
                axis=0,
            ).drop_duplicates(subset=[colindex])
        elif isinstance(query_expr, str) and '`' in query_expr:
            data1 = data.query(expr=query_expr)
        elif isinstance(query_expr, str) and '`' not in query_expr:
            ## column with categories
            data1 = data.dropna(subset=[query_expr])
        elif isinstance(query_expr, dict):
            ## subset
            data1 = data.rd.filter_rows(query_expr)
        if verbose:
            print(data1)

    # plot
    if show_outlines is not None:        
        if not isinstance(show_outlines, str):
            # borders
            ax = sns.scatterplot(
                data=data1,
                x=colx,
                y=coly,
                # hue=show_outlines if isinstance(show_outlines, str) else None,
                ec="k",
                # ec="face",
                lw=1,
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
            if legend:
                ax.legend(
                    title=column_outlines,
                    **legend_kws,
                )
            else:
                logging.warning("set legend=True to show legends for outlines ..")
            
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
            kws_annot_side={
                    **dict(
                        colx=colx,
                        coly=coly,
                        col_label=collabel,
                        kind='curved',
                    ),
                    **labels_kws,
                }
            # print(kws_annot_side)
            ax = annot_side(
                data1,
                ax=ax,
                **kws_annot_side
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

def plot_volcano_split(
    data,
    col_label,
    row,
    row_order=None,
    annot_expr=None,

    signi_lt=0.05,
    xlim=None,
    text_x_offs=[0.25,-1],
    
    axd=None,
    figsize=[5,2],
    kws_annot={},
    **kws_volcano,
    ):
    if row_order is None:
        row_order=data[row].unique().tolist()
        
    if axd is None:
        fig,axd=plt.subplot_mosaic(
            [
                row_order
            ],
            figsize=figsize,
        )
        
    from roux.viz.ax_ import format_ax
    from roux.viz.figure import to_pos_in_ax2
    for rowi,title in enumerate(row_order):
        ax=axd[title]
        loc='right' if rowi==0 else 'left'
        data_=data.groupby(row,sort=False).get_group(title)
        
        # from roux.viz.scatter import plot_volcano
        ax,data_volcano=plot_volcano(
            data_,
            **{
                **dict(
                    colx='LFC',
                    coly='Q',
                    colindex='gene set id',

                    line_pvalue=signi_lt,
                    xlim=xlim,          
                    outmore=True,
                    show_text=False,
                    show_n=False,
            
                    alpha=0.5,
                    ),
                **kws_volcano,
            },
            ax=ax
        )
        from roux.viz.annot import annot_side
        data_annot=(
            data_volcano
                .log.query(
                    expr="Q < 0.05"
                )
                .sort_values(
                    [
                        'LFC',
                        'significance\n(-log10(Q))',
                    ],
                    ascending=[False,False]
                )
            )
        if annot_expr is not None:
            data_annot=(
                data_annot
                    .log.query(
                        expr=annot_expr,
                    )
            )
        if len(data_annot)==0:
            continue
        
        kws_annot_side=kws_annot.copy()
        if loc=='left':
            annot_ax2={
                k:to_pos_in_ax2(
                pt=pt,
                ax=axd[row_order[0]],
                ax2=axd[row_order[-1]],
                ) for k,pt in annot_ax1.items()
            }
            annot_df=(
                pd.DataFrame(annot_ax2,index=['text_x','text_y']).T
                    .rename_axis(col_label)
                    .reset_index(0)
            )
            assert data_annot[col_label].isin(annot_df[col_label]).all(), data_annot[col_label].unique()
            data_annot=(
                data_annot
                .log.merge(
                    right=annot_df,
                    on=col_label,
                    how='inner',
                    validate='m:1',
                )
                .assign(
                    **{
                        col_label: '',
                        'text_x': lambda df: df['text_x']+text_x_offs[1],
                    }
                )
            )
            kws_annot_side={
                **kws_annot_side,
                **dict(
                    text_x='text_x',
                    text_y='text_y',
                ),
            }        
        
        annot_side(
            data_annot,
            colx='LFC',
            coly='significance\n(-log10(Q))',
            col_label=col_label,
            ax=ax,
            loc='left',
            kind='curved',
            off=text_x_offs[0],
            
            **{
                **dict(
                    size=10,
                    linespacing=0.75,
                    ha='center',
                    limf=[0.1,0.9],
                    kws_line=dict(
                        zorder=5,
                        ),
                    zorder=10,
                 ),
                **kws_annot_side,
            },
        )
        format_ax(
            ax=ax,
            xlabel='Enrichment (Log$_\mathrm{2}$(OR))',
            # rotate_ylabel=True,
            title=title,
            # xticks=list(sorted(set([0]+list(ax.get_xticks())))),
            xticks=xlim,
            xlim=xlim,
            ylim=[0,ax.get_ylim()[1]],
        )
        if loc=='right':
            ax.invert_xaxis()
            from roux.viz.figure import get_text
            
            annot_pos={
                s: get_text(
                    s,
                    ax=ax,
                )[0] for s in data_annot[col_label].astype(str).unique()
            }
            [t.set_zorder(100) for t in annot_pos.values()]
            annot_pos={k: (
                    t
                    .get_window_extent(
                        renderer=plt.gcf().canvas.get_renderer()
                        )
                    .transformed(ax.transData.inverted())
                ) for k,t in annot_pos.items()}
            annot_ax1={k:[t.xmin,np.mean([t.ymin,t.ymax])]for k,t in annot_pos.items()}
        else:
            ax.set(
                # ylabel=None,
                yticklabels=[],
            )        
        # g: 2. Move the Y-axis label text to the right side
        ax.yaxis.set_label_position('left' if loc=='right' else 'right')
        
        # g: 3. Configure tick visibility (ensures left ticks are off 
        # g:    and right ticks are explicitly on, though tick_right often handles this)
        ax.tick_params(axis='y', left=loc=='right', right=(loc=='left'))
        
        # g: 4. The essential fix: Hide the left spine and show the right spine.
        ax.spines[loc].set_visible(False)
        ax.spines['left' if loc=='right' else 'right'].set_visible(True)
        
        # ax.legend()
        # break
    return axd
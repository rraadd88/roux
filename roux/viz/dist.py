"""For distribution plots."""

import logging
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import roux.lib.df as rd  # noqa
from roux.lib.set import dropna
from roux.viz.colors import get_colors_default
from roux.viz.annot import pval2annot
from roux.viz.ax_ import get_axlims, get_ticklabel_position, rename_ticklabels
import seaborn as sns


## single distributions.
def hist_annot(
    dplot: pd.DataFrame,
    colx: str,
    colssubsets: list = [],
    bins: int = 100,
    subset_unclassified: bool = True,
    cmap: str = "hsv",
    ymin=None,
    ymax=None,
    ylimoff: float = 1,
    ywithinoff: float = 1.2,
    annotaslegend: bool = True,
    annotn: bool = True,
    params_scatter: dict = {"zorder": 2, "alpha": 0.1, "marker": "|"},
    xlim: tuple = None,
    ax: plt.Axes = None,
    **kws,
) -> plt.Axes:
    """Annoted histogram.

    Args:
        dplot (pd.DataFrame): input dataframe.
        colx (str): x column.
        colssubsets (list, optional): columns indicating subsets. Defaults to [].
        bins (int, optional): bins. Defaults to 100.
        subset_unclassified (bool, optional): call non-annotated subset as 'unclassified'. Defaults to True.
        cmap (str, optional): colormap. Defaults to 'Reds_r'.
        ylimoff (float, optional): y-offset for y-axis limit . Defaults to 1.2.
        ywithinoff (float, optional): y-offset for the distance within labels. Defaults to 1.2.
        annotaslegend (bool, optional): convert labels to legends. Defaults to True.
        annotn (bool, optional): annotate sample sizes. Defaults to True.
        params_scatter (_type_, optional): parameters of the scatter plot. Defaults to {'zorder':2,'alpha':0.1,'marker':'|'}.
        xlim (tuple, optional): x-axis limits. Defaults to None.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `hist` function.

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs:
        For scatter, use `annot_side` with `loc='top'`.
    """
    if xlim is not None:
        logging.warning("colx adjusted to xlim")
        dplot.loc[(dplot[colx] < xlim[0]), colx] = xlim[0]
        dplot.loc[(dplot[colx] > xlim[1]), colx] = xlim[1]
    if ax is None:
        ax = plt.subplot(111)
    ax = dplot[colx].hist(
        bins=bins,
        ax=ax,
        zorder=1,
        **kws,
    )
    ax.set_xlabel(colx)
    ax.set_ylabel("count")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(0, ax.get_ylim()[1] * ylimoff)
    from roux.viz.colors import get_ncolors

    colors = get_ncolors(len(colssubsets), cmap=cmap)
    for colsubsetsi, (colsubsets, color) in enumerate(zip(colssubsets, colors)):
        subsets = [
            s
            for s in dropna(dplot[colsubsets].unique())
            if not (subset_unclassified and s == "unclassified")
        ]
        for subseti, subset in enumerate(subsets):
            y = (
                (ax.set_ylim()[1] if ymax is None else ymax)
                - (ax.set_ylim()[0] if ymin is None else ymin)
            ) * (
                (10 - (subseti * ywithinoff + colsubsetsi)) / 10 - 0.05
            ) + ax.set_ylim()[0]
            X = dplot.loc[(dplot[colsubsets] == subset), colx]
            Y = [y for i in X]
            ax.scatter(X, Y, color=color, **params_scatter)
            ax.text(
                (max(X) if not annotaslegend else ax.get_xlim()[1]),
                max(Y),
                f"{subset}\n(n={len(X)})" if annotn else f" {subset}",
                ha="left",
                va="center",
            )
    #     break
    #     ax=reset_legend_colors(ax)
    #     ax.legend(bbox_to_anchor=[1,1])
    return ax


def plot_gmm(
    x: pd.Series,
    coff: float = None,
    mix_pdf: object = None,
    two_pdfs: tuple = None,
    weights: tuple = None,
    n_clusters: int = 2,
    bins: int = 20,
    show_cutoff: bool = True,
    show_cutoff_line: bool = True,
    colors: list = ["gray", "gray", "lightgray"],
    out_coff: bool = False,
    hist: bool = True,
    test: bool = False,
    ax: plt.Axes = None,
    kws_axvline=dict(color="k"),
    **kws,
) -> plt.Axes:
    """Plot Gaussian mixture Models (GMMs).

    Args:
        x (pd.Series): input vector.
        coff (float, optional): intersection between two fitted distributions. Defaults to None.
        mix_pdf (object, optional): Probability density function of the mixed distribution. Defaults to None.
        two_pdfs (tuple, optional): Probability density functions of the separate distributions. Defaults to None.
        weights (tuple, optional): weights of the individual distributions. Defaults to None.
        n_clusters (int, optional): number of distributions. Defaults to 2.
        bins (int, optional): bins. Defaults to 50.
        colors (list, optional): colors of the invividual distributions and of the mixed one. Defaults to ['gray','gray','lightgray'].
        'gray'
        out_coff (bool,False): return the cutoff. Defaults to False.
        hist (bool, optional): show histogram. Defaults to True.
        test (bool, optional): test mode. Defaults to False.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `hist` function.
        kws_axvline: parameters provided to the `axvline` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if mix_pdf is None:
        from roux.stat.cluster import cluster_1d

        d_ = cluster_1d(
            x,
            n_clusters=n_clusters,
            clf_type="gmm",
            random_state=88,
            test=False,
            returns=["coff", "mix_pdf", "two_pdfs", "weights"],
        )
        coff, mix_pdf, two_pdfs, weights = (
            d_["coff"],
            d_["mix_pdf"],
            d_["two_pdfs"],
            d_["weights"],
        )
    if ax is None:
        plt.figure(figsize=[2.5, 2.5])
        ax = plt.subplot()
    # plot histogram
    if hist:
        pd.Series(x).hist(density=True, histtype="step", bins=bins, ax=ax, **kws)
    # plot fitted distributions
    ax.plot(x, mix_pdf.ravel(), c=colors[-1])
    for i in range(n_clusters):
        ax.plot(x, two_pdfs[i] * weights[i], c=colors[i])
    #     ax.plot(x,two_pdfs[1]*weights[1], c='gray')
    if n_clusters == 2:
        if show_cutoff_line:
            ax.axvline(coff, **kws_axvline)
        if show_cutoff:
            ax.text(coff, ax.get_ylim()[1], f"{coff:.1f}", ha="center", va="bottom")
    if not out_coff:
        return ax
    else:
        return ax, coff


def plot_normal(
    x: pd.Series,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot normal distribution.

    Args:
        x (pd.Series): input vector.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if ax is not None:
        fig, ax = plt.subplots(figsize=[3, 3])
    ax = sns.distplot(
        x,
        hist=True,
        kde_kws={
            "shade": True,
            "lw": 1,
        },
        fit=sc.stats.norm,
        label="residuals",
    )
    ax.set_title(
        "SW test "
        + pval2annot(sc.stats.shapiro(x)[1], alpha=0.05, fmt="<", linebreak=False)
    )
    ax.legend()
    return ax


## annotations
def get_jitter_positions(
    ax,
    df1,
    order,
    column_category,
    column_position,
):
    ## filtering
    df1 = df1.loc[df1[column_category].isin(order), :]
    ## pos
    d1 = dict(zip(order, [c.get_offsets()[:, 0 if column_position =='x' else 1].data for c in ax.collections]))
    ## mapping
    return (
        df1.groupby(column_category, as_index=False)
        .apply(lambda df: df.assign(**{column_position: (d1[df.name])}))
        .reset_index(drop=True)
    )


## paired distributions.
def plot_dists(
    df1: pd.DataFrame,
    x: str,
    y: str,
    colindex: str,
    hue: str = None,
    order: list = None,
    hue_order: list = None,
    kind: str = "box",
    show_p: bool = True,
    show_n: bool = True,
    show_n_prefix: str = "",
    show_n_ha=None,
    show_n_ticklabels: bool = True,
    show_outlines: bool = False,
    kws_outlines: dict = {},
    alternative: str = "two-sided",
    offx_n: float = 0,
    axis_cont_lim: tuple = None,
    axis_cont_scale: str = "linear",
    offs_pval: dict = None,
    fmt_pval: str = "<",
    alpha: float = 0.5,
    # saturate_color_alpha: float=1.5,
    ax: plt.Axes = None,
    test: bool = False,
    kws_stats: dict = {},
    **kws,
) -> plt.Axes:
    """Plot distributions.

    Args:
        df1 (pd.DataFrame): input data.
        x (str): x column.
        y (str): y column.
        colindex (str): index column.
        hue (str, optional): column with values to be encoded as hues. Defaults to None.
        order (list, optional): order of categorical values. Defaults to None.
        hue_order (list, optional): order of values to be encoded as hues. Defaults to None.
        kind (str, optional): kind of distribution. Defaults to 'box'.
        show_p (bool, optional): show p-values. Defaults to True.
        show_n (bool, optional): show sample sizes. Defaults to True.
        show_n_prefix (str, optional): show prefix of sample size label i.e. `n=`. Defaults to ''.
        offx_n (float, optional): x-offset for the sample size label. Defaults to 0.
        axis_cont_lim (tuple, optional): x-axis limits. Defaults to None.
        offs_pval (float, optional): x and y offsets for the p-value labels.
        # saturate_color_alpha (float, optional): saturation of the color. Defaults to 1.5.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        test (bool, optional): test mode. Defaults to False.
        kws_stats (dict, optional): parameters provided to the stat function. Defaults to {}.

    Keyword Args:
        kws: parameters provided to the `seaborn` function.

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs:
        1. Sort categories.
        2. Change alpha of the boxplot rather than changing saturation of the swarmplot.

    """
    if isinstance(colindex, str):
        colindex = [colindex]
    ## y is expected to be categorical (str/bool) for horizontal orientation which is preferred and also for calculating stats
    ## if it is not, switch between x and y
    # print(df1[y].dtype)
    if df1[y].dtype not in [int, float]:
        x_stat, y_stat = x, y
        axis_desc, axis_cont = "y", "x"
    else:
        x_stat, y_stat = y, x
        axis_desc, axis_cont = "x", "y"

    if test:
        logging.info(x_stat, y_stat)

    ## formatting the table
    df1 = df1.log.dropna(subset=colindex + [x, y]).assign(
        **{y_stat: lambda df: df[y_stat].astype(str)}
    )

    ## set order of the categories
    if order is None:
        order = df1[y_stat].unique().tolist()
        for l in [["True", "False"], ["yes", "no"]]:
            if df1[y_stat].isin(l).all():
                order = l
                break
    if test:
        logging.info(order)
    ## set order of the colors
    if hue is not None and hue_order is None:
        hue_order = df1[hue].unique().tolist()

    ## get stats
    if show_p:
        if (hue is None) and (isinstance(show_p, bool)):
            from roux.stat.diff import get_stats

            df2 = get_stats(
                df1,
                colindex=colindex,
                colsubset=y_stat,
                cols_value=[x_stat],
                subsets=order,
                # alpha=0.05
                axis=0,
                **kws_stats,
            )
            if df2 is None:
                logging.error("get_stats failed.")
                d1 = {}
            else:
                df2 = df2.reset_index()
                # df1=df1.rd.renameby_replace({f"{} ":''})
                df2 = df2.loc[(df2["subset1"] == order[0]), :]
                # print(df2)
                d1 = df2.rd.to_dict(["subset2", "P (MWU test)"])
        elif (hue is not None) and (isinstance(show_p, bool)):
            from roux.stat.diff import get_stats_groupby

            df2 = get_stats_groupby(
                df1.loc[df1[hue].isin(hue_order), :],
                cols_group=[y],
                colsubset=hue,
                cols_value=[x],
                colindex=colindex,
                alpha=0.05,
                axis=0,
                **kws_stats,
            ).reset_index()
            # df1=df1.rd.renameby_replace({f"{} ":''})
            # df2=df2.loc[(df2['subset1']==hue_order[1]),:]
            # d1=df2.rd.to_dict([y,'P (MWU test)'])
            d1 = df2.set_index(y)["P (MWU test)"].to_dict()
            if test:
                logging.info(d1)
        if df2 is not None:
            # print(df2.set_index(['subset1','subset2']).T)
            ## stats printing
            stats = df2.set_index(["subset1", "subset2"]).rd.dropby_patterns(
                ["median ", "mean ", "var ", "variable"], verbose=False
            )
            logging.info(stats)
            del stats
        else:
            show_p = False
            logging.error(
                "p-value could not be estimated. likely lack of >1 categories."
            )
    ## axes
    if ax is None:
        ax = plt.gca()

    ## distributions
    if isinstance(kind, str):
        kind = {kind: {}}
    elif isinstance(kind, list):
        kind = {k: {} for k in kind}
    for k in kind:
        kws_ = kws.copy()
        # print(kws['palette'],kind)
        # if 'palette' in kws and any([k_ in kind for k_ in ['swarm','strip']]):
        #     from roux.viz.colors import saturate_color
        #     kws['palette']=[saturate_color(color=c, alpha=saturate_color_alpha-1) for c in kws['palette']]
        # if 'palette' in kws and k in ['swarm','strip']:
        # from roux.viz.colors import saturate_color
        # kws['palette']=[saturate_color(color=c, alpha=saturate_color_alpha+0.5) for c in kws['palette']]
        if k == "box" and (("swarm" in kind) or ("strip" in kind)):
            kws_["showfliers"] = False
            kws_["boxprops"] = dict(alpha=alpha)
        if k in ["swarm", "strip"] and ("box" in kind):
            kws_["alpha"] = alpha
        if hue is None:
            kws_["color"] = get_colors_default()[0]
        getattr(sns, k + "plot")(
            data=df1,
            x=x,
            y=y,
            hue=hue,
            order=order,
            hue_order=hue_order,
            **{**kws_, **kind[k]},  ## combine overwrite with user provided
            ax=ax,
        )
    ax.set(
        **{
            f"{axis_desc}label": y_stat,  # None if hue is None else y_stat,
            f"{axis_cont}scale": axis_cont_scale,
            f"{axis_cont}lim": axis_cont_lim,
        },
    )
    ticklabel2position = get_ticklabel_position(ax, axis_desc)
    d3 = get_axlims(ax)
    ## show p-value
    if isinstance(show_p, (bool, dict)):
        if isinstance(show_p, bool) and show_p:
            d1 = {
                k: pval2annot(
                    d1[k],
                    alternative=alternative,
                    fmt=fmt_pval,
                    linebreak=False,
                )
                for k in d1
            }
        else:
            #     d1=show_p
            d1 = {}
        if offs_pval is None:
            offs_pval = {}
        offs_pval = {**{"x": 0, "y": 0}, **offs_pval}

        if hue is None and len(d1) == 1:
            offs_pval[axis_desc] += -0.5
        if test:
            logging.info(offs_pval)
        if isinstance(d1, dict):
            if test:
                logging.info(d1, ticklabel2position, d3)
            for k, s in d1.items():
                ax.text(
                    **{
                        axis_cont: d3[axis_cont]["max"]
                        + offs_pval[
                            axis_cont
                        ],  # +((d3[axis_cont]['len']*offx_pval) if axis_desc=='y' else 0),
                        axis_desc: ticklabel2position[k] + offs_pval[axis_desc],
                    },
                    s=s,
                    va="center" if axis_desc == "y" else "top",
                    ha="right" if axis_desc == "y" else "center",
                    zorder=5,
                )

    ## show sample sizes
    if show_n:
        df1_ = (
            df1.groupby(y_stat)
            .apply(lambda df: df.groupby(colindex).ngroups)
            .to_frame("n")
            .reset_index()
        )
        df1_[axis_desc] = df1_[y_stat].map(ticklabel2position)
        if test:
            logging.info(df1_)
        if show_n_ticklabels:
            df1_["label"] = df1_.apply(
                lambda x: f"{x[y_stat]}\n({show_n_prefix}{x['n']})", axis=1
            )
            ax = rename_ticklabels(
                ax=ax, axis=axis_desc, rename=df1_.rd.to_dict([y_stat, "label"])
            )
        else:
            import matplotlib.transforms as transforms

            df1_.apply(
                lambda x: ax.text(
                    **{
                        axis_cont: 1.1 + offx_n,
                        axis_desc: x[axis_desc],
                    },
                    s=show_n_prefix + str(x["n"]),
                    va="center" if axis_desc == "y" else "top",
                    ha=show_n_ha
                    if show_n_ha is not None
                    else "left"
                    if axis_desc == "y"
                    else "center",
                    transform=transforms.blended_transform_factory(
                        **{
                            f"{axis_cont}_transform": ax.transAxes,
                            f"{axis_desc}_transform": ax.transData,
                        }
                    ),
                ),
                axis=1,
            )
    ax.tick_params(axis=axis_desc, colors="k")
    if hue is not None:
        o1 = ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1),
            # frameon=True,
            title=hue,
        )
        # o1.get_frame().set_edgecolor((0.95,0.95,0.95))
    if show_outlines:
        column_outlines = show_outlines
        ## get jitter positions and plot outlines
        from roux.viz.annot import show_outlines
        df_jit=get_jitter_positions(
                ax,
                df1,
                order=order,
                column_category=x if axis_desc == "x" else y,
                column_position=("x" if axis_desc == "x" else "y"),  # jitter is along the axis with decrete values
            )
        # df_jit=df_jit.dropna([column_outlines])
        show_outlines(
            df_jit,
            colx="x" if axis_desc == "x" else x,
            coly=y if axis_desc == "x" else 'y',
            column_outlines=column_outlines,
            **kws_outlines,
            ax=ax,
        )
    return ax


def pointplot_groupbyedgecolor(
    data: pd.DataFrame, ax: plt.Axes = None, **kws
) -> plt.Axes:
    """Plot seaborn's `pointplot` grouped by edgecolor of points.

    Args:
        data (pd.DataFrame): input data.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Keyword Args:
        kws: parameters provided to the `seaborn`'s `pointplot` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    ax = plt.subplot() if ax is None else ax
    ax = sns.pointplot(data=data, ax=ax, **kws)
    plt.setp(ax.collections, sizes=[100])
    for c in ax.collections:
        if c.get_label().startswith(kws["hue_order"][0].split(" ")[0]):
            c.set_linewidth(2)
            c.set_edgecolor("k")
        else:
            c.set_linewidth(2)
            c.set_edgecolor("w")
    ax.legend(bbox_to_anchor=[1, 1])
    return ax

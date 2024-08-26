"""For plotting sets."""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import roux.lib.dfs as rd  # noqa
from roux.lib.df import to_map_binary
from roux.viz.ax_ import set_axlims, set_ylabel


def plot_venn(
    ds1: pd.Series,
    ax: plt.Axes = None,
    figsize: tuple = [2.5, 2.5],
    show_n: bool = True,
    outmore=False,
    **kws,
) -> plt.Axes:
    """Plot Venn diagram.

    Args:
        ds1 (pd.Series): input pandas.Series or dictionary. Subsets in the index levels, mapped to counts.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to [2.5,2.5].
        show_n (bool, optional): show sample sizes. Defaults to True.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    # if ds1.dtypes!='int':
    if isinstance(ds1, dict):
        from roux.lib.df import dict2df

        df_ = to_map_binary(
            dict2df(ds1).explode("value"), colgroupby="key", colvalue="value"
        )
        ds2 = df_.groupby(df_.columns.tolist()).size()
    elif isinstance(ds1, pd.Series):
        # assert not ds1._is_view, "input series should be a copy not a view"
        ds2 = ds1.copy()
    assert isinstance(ds2, pd.Series)
    assert ds2.dtypes == "int"
    assert len(ds2.index.names) >= 2
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if show_n:
        from roux.lib.df import get_totals

        d1 = get_totals(ds2)
        ds2.index.names = [f"{k}\n({d1[k]})" for k in ds2.index.names]
    set_labels = list(ds2.index.names)
    if len(set_labels) == 1 or len(set_labels) > 3:
        logging.warning("need 2 or 3 sets")
        return
    ds2.index = ["".join([str(int(i)) for i in list(t)]) for t in ds2.index]
    import matplotlib_venn as mv

    venn = getattr(mv, f"venn{len(set_labels)}")(
        subsets=ds2.to_dict(),
        set_labels=set_labels,
        ax=ax,
        **kws,
    )
    # if len(set_labels)==2:
    #     ds1={key:sorted(val) for key,val in ds1.items()}
    #     if hasattr(venn.get_label_by_id('11'),'set_text'):
    #         venn.get_label_by_id('11').set_text('\n'.join(sorted(set(ds1[list(ds1.keys())[0]])&set(ds1[list(ds1.keys())[1]]))+[f"({venn.get_label_by_id('11').get_text()})"]))
    #     if hasattr(venn.get_label_by_id('10'),'set_text'):
    #         venn.get_label_by_id('10').set_text('\n'.join(sorted(set(ds1[list(ds1.keys())[0]])-set(ds1[list(ds1.keys())[1]]))+[f"({venn.get_label_by_id('10').get_text()})"]))
    #     if hasattr(venn.get_label_by_id('01'),'set_text'):
    #         venn.get_label_by_id('01').set_text('\n'.join(sorted(set(ds1[list(ds1.keys())[1]])-set(ds1[list(ds1.keys())[0]]))+[f"({venn.get_label_by_id('01').get_text()})"]))
    if not outmore:
        return ax
    else:
        return ax, venn


def plot_intersection_counts(
    df1: pd.DataFrame,
    cols: list = None,
    kind: str = "table",
    method: str = None,  #'chi2'|fe
    show_counts: bool = True,
    show_pval: bool = True,
    confusion: bool = False,
    rename_cols: bool = False,
    sort_cols: tuple = [True, True],
    order_x: list = None,
    order_y: list = None,
    cmap: str = "Reds",
    ax: plt.Axes = None,
    kws_show_stats: dict = {},
    **kws_plot,
) -> plt.Axes:
    """Plot counts for the intersection between two sets.

    Args:
        df1 (pd.DataFrame): input data
        cols (list, optional): columns. Defaults to None.
        kind (str, optional): kind of plot: table or barplot. Detaults to table.
        method (str, optional): method to check the association ['chi2','FE']. Defaults to None.
        rename_cols (bool, optional): rename the columns. Defaults to True.
        show_pval (bool, optional): annotate p-values. Defaults to True.
        cmap (str, optional): colormap. Defaults to 'Reds'.
        kws_show_stats (dict, optional): arguments provided to stats function. Defaults to {}.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Raises:
        ValueError: `show_pval` position should be the allowed one.

    Keyword Args:
        kws_plot: keyword arguments provided to the plotting function.

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs:
        1. Use `compare_classes` to get the stats.
    """
    if cols is not None:
        for i, c in enumerate(cols):
            df1 = df1.log.dropna(subset=cols).assign(
                **{c: lambda df: df[c].astype(str)}
            )
            for l in [["True", "False"], ["yes", "no"]]:
                if df1[c].isin(l).all():
                    sort_cols[i] = False
                    break
        dplot = pd.crosstab(df1[cols[0]], df1[cols[1]])
    else:
        dplot = df1.copy()

    dplot = dplot.sort_index(axis=0, ascending=sort_cols[0]).sort_index(
        axis=1, ascending=sort_cols[1]
    )
    if dplot.shape == (2, 2) and rename_cols:
        dplot = dplot.rename(
            columns={True: dplot.columns.name, False: "not"},
            index={True: dplot.index.name, False: "not"},
        )
        dplot.columns.name = None
        dplot.index.name = None
        if "not" in dplot.columns:
            dplot = dplot.loc[:, [s for s in dplot.columns if s != "not"] + ["not"]]
        if "not" in dplot.index:
            dplot = dplot.loc[[s for s in dplot.index if s != "not"] + ["not"], :]
    # dplot=dplot.sort_index(ascending=False,axis=1).sort_index(ascending=False,axis=0)
    if order_y is None:
        order_y = dplot.index.tolist()
    if order_x is None:
        order_x = dplot.columns.tolist()
    dplot = dplot.loc[order_y, order_x]
    if kind == "table":
        from roux.viz.heatmap import plot_table

        ax = plot_table(
            dplot,
            cmap=cmap,
            ax=ax,
            **kws_plot,
        )
    elif kind == "bar":
        ax = dplot.plot.barh(stacked=True, ax=ax)
        ax.set(xlabel="count")
        if show_counts:
            ## show counts
            for pa, n in zip(ax.get_children()[:4], dplot.melt()["value"].tolist()):
                bbox = pa.get_bbox()  # left, bottom, width, height
                x, y = np.mean([bbox.x0, bbox.x1]), np.mean([bbox.y0, bbox.y1])
                ax.text(s=n, x=x, y=y, va="center", ha="center")
        if "loc" not in kws_show_stats:
            kws_show_stats["loc"] = "center"
    else:
        raise ValueError(kind)
    if show_pval:
        from roux.viz.annot import show_crosstab_stats

        show_crosstab_stats(
            df1,
            cols=cols,
            ax=ax,
            **kws_show_stats,
        )
    return ax


def plot_intersections(
    ds1: pd.Series,
    item_name: str = None,
    figsize: tuple = [4, 4],
    text_width: float = 2,
    yorder: list = None,
    sort_by: str = "cardinality",
    sort_categories_by: str = None,  #'cardinality',
    element_size: int = 40,
    facecolor: str = "gray",
    bari_annot: int = None,  # 0, 'max_intersections'
    totals_bar: bool = False,
    totals_text: bool = True,
    intersections_ylabel: float = None,
    intersections_min: float = None,
    test: bool = False,
    annot_text: bool = False,
    set_ylabelx: float = -0.25,
    set_ylabely: float = 0.5,
    **kws,
) -> plt.Axes:
    """Plot upset plot.

    Args:
        ds1 (pd.Series): input vector.
        item_name (str, optional): name of items. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to [4,4].
        text_width (float, optional): max. width of the text. Defaults to 2.
        yorder (list, optional): order of y elements. Defaults to None.
        sort_by (str, optional): sorting method. Defaults to 'cardinality'.
        sort_categories_by (str, optional): sorting method. Defaults to None.
        element_size (int, optional): size of elements. Defaults to 40.
        facecolor (str, optional): facecolor. Defaults to 'gray'.
        bari_annot (int, optional): annotate nth bar. Defaults to None.
        totals_text (bool, optional): show totals. Defaults to True.
        intersections_ylabel (float, optional): y-label of the intersections. Defaults to None.
        intersections_min (float, optional): intersection minimum to show. Defaults to None.
        test (bool, optional): test mode. Defaults to False.
        annot_text (bool, optional): annotate text. Defaults to False.
        set_ylabelx (float, optional): x position of the ylabel. Defaults to -0.25.
        set_ylabely (float, optional): y position of the ylabel. Defaults to 0.5.

    Keyword Args:
        kws: parameters provided to the `upset.plot` function.

    Returns:
        plt.Axes: `plt.Axes` object.

    Notes:
        sort_by:{‘cardinality’, ‘degree’}
        If ‘cardinality’, subset are listed from largest to smallest. If ‘degree’, they are listed in order of the number of categories intersected.
        sort_categories_by:{‘cardinality’, None}
        Whether to sort the categories by total cardinality, or leave them in the provided order.

    References:
        https://upsetplot.readthedocs.io/en/stable/api.html
    """
    assert isinstance(ds1, pd.Series)
    if (item_name is None) and (ds1.name is not None):
        item_name = ds1.name
    if intersections_min is None:
        intersections_min = len(ds1)
    if yorder is not None:
        yorder = [c for c in yorder if c in ds1.index.names][::-1]
        ds1.index = ds1.index.reorder_levels(yorder)
    ds2 = (ds1 / ds1.sum()) * 100
    import upsetplot as up

    d = up.plot(
        ds2,
        figsize=figsize,
        text_width=text_width,
        sort_by=sort_by,
        sort_categories_by=sort_categories_by,
        facecolor=facecolor,
        element_size=element_size,
        **kws,
    )
    d["totals"].set_visible(totals_bar)
    if totals_text:
        from roux.lib.df import get_totals

        d_ = get_totals(ds1)
        d["matrix"].set_yticklabels(
            [
                f"{s.get_text()} (n={d_[s.get_text()]})"
                for s in d["matrix"].get_yticklabels()
            ],
        )
    if totals_bar:
        d["totals"].set(ylim=d["totals"].get_ylim()[::-1], xlabel="%")
    set_ylabel(
        ax=d["intersections"],
        s=(f"{item_name}s " if item_name is not None else "")
        + f"%\n(total={ds1.sum()})",
        x=set_ylabelx,
        y=set_ylabely,
    )
    d["intersections"].set(
        xlim=[-0.5, intersections_min - 0.5],
    )
    if sort_by == "cardinality":
        y = ds2.max()
    elif sort_by == "degree":
        y = ds2.loc[tuple([True for i in ds2.index.names])]
    #     if bari_annot=='max_intersections':
    #         l1=[i for i,t in enumerate(ds1.index) if t==tuple(np.repeat(True,len(ds1.index.names)))]
    #         if len(l1)==1:
    #             bari_annot=l1[0]
    #             print(bari_annot)
    #     print(sum(ds1==ds1.max()))
    #     print(bari_annot)
    if sum(ds1 == ds1.max()) != 1:
        bari_annot = None
    if isinstance(bari_annot, int):
        bari_annot = [bari_annot]
    if isinstance(bari_annot, list):
        #         print(bari_annot)
        for i in bari_annot:
            d["intersections"].get_children()[i].set_color("#f55f5f")
    if annot_text and bari_annot == 0:
        d["intersections"].text(
            bari_annot - 0.25,
            y,
            f"{y:.1f}%",
            ha="left",
            va="bottom",
            color="#f55f5f",
            zorder=10,
        )

    #     if intersections_ylabel
    #     if not post_fun is None: post_fun(ax['intersections'])
    return d


def plot_enrichment(
    data: pd.DataFrame,
    x: str,
    y: str,
    s: str,
    hue="Q",
    xlabel=None,
    ylabel="significance\n(-log10(Q))",
    size: int = None,
    color: str = None,
    annots_side: int = 5,
    annots_side_labels=None,
    coff_fdr: float = None,
    xlim: tuple = None,
    xlim_off: float = 0.2,
    ylim: tuple = None,
    ax: plt.Axes = None,
    break_pt: int = 25,
    annot_coff_fdr: bool = False,
    kws_annot: dict = dict(
        loc="right",
        # annot_count_max=5,
        offx3=0.15,
    ),
    returns="ax",
    **kwargs,
) -> plt.Axes:
    """Plot enrichment stats.

    Args:
        data (pd.DataFrame): input data.
        x (str): x column.
        y (str): y column.
        s (str): size column.
        size (int, optional): size of the points. Defaults to None.
        color (str, optional): color of the points. Defaults to None.
        annots_side (int, optional): how many labels to show on side. Defaults to 5.
        coff_fdr (float, optional): FDR cutoff. Defaults to None.
        xlim (tuple, optional): x-axis limits. Defaults to None.
        xlim_off (float, optional): x-offset on limits. Defaults to 0.2.
        ylim (tuple, optional): y-axis limits. Defaults to None.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        break_pt (int, optional): break point ('\n') for the labels. Defaults to 25.
        annot_coff_fdr (bool, optional): show FDR cutoff. Defaults to False.
        kws_annot (dict, optional): parameters provided to the `annot_side` function. Defaults to dict( loc='right', annot_count_max=5, offx3=0.15, ).

    Keyword Args:
        kwargs: parameters provided to the `sns.scatterplot` function.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if coff_fdr is None:
        coff_fdr = 1
    if any(data[y] == 0):
        logging.warning(
            f"found {sum(data[y]==0)} zeros in among y values; replaced them with the next minimum which is {data[y].replace(0,np.nan).min()}."
        )
        data[y] = data[y].replace(0, data[y].replace(0, np.nan).min())
    from roux.stat.transform import log_pval

    # if y.startswith('P '):
    data[ylabel] = data[y].apply(log_pval)
    data[hue] = pd.cut(
        x=data[y],
        bins=[
            # data['P (FE test, FDR corrected)'].min(),
            0,
            0.01,
            0.05,
            coff_fdr,
        ],
        right=False,
    )
    y = ylabel
    if size is not None:
        if not data[size].dtype == "category":
            data[size] = pd.qcut(data[size], q=3, duplicates="drop")
        data = data.sort_values(size, ascending=False)
        data[size] = data[size].apply(lambda x: f"({x.left:.0f}, {x.right:.0f}]")
    if ax is None:
        fig, ax = plt.subplots()  # (figsize=[1.5,4])
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        size=size if size is not None else None,
        size_order=data[size].unique() if size is not None else None,
        hue=hue,
        # color=color,
        zorder=2,
        ax=ax,
        **kwargs,
    )
    if size is not None:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.1, 0.1),
            # title=size,
            frameon=True,
            #          nrow=3,
            ncol=2,
        )
    if xlim is None:
        ax = set_axlims(ax, off=xlim_off, axes=["x"])
    else:
        ax.set(xlim=xlim)
    # if ylim is None:
    #     ax.set(ylim=(log_pval(coff_fdr),ax.get_ylim()[1]),
    # #               xlim=(data[x].min(),data[x].max()),
    #           )
    # else:
    #     ax.set(ylim=ylim)
    if annot_coff_fdr:
        ax.annotate(
            f"Q={coff_fdr}",
            xy=(ax.get_xlim()[0], log_pval(coff_fdr)),
            xycoords="data",
            #             xytext=(-10,log_pval(coff_fdr)), textcoords='data',
            xytext=(0.01, 0.1),
            textcoords="figure fraction",
            ha="left",
            va="top",
            arrowprops=dict(
                arrowstyle="->",
                color="0.5",
                #                             shrinkA=5, shrinkB=5,
                #                             patchA=None, patchB=None,
                connectionstyle="arc3,rad=-0.3",
            ),
        )
    from roux.viz.annot import annot_side

    ax = annot_side(
        ax=ax,
        df1=data.sort_values(y, ascending=False).head(annots_side)
        if annots_side_labels is None
        else data.loc[data[s].isin(annots_side_labels), :],
        colx=x,
        coly=y,
        cols=s,
        break_pt=break_pt,
        # offymin=kws_annot['offymin'] if 'offymin' in kws_annot else if not size is None else 0,
        zorder=3,
        **kws_annot,
    )
    return locals()[returns]


## subsets
def _to_data_plot_pie(
    data: pd.DataFrame,
    rename: dict = None,
    colors: list = None,
    explode: str = None,
    show_n: bool = False,
    order: list = None,
) -> pd.DataFrame:
    """Preprocess data for the pie plot.

    Args:
        data (pd.DataFrame): input data
        rename (dict, optional): rename the subsets. Defaults to None.
        colors (list, optional): colors of the subsets. Defaults to None.
        explode (str, optional): separate subset/s. Defaults to None.
        show_n (bool, optional): show the sample size. Defaults to False.
        order (list, optional): order of the subsets. Defaults to None.

    Raises:
        ValueError: explode parameter should be a either 'first' or 'last'

    Returns:
        pd.DataFrame: output table
    """
    data_ = data.to_frame("count").rename_axis("name").reset_index()
    if order is None:
        if rename is not None:
            order = list(rename.values())
        else:
            order = data_["name"].tolist()
        logging.warning(f"inferred `order`={order}")
    if rename is not None:
        data_ = data_.assign(
            **{
                "name": lambda df: df["name"].map(rename),
            }
        ).rd.sort_valuesby_list(by="name", cats=order)
    if colors is not None:
        if isinstance(colors, list):
            colors = dict(zip(order, colors))
        data_ = data_.assign(
            **{
                "color": lambda df: df["name"].map(colors),
            }
        )
    if explode is not None:
        if explode == "first":
            explode = order[0]
        elif explode == "last":
            explode = order[-1]
        else:
            raise ValueError(explode)
        data_ = data_.assign(
            **{
                "explode": lambda df: df["name"].apply(
                    lambda x: 0.1 if explode == x else 0
                ),
            }
        )
    if show_n:
        data_["perc"] = 100.0 * data_["count"] / data_["count"].sum()
        data_ = (
            data_.assign(
                **{
                    "name": lambda df: df.apply(
                        lambda x: f"{x['name']}\n{x['perc']:.0f}% ({x['count']})",
                        axis=1,
                    ),
                }
            )
        ).drop(["perc"], axis=1)

    return data_.rename(
        columns={
            "name": "labels",
            "count": "counts",
            "color": "colors",
        },
        errors="ignore",
    )


def plot_pie(
    counts: list,
    labels: list,
    scales_line_xy: tuple = (1.1, 1.1),
    remove_wedges: list = None,
    remove_wedges_index: list = [],
    line_color: str = "k",
    annot_side: bool = False,
    kws_annot_side: dict = {},
    ax: plt.Axes = None,
    **kws_pie,
) -> plt.Axes:
    """Pie plot.

    Args:
        counts (list): counts.
        labels (list): labels.
        scales_line_xy (tuple, optional): scales for the lines. Defaults to (1.1,1.1).
        remove_wedges (list, optional): remove wedge/s. Defaults to None.
        remove_wedges_index (list, optional): remove wedge/s by index. Defaults to [].
        line_color (str, optional): line color. Defaults to 'k'.
        annot_side (bool, optional): annotations on side using the `annot_side` function. Defaults to False.
        kws_annot_side (dict, optional): keyword arguments provided to the `annot_side` function. Defaults to {}.
        ax (plt.Axes, optional): subplot. Defaults to None.

    Keyword Args:
        kws_pie: keyword arguments provided to the `pie` chart function.

    Returns:
        plt.Axes: subplot

    References:
        https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_and_donut_labels.html
    """
    if ax is None:
        ax = plt.gca()
    wedges, texts = ax.pie(
        counts,
        startangle=90,
        counterclock=False,
        **kws_pie,
    )

    kws_annotate = dict(
        arrowprops=dict(arrowstyle="-", color=line_color, shrinkB=0),
        zorder=0,
        va="center",
    )
    if remove_wedges is not None:
        remove_wedges_index += [labels.index(k) for k in remove_wedges]
    if annot_side:
        ## collect inputs
        ks, xs, ys = [], [], []
    for i, (p, t) in enumerate(zip(wedges, texts)):
        ## remove wedge/s
        if i in remove_wedges_index:
            p.set_visible(False)
            t.set_visible(False)
            continue
        ## labels
        ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        if not annot_side:
            horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
            # horizontalalignment = 'center'
            connectionstyle = f"angle,angleA=0,angleB={ang}"
            kws_annotate["arrowprops"].update({"connectionstyle": connectionstyle})

            ax.annotate(
                labels[i],
                xy=(x, y),
                xytext=(scales_line_xy[0] * np.sign(x), scales_line_xy[1] * y),
                horizontalalignment=horizontalalignment,
                **kws_annotate,
            )
        else:
            ks.append(labels[i])
            xs.append(x)
            ys.append(y)
    if annot_side:
        df1 = pd.DataFrame(dict(labels=ks, xs=xs, ys=ys))
        from roux.viz.annot import annot_side

        ## right
        df1_ = df1.query("`xs` >= 0")
        ax = annot_side(
            ax=ax,
            df1=df1_,
            loc="right",
            colx="xs",
            coly="ys",
            cols="labels",
            color=line_color,
            kws_line=dict(lw=1),
            **(
                kws_annot_side
                if len(df1_) != 1
                else {
                    k: v
                    for k, v in kws_annot_side.items()
                    if k not in ["offymin", "offymax"]
                }
            ),
        )
        ## left
        df1_ = df1.query("`xs` < 0")
        ax = annot_side(
            ax=ax,
            loc="left",
            df1=df1_,
            colx="xs",
            coly="ys",
            cols="labels",
            color=line_color,
            kws_line=dict(lw=1),
            **(
                kws_annot_side
                if len(df1_) != 1
                else {
                    k: v
                    for k, v in kws_annot_side.items()
                    if k not in ["offymin", "offymax"]
                }
            ),
        )
    return ax

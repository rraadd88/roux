"""For comparative plots."""

import matplotlib.pyplot as plt
from roux.viz.colors import get_colors_default
from roux.viz.io import to_plot
from pathlib import Path


def plot_comparisons(
    plot_data,
    x,
    ax=None,
    output_dir_path=None,
    force=False,
    return_path=False,
):
    """
    Parameters:
        plot_data: output of `.stat.compare.get_comparison`
    Notes:
        `sample type`: different sample of the same data.
    """
    ## get the sample type
    ## get the data
    ## get the columns
    if output_dir_path is not None:
        output_path = f"{output_dir_path}/{x['comparison type']}/{x['variable x']}/{x['variable y']}/{x['sample type']}"
        if Path(output_path + ".pdf").exists() and not force:
            return
    if ax is None:
        fig, ax = plt.subplots(figsize=[2, 2])
    if x["comparison type"].startswith("correlation "):
        from roux.viz.scatter import plot_scatter

        ax = plot_scatter(
            dplot=plot_data[x["sample type"]]["data"],
            colx=x["variable y"],
            coly=x["variable x"],
            trendline_method="lowess",
            stat_method="spearman",
            resample=True,
            params_plot={
                "color": get_colors_default()[0],
                "ec": get_colors_default()[0],
                "fc": "none",
                "linewidth": 1,
            },
            params_plot_trendline={"linestyle": ":"},
            params_set_label={"loc": 2, "off_loc": 0.01},
            ax=ax,
            # **kws,
        )
    elif x["comparison type"].startswith("difference "):
        if plot_data[x["sample type"]]["data"][x["variable y"]].dtypes in (float, int):
            colx, coly = x["variable x"], x["variable y"]
        else:
            colx, coly = x["variable y"], x["variable x"]
        from roux.viz.dist import plot_dists

        ax = plot_dists(
            df1=plot_data[x["sample type"]]["data"],
            x=x["variable y"],
            y=x["variable x"],
            colindex=plot_data[x["sample type"]]["columns"]["cols_index"],
            hue=None,
            # order=['high','non-redistribution','low'],
            hue_order=None,
            palette=get_colors_default()[:2],
            kind=["box", "swarm"],
            show_p=True,
            show_n=True,
            show_n_prefix="",
            offx_n=0,
            xlim=None,
            # xscale='log',
            offx_pval=0.05,
            # offy_pval=-0.5,
            saturate_color_alpha=1,
            ax=ax,
            # kws_stats=dict(fun=ttest),
            # **kws,
        )
        ax.set(ylabel=x["variable x"])
    elif x["comparison type"].startswith("association "):
        # %run ../../../code/roux/roux/viz/heatmap.py
        from roux.viz.heatmap import plot_crosstab

        plot_crosstab(
            plot_data[x["sample type"]]["data"],
            [x["variable x"], x["variable y"]],
            method="fe",
            fmt="d",
            ax=ax,
            # ax=axs[rowi,coli],
        )
    else:
        raise ValueError(x["comparison type"])
    if output_dir_path is not None:
        output_path = to_plot(
            f"{output_dir_path}/{x['comparison type']}/{x['variable x']}/{x['variable y']}/{x['sample type']}",
            fmts=["pdf", "png"],
        )
        plt.close(fig)
        if return_path:
            return output_path
    return ax

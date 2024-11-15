"""For setting up subplots."""

# import seaborn as sns
import pandas as pd
import numpy as np
import logging

## viz basic
import matplotlib.pyplot as plt

from roux.lib.str import replace_many


def set_axes_minimal(
    ax,
    xlabel=None,
    ylabel=None,
    off_axes_pad=0,
) -> plt.Axes:
    """
    Set minimal axes labels, at the lower left corner.
    """
    if xlabel is None:
        xlabel = ax.get_xlabel()
    if ylabel is None:
        ylabel = ax.get_ylabel()

    ax.arrow(
        x=off_axes_pad,
        y=off_axes_pad,
        dx=0.1,
        dy=0,
        head_width=0.02,
        transform=ax.transAxes,  # fig.transFigure,
        clip_on=False,
        color="k",
        lw=1,
    )
    ax.arrow(
        x=off_axes_pad,
        y=off_axes_pad,
        dx=0,
        dy=0.1,
        head_width=0.02,
        transform=ax.transAxes,  # fig.transFigure,
        clip_on=False,
        color="k",
        lw=1,
    )
    ax.text(
        x=off_axes_pad,
        y=off_axes_pad - 0.01,
        s=xlabel,
        transform=ax.transAxes,  # fig.transFigure,
        ha="left",
        va="top",
    )
    ax.text(
        y=off_axes_pad,
        x=off_axes_pad - 0.01,
        s=ylabel,
        transform=ax.transAxes,  # fig.transFigure,
        rotation=90,
        ha="right",
        va="bottom",
    )
    return ax


def set_axes_arrows(
    ax: plt.Axes,
    length: float = 0.1,
    pad: float = 0.2,
    color: str = "k",
    head_width: float = 0.03,
    head_length: float = 0.02,
    length_includes_head: bool = True,
    clip_on: bool = False,
    **kws_arrow,
):
    """
    Set arrows next to the axis labels.

    Parameters:
        ax (plt.Axes): subplot.
        color=
    """
    kws = {
        **dict(
            fc=color,
            ec=color,
            head_width=head_width,
            head_length=head_length,
            length_includes_head=length_includes_head,
            clip_on=clip_on,
            transform=ax.transAxes,
        ),
        ## overwrite
        **kws_arrow,
    }
    ax.arrow(1 - length, -1 * (length * (1 + pad)), length, 0, **kws)
    ax.arrow(-1 * length, 1 + (length * pad), 0.0, length, **kws)
    return ax


# labels
def set_label(
    s: str,
    ax: plt.Axes,
    x: float = 0,
    y: float = 0,
    ha: str = "left",
    va: str = "top",
    loc=None,
    off_loc=0.01,
    title: bool = False,
    **kws,
) -> plt.Axes:
    """Set label on a plot.

    Args:
        x (float): x position.
        y (float): y position.
        s (str): label.
        ax (plt.Axes): `plt.Axes` object.
        ha (str, optional): horizontal alignment. Defaults to 'left'.
        va (str, optional): vertical alignment. Defaults to 'top'.
        loc (int, optional): location of the label. 1:'upper right', 2:'upper left', 3:'lower left':3, 4:'lower right'
        offs_loc (tuple,optional): x and y location offsets.
        title (bool, optional): set as title. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if title:
        ax.set_title(s, **kws)
    elif loc is not None:
        if loc == 1 or loc == "upper right":
            x = 1 - off_loc
            y = 1 - off_loc
            ha = "right"
            va = "top"
        elif loc == 2 or loc == "upper left":
            x = 0 + off_loc
            y = 1 - off_loc
            ha = "left"
            va = "top"
        elif loc == 3 or loc == "lower left":
            x = 0 + off_loc
            y = 0 + off_loc
            ha = "left"
            va = "bottom"
        elif loc in [0, 4] or loc == "lower right":
            x = 1 - off_loc
            y = 0 + off_loc
            ha = "right"
            va = "bottom"
        else:
            raise ValueError(loc)
    ax.text(s=s, x=x, y=y, ha=ha, va=va, transform=ax.transAxes, **kws)
    return ax


def set_ylabel(
    ax: plt.Axes,
    s: str = None,
    x: float = -0.1,
    y: float = 1.02,
    xoff: float = 0,
    yoff: float = 0,
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
    if s is not None:  # and (ax.get_ylabel()=='' or ax.get_ylabel() is None):
        ax.set_ylabel(s)
    ax.set_ylabel(ax.get_ylabel(), rotation=0, ha="right", va="center")
    ax.yaxis.set_label_coords(x + xoff, y + yoff)
    return ax


def get_ax_labels(
    ax: plt.Axes,
):
    labels = []
    for k in ["get_xlabel", "get_ylabel", "get_title", "legend_"]:
        if hasattr(ax, k):
            if k != "legend_":
                # plotp=f"{plotp}_"+(getattr(ax,k)()).replace('.','_')
                labels.append(getattr(ax, k)())
            else:
                if ax.legend_ is not None:
                    # plotp=f"{plotp}_"+(ax.legend_.get_title().get_text()).replace('.','_')
                    labels.append(ax.legend_.get_title().get_text())
    return labels


def format_labels(
    ax,
    axes: list = ["x", "y"],
    fmt="cap1",
    title_fontsize=15,
    rename_labels=None,
    rotate_ylabel=True,
    y=1.05,
    test=False,
    textwrap_width=None,
):
    def cap1(s):
        return s[0].upper() + s[1:]

    for k in ["legend"] + [f"{k}label" for k in axes] + ["title"]:
        if k == "title":
            kws = dict(fontdict=dict(fontsize=title_fontsize))
        else:
            kws = {}
        if hasattr(ax, "get_" + k):
            if k == "legend":
                ## adjust legend first, because setting other labels can have unexpected effects on the legend.
                if ax.legend_ is not None:
                    label = ax.legend_.get_title().get_text()
                    if rename_labels is not None:
                        label = replace_many(label, rename_labels, ignore=True)
                    if test:
                        print(label)
                    if fmt == "cap1":
                        if isinstance(label, str):
                            if label != "":
                                ax.legend(title=cap1(label))
            else:
                label = getattr(ax, "get_" + k)()
                if rename_labels is not None:
                    label = replace_many(label, rename_labels, ignore=True)
                if isinstance(label, str):
                    if label != "":
                        if fmt == "cap1":
                            getattr(ax, "set_" + k)(cap1(label), **kws)
    if rotate_ylabel:
        ax.set_ylabel(
            ax.get_ylabel(),
            ha="left",
            y=y,
            rotation=0,
            labelpad=0,
        )
        ax.yaxis.set_label_coords(-0.05, 1.02)
        set_axes_arrows(ax=ax)
        
    if isinstance(textwrap_width,int):
        from roux.lib.str import linebreaker
        ax.set(
            xlabel=linebreaker(
                ax.get_xlabel(),
                width=textwrap_width,
                ),
            ylabel=linebreaker(
                ax.get_ylabel(),
                width=textwrap_width,
                ),                
        )
    return ax


## ticklabels
def rename_ticklabels(
    ax: plt.Axes,
    axis: str,
    rename: dict = None,
    replace: dict = None,
    ignore: bool = False,
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
    k = f"{axis}ticklabels"
    if replace is not None:
        from roux.lib.str import replace_many

        _ = getattr(ax, f"set_{k}")(
            [
                replace_many(t.get_text(), replace, ignore=ignore)
                for t in getattr(ax, f"get_{k}")()
            ]
        )
    elif rename is not None:
        _ = getattr(ax, f"set_{k}")(
            [rename[t.get_text()] for t in getattr(ax, f"get_{k}")()]
        )
    else:
        raise ValueError("either `rename` or `replace` should be provided.")
    return ax


def get_ticklabel_position(
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
    return dict(
        zip(
            [t.get_text() for t in getattr(ax, f"get_{axis}ticklabels")()],
            getattr(ax, f"{axis}axis").get_ticklocs(),
        )
    )


def set_ticklabels_color(
    ax: plt.Axes, ticklabel2color: dict, axis: str = "y"
) -> plt.Axes:
    """Set colors to ticklabels.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        ticklabel2color (dict): colors of the ticklabels.
        axis (str): axis (x|y).

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    for tick in getattr(ax, f"get_{axis}ticklabels")():
        if tick.get_text() in ticklabel2color.keys():
            tick.set_color(ticklabel2color[tick.get_text()])
    return ax


def format_ticklabels(
    ax: plt.Axes,
    axes: tuple = ["x", "y"],
    interval: float = None,
    n: int = None,
    fmt: str = None,
    font: str = None,  #'DejaVu Sans Mono',#"Monospace"
) -> plt.Axes:
    """format_ticklabels

    Args:
        ax (plt.Axes): `plt.Axes` object.
        axes (tuple, optional): axes. Defaults to ['x','y'].
        n (int, optional): number of ticks. Defaults to None.
        fmt (str, optional): format e.g. '.0f'. Defaults to None.
        font (str, optional): font. Defaults to 'DejaVu Sans Mono'.

    Returns:
        plt.Axes: `plt.Axes` object.

    TODOs:
        1. include color_ticklabels
    """
    if isinstance(n, int):
        n = {"x": n, "y": n}
    if isinstance(fmt, str):
        fmt = {"x": fmt, "y": fmt}
    for axis in axes:
        if n is not None:
            getattr(ax, axis + "axis").set_major_locator(plt.MaxNLocator(n[axis]))
        elif interval is not None:
            getattr(ax, axis + "axis").set_major_locator(plt.MultipleLocator(interval))
        if fmt is not None:
            if fmt[axis] == "counts":
                max_val = getattr(ax, f"get_{axis}lim")()[1]
                if max_val <= 10:
                    interval = 1
                elif max_val <= 100:
                    interval = 10
                elif max_val <= 1000:
                    interval = 100
                else:
                    interval = 1000

                import matplotlib.ticker as ticker

                locator = ticker.MultipleLocator(interval)
                # locator
                getattr(ax, f"{axis}axis").set_major_locator(locator)
                ## start with 1
                ticks = getattr(ax, f"get_{axis}ticks")()
                getattr(ax, f"set_{axis}ticks")(np.where(ticks == 0, 1, ticks))
                ## as integers
                getattr(ax, axis + "axis").set_major_formatter(
                    plt.FormatStrFormatter("%d")
                )
                getattr(ax, f"set_{axis}lim")(1, max_val)

            # elif fmt[axis] is not None:
            else:
                getattr(ax, axis + "axis").set_major_formatter(
                    plt.FormatStrFormatter(fmt[axis])
                )

        if font is not None:
            for tick in getattr(ax, f"get_{axis}ticklabels")():
                tick.set_fontname(font)
    return ax


def split_ticklabels(
    ax: plt.Axes,
    fmt: str,
    axis="x",
    group_x=-0.45,  # x-position of the group labels
    group_y=-0.25,  # x-position of the group labels
    group_prefix=None,
    group_suffix=False,
    group_loc="center",
    # group_pad=0.02,
    group_colors=None,
    group_alpha=0.2,
    show_group_line=True,
    group_line_off_x=0.15,
    group_line_off_y=0.1,
    show_group_span=False,
    group_span_kws={},
    sep: str = "-",
    pad_major=6,
    off: float = 0.2,
    test: bool = False,
    **kws,
) -> plt.Axes:
    """Split ticklabels into major and minor. Two minor ticks are created per major tick.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        fmt (str): 'group'-wise or 'pair'-wise splitting of the ticklabels.
        axis (str): name of the axis: x or y.
        sep (str, optional): separator within the tick labels. Defaults to ' '.
        test (bool, optional): test-mode. Defaults to False.
    Returns:
        plt.Axes: `plt.Axes` object.
    """
    ax.set(**{f"{axis}label": None})
    ticklabels = getattr(ax, f"get_{axis}ticklabels")()
    if fmt.startswith("pair"):
        # kws={
        #     f"{axis}ticks":flatten([[i-off,i+off] for i in range(0,len(ticklabels))]),
        #     f"{axis}ticklabels":flatten([t.get_text().split('-') for t in ticklabels]),
        # }
        # print(kws)
        getattr(ax, f"set_{axis}ticklabels")(
            [s.get_text().replace(sep, "\n") for s in ticklabels],
            # **kws,
        )
        # ax.set(**kws)
    elif fmt.startswith("group"):
        axlims = get_axlims(ax)
        # if axis=='x':logging.warning(f'axis={axis} is not tested.')
        from roux.lib.df import dict2df

        df0_ = dict2df(
            get_ticklabel_position(ax=ax, axis=axis),
            colkey=axis + "ticklabel",
            colvalue=axis,
        )
        if sep is not None:
            df0_[axis + "ticklabel major"] = df0_[axis + "ticklabel"].str.split(
                sep, n=1, expand=True
            )[0]
            df0_[axis + "ticklabel minor"] = df0_[axis + "ticklabel"].str.split(
                sep, n=1, expand=True
            )[1]
        else:
            df0_[axis + "ticklabel major"] = df0_[axis + "ticklabel"].tolist()
            df0_[axis + "ticklabel minor"] = ["" for k in df0_[axis + "ticklabel"]]
        df_ = (
            df0_.groupby(axis + "ticklabel major")
            .agg(
                {
                    axis: ["min", "max", len],
                }
            )
            .rd.flatten_columns()
        )
        if test:
            print(df0_)
            print(df_)
        # if group_loc=='left':
        #     group_x=group_x
        #     group_x=axlims[axis]['min']-(axlims[axis]['len']*group_pad)
        #     group_xlabel=axlims[axis]['min']-(axlims[axis]['len']*group_pad-0.1)
        # elif group_loc=='right':
        #     group_x=axlims[axis]['max']+(axlims[axis]['len']*group_pad)
        #     group_xlabel=axlims[axis]['max']+(axlims[axis]['len']*group_pad+0.1)
        # print(axlims[axis]['min']-(group_pad*5.5))
        import matplotlib.transforms as transforms

        if show_group_line:
            df_.apply(
                lambda x: ax.plot(
                    *(
                        [
                            [x[axis + " min"] - 0.3, x[axis + " max"] + 0.3],
                            [group_y + group_line_off_y, group_y + group_line_off_y],
                        ]
                        if axis == "x"
                        else [
                            [group_x + group_line_off_x, group_x + group_line_off_x],
                            [x[axis + " min"] - 0.3, x[axis + " max"] + 0.3],
                        ]
                        if axis == "y"
                        else logging.error(axis)
                    ),
                    clip_on=False,
                    lw=0.5,
                    color="k",
                    transform=transforms.blended_transform_factory(
                        *(
                            (ax.transAxes, ax.transData)
                            if axis == "y"
                            else (ax.transData, ax.transAxes)
                        )
                    ),
                ),
                axis=1,
            )
        if show_group_span:
            # print(axlims[axis]['min']-(group_pad*5.5))
            # print(axlims[axis]['min'])
            axhspan_kws = dict(
                zip(["xmin", "xmax"], sorted([group_x, axlims[axis]["min"]]))
            )
            # print(axhspan_kws)
            df_.apply(
                lambda x: ax.axhspan(
                    # xmin=(group_x/axlims[axis]['len'])*0.5,
                    # xmin=axlims[axis]['min']-(group_pad*5.5),
                    # xmin=group_x,
                    # xmax=axlims[axis]['min'],
                    **dict(
                        zip(["xmin", "xmax"], sorted([group_x, axlims[axis]["min"]]))
                    ),
                    ymin=x[axis + " min"] - 0.5,
                    ymax=x[axis + " max"] + 0.5,
                    transform="axes",
                    clip_on=False,
                    zorder=0,
                    color=None if group_colors is None else group_colors[x.name],
                    edgecolor="none",
                    alpha=group_alpha,
                    **{**group_span_kws},
                ),
                axis=1,
            )
        df_.apply(
            lambda x: ax.text(
                **(
                    dict(x=group_x, y=np.mean([x[axis + " min"], x[axis + " max"]]))
                    if axis == "y"
                    else dict(
                        x=np.mean([x[axis + " min"], x[axis + " max"]]), y=group_y
                    )
                ),
                s=(group_prefix + "\n" if group_prefix is not None else "")
                + f"{x.name}".replace(" ", "\n")
                + (("\n" + f"(n={int(x[axis+' len'])})") if group_suffix else ""),
                color="k",
                # ha=get_alt(['left','right'],group_loc),
                ha=group_loc,
                va="center",
                transform=transforms.blended_transform_factory(
                    *(
                        (ax.transAxes, ax.transData)
                        if axis == "y"
                        else (ax.transData, ax.transAxes)
                    )
                ),
                # transform=ax.transAxes,
            ),
            axis=1,
        )
        getattr(ax, f"set_{axis}ticklabels")(
            [s.get_text().split(sep, 1)[1] for s in ticklabels],
            **kws,
        )
    else:
        import pandas as pd

        if axis == "y":
            logging.warning(f"axis={axis} is not tested.")
        ticklabels_major = pd.unique(
            ["\u2014\n" + s.get_text().split(sep)[0] for s in ticklabels]
        )
        ticklabels_minor = [s.get_text().split(sep)[1] for s in ticklabels]

        ticks_minor = getattr(ax, f"get_{axis}ticks")()
        ticks_major = ticks_minor.reshape(int(len(ticks_minor) / 2), 2).mean(axis=1)
        _ = getattr(ax, f"set_{axis}ticks")(ticks_major, minor=False)
        getattr(ax, f"set_{axis}ticklabels")(
            ticklabels_major,
            minor=False,
            **kws,
        )
        _ = ax.set_xticks(ticks_minor, minor=True)
        getattr(ax, f"set_{axis}ticklabels")(
            ticklabels_minor,
            minor=True,
            **kws,
        )
        ax.tick_params(axis=axis, which="minor", bottom=True, pad=0)
        ax.tick_params(axis=axis, which="major", bottom=False, pad=pad_major)
    return ax


# axis limits
def get_axlimsby_data(
    X: pd.Series, Y: pd.Series, off: float = 0.2, equal: bool = False
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
    logging.warning("Check ax.autoscale.")
    try:
        xmin = np.min(X)
        xmax = np.max(X)
    except:
        print(X)
    xlen = xmax - xmin
    ymin = np.min(Y)
    ymax = np.max(Y)
    ylen = ymax - ymin
    xlim = (xmin - off * xlen, xmax + off * xlen)
    ylim = (ymin - off * ylen, ymax + off * ylen)
    if not equal:
        return xlim, ylim
    else:
        lim = [np.min([xlim[0], ylim[0]]), np.max([xlim[1], ylim[1]])]
        return lim, lim


def get_axlims(ax: plt.Axes) -> plt.Axes:
    """Get axis limits.

    Args:
        ax (plt.Axes): `plt.Axes` object.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    d1 = {}
    for axis in ["x", "y"]:
        d1[axis] = {}
        d1[axis]["min"], d1[axis]["max"] = getattr(ax, f"get_{axis}lim")()
        if d1[axis]["min"] > d1[axis]["max"]:
            d1[axis]["min"], d1[axis]["max"] = d1[axis]["max"], d1[axis]["min"]
        d1[axis]["len"] = abs(d1[axis]["min"] - d1[axis]["max"])
    return d1


def set_equallim(
    ax: plt.Axes,
    diagonal: bool = False,
    difference: float = None,
    format_ticks: bool = True,
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
    min_, max_ = (
        np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
        np.max([ax.get_xlim()[1], ax.get_ylim()[1]]),
    )
    if diagonal:
        ax.plot([min_, max_], [min_, max_], ":", color="gray", zorder=5)
    if difference is not None:
        off = np.sqrt(difference**2 + difference**2)
        ax.plot([min_ + off, max_ + off], [min_, max_], ":", color="gray", zorder=5)
        ax.plot([min_ - off, max_ - off], [min_, max_], ":", color="gray", zorder=5)
    if format_ticks and len(ax.get_xticklabels()) != 0:
        ax = format_ticklabels(ax, n=len(ax.get_xticklabels()), **kws_format_ticklabels)
        # logging.warning('format_ticklabels failed possibly because of shared axes (?).')
    ax.set_xlim(min_, max_)
    ax.set_ylim(min_, max_)
    #     ax.set_xticks(ax.get_yticks())
    ax.set_aspect("equal", "box")
    return ax


def set_axlims(
    ax: plt.Axes,
    off: float,
    axes: list = ["x", "y"],
    equal=False,
    **kws_set_equallim,
) -> plt.Axes:
    """Set axis limits.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        off (float): offset.
        axes (list, optional): axis name/s. Defaults to ['x','y'].

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if not equal:
        logging.warning("prefer `ax.margins`")
        d1 = get_axlims(ax)
        for k in axes:
            off_ = (d1[k]["len"]) * off
            if not getattr(ax, f"{k}axis").get_inverted():
                getattr(ax, f"set_{k}lim")(d1[k]["min"] - off_, d1[k]["max"] + off_)
            else:
                getattr(ax, f"set_{k}lim")(d1[k]["max"] + off_, d1[k]["min"] - off_)
    else:
        ax = set_equallim(
            ax=ax,
            **kws_set_equallim,
        )
    return ax


def set_grids(ax: plt.Axes, axis: str = None) -> plt.Axes:
    """Show grids based on the shape (aspect ratio) of the plot.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        axis (str, optional): axis name. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    w, h = ax.figure.get_size_inches()
    if w / h >= 1.1 or axis == "y" or axis == "both":
        ax.set_axisbelow(True)
        ax.yaxis.grid(color="gray", linestyle="dashed")
    if w / h <= 0.9 or axis == "x" or axis == "both":
        ax.set_axisbelow(True)
        ax.xaxis.grid(color="gray", linestyle="dashed")
    return ax


# legends
def format_legends(
    ax: plt.Axes,
    **kws_legend,
) -> plt.Axes:
    """Format legend text.

    Args:
        ax (plt.Axes): `plt.Axes` object.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    handles, labels = ax.get_legend_handles_labels()
    labels = [str(s).capitalize() for s in labels]
    kws_legend = {
        **dict(
            borderpad=0,
            handletextpad=0.1,
            labelspacing=0.01,
            columnspacing=0.1,
            handlelength=0.8,
        ),
        **kws_legend,
    }
    return ax.legend(
        handles=handles,
        labels=labels,
        **{
            ## inferred
            **dict(
                title=ax.get_legend().get_title().get_text().capitalize()
                    if ax.get_legend() is not None
                    else None,
            ),
            ## custom
            **kws_legend,
        }
    )


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
    labels = [str(s) for s in labels]
    if len(set(labels) - set(replaces.keys())) == 0:
        labels = [replaces[s] for s in labels]
    else:
        labels = [replace_many(s, replaces) for s in labels]
    return ax.legend(
        handles=handles,
        labels=labels,
        **{
            **dict(
                bbox_to_anchor=ax.get_legend().get_bbox_to_anchor()._bbox.bounds,
                title=ax.get_legend().get_title().get_text(),
            ),
            **kws_legend,
        },
    )


def append_legends(ax: plt.Axes, labels: list, handles: list, **kws) -> plt.Axes:
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
    ax.legend(handles=h1 + handles, labels=l1 + labels, **kws)
    return ax


def sort_legends(ax: plt.Axes, sort_order: list = None, **kws) -> plt.Axes:
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
        handles, labels = zip(*sorted(zip(handles, labels), key=lambda t: t[1]))
    else:
        if all([isinstance(i, str) for i in sort_order]):
            sort_order = [labels.index(s) for s in sort_order]
        if not all([isinstance(i, int) for i in sort_order]):
            raise ValueError("sort_order should contain all integers")
        handles, labels = (
            [handles[idx] for idx in sort_order],
            [labels[idx] for idx in sort_order],
        )
        # print(handles,labels)
    return ax.legend(handles, labels, **kws)


def drop_duplicate_legend(ax, **kws):
    return sort_legends(ax=ax, sort_order=None, **kws)


def reset_legend_colors(ax):
    """Reset legend colors.

    Args:
        ax (plt.Axes): `plt.Axes` object.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    #         lh._legmarker.set_alpha(1)
    return ax


def set_legends_merged(
    axs,
    **kws_legend,
):
    """Reset legend colors.

    Args:
        axs (list): list of `plt.Axes` objects.

    Returns:
        plt.Axes: first `plt.Axes` object in the list.
    """
    df_ = pd.concat(
        [pd.DataFrame(ax.get_legend_handles_labels()[::-1]).T for ax in axs], axis=0
    )
    df_["fc"] = df_[1].apply(lambda x: x.get_fc())
    df_ = df_.log.drop_duplicates(subset=[0, "fc"])
    if df_[0].duplicated().any():
        logging.error("duplicate legend labels")
    return axs[1].legend(
        handles=df_[1].tolist(),
        labels=df_[0].tolist(),
        **kws_legend,
        # bbox_to_anchor=[-0.2,0],loc=2,
        # frameon=False,
    )  # .get_frame().set_edgecolor((0.95,0.95,0.95))


def set_legend_custom(
    ax: plt.Axes,
    legend2param: dict,
    param: str = "color",
    lw: float = 1,
    marker: str = "o",
    markerfacecolor: bool = True,
    size: float = 10,
    color: str = "k",
    linestyle: str = "",
    title_ha: str = "center",
    # frameon: bool=False,
    **kws,
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

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color=color if param != "color" else legend2param[k],
            markeredgecolor=(color if param != "color" else legend2param[k]),
            markerfacecolor=(color if param != "color" else legend2param[k])
            if markerfacecolor is not None
            else "none",
            markersize=(size if param != "size" else legend2param[k]),
            label=k,
            lw=(lw if param != "lw" else legend2param[k]),
            linestyle=legend2param[k]
            if param == "linestyle"
            else linestyle
            if param != "lw"
            else "-",
        )
        for k in legend2param
    ]
    o1 = ax.legend(
        handles=legend_elements,
        # frameon=frameon,
        **kws,
    )
    # o1._legend_box.align=title_ha
    # o1.get_frame().set_edgecolor((0.95,0.95,0.95))
    return ax


# line round
def get_line_cap_length(ax: plt.Axes, linewidth: float) -> plt.Axes:
    """Get the line cap length.

    Args:
        ax (plt.Axes): `plt.Axes` object
        linewidth (float): width of the line.

    Returns:
        plt.Axes: `plt.Axes` object
    """
    radius = linewidth / 2
    ppd = 72.0 / ax.figure.dpi  # points per dot
    trans = ax.transData.inverted().transform
    x_radius = (trans((radius / ppd, 0)) - trans((0, 0)))[0]
    y_radius = (trans((0, radius / ppd)) - trans((0, 0)))[1]
    return x_radius, y_radius


# colorbar
def set_colorbar(
    axc: plt.Axes,
    orientation: str = "horizontal",
    bounds=[0.2,-0.3,0.6,0.05], #[x0, y0, width, height],
    x0: float = None, y0: float = None, width: float = None, height: float = None,
    label: str=None,
    fig: object =None,
    ax: plt.Axes =None,
    # bbox_to_anchor: tuple = (0.05, 0.5, 1, 0.45),
    # width='50%',
    # height='5%',
    kws_ins={},
    **kws
):
    """Set colorbar.

    Args:
        fig (object): figure object.
        ax (plt.Axes): `plt.Axes` object.
        axc (plt.Axes): `plt.Axes` object for the colorbar.
        label (str): label
        bbox_to_anchor (tuple, optional): location. Defaults to (0.05, 0.5, 1, 0.45).
        orientation (str, optional): orientation. Defaults to "vertical".

    Returns:
        figure object.
    """

    # if orientation == "vertical":
    #     width, height = "5%", "50%"
    # else:
    # if orientation == "vertical":
    #     width, height = height, width
        
    # from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # axins = inset_axes(
    #     ax,
    #     width=width,  # width = 5% of parent_bbox width
    #     height=height,  # height : 50%
    #     loc=2,
    #     bbox_to_anchor=bbox_to_anchor,
    #     bbox_transform=ax.transAxes,
    #     borderpad=0,
    # )
    for i,n in enumerate([x0, y0, width, height]):
        if n is not None:
            bounds[i]=n
    if ax is None:
        ax=plt.gca()
    if fig is None:
        fig=ax.get_figure()
        
    axins=ax.inset_axes(
        bounds,
        **kws_ins
    )
    
    cb=fig.colorbar(
        axc,
        cax=axins,
        label=label,
        orientation=orientation,
        pad=0,
        **kws,
    )
    cb.outline.set_visible(False)
    return cb


def set_colorbar_label(ax: plt.Axes, label: str) -> plt.Axes:
    """Find colorbar and set label for it.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        label (str): label.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    for a in ax.figure.get_axes()[::-1]:
        if a.properties()["label"] == "<colorbar>":
            if hasattr(a, "set_ylabel"):
                a.set_ylabel(label)
                break
    return ax


## meta
def format_ax(
    ax=None,
    kws_fmt_ticklabels={},
    kws_fmt_labels={},
    kws_legend={},
    rotate_ylabel=False,
    textwrap_width=None,
):
    if ax is None:
        ax = plt.gca()

    format_ticklabels(
        ax,
        **kws_fmt_ticklabels,
    )
    format_labels(
        ax,
        textwrap_width=textwrap_width,
        rotate_ylabel=rotate_ylabel,
        **kws_fmt_labels,
        # fmt='cap1',
        # title_fontsize=15,
        # rename_labels=None,
        # test=False,
    )
    format_legends(
        ax=ax,
        **kws_legend,
    )
    try:
        import seaborn as sns

        sns.despine(trim=False)
    except ImportError:
        logging.warning(
            "Optional dependency seaborn missing, install by running: pip install roux[viz]"
        )
    return ax

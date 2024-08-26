"""Theming."""

import matplotlib.pyplot as plt


def set_theme(
    font: str = "Myriad Pro",
    # settings
    fontsize: int = 12,
    pad: int = 2,
    palette: list = [
        "#50AADC",  # blue
        "#D3DDDC",  # gray
        "#F1D929",  # yellow
        "#f55f5f",  # red
        "#046C9A",  # blue
        "#00A08A",
        "#F2AD00",
        "#F98400",
        "#5BBCD6",
        "#ECCBAE",
        "#D69C4E",
        "#ABDDDE",
        "#000000",
    ],
):
    """
    Set the theme.

    Parameters:
        font (str): font name.
        fontsize (int): font size.
        pad (int): padding.

    TODOs:
        Addition of `palette` options.
    """
    plt.set_loglevel("error")
    plt.style.use("ggplot")
    # https://matplotlib.org/stable/users/explain/customizing.html#the-default-matplotlibrc-file

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["font.family"] = font
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["legend.frameon"] = False
    
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler(
        "color",
        palette,
    )
    # plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
    # ticks
    # plt.rcParams['xtick.color']=[0.95,0.95,0.95]
    plt.rc(
        "axes",
        grid=False,
        axisbelow=True,
        unicode_minus=False,
        labelsize=fontsize,
        labelcolor="k",
        labelpad=pad,
        titlesize=fontsize,
        facecolor="w",
        edgecolor="k",
        linewidth=0.5,
    )
    # plt.rcParams['axes.formatter.use_mathtext'] = True
    plt.rcParams["axes.formatter.limits"] = -3, 3
    plt.rcParams["axes.formatter.min_exponent"] = 3

    plt.rcParams["xtick.major.size"] = pad
    plt.rcParams["ytick.major.size"] = pad
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.major.pad"] = pad
    plt.rcParams["ytick.major.pad"] = pad
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    ## tick labels
    plt.rcParams["xtick.labelcolor"] = "k"
    plt.rcParams["ytick.labelcolor"] = "k"
    # legend
    ## Dimensions as fraction of font size:
    plt.rcParams["legend.borderpad"] = 0  # border whitespace
    plt.rcParams["legend.labelspacing"] = (
        0.1  # the vertical space between the legend entries
    )
    # legend.handlelength:  2.0  # the length of the legend lines
    # legend.handleheight:  0.7  # the height of the legend handle
    plt.rcParams["legend.handletextpad"] = (
        0.1  # the space between the legend line and legend text
    )
    # legend.borderaxespad: 0.5  # the border between the axes and legend edge
    # legend.columnspacing: 2.0  # column separation

    # scale
    plt.rc("figure", figsize=(3, 3))
    plt.rc("figure.subplot", wspace=0.3, hspace=0.3)
    # sns.set_context('notebook') # paper < notebook < talk < poster


set_theme()

"""Theming."""

import matplotlib.pyplot as plt
from cycler import cycler

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
    figsize=(3,3),
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

    # g: Consolidated settings into a single update call for brevity
    plt.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": font,
        "font.size": fontsize,
        "legend.frameon": False,
        "axes.prop_cycle": cycler("color", palette),
        # plt.rc('grid', lw=0.2,linestyle="-", color=[0.98,0.98,0.98])
        # ticks
        # plt.rcParams['xtick.color']=[0.95,0.95,0.95]
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.unicode_minus": False,
        "axes.labelsize": fontsize,
        "axes.labelcolor": "k",
        "axes.labelpad": pad,
        "axes.titlesize": fontsize,
        "axes.facecolor": "w",
        "axes.edgecolor": "k",
        "axes.linewidth": 0.5,
        # plt.rcParams['axes.formatter.use_mathtext'] = True
        "axes.formatter.limits": (-3, 3),
        "axes.formatter.min_exponent": 3,
        "xtick.major.size": pad,
        "ytick.major.size": pad,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.pad": pad,
        "ytick.major.pad": pad,
        "xtick.direction": "in",
        "ytick.direction": "in",
        ## tick labels
        "xtick.labelcolor": "k",
        "ytick.labelcolor": "k",
        # legend
        ## Dimensions as fraction of font size:
        "legend.borderpad": 0,  # border whitespace
        "legend.labelspacing": 0.1,  # the vertical space between the legend entries
        # legend.handlelength:  2.0  # the length of the legend lines
        # legend.handleheight:  0.7  # the height of the legend handle
        "legend.handletextpad": 0.1,  # the space between the legend line and legend text
        # legend.borderaxespad: 0.5  # the border between the axes and legend edge
        # legend.columnspacing: 2.0  # column separation
        # scale
        "figure.figsize": figsize,
        "figure.subplot.wspace": 0.3,
        "figure.subplot.hspace": 0.3,
        "figure.titlesize": fontsize,
        # g: Set global font family which figure.supxlabel will inherit
        # 'font.family': 'sans-serif',
        # 'font.sans-serif': ['Noto Sans'], # g: Using Noto Sans as per environment defaults
        "figure.labelsize": fontsize,     # g: Controls size of supxlabel/supylabel
        # 'figure.labelweight': 'bold'      # g: Controls weight of supxlabel/supylabel
    })
    # plt.rcParams['figure.title_fontfamily'] = font
    # sns.set_context('notebook') # paper < notebook < talk < poster

set_theme()
"""For setting up colors."""

import matplotlib.pyplot as plt
from matplotlib import colors, cm
import numpy as np
import pandas as pd

# colors
from matplotlib.colors import to_hex
from matplotlib.colors import ColorConverter

to_rgb = ColorConverter.to_rgb


def rgbfloat2int(rgb_float):
    return [int(round(i * 255)) for i in rgb_float]


# aliases
rgb2hex = to_hex
hex2rgb = to_rgb


# colors
def get_colors_default() -> list:
    """get default colors.

    Returns:
        list: colors.
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def get_ncolors(
    n: int,
    cmap: str = "Spectral",
    ceil: bool = False,
    test: bool = False,
    N: int = 20,
    out: str = "hex",
    **kws_get_cmap_section,
) -> list:
    """Get colors.

    Args:
        n (int): number of colors to get.
        cmap (str, optional): colormap. Defaults to 'Spectral'.
        ceil (bool, optional): ceil. Defaults to False.
        test (bool, optional): test mode. Defaults to False.
        N (int, optional): number of colors in the colormap. Defaults to 20.
        out (str, optional): output. Defaults to 'hex'.

    Returns:
        list: colors.
    """
    if cmap is None:
        cmap="Spectral"
    if isinstance(cmap, str):
        cmap = get_cmap_section(cmap, **kws_get_cmap_section)
    elif isinstance(cmap, list):
        cmap = make_cmap(cmap, N=N)
    #         cmap = cm.get_cmap(cmap)
    if test:
        print(np.arange(1 if ceil else 0, n + (1 if ceil else 0), 1))
        print(np.arange(1 if ceil else 0, n + (1 if ceil else 0), 1) / n)
    colors = [
        cmap(i) for i in np.arange(1 if ceil else 0, n + (1 if ceil else 0), 1) / n
    ]
    assert n == len(colors)
    if out == "hex":
        colors = [rgb2hex(c) for c in colors]
    return colors


def get_val2color(
    ds: pd.Series, vmin: float = None, vmax: float = None, cmap: str = "Reds"
) -> dict:
    """Get color for a value.

    Args:
        ds (pd.Series): values.
        vmin (float, optional): minimum value. Defaults to None.
        vmax (float, optional): maximum value. Defaults to None.
        cmap (str, optional): colormap. Defaults to 'Reds'.

    Returns:
        dict: output.
    """
    if vmin is None:
        vmin = min(ds)
    if vmax is None:
        vmax = max(ds)
    colors = [
        (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap)(
            (i - vmin) / (vmax - vmin)
        )
        for i in ds
    ]
    legend2color = {
        i: (plt.get_cmap(cmap) if isinstance(cmap, str) else cmap)(
            (i - vmin) / (vmax - vmin)
        )
        for i in [vmin, np.mean([vmin, vmax]), vmax]
    }
    return dict(zip(ds, colors)), legend2color


#    columns=['value','c']


def saturate_color(color, alpha: float) -> object:
    """Saturate a color.

    Args:
        color (_type_):
        alpha (float): alpha level.

    Returns:
        object: output.

    References:
        https://stackoverflow.com/a/60562502/3521099
    """
    import colorsys
    from roux.stat.transform import rescale

    alpha = rescale(alpha, [0, 2], [1.6, 0.4])
    if isinstance(color, str):
        color = hex2rgb(color)
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*color)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * alpha), s=s)


def mix_colors(d: dict) -> str:
    """Mix colors.

    Args:
        d (dict): colors to alpha map.

    Returns:
        str: hex color.

    References:
        https://stackoverflow.com/a/61488997/3521099
    """
    if isinstance(d, list):
        d = {k: 1.0 for k in d}
    d = {k.replace("#", ""): d[k] for k in d}
    d_items = sorted(d.items())
    tot_weight = sum(d.values())
    red = int(sum([int(k[:2], 16) * v for k, v in d_items]) / tot_weight)
    green = int(sum([int(k[2:4], 16) * v for k, v in d_items]) / tot_weight)
    blue = int(sum([int(k[4:6], 16) * v for k, v in d_items]) / tot_weight)
    zpad = lambda x: x if len(x) == 2 else "0" + x  # noqa
    c = zpad(hex(red)[2:]) + zpad(hex(green)[2:]) + zpad(hex(blue)[2:])
    return f"#{c}"


# colormaps


def make_cmap(cs: list, N: int = 20, **kws):
    """Create a colormap.

    Args:
        cs (list): colors
        N (int, optional): resolution i.e. number of colors. Defaults to 20.

    Returns:
        cmap.
    """
    return colors.LinearSegmentedColormap.from_list("custom", colors=cs, N=N, **kws)


def get_cmap_section(
    cmap, vmin: float = 0.0, vmax: float = 1.0, n: int = 100
) -> object:
    """Get section of a colormap.

    Args:
        cmap (object| str): colormap.
        vmin (float, optional): minimum value. Defaults to 0.0.
        vmax (float, optional): maximum value. Defaults to 1.0.
        n (int, optional): resolution i.e. number of colors. Defaults to 100.

    Returns:
        object: cmap.
    """
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    new_cmap = colors.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=vmin, b=vmax),
        cmap(np.linspace(vmin, vmax, n)),
    )
    return new_cmap


def append_cmap(
    cmap: str = "Reds",
    color: str = "#D3DDDC",
    cmap_min: float = 0.2,
    cmap_max: float = 0.8,
    ncolors: int = 100,
    ncolors_min: int = 1,
    ncolors_max: int = 0,
):
    """Append a color to colormap.

    Args:
        cmap (str, optional): colormap. Defaults to 'Reds'.
        color (str, optional): color. Defaults to '#D3DDDC'.
        cmap_min (float, optional): cmap_min. Defaults to 0.2.
        cmap_max (float, optional): cmap_max. Defaults to 0.8.
        ncolors (int, optional): number of colors. Defaults to 100.
        ncolors_min (int, optional): number of colors minimum. Defaults to 1.
        ncolors_max (int, optional): number of colors maximum. Defaults to 0.

    Returns:
        cmap.

    References:
        https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html
    """
    viridis = cm.get_cmap(cmap, ncolors)
    newcolors = viridis(np.linspace(cmap_min, cmap_max, ncolors))
    #     pink = np.array([248/256, 24/256, 148/256, 1])
    pink = np.append(np.array(hex2rgb(color)) / 256, 1)
    if ncolors_min != 0:
        newcolors[:ncolors_min, :] = pink
    elif ncolors_max != 0:
        newcolors[ncolors_max:, :] = pink
    newcmp = colors.ListedColormap(newcolors)
    return newcmp

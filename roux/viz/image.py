import matplotlib.pyplot as plt
from os.path import splitext

def plot_image(
    imp: str,
    ax: plt.Axes=None,
    force=False,
    margin=0,
    axes=False,
    test=False,
    **kwarg
    ) -> plt.Axes:
    """Plot image e.g. schematic.

    Args:
        imp (str): path of the image.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        force (bool, optional): overwrite output. Defaults to False.
        margin (int, optional): margins. Defaults to 0.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.

    :param kwarg: cairosvg: {'dpi':500,'scale':2}; imagemagick: {'trim':False,'alpha':False}

    """
    if splitext(imp)[1]=='.png':
        pngp=imp
    else:
#         if splitext(imp)[1]=='.svg' or force:
#             from roux.lib.figs.convert import svg2png
#             pngp=svg2png(imp,force=force,**kwarg)
#         else:
        from roux.viz.io import to_raster
        pngp=to_raster(imp,force=force,**kwarg)
    ax=plt.subplot() if ax is None else ax
    im=plt.imread(pngp)
    ax.imshow(im,interpolation='catrom')
    ax.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
    ax.margins(margin)
    if not axes:
        ax.axis('off')
    return ax 
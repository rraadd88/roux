"""For setting up figures."""
import numpy as np

import matplotlib.pyplot as plt
import logging

def labelplots(
    axes: list=None,
    fig=None,
    labels: list=None,
    xoff: float=0,
    yoff: float=0,
    auto: bool=False,
    xoffs: dict={},
    yoffs: dict={},
    va:str = 'center',
    ha:str = 'left',
    verbose: bool=True,
    test: bool=False,
    # transform='ax',
    **kws_text,
    ):
    """Label (sub)plots.

    Args:
        fig : `plt.figure` object. 
        axes (_type_): list of `plt.Axes` objects.
        xoff (int, optional): x offset. Defaults to 0.
        yoff (int, optional): y offset. Defaults to 0.
        params_alignment (dict, optional): alignment parameters. Defaults to {}.
        params_text (dict, optional): parameters provided to `plt.text`. Defaults to {'size':20,'va':'bottom', 'ha':'right' }.
        test (bool, optional): test mode. Defaults to False.
        
    Todos:
        1. Get the x coordinate of the ylabel.
    """
    if axes is None:
        axes=fig.axes
    if labels is None:
        import string
        labels=string.ascii_uppercase[:len(axes)]
    else:
        assert len(axes)==len(labels) 
    label2ax=dict(zip(labels,axes))
    axi2xy={}
    for axi,label in enumerate(label2ax.keys()):
        ax=label2ax[label]
    if auto:
        fig.draw_without_rendering() ## get positions after drawing, applicable to ylabel's x.
        for label,ax in label2ax.items():
            x,y=ax.yaxis.get_label().get_position()[0], (ax.transAxes.transform(ax.title.get_position())[1] if hasattr(ax,'title') else 1.0)
            if verbose: logging.info(f"x,y={x},{y}")
            ax.annotate(label,
                        xy=(x, y),
                        xycoords='figure pixels'
                       )

    else:
        ## manual
        # if len(xoffs)!=0 or len(xoffs)!=0:
        for label,ax in label2ax.items():
            x,y=0, (ax.title.get_position()[1] if hasattr(ax,'title') else 1.0)
            ax.text(s=label,
                    x=x+xoff+(xoffs[label] if label in xoffs else 0),
                    y=y+0.045+yoff+(yoffs[label] if label in yoffs else 0),
                    **kws_text,
                    transform=ax.transAxes,
                    )
            
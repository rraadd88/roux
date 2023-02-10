"""For setting up figures."""
import numpy as np

import matplotlib.pyplot as plt
import logging

def labelplots(
    fig,
    axes: list=None,
    labels: list=None,
    xoff: float=0,
    yoff: float=0,
    custom_positions: dict={},
    size:float = None,
    va:str = 'center',
    ha:str = 'right',
    test: bool=False,
    transform='ax',
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
    
    # import matplotlib.transforms as transforms
    # trans = transforms.blended_transform_factory(
    #             # fig.transFigure, 
    #             ax.transAxes,
    #             ax.transAxes,
    # )
    if transform=='ax':
        for label,ax in label2ax.items():
            ## x position
            logging.info(f"ax.yaxis.get_label().get_position()={ax.yaxis.get_label().get_position()}")
            t=ax.yaxis.get_label()
            
            logging.info(f"t.get_position()={t.get_position()}")
            ## y position
            x,y=np.array(ax.get_position())[0][0], (ax.title.get_position()[1] if hasattr(ax,'title') else 1.0)+yoff
            if test: logging.info(f"ax.title.get_position()={ax.title.get_position()}")
            if test: logging.info(f"x,y={x},{y}")
            
            ax.set_title(
                label,
                loc='left',
                # x=x,
                x=-0.175+xoff,
                # x=np.array(ax.get_position())[0][0],
                y= (ax.title.get_position()[1] if hasattr(ax,'title') else 1.0)+yoff,
                ha=ha,
                # va=va,
                # transform=trans,
                )
    elif transform=='figure':
        for axi,label in enumerate(label2ax.keys()):
            axi2xy[axi]=ax.get_position(original=True).xmin+xoff,ax.get_position(original=False).ymax+yoff
            label2ax[label].text(
                *(axi2xy[axi] if not axi in custom_positions else custom_positions[axi]),
                f"{label}",
                size=None,
                ha=ha,
                va=va,
                transform=fig.transFigure,
                **kws_text,
            )    
        

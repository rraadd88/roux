"""For setting up figures."""
import numpy as np

import matplotlib.pyplot as plt
import logging

def get_children(fig):
    """
    Get all the individual objects included in the figure.
    """
    from tqdm import tqdm
    from roux.lib.set import flatten
    ## figure
    l1=[]
    for ax in tqdm(fig.get_children()):
        if isinstance(ax, plt.Subplot):
            l1+=ax.get_children()
        else:
            l1+=[ax]
    l1=flatten(l1)
    ## subfigure
    l2=[]
    for ax in tqdm(l1):
        if isinstance(ax, plt.Subplot):
            l2+=ax.get_children()
        else:
            l2+=[ax]
    return flatten(l2)

def get_child_text(
    search_name,
    all_children=None,
    fig=None,
    ):
    """
    Get text object.
    """
    if all_children is None:
        all_children=get_children(fig=fig)
    child=None
    for c in all_children:
        if isinstance(c, plt.Text) and c.get_text() == search_name:
            child = c
            break
    assert not child is None, (search_name,all_children)
    return child

def align_texts(
    fig,
    texts: list,
    align: str,
    test=False,
    ):
    """
    Align text objects.
    """
    all_children=get_children(fig=fig)
    x_px_set,y_px_set=None,None
    for i,search_name in enumerate(texts):
        text=get_child_text(
            search_name=search_name,
            all_children=all_children,
            )
        extent = text.get_window_extent(renderer=fig.canvas.get_renderer())
        x_px,y_px=np.array(extent)[0][0],np.array(extent)[1][1]
        if test:
            print(x_px,y_px)
        if i==0 and align == 'v':
            y_px_set=y_px
            continue
        elif i==0 and align == 'h':
            x_px_set=x_px
            continue
        else:
            ## set
            if not x_px_set is None:
                x_px=x_px_set
            if not y_px_set is None:
                y_px=y_px_set
            ax_=text.axes
            x_data, y_data = ax_.transData.inverted().transform((x_px,y_px))
            if test:
                print(x_data, y_data)
            # Add text to the subplot using data coordinates
            _ = ax_.text(x_data, y_data, s=text.get_text(),
                         fontsize=text.get_fontsize(),
                         ha='left',#text.get_ha(),
                         va='top',#text.get_va(),
                         color=text.get_color(), ##TODO transfer all the properties
                        )
            if not test:
                text.remove()
                
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
            
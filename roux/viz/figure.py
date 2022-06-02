import matplotlib.pyplot as plt

## set subplots
def get_subplots(
    nrows: int,
    ncols: int,
    total: int=None
    ) -> list:
    """Get subplots.

    Args:
        nrows (int): number of rows.
        ncols (int): number of columns.
        total (int, optional): total subplots. Defaults to None.

    Returns:
        list: list of `plt.Axes` objects.
    """
    idxs=list(itertools.product(range(nrows),range(ncols)))
    if not total is None:
        idxs=idxs[:total]
    print(idxs)
    return [plt.subplot2grid([nrows,ncols],idx,1,1) for idx in idxs]

def labelplots(
    fig,
    axes: list=None,
    labels: list=None,
    xoff: float=0,
    yoff: float=0,
    params_alignment: dict={},
    size:float = 20,
    va:str = 'bottom',
    ha:str = 'right',
    test: bool=False,
    **kws,
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
        axi2xy[axi]=ax.get_position(original=True).xmin+xoff,ax.get_position(original=False).ymax+yoff
    for pair in params_alignment:
        axi2xy[pair[1]]=[axi2xy[pair[0 if 'x' in params_alignment[pair] else 1]][0],
                         axi2xy[pair[0 if 'y' in params_alignment[pair] else 1]][1]]
    for axi,label in enumerate(label2ax.keys()):
        label2ax[label].text(*axi2xy[axi],f"{label}",
                             size=size,
                             ha=ha,
                             va=va,
                             transform=fig.transFigure,
                             **kws)    


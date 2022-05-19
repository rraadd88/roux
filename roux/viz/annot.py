import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
from roux.lib.str import *

# redirects
from roux.stat.io import perc_label,pval2annot

# labels 
def set_label(
    s: str,
    ax: plt.Axes,
    x: float= 0,
    y: float= 0,
    ha: str='left',
    va: str='top',
    loc=None,
    off_loc=0.01,
    title: bool=False,
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
        ax.set_title(s,**kws)
    elif not loc is None:
        if loc==1 or loc=='upper right':
            x=1-off_loc
            y=1-off_loc
            ha='right'
            va='top'
        elif loc==2 or loc=='upper left':
            x=0+off_loc
            y=1-off_loc
            ha='left'
            va='top'
        elif loc==3 or loc=='lower left':
            x=0+off_loc
            y=0+off_loc
            ha='left'
            va='bottom'            
        elif loc==4 or loc=='lower right':
            x=1-off_loc
            y=0+off_loc
            ha='right'
            va='bottom'            
        else:
            raise ValueError(loc)
    ax.text(s=s,transform=ax.transAxes,
            x=x,y=y,ha=ha,va=va,
            **kws)
    return ax

def annot_side(
    ax: plt.Axes,
    df1: pd.DataFrame,
    colx: str,
    coly: str,
    cols: str=None,
    hue: str=None,
    loc: str='right',
    scatter=False,
    lines=True,
    text=True,
    invert_xaxis: bool=False, 
    offx3: float=0.15,
    offymin: float=0.1,
    offymax: float=0.9,
    break_pt: int=25,
    length_axhline: float=3,
    va: str='bottom',
    zorder: int=1,
    color: str='gray',
    kws_line: dict={},
    kws_scatter: dict={'zorder':2,'alpha':0.75,'marker':'|','s':100},
    **kws_text,
    ) -> plt.Axes:
    """Annot elements of the plots on the of the side plot.

    Args:
        df1 (pd.DataFrame): input data
        colx (str): column with x values.
        coly (str): column with y values.
        cols (str): column with labels.
        hue (str): column with colors of the labels.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        loc (str, optional): location. Defaults to 'right'.
        invert_xaxis (bool, optional): invert xaxis. Defaults to False.
        offx3 (float, optional): x-offset for bend position of the arrow. Defaults to 0.15.
        offymin (float, optional): x-offset minimum. Defaults to 0.1.
        offymax (float, optional): x-offset maximum. Defaults to 0.9.
        break_pt (int, optional): break point of the labels. Defaults to 25.
        length_axhline (float, optional): length of the horizontal line i.e. the "underline". Defaults to 3.
        zorder (int, optional): z-order. Defaults to 1.
        color (str, optional): color of the line. Defaults to 'gray'.
        kws_line (dict, optional): parameters for formatting the line. Defaults to {}.

    Keyword Args:
        kws: parameters provided to the `ax.text` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if len(df1)==0: 
        logging.warning("annot_side: no data found")
        return
    if isinstance(colx,float):
        df1['colx']=colx
        colx='colx'
    if isinstance(coly,float):
        df1['coly']=coly
        coly='coly'
    # assert not 'y' in df1, 'table should not contain a column named `y`'
    df1=df1.sort_values(coly if loc!='top' else colx,ascending=True)
    from roux.viz.ax_ import get_axlims
    d1=get_axlims(ax)
    # if loc=='top', annotations x is y and y is x
    df1['y']=np.linspace(d1['y' if loc!='top' else 'x']['min']+((d1['y' if loc!='top' else 'x']['len'])*offymin),
                        d1['y' if loc!='top' else 'x']['max']*offymax,
                        len(df1))
    x2=d1['x'if loc!='top' else 'y']['min' if not invert_xaxis else 'max'] if loc=='left' else d1['x'if loc!='top' else 'y']['max' if not invert_xaxis else 'min']
    x3=d1['x']['min']-(d1['x']['len']*offx3) if (loc=='left' and not invert_xaxis) else \
        d1['y']['max']+(d1['y']['len']*offx3) if loc=='top' else \
        d1['x']['max']+(d1['x']['len']*offx3)
    # print(x2,x3)
    # line#1
    # print(df1.loc[:,[colx,coly,'y']].iloc[0,:])
    if lines:
        df1.apply(lambda x: ax.plot([x[colx],x2] if loc!='top' else [x[colx], x['y']],
                                   [x[coly],x['y']] if loc!='top' else [x[coly], x2],
                                   color=color,lw=1,**kws_line,
                                   zorder=zorder,
                                   ),axis=1)
    if scatter:
        df1.plot.scatter(x=colx,y=coly,ax=ax,
                        **kws_scatter)
        
    ## text
    if text:
        df1.apply(lambda x: ax.text(x3 if loc!='top' else x['y'],
                                    x['y'] if loc!='top' else x3,
                                    linebreaker(x[cols],break_pt=break_pt,),
                                    ha='right' if loc=='left' else 'center' if loc=='top' else 'left',
                                    va=va,
                                    color=x[hue] if not hue is None else 'k',
                                    # **{k:v for k,v in kws_text.items() if not (k==color and not hue is None)},
                                    rotation=0  if loc!='top' else 90,
                                    **kws_text,
                                  zorder=2),axis=1)
    # line #2
    if lines:
        if loc!='top':
            df1.apply(lambda x:ax.axhline(y = x['y'], 
                                         xmin=0 if loc=='left' else 1,
                                         xmax=0-(length_axhline-1)-offx3 if loc=='left' else length_axhline+offx3,
                                                 clip_on = False,color=color,lw=1,
                                        ),axis=1)
        else:
            df1.apply(lambda x:ax.axvline(x = x['y'], 
                                         ymin=0 if loc=='left' else 1,
                                         ymax=0-(length_axhline-1)-offx3 if loc=='left' else length_axhline+offx3,
                                                 clip_on = False,color=color,lw=1,
                                        ),axis=1)
    if loc=='left':
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    ax.set(xlim=[d1['x']['min'],d1['x']['max']],
           ylim=[d1['y']['min'],d1['y']['max']],
            )
    return ax

def annot_corners(
    ax : plt.Axes,
    df1 : pd.DataFrame,
    colx : str,
    coly : str,
    coltext : str,
    off : float=0.1,
    **kws,
    ) -> plt.Axes:
    """
    Annotate points above and below the diagonal.
    """
    df1['diff']=df1[coly]-df1[colx]
    # above diagonal
    df1['loc']=df1['diff'].apply(lambda x: 'above' if x>0 else 'below')
    from roux.viz.ax_ import get_axlims
    axlims=get_axlims(ax=ax)
    for loc,df1_ in df1.groupby('loc'):
        df1_=df1_.sort_values([colx,coly])        
        offx=axlims['x']['len']*off
        offy=axlims['y']['len']*off
        # upper
        df1_['x text' if loc=='above' else 'y text']=np.linspace(axlims['x' if loc=='above' else 'y']['min']+offx,
                                (axlims['x' if loc=='above' else 'y']['min']+axlims['x' if loc=='above' else 'y']['len']*0.5)-offx,
                                len(df1_))
        df1_['y text' if loc=='above' else 'x text']=np.linspace(
                                (axlims['y' if loc=='above' else 'x']['min']+axlims['y' if loc=='above' else 'x']['len']*0.5)+offy,
                                axlims['y' if loc=='above' else 'x']['max']-offy,
                                len(df1_))
        df1_.apply(lambda x: ax.annotate(s=x[coltext], 
                                        xytext=(x['x text'],x['y text']), 
                                        xy=(x[colx],x[coly]), 
                                        va='center',
                                        ha='center',
                                         **kws,
                                        ),axis=1)
    return ax

# variance
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters:
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    
    References
    ----------
    https://matplotlib.org/3.5.0/gallery/statistics/confidence_ellipse.html
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

## heatmaps
def show_box(
    ax: plt.Axes,
    xy: tuple,
    width: float,
    height: float,
    fill: str=None, 
    alpha: float=1,
    lw: float=1.1,
    ec: str='k',
    clip_on: bool=False,
    scale_width: float=1,
    scale_height: float=1,
    xoff: float=0,
    yoff: float=0,
    **kws,
    ) -> plt.Axes:
    """Highlight sections of a plot e.g. heatmap by drawing boxes.

    Args:
        xy (tuple): position of left, bottom corner of the box.
        width (float): width.
        height (float): height.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        fill (str, optional): fill the box with color. Defaults to None.
        alpha (float, optional): alpha of color. Defaults to 1.
        lw (float, optional): line width. Defaults to 1.1.
        ec (str, optional): edge color. Defaults to 'k'.
        clip_on (bool, optional): clip the boxes by the axis limit. Defaults to False.
        scale_width (float, optional): scale width. Defaults to 1.
        scale_height (float, optional): scale height. Defaults to 1.
        xoff (float, optional): x-offset. Defaults to 0.
        yoff (float, optional): y-offset. Defaults to 0.

    Keyword Args:
        kws: parameters provided to the `Rectangle` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from matplotlib.patches import Rectangle
    return ax.add_patch(Rectangle(
                            xy=[xy[0]+xoff,xy[1]+yoff], 
                            width=width*scale_width, height=height*scale_height, 
                            fill=fill, alpha=alpha,
                            lw=lw,ec=ec,clip_on=clip_on,
                            **kws,
                            ))

def annot_confusion_matrix(
    df_: pd.DataFrame,
    ax: plt.Axes=None,
    off: float=0.5
    ) -> plt.Axes:
    """Annotate a confusion matrix.

    Args:
        df_ (pd.DataFrame): input data.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        off (float, optional): offset. Defaults to 0.5.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.stat.binary import get_stats_confusion_matrix
    df1=get_stats_confusion_matrix(df_)
    df2=pd.DataFrame({
                    'TP': [0,0],
                    'TN': [1,1],
                    'FP': [0,1],
                    'FN': [1,0],
                    'TPR':[0,2],
                    'TNR': [1,2],
                    'PPV': [2,0],
                    'NPV': [2,1],
                    'FPR': [1,3],
                    'FNR': [0,3],
                    'FDR': [3,0],
                    'ACC': [2,2],
                    },
                     index=['x','y']).T
    df2.index.name='variable'
    df2=df2.reset_index()
    df3=df1.merge(df2,
              on='variable',
              how='inner',
              validate="1:1")
    
    _=df3.loc[(df3['variable'].isin(['TP','TN','FP','FN'])),:].apply(lambda x: ax.text(x['x']+off,
                                                                                       x['y']+(off*2),
    #                               f"{x['variable']}\n{x['value']:.0f}",
                                  x['variable'],
    #                               f"({x['T|F']+x['P|N']})",
                                ha='center',va='bottom',
                               ),axis=1)
    _=df3.loc[~(df3['variable'].isin(['TP','TN','FP','FN'])),:].apply(lambda x: ax.text(x['x']+off,
                                                                                        x['y']+(off*2),
                                  f"{x['variable']}\n{x['value']:.2f}",
    #                               f"({x['T|F']+x['P|N']})",
                                ha='center',va='bottom',
                               ),axis=1)
    return ax


def get_logo_ax(
    ax: plt.Axes,
    size: float=0.5,
    bbox_to_anchor: list=None,
    loc: str=1,
    axes_kwargs: dict={'zorder':-1},
    ) -> plt.Axes:
    """Get `plt.Axes` for placing the logo.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        size (float, optional): size of the subplot. Defaults to 0.5.
        bbox_to_anchor (list, optional): location. Defaults to None.
        loc (str, optional): location. Defaults to 1.
        axes_kwargs (_type_, optional): parameters provided to `inset_axes`. Defaults to {'zorder':-1}.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    width, height,aspect_ratio=get_subplot_dimentions(ax)
    axins = inset_axes(ax, 
                       width=size, height=size,
                       bbox_to_anchor=[1,1,0,size/(height)] if bbox_to_anchor is None else bbox_to_anchor,
                       bbox_transform=ax.transAxes, 
                       loc=loc, 
                       borderpad=0,
                      axes_kwargs=axes_kwargs)
    return axins

def set_logo(
    imp: str,
    ax: plt.Axes,
    size: float=0.5,
    bbox_to_anchor: list=None,
    loc: str=1,
    axes_kwargs: dict={'zorder':-1},
    params_imshow: dict={'aspect':'auto','alpha':1,
    #                             'zorder':1,
    'interpolation':'catrom'},
    test: bool=False,
    force: bool=False
    ) -> plt.Axes:
    """Set logo.

    Args:
        imp (str): path to the logo file.
        ax (plt.Axes): `plt.Axes` object.
        size (float, optional): size of the subplot. Defaults to 0.5.
        bbox_to_anchor (list, optional): location. Defaults to None.
        loc (str, optional): location. Defaults to 1.
        axes_kwargs (_type_, optional): parameters provided to `inset_axes`. Defaults to {'zorder':-1}.
        params_imshow (_type_, optional): parameters provided to the `imshow` function. Defaults to {'aspect':'auto','alpha':1, 'interpolation':'catrom'}.
        test (bool, optional): test mode. Defaults to False.
        force (bool, optional): overwrite file. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    from roux.lib.figs.convert import vector2raster
    if isinstance(imp,str):
        if splitext(imp)[1]=='.svg':
            pngp=vector2raster(imp,force=force)
        else:
            pngp=imp
        if not exists(pngp):
            logging.error(f'{pngp} not found')
            return
        im = plt.imread(pngp)
    elif isinstance(imp,np.ndarray):
        im = imp
    else:
        loggin.warning('imp should be path or image')
        return
    axins=get_logo_ax(ax,size=size,bbox_to_anchor=bbox_to_anchor,loc=loc,
             axes_kwargs=axes_kwargs,)
    axins.imshow(im, **params_imshow)
    if not test:
        axins.set(**{'xticks':[],'yticks':[],'xlabel':'','ylabel':''})
        axins.margins(0)    
        axins.axis('off')    
        axins.set_axis_off()
    else:
        print(width, height,aspect_ratio,size/(height*2))
    return axins

## color
def color_ax(
    ax: plt.Axes,
    c: str,
    linewidth: float=None
    ) -> plt.Axes:
    """Color border of `plt.Axes`.

    Args:
        ax (plt.Axes): `plt.Axes` object.
        c (str): color.
        linewidth (float, optional): line width. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    plt.setp(ax.spines.values(), color=c)
    if not linewidth is None:
        plt.setp(ax.spines.values(), linewidth=linewidth)
    plt.setp([ax.get_xticklines(), ax.get_yticklines()], color=c)
    return ax

## stats
def annot_n_legend(
    ax,
    df1: pd.DataFrame,
    colid: str,
    colgroup: str,
    **kws,
    ):
    from roux.viz.ax_ import rename_legends
    replaces={str(k):str(k)+'\n'+f'(n={v})' for k,v in df1.groupby(colgroup)[colid].nunique().items()}
    return rename_legends(ax,
                  replaces=replaces,
                  **kws)
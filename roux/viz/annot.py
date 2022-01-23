import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import logging

from roux.lib.str import *

def add_corner_labels(fig,pos,xoff=0,yoff=0,test=False,kw_text={}):
    import string
    label2pos=dict(zip(string.ascii_uppercase[:len(pos)],pos))
    for label in label2pos:
        pos=label2pos[label]
        t=[(i,j) for i in np.arange(0,1,1/pos[1]) for j in np.arange(0.95,0,-1/float(pos[0]))]

        dpos=pd.DataFrame(t).sort_values(by=1,ascending=False)
        dpos.columns=['x','y']
        dpos.index=dpos.reset_index().index+1
        if test:
            print(dpos.loc[pos[2],'x'],dpos.loc[pos[2],'y'])
        fig.text(dpos.loc[pos[2],'x']+xoff,dpos.loc[pos[2],'y']+yoff,label,va='baseline' ,**kw_text)
        del dpos
    return fig

from roux.lib.set import dropna
def dfannot2color(df,colannot,cmap='Spectral',
                  renamecol=True,
                  test=False,):
    annots=dropna(df[colannot].unique())
    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annots_keys=annots
        annots=[ii for ii, i in enumerate(annots)]
    if test:
        print(annots_keys,annots)

    import matplotlib
    cmap = matplotlib.cm.get_cmap(cmap)
    norm = matplotlib.colors.Normalize(vmin=np.min(annots), vmax=np.max(annots))
    rgbas = [cmap(norm(a)) for a in annots]

    if df.dtypes[colannot]=='O' or df.dtypes[colannot]=='S' or df.dtypes[colannot]=='a':
        annot2color=dict(zip(annots_keys,rgbas))
    else:
        if test:
            print('non string data')
        annot2color=dict(zip(annots,rgbas)) 
    if renamecol:
        colcolor=colannot
    else:
        colcolor=f"{colannot} color"
    if test:
        print(annot2color)
    df[colcolor]=df[colannot].apply(lambda x : annot2color[x] if not pd.isnull(x) else x)
    return df,annot2color

def get_dmetrics(df,metricsby,colx,coly,colhue,xs,hues,alternative,
                test=False):
    from scipy.stats import mannwhitneyu
    if len(hues)==0:
        hues=['a']
        colhue='a'
        df['a']='a'
    dmetrics=pd.DataFrame(columns=xs,
                          index=hues)
    if metricsby=='hues':
        # first hue vs rest    
        for hue in dmetrics.index[1:]:  
            for xi,x in enumerate(dmetrics.columns):
                X=df.loc[((df[colhue]==hues[0]) & (df[colx]==x)),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if test:
                    print(len(X),len(Y))
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                     alternative=alternative if isinstance(alternative,str) else alternative[xi])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    # first x vs rest
    elif metricsby=='xs':
        # first hue vs rest    
        for huei,hue in enumerate(dmetrics.index):
            for x in dmetrics.columns[1:]:
                X=df.loc[((df[colhue]==hue) & (df[colx]==xs[0])),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                    alternative=alternative if isinstance(alternative,str) else alternative[huei])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    # first y vs rest
    elif metricsby=='ys':
        # first hue vs rest    
        for huei,hue in enumerate(dmetrics.index):
            for x in dmetrics.columns[1:]:
                X=df.loc[((df[colhue]==hue) & (df[colx]==xs[0])),coly]
                Y=df.loc[((df[colhue]==hue) & (df[colx]==x)),coly]
                if not (len(X)==0 or len(Y)==0):
                    dmetrics.loc[hue,x]=mannwhitneyu(X,Y,
                    alternative=alternative if isinstance(alternative,str) else alternative[huei])[1]
                else:
                    logging.warning('length of X or Y is 0')
                    dmetrics.loc[hue,x]=np.nan
    if test:
        print(dmetrics)
    return dmetrics
    
def annot_boxplot(ax,dmetrics,xoffwithin=0.85,xoff=1.6,
                  yoff=0,annotby='xs',position='top',
                  test=False):
    """
    :param dmetrics: hue in index, x in columns
    
    #todos
    #x|y off in %
    xmin,xmax=ax.get_xlim()
    (xmax-xmin)+(xmax-xmin)*0.35+xmin
    """
    xlabel=ax.get_xlabel()
    ylabel=ax.get_ylabel()
    if test:
        dmetrics.index.name='index'
        dmetrics.columns.name='columns'
        dm=dmetrics.melt()
        dm['value']=1
        ax=sns.boxplot(data=dm,x='columns',y='value')
    for huei,hue in enumerate(dmetrics.index):  
        for xi,x in enumerate(dmetrics.columns):
            if not pd.isnull(dmetrics.loc[hue,x]):
                xco=xi+(huei*xoffwithin/len(dmetrics.index)+(xoff/len(dmetrics.index)))
                yco=ax.get_ylim()[1]+yoff
                if annotby=='ys':
                    xco,yco=yco,xco
                ax.text(xco,yco,dmetrics.loc[hue,x],ha='center')
#                 print(xco)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax

## heatmaps
def annot_heatmap_boxes(ax,
                        xy,
                        width,
                        height,
                        fill=None, alpha=1,
                        lw=1.1,ec='k',clip_on=False,
                        scale_width=1,
                        scale_height=1,
                        xoff=0,yoff=0,
                        **kws,
                ):
    """
    """
    from matplotlib.patches import Rectangle
    return ax.add_patch(Rectangle(
                            xy=[xy[0]+xoff,xy[1]+yoff], 
                            width=width*scale_width, height=height*scale_height, 
                            fill=fill, alpha=alpha,
                            lw=lw,ec=ec,clip_on=clip_on,
                            **kws,
                            ))

def annot_heatmap_text(ax,dannot,
                  xoff=0,yoff=0,
                  kws_text={},# zip
                  annot_left='(',annot_right=')',
                  annothalf='upper',
                ):
    """
    kws_text={'marker','s','linewidth','facecolors','edgecolors'}
    """
    for xtli,xtl in enumerate(ax.get_xticklabels()):
        xtl=xtl.get_text()
        for ytli,ytl in enumerate(ax.get_yticklabels()):
            ytl=ytl.get_text()
            if annothalf=='upper':
                ax.text(xtli+0.5+xoff,ytli+0.5+yoff,dannot.loc[xtl,ytl],**kws_text,ha='center')
            else:
                ax.text(ytli+0.5+yoff,xtli+0.5+xoff,dannot.loc[xtl,ytl],**kws_text,ha='center')                
    return ax


# stats 
def perc_label(a,b=None,bracket=True): 
    if b is None:
        b=len(a)
        a=sum(a)
    return f"{(a/b)*100:.0f}%"+(f" ({num2str(a)}/{num2str(b)})" if bracket else "")

def pval2annot(pval,alternative=None,alpha=None,fmt='*',#swarm=False
               linebreak=False,
               q=False,
              ):
    """
    fmt: *|<|'num'    
    """
    if alternative is None and alpha is None:
        ValueError('both alternative and alpha are None')
    if alpha is None:
        alpha=0.025 if alternative=='two-sided' else 0.05
    if pd.isnull(pval):
        annot= ''
    elif pval < 0.0001:
        annot= "****" if fmt=='*' else f"P<\n{0.0001:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}"  if not linebreak else f"P={pval:.1g}"
    elif (pval < 0.001):
        annot= "***"  if fmt=='*' else f"P<\n{0.001:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    elif (pval < 0.01):
        annot= "**" if fmt=='*' else f"P<\n{0.01:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    elif (pval < alpha):
        annot= "*" if fmt=='*' else f"P<\n{alpha}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    else:
        annot= "ns" if fmt=='*' else f"P=\n{pval:.0e}" if fmt=='<' else f"P={pval:.1g}" if len(f"P={pval:.1g}")<6 else f"P=\n{pval:.1g}" if not linebreak else f"P={pval:.1g}"
    annot=annot if linebreak else annot.replace('\n','')
    annot=annot if not q else annot.replace('P','Q') 
    return annot 

def pval2stars(pval,alternative): return pval2annot(pval,alternative=alternative,fmt='*',)

               
def annot_many(ax,df,axis2col,colannot,
               colcolor=None,color_annot='k',color_line='gray',
               axis2cutoffs={'x':None,'y':None},
               xoff_bend=0.1,xoff_annot=0.05,off_axes=1,
               test=False):
    from roux.lib.str import linebreaker
    axis2params={axis:{} for axis in ['x','y']}
    for axis in ['x','y']:
        axis2params[axis]['cutoff']=df[axis2col[axis]].median() if axis2cutoffs[axis] is None else axis2cutoffs[axis]
        axis2params[axis]['original']=getattr(ax,f'get_{axis}lim')()
        axis2params[axis]['length']=axis2params[axis]['original'][1]-axis2params[axis]['original'][0]
        axis2params[axis]['bend']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*xoff_bend,
                                 axis2params[axis]['original'][1]+axis2params[axis]['length']*xoff_bend]
        axis2params[axis]['annot']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*(xoff_bend+xoff_annot),
                                    axis2params[axis]['original'][1]+axis2params[axis]['length']*(xoff_bend+xoff_annot)]
        axis2params[axis]['expanded']=[axis2params[axis]['original'][0]-axis2params[axis]['length']*(xoff_bend+xoff_annot+(off_axes if axis=='x' else off_axes)),
                                     axis2params[axis]['original'][1]+axis2params[axis]['length']*(xoff_bend+xoff_annot+(off_axes if axis=='x' else off_axes))]
        axis2params[axis]['expanded length']=axis2params[axis]['expanded'][1]-axis2params[axis]['expanded'][0]
        df[f"{axis} high"]=df[axis2col[axis]].apply(lambda x: x>axis2params[axis]['cutoff'])
    
    df=df.sort_values(by=[axis2col['y']],ascending=[True])
    df['x annot']=df['x high'].apply(lambda x : axis2params['x']['annot'][1] if x else axis2params['x']['annot'][0])
    def getys(x):
        if len(x)>2:
            x['y annot']=np.linspace(axis2params['y']['expanded'][0]+(axis2params['y']['expanded length']/len(x)),
                                     axis2params['y']['expanded'][1]-(axis2params['y']['expanded length']/len(x)),
                                     len(x))
        else:
            x['y annot']=x[axis2col['y']]
        return x
    df=df.groupby(['x annot'],as_index=False).apply(getys)
    if test:
        print(axis2params['x'])
        print(df.loc[:,['x annot','y annot']].head())
    df.apply(lambda x: ax.text(x['x annot'],x['y annot'],linebreaker(x[colannot],break_pt=25,),
                              ha='right' if not x['x high'] else 'left',va='center',
                              color=color_annot if colcolor is None else x[colcolor]),axis=1)
    df.apply(lambda x: ax.plot([x[axis2col['x']],axis2params['x']['bend'][1] if x['x high'] else axis2params['x']['bend'][0],x['x annot']],
                               [x[axis2col['y']],x['y annot'],x['y annot']],
                              color='gray',lw=1),axis=1)
    ax.set(**{f'{axis}lim':axis2params[axis]['expanded'] for axis in ['x','y']})
    return ax

def annot_subsets(dplot,colx,colsubsets,
                off_ylim=1.2,
                params_scatter={'zorder':2,'alpha':0.1,'marker':'|'},
                cmap='tab10',
                ax=None):
    if ax is None:ax=plt.subplot()
    ax.set_ylim(0,ax.get_ylim()[1]*off_ylim)
    from roux.viz.colors import get_ncolors
#     dplot=dplot.loc[:,[colx]+colsubsets].dropna(how='all',axis=1)
    colsubsets=[c for c in colsubsets if dplot[c].isnull().sum()!=len(dplot)]
    subsets=dplot.loc[:,colsubsets].melt()['value'].dropna().unique().tolist()
    subset2color=dict(zip(subsets,get_ncolors(len(subsets),cmap=cmap)))
    for coli,col in enumerate(colsubsets):
        subsets=dplot[col].dropna().unique().tolist()
        for subseti,subset in enumerate(subsets):
            y=(ax.set_ylim()[1]-ax.set_ylim()[0])*((10-(subseti+coli))/10-0.05)+ax.set_ylim()[0]
            X=dplot.loc[(dplot[col]==subset),colx]
            Y=[y for i in X]
            ax.scatter(X,Y,label=subset,color=subset2color[subset],**params_scatter)
            ax.text(ax.get_xlim()[1],y,subset,ha='left')
    return ax

def annot_errorbar(ax, xdata, ydata, caps="  ",color='lightgray'):
    import matplotlib as mpl
    line = ax.add_line(mpl.lines.Line2D(xdata, ydata,color=color))
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 24,
        'color': line.get_color()
    }
    a0 = ax.annotate(caps[0], xy=(xdata[0], ydata[0]), **anno_args)
    a1 = ax.annotate(caps[1], xy=(xdata[1], ydata[1]), **anno_args)
    return ax

def annot_corr(x, y,method='spearman',**kws):
    ax = plt.gca()
    from roux.stat.corr import get_corr
    from roux.viz.ax_ import set_label
    ax=set_label(ax=ax,s=get_corr(x,y,method=method,
                                   bootstrapped=True,
                                   outstr=True),
                 x=0,y=1,
                )
    return ax

def annot_contourf(colx,coly,colz,dplot,annot,ax=None,fig=None,vmin=0.2,vmax=1):
    """
    annot can be none, dict,list like anything..
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes    
    ax=plt.subplot() if ax is None else ax
    fig=plt.figure() if fig is None else fig
    if isinstance(annot,dict):
        #kdeplot
        # annot is a colannot
        for ann in annot:
            if not ann in ['line']:
                for subseti,subset in enumerate(list(annot[ann])[::-1]):
                    df_=dplot.loc[(dplot[ann]==subset),:]
                    sns.kdeplot(df_[colx],df_[coly],ax=ax,
                              shade_lowest=False,
                                n_levels=5,
                              cmap=get_cmap_subset(annot[ann][subset], vmin, vmax),
#                               cbar_ax=fig.add_axes([0.94+subseti*0.25, 0.15, 0.02, 0.32]), #[left, bottom, width, height]
                              cbar_ax=inset_axes(ax,
                                               width="5%",  # width = 5% of parent_bbox width
                                               height="50%",  # height : 50%
                                               loc=2,
                                               bbox_to_anchor=(1.36-subseti*0.35, -0.4, 0.5, 0.8),
                                               bbox_transform=ax.transAxes,
                                               borderpad=0,
                                               ),                                
                              cbar_kws={'label':subset},
                              linewidths=3,
                             cbar=True)
            if ann=='line':
                dannot=annot[ann]
                for subset in dannot:
                    ax.plot(dannot[subset]['x'],dannot[subset]['y'],marker='o', linestyle='-',
                            color=saturate_color(dannot[subset]['color'][0],1),
                           )
                    va_top=True
                    for x,y,s,c in zip(dannot[subset]['x'],dannot[subset]['y'],dannot[subset]['text'],dannot[subset]['color']):
                        ax.text(x,y,f" {s}",color=saturate_color(c,1.1),weight = 'bold',va='top' if va_top else 'bottom')
                        va_top=False if va_top else True #flip
    return fig,ax

def annot_corners(labels,X,Y,ax,space=-0.2,fontsize=18):
    xlims,ylims=get_axlims(X,Y,space=space)
    print('corners xlims,ylims=',xlims,ylims)
    
    labeli=0
    for x in xlims:
        for y in ylims:
            ax.text(x,y,labels[labeli],
                color='k',
                fontsize=fontsize,
                ha='center',
                va='center',
                bbox=dict(facecolor='w',edgecolor='none',alpha=0.4),
                )
            labeli+=1
    return ax

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
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
    Ref:
    1. https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
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

def annot_side(ax,
    df1,
    colx,#=params['text']['x'],
    coly,#=params['text']['y'],
    cols,#=params['text']['s'],
    loc='right',
    annot_count_max=5,
    offx3=0.15,
    offymin=0.1,
    offymax=0.9,
    break_pt=25,
    length_axhline=3,
    zorder=1,
    color='gray',
    kws_line={},
    **kws_text,
## text
#     style='normal',
    ):
    if len(df1)==0: 
        logging.warning("annot_side: no data found")
        return
    df1=df1.sort_values(coly,ascending=True)
    from roux.viz.ax_ import get_axlims
    d1=get_axlims(ax)
    df1['y']=np.linspace(d1['y']['min']+((d1['y']['len'])*offymin),
                        d1['y']['max']*offymax,
                        len(df1))
    x2=d1['x']['min'] if loc=='left' else d1['x']['max']
    x3=d1['x']['min'] if loc=='left' else d1['x']['max']+d1['x']['len']*offx3
    df1.apply(lambda x: ax.plot([x[colx],x2],
                               [x[coly],x['y']],
                              color=color,lw=1,**kws_line,
                               zorder=zorder,
                               ),axis=1)
    df1.apply(lambda x: ax.text(x3,x['y'],linebreaker(x[cols],break_pt=break_pt,),
                              ha='right' if loc=='left' else 'left',
                               va='bottom',**kws_text,
    #                           color=element2color[x['comparison type']],
                              zorder=2),axis=1)
    df1.apply(lambda x:ax.axhline(y = x['y'], 
                                 xmin=0 if loc=='left' else 1,
                                 xmax=0-offx3 if loc=='left' else length_axhline+offx3,
                                         clip_on = False,color=color,lw=1,
                                ),axis=1)
    ax.set_xlim([d1['x']['min'],d1['x']['max']])
    return ax

# unicode
def get_circled_digit(i): 
    d1={1:'\u2460',
 2:'\u2461',
 3:'\u2462',
 4:'\u2463',
 5:'\u2464',
 6:'\u2465',
 7:'\u2466',
 8:'\u2467',
 9:'\u2468',
 10:'\u2469',
    }
    return d1[i]
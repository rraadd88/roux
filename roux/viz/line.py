import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from os.path import exists, basename,dirname
from icecream import ic as info
from roux.viz.ax_ import *

def plot_range(df00,colvalue,
               colindex,
               k,
               headsize=15,
               headcolor='lightgray',
               ax=None):
    df00['rank']=df00[colvalue].rank()
    x,y=df00.rd.filter_rows({colindex:k}).iloc[0,:]['rank'],df00.rd.filter_rows({colindex:k}).iloc[0,:][colvalue]
    if ax is None:
        fig,ax=plt.subplots(figsize=[1,1])
    ax=df00.set_index('rank').sort_index(0)[colvalue].plot.area(ax=ax)
    ax.annotate('', xy=(x, y),  xycoords='data',
                xytext=(x, ax.get_ylim()[1]), textcoords='data',
                arrowprops=dict(facecolor=headcolor, shrink=0,
                               width=0,ec='none',
                               headwidth=headsize,
                               headlength=headsize,
                               ),
                horizontalalignment='right', verticalalignment='top',
                )
    d_=get_axlims(ax)
    ax.text(x,y+(d_['y']['len'])*0.25,int(y),#f"{y:.1f}",
                    # transform=ax.transAxes,
                    va='bottom',ha='center',
                   )
    ax.text(0.5,0,colvalue,
                    transform=ax.transAxes,
                    va='top',ha='center',
                   )
    ax.axis(False)
    return ax

def plot_summarystats(df,cols=['mean','min','max','50%'],plotp=None,ax=None,value_name=None):
    if ax is None:ax=plt.subplot(111)
    if not any([True if c in df else False for c in cols]):
        df=df.describe().T
    ax=df.loc[:,cols].plot(ax=ax)
    ax.fill_between(df.index, df['mean']-df['std'], df['mean']+df['std'], color='b', alpha=0.2,label='std')
    ax.legend(bbox_to_anchor=[1,1])
    if value_name is None:
        ax.set_ylabel('value')
    else:
        ax.set_ylabel(value_name)
    ax.set_xticklabels(df.index)    
    return ax
    
def plot_mean_std(df,cols=['mean','min','max','50%'],plotp=None):
    return plot_summarystats(df,cols=cols,plotp=plotp)
    
def plot_connections(dplot,label2xy,colval='$r_{s}$',line_scale=40,legend_title='similarity',
                        label2rename=None,
                        element2color=None,
                         xoff=0,yoff=0,
                     rectangle={'width':0.2,'height':0.32},
                     params_text={'ha':'center','va':'center'},
                     params_legend={'bbox_to_anchor':(1.1, 0.5),
                                  'ncol':1,
                                  'frameon':False},
                     legend_elements=[],
                     params_line={'alpha':1},
                     ax=None,
                    test=False):
    import matplotlib.patches as mpatches
    from matplotlib.collections import PatchCollection
    label2xy={k:[label2xy[k][0]+xoff,label2xy[k][1]+yoff] for k in label2xy}
    dplot['index xy']=dplot['index'].map(label2xy)
    dplot['column xy']=dplot['column'].map(label2xy)
    
    ax=plt.subplot() if ax is None else ax
    from roux.viz.ax_ import set_logos,get_subplot_dimentions
    patches=[]
    label2xys_rectangle_centers={}
    for label in label2xy:
        xy=label2xy[label]
        rect = mpatches.Rectangle(xy, **rectangle, fill=False,fc="none",lw=2,
                                  ec=element2color[label] if not element2color is None else 'k',
                                 zorder=0)

        patches.append(rect)
        line_xys=[np.transpose(np.array(rect.get_bbox()))[0],np.transpose(np.array(rect.get_bbox()))[1][::-1]]
        label2xys_rectangle_centers[label]=[np.mean(line_xys[0]),np.mean(line_xys[1])]
        inset_width=0.2
        inset_height=inset_width/get_subplot_dimentions(ax)[2]
        axin=ax.inset_axes([*[l-(off*0.5) for l,off in zip(label2xys_rectangle_centers[label],[inset_width,inset_height])],
                            inset_width,inset_height])
        if not test:
            axin=set_logos(label=label,element2color=element2color,ax=axin,test=test)
        axin.text(np.mean(axin.get_xlim()),np.mean(axin.get_ylim()),
                 label2rename[label] if not label2rename is None else label,
                  **params_text,
                 )
    dplot.apply(lambda x: ax.plot(*[[label2xys_rectangle_centers[x[k]][0] for k in ['index','column']],
                                  [label2xys_rectangle_centers[x[k]][1] for k in ['index','column']]],
                                  lw=(x[colval]-0.49)*line_scale,
                                  linestyle=params_line['linestyle'],
                                  color='k',zorder=-1,
                                  alpha=params_line['alpha'],
                                ),axis=1)            
    if params_line['annot']:
        def set_text_position(ax,x):
            xs,ys=[[label2xys_rectangle_centers[x[k]][i] for k in ['index','column']] for i in [0,1]]
            xy=[np.mean(xs),np.mean(ys)]
            if np.subtract(*xs)==0 or np.subtract(*ys)==0:
                ha,va='center','center'
                rotation=0
            else:
                if np.subtract(*xs)<0:      
                    ha,va='right','bottom'
                    xy[1]=xy[1]+0.025
                    rotation=-45
                else:
                    ha,va='right','top'
                    xy[1]=xy[1]-0.025
                    rotation=45
            ax.text(xy[0],xy[1],f"{x[colval]:.2f}",
                    ha=ha,va=va,
                    color='k',rotation=rotation,
                   bbox=dict(boxstyle="round",
                   fc='lightgray',ec=None,)
                   )
            return ax
            
        dplot.apply(lambda x: set_text_position(ax,x),axis=1)            
    from matplotlib.lines import Line2D
    legend_elements=legend_elements+[Line2D([0], [0], color='k', linestyle='solid', lw=(i-0.49)*line_scale, 
                                alpha=params_line['alpha'],
                                label=f' {colval}={i:1.1f}') for i in [1.0,0.8,0.6]]
    ax.legend(handles=legend_elements,
              title=legend_title,**params_legend)
    ax.set(**{'xlim':[0,1],'ylim':[0,1]})
    if not test:
        ax.set_axis_off()      
    return ax

def plot_groupby_qbin(dplot,bins,
                      colindex,colx,coly,
                      colhue=None,
                      ax=None,
                      aggfunc=None,
                      ticklabels_precision=1,
                      **params_pointplot,
                     ):
    from roux.stat.transform import get_qbins
    d=get_qbins(dplot.set_index(colindex)[f"{colx}"],bins, 'mid')
    d={k:"{:.{}f}".format(d[k],ticklabels_precision) for k in d}
    d={k:int(d[k]) if ticklabels_precision==0 else float(d[k]) for k in d}
    dplot[f"{colx}\n(midpoint of qbin)"]=dplot[colindex].map(d)
    if not aggfunc is None: 
        dplot=dplot.groupby([colhue,f"{colx}\n(midpoint of qbin)"]).agg({coly:aggfunc}).reset_index()
    if ax is None: ax=plt.subplot()
    sns.pointplot(data=dplot,
                  x=f"{colx}\n(midpoint of qbin)",
                  y=coly,
                  hue=colhue,
                  ax=ax,
                  **params_pointplot)
    return ax

def plot_kinetics(df1, x, y, hue, cmap='Reds_r',
                 ax=None,
                test=False,
                  kws_legend={},
                  **kws_set,
                 ):
    from roux.viz.ax_ import rename_legends
    from roux.viz.colors import get_ncolors
    df1=df1.sort_values(hue,ascending=False)
    info(df1[hue].unique())
    if ax is None: fig,ax=plt.subplots(figsize=[2.5,2.5])
    label2color=dict(zip(df1[hue].unique(),get_ncolors(df1[hue].nunique(),
                                                            ceil=False,
                                                            cmap=cmap,
                                                                )))
    df2=df1.groupby([hue,x],sort=False).agg({c:[np.mean,np.std] for c in [y]}).rd.flatten_columns().reset_index()
    d1=df1.groupby([hue,x],sort=False,as_index=False).size().groupby(hue)['size'].agg([min,max]).T.to_dict()
    d2={str(k):str(k)+'\n'+(f"(n={d1[k]['min']})" if d1[k]['min']==d1[k]['max'] else f"(n={d1[k]['min']}-{d1[k]['max']})") for k in d1}
    if test:info(d2)
    df2.groupby(hue,sort=False).apply(lambda df: df.sort_values(x).plot(x=x,
                                                            y=f"{y} mean",
                                                            yerr=f"{y} std",
                                                                        elinewidth=0.3,
                                                            label=df.name,
                                                            color=label2color[df.name],
                                                            lw=2,
                                                           ax=ax))
    ax=rename_legends(ax,replaces=d2,title=hue,
                     **kws_legend)
    ax.set(**kws_set)
    return ax
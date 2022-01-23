import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from roux.viz.ax_ import *

def plot_value_counts(df,col,logx=False,
                      kws_hist={'bins':10},
                      kws_bar={},
                     grid=False,
                     axes=None,fig=None,
                     hist=True):
    dplot=pd.DataFrame(df[col].value_counts()).sort_values(by=col,ascending=True)
    figsize=(4, len(dplot)*0.4)
    if axes is None:
        if hist:
            fig, axes = plt.subplots(2,1, sharex=True, 
                                       gridspec_kw={'height_ratios': [1, 6]},
                                       figsize=figsize,
                                    )
        else:
            fig, axes = plt.subplots(1,1, 
                                    figsize=figsize,
                                    )
    if hist:
        _=dplot.plot.hist(ax=axes[0],legend=False,**kws_hist
        #                   orientation="horizontal"
                         )
        axbar=axes[1]
    else:
        axbar=axes
    _=dplot.plot.barh(ax=axbar,legend=False,**kws_bar)
    axbar.set_xlabel('count')
    from roux.lib.str import linebreaker
    axbar.set_ylabel(col.replace(' ','\n'))
    if logx:
        if hist:
            for ax in axes.flat:
                ax.set_xscale("log")
                if grid:
                    ax.set_axisbelow(False)
        else:
            axes.set_xscale("log")
            if grid:
                axes.set_axisbelow(False)
            
# def plot_barh_stacked_percentage(df1,cols_y,ax=None):
#     dplot=pd.DataFrame({k:df1.drop_duplicates(k)[f'{k} '].value_counts() for k in cols_y})
#     dplot_=dplot.apply(lambda x:x/x.sum()*100,axis=0)
#     d=dplot.sum().to_dict()
#     dplot_=dplot_.rename(columns={k:f"{k}\n(n={d[k]})" for k in d})
#     dplot_.index.name='subset'
#     if ax is None: ax=plt.subplot()
#     dplot_.T.plot.barh(stacked=True,ax=ax)
#     ax.legend(bbox_to_anchor=[1,1])
#     ax.set(xlim=[0,100],xlabel='%')
#     _=[ax.text(1,y,f"{s:.0f}%",va='center') for y,s in enumerate(dplot_.iloc[0,:])]
#     return ax

def plot_barh_stacked_percentage(df1,coly,colannot,
                     color=None,
                     yoff=0,
                     ax=None,
                     ):
    """
    :param dplot: values in rows sum to 100% 
    :param coly: yticklabels, e.g. retained and dropped 
    :param colannot: col to annot
    """
    from roux.viz.ax_ import get_ticklabel2position
    from roux.viz.colors import get_colors_default
    if color is None:
        color=get_colors_default()[0]
    ax=plt.subplot() if ax is None else ax
    df2=df1.set_index(coly).apply(lambda x: (x/sum(x))*100, axis=1)
    ax=df2.plot.barh(stacked=True,ax=ax)
    ticklabel2position=get_ticklabel2position(ax,'y')
    from roux.viz.colors import saturate_color
    _=df2.reset_index().apply(lambda x: ax.text(1,
                                                  ticklabel2position[x[coly]]-yoff,
                                                  f"{x[colannot]:.1f}%",ha='left',va='center',
                                                 color=saturate_color(color,2),
                                               ),
                                axis=1)
    ax.legend(bbox_to_anchor=[1,1],title=df1.columns.name)
    d1=df1.set_index(coly).T.sum().to_dict()
    ax.set(xlim=[0,100],xlabel='%',
          yticklabels=[f"{t.get_text()}\n(n={d1[t.get_text()]})" for t in ax.get_yticklabels()])
    return ax

def plot_bar_serial(ax,d1,polygon=False,
             polygon_x2i=0,
             labelis=[],
             y=0,
             ylabel=None,
             off_arrowy=0.15,
             **kws_rectangle):
    kws_rectangle=dict(height=0.5,linewidth=1)
    kws1=dict(
    xs=[(list(d1.values())[i]/sum(list(d1.values())))*100 for i in range(len(d1))],
    cs=['#f55f5f', '#D3DDDC'] if len(d1)==2 else ['#f55f5f','#e49e9d', '#D3DDDC'],
    labels=[],
    size=sum(list(d1.values())),
    xmax=100,
    y=y,
    )
    
    import matplotlib.patches as patches
    l1=[
        patches.Rectangle((0,kws1['y']), kws1['xmax'],  
                          fc='none', ec='k',alpha=1,zorder=2,**kws_rectangle),
    ]
    x=0
    for i,(x_,c,s) in enumerate(zip(kws1['xs'],kws1['cs'],d1.keys())):
        print(x,x_)
        l1.append(patches.Rectangle((x, kws1['y']), x_, 
                          fc=c, ec='none',alpha=1,**kws_rectangle))
        if i in labelis:
            ax.text(x if i!=len(d1.keys())-1 else x+x_,
                    kws1['y']+kws_rectangle['height']*0.5,
                    f"{s} ({x_:.0f})%",
                    ha='left' if i!=len(d1.keys())-1 else 'right',
                    va='center')
        x+=x_
        if polygon:
            if polygon_x2i==i:
                l1.append(patches.Polygon(xy=[(0,kws1['y']),
                                              (x,kws1['y']),
                                              (100,kws1['y']-1),
                                              (0,kws1['y']-1),], closed=True,fc=[0.95,0.95,0.95],
                                          ec='gray',
                                          lw=0.1,
                                          zorder=-2))        
    #     break
    s=num2str(kws1['size'],magnitude=True,decimals=1)
    ax.annotate(s=s, xy=(0,kws1['y']-off_arrowy), xytext=(50,kws1['y']-off_arrowy), arrowprops=dict(arrowstyle='->',shrinkA=0,shrinkB=0,color='k'),zorder=-1,
               va='center',ha='center')
    ax.annotate(s=' '*(len(s)*2), xy=(100,kws1['y']-off_arrowy), xytext=(50,kws1['y']-off_arrowy), arrowprops=dict(arrowstyle='->',shrinkA=0,shrinkB=0,color='k'),zorder=-1,
               va='center',ha='center')
    _=[ax.add_patch(o) for o in l1]
    if not ylabel is None:
        ax.text(-2.5,kws1['y']+kws_rectangle['height']*0.5,ylabel,ha='right',va='center')
            
    return ax

def plot_barh_stacked_percentage_intersections(df0,
                                               colxbool='paralog',
                                               colybool='essential',
                                               colvalue='value',
                                               colid='gene id',
                                               colalt='singleton',
                                               coffgroup=0.95,
                                               colgroupby='tissue',
                                              ):
    ##1 threshold for value by group
    def apply_(df):
        coff=np.quantile(df.loc[df[colybool],colvalue],coffgroup)
        df[colybool]=df[colvalue]<coff
        return df
    df1=df0.groupby(colgroupby).progress_apply(apply_)
    ##2 % 
    df2=df1.groupby([colid,colxbool]).agg({colybool: perc}).reset_index().rename(columns={colybool:f'% {colgroupby}s with {colybool}'},
                                                                                        errors='raise')
    coly=f"% of {colgroupby}s"
    ##3 bin y
    df2[coly]=pd.cut(df2[f'% {colgroupby}s with {colybool}'],bins=pd.interval_range(0,100,4),)
    ##3 % sum
    df3=df2.groupby(coly)[colxbool].agg([sum]).rename(columns={'sum':colxbool})
    dplot=df3.join(df2.groupby(coly).size().to_frame('total'))
    dplot[colalt]=dplot['total']-dplot[colxbool]
    dplot.index=[str(i) for i in dplot.index]
    dplot.index.name=coly
    dplot.columns.name=f"{colid} type"
    dplot=dplot.sort_values(coly,ascending=False)
    dplot=dplot.reset_index()
#     from roux.viz.bar import plot_barh_stacked_percentage
    fig,ax=plt.subplots(figsize=[3,3])
    plot_barh_stacked_percentage(df1=dplot.loc[:,[coly,colxbool,colalt]],
                                coly=coly,
                                colannot=colxbool,
                                ax=ax)
    set_ylabel(ax)
    ax.set(xlabel=f'% of {colid}s',
          ylabel=None)
    return ax

def plot_intersections(ds1,
                            item_name=None,
                            figsize=[4,4],text_width=2,
                            yorder=None,
                            sort_by='cardinality',
                            sort_categories_by=None,#'cardinality',
                            element_size=40,
                            facecolor='gray',
                            bari_annot=None, # 0, 'max_intersections'
                            totals_bar=False,
                            totals_text=True,                           
                            intersections_ylabel=None,
                            intersections_min=None,
                            test=False,
                            annot_text=False,
                           set_ylabelx=-0.25,set_ylabely=0.5,
#                             post_fun=None,
                            **kws,
                          ):
    """
    upset
    sort_by:{‘cardinality’, ‘degree’}
    If ‘cardinality’, subset are listed from largest to smallest. If ‘degree’, they are listed in order of the number of categories intersected.

    sort_categories_by:{‘cardinality’, None}
    Whether to sort the categories by total cardinality, or leave them in the provided order.

    Ref: https://upsetplot.readthedocs.io/en/stable/api.html
    """
    assert(isinstance(ds1,pd.Series))
    if (item_name is None) and (not ds1.name is None):
        item_name=ds1.name
    if intersections_min is None:
        intersections_min=len(ds1)
    if not yorder is None:
        yorder= [c for c in yorder if c in ds1.index.names][::-1]
        ds1.index = ds1.index.reorder_levels(yorder) 
    ds2=(ds1/ds1.sum())*100
    import upsetplot as up
    d=up.plot(ds2,
              figsize=figsize,
              text_width=text_width,
              sort_by=sort_by,
              sort_categories_by=sort_categories_by,
              facecolor=facecolor,element_size=element_size,
              **kws,
             ) 
    d['totals'].set_visible(totals_bar)
    if totals_text:
        from roux.lib.df import get_totals
        d_=get_totals(ds1)
        d['matrix'].set_yticklabels([f"{s.get_text()} (n={d_[s.get_text()]})" for s in d['matrix'].get_yticklabels()],
                                          )
    if totals_bar:
        d['totals'].set(ylim=d['totals'].get_ylim()[::-1],
                       xlabel='%')
    set_ylabel(ax=d['intersections'],
              s=(f"{item_name}s " if not item_name is None else "")+f"%\n(total={ds1.sum()})",
               x=set_ylabelx, y=set_ylabely,
              )        
    d['intersections'].set(
                          xlim=[-0.5,intersections_min-0.5],
                          )
    if sort_by=='cardinality':
        y=ds2.max()
    elif sort_by=='degree':
        y=ds2.loc[tuple([True for i in ds2.index.names])]
#     if bari_annot=='max_intersections':
#         l1=[i for i,t in enumerate(ds1.index) if t==tuple(np.repeat(True,len(ds1.index.names)))]
#         if len(l1)==1:
#             bari_annot=l1[0]
#             print(bari_annot)
#     print(sum(ds1==ds1.max()))
#     print(bari_annot)
    if sum(ds1==ds1.max())!=1:
        bari_annot=None
    if isinstance(bari_annot,int):
        bari_annot=[bari_annot]
    if isinstance(bari_annot,list):
#         print(bari_annot)
        for i in bari_annot:
            d['intersections'].get_children()[i].set_color("#f55f5f")
    if annot_text and bari_annot==0:
        d['intersections'].text(bari_annot-0.25,y,f"{y:.1f}%",
                                ha='left',va='bottom',color="#f55f5f",zorder=10)
    
#     if intersections_ylabel    
#     if not post_fun is None: post_fun(ax['intersections'])
    return d

def plot_text(df1,
            xlabel='overlap %',
            ylabel='gene set name',
            cols='s',
            colannot='Q',
              offx=0,
              xmin=None,
            ax=None,
          ):
    from roux.viz.annot import pval2annot
    df1['y']=range(len(df1[ylabel]))#[::-1]
    x1=(df1[xlabel].min() if xmin is None else xmin)+((df1[xlabel].max()-(df1[xlabel].min() if xmin is None else xmin))*offx)
    if ax is None:
        fig,ax=plt.subplots(figsize=[4.45,len(df1)*0.33])
    ax=df1.set_index(ylabel)[xlabel].plot.barh(color='#60badc',#'#0094cc',#'#0c7ec2',
                                                                  width=0.8,
                                                                  ax=ax)
    _=df1.apply(lambda x: ax.text(x1,x['y'],x[ylabel],
                                va='center'),axis=1)
    _=df1.apply(lambda x: ax.text(ax.get_xlim()[1],
                                   x['y'],f" {x[cols]}",
                                va='center'),axis=1)
    ax.get_yaxis().set_visible(False)
    ax.set(xlabel=xlabel,
          )
    return ax

# def plot_text(df1,
#             xlabel='overlap %',
#             ylabel='gene set name',
#             cols='s',
#             colannot='Q',
#               offx=0.05,
#               xmin=None,
#             ax=None,
#           ):
#     from roux.viz.annot import pval2annot
#     d1=get_axlims(ax)
#     print(xmin)
#     print(df1[xlabel].min() if xmin is None else xmin)
#     x1=(df1[xlabel].min() if xmin is None else xmin)+((df1[xlabel].max()-(df1[xlabel].min() if xmin is None else xmin))*offx)
#     if ax is None:
#         fig,ax=plt.subplots(figsize=[4.45,len(df1)*0.33])
#     ax=df1.set_index(ylabel)[xlabel].plot.barh(color='#60badc',#'#0094cc',#'#0c7ec2',
#                                                                   width=0.8,
#                                                                   ax=ax)
#     _=df1.apply(lambda x: ax.text(x1,x['y'],x[ylabel],
#                                 va='center'),axis=1)
#     _=df1.apply(lambda x: ax.text(ax.get_xlim()[1],
#                                    x['y'],f" {x[cols]}",
#                                 va='center'),axis=1)
#     ax.get_yaxis().set_visible(False)
#     ax.set(xlabel=xlabel,
#           )
#     return ax
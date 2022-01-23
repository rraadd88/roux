from roux.global_imports import *
from roux.lib.str import make_pathable_string
# from roux.lib import to_class,rd

#rasterized=True
# sns scatter_kws={'s':5, 'alpha':0.3, 'rasterized':True}

# stats
def annot_stats(dplot,colx,coly,colz,
            stat_method=[],
            bootstrapped=False,
            params_set_label={},
            ax=None):
    stat_method = [stat_method] if isinstance(stat_method,str) else stat_method
    from roux.viz.ax_ import set_label
    if 'mlr' in stat_method:
        from roux.lib.stat.poly import get_mlr_2_str
        ax=set_label(ax,label=get_mlr_2_str(dplot,colz,[colx,coly]),
                    title=True,params={'loc':'left'})
    if 'spearman' in stat_method or 'pearson' in stat_method:
        from roux.stat.corr import get_corr
        ax=set_label(ax,label=get_corr(dplot[colx],dplot[coly],method=stat_method[0],
                                       bootstrapped=bootstrapped,
                                       outstr=True,n=True),
                    **params_set_label)
    return ax

def plot_trendline(dplot,colx,coly,
                    params_plot={'color':'r','linestyle':'solid','lw':2},
                    poly=False,lowess=True,
                    params_poly={'deg':1},
                    params_lowess={'frac':0.7,'it':5},
                    ax=None):
    """
    TODO label with goodness of fit, r (y_hat vs y)
    """
    ax= plt.subplot() if ax is None else ax    
    if poly:
        coef = np.polyfit(dplot[colx], dplot[coly],**params_poly)
        poly1d_fn = np.poly1d(coef)
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        ax.plot(dplot[colx], poly1d_fn(dplot[colx]), **params_plot)
    if lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xys_lowess=lowess(dplot[coly], dplot[colx],frac=0.7,it=5)
        ax.plot(xys_lowess[:,0],xys_lowess[:,1], **params_plot)
    return ax

# @to_class(rd)
def plot_scatter(dplot,colx,coly,colz=None,
            kind='hexbin',
            trendline_method='poly',
            stat_method="spearman",
            bootstrapped=False,
            cmap='Reds',label_colorbar=None,
            gridsize=25,
            params_plot={},
            params_plot_trendline={},
            params_set_label=dict(x=0,y=1),
            ax=None,
            **kws,
                ):
    """
    trendline:['poly','lowess']
    stat_method=['mlr',"spearman"],
    """
    ax= plt.subplot() if ax is None else ax
    
    trendline_method = [trendline_method] if isinstance(trendline_method,str) else trendline_method
    stat_method = [stat_method] if isinstance(stat_method,str) else stat_method
    
    dplot=dplot.dropna(subset=[colx,coly]+[] if colz is None else [colz],how='any')
    if kind in ['hexbin']:
        if colz is None: 
            colz='count'
            dplot[colz]=1
        if colz in dplot:
            params_plot['C']=colz
            params_plot['reduce_C_function']=len if colz=='count' else params_plot['reduce_C_function'] if 'reduce_C_function' in params_plot else np.mean        
            params_plot['gridsize']=params_plot['gridsize'] if 'gridsize' in params_plot else gridsize
            params_plot['cmap']=params_plot['cmap'] if 'cmap' in params_plot else cmap
            print(params_plot)
        ax=dplot.plot(kind=kind, x=colx, y=coly, 
    #         C=colz,
            ax=ax,
            **params_plot,
                      **kws,
            )
    else:
        ax=sns.scatterplot(data=dplot,
                            x=colx,
                            y=coly,
                            hue=colz,       
                            palette=cmap,
                            ax=ax,
                           **params_plot,
                           **kws)
        if not colz is None:
            ax.legend(bbox_to_anchor=[1,1],title=colz)
    from roux.viz.ax_ import set_label_colorbar
#     print(colz)
    ax=set_label_colorbar(ax,colz if label_colorbar is None else label_colorbar)
    from roux.viz.ax_ import set_label
    if 'mlr' in stat_method:
        from roux.lib.stat.poly import get_mlr_2_str
        ax=set_label(ax,label=get_mlr_2_str(dplot,colz,[colx,coly]),
                    title=True,params={'loc':'left'})
    if 'spearman' in stat_method or 'pearson' in stat_method:
        from roux.stat.corr import get_corr
        ax=set_label(ax=ax,s=get_corr(dplot[colx],dplot[coly],method=stat_method[0],
                                       bootstrapped=bootstrapped,
                                       outstr=True,n=True),
                    **params_set_label)
    from roux.viz.colors import saturate_color
    plot_trendline(dplot,colx,coly,
                    params_plot={'color':saturate_color(params_plot['color'],alpha=1.5) if 'color' in params_plot else None,
                                 'linestyle':'solid','lw':2},
                    poly='poly' in trendline_method,
                    lowess='lowess' in trendline_method,
                   ax=ax, 
                   **params_plot_trendline,
                    )    
    return ax

def plot_reg(d,xcol,ycol,textxy=[0.65,1],
             scafmt='hexbin',
             method="spearman",pval=True,bootstrapped=False,
             trendline=False,
             trendline_lowess=False,
             cmap='Reds',vmax=10,cbar_label=None,title_stat=False,
             axscale_log=False,
             params_scatter={},
            ax=None,
            plotp=None,plotsave=False):
    d=d.dropna(subset=[xcol,ycol],how='any')
    if ax is None:
        ax=plt.subplot(111)
    if scafmt=='hexbin':
        cbar_label='count'
        hb = ax.hexbin(d[xcol], d[ycol], gridsize=25, cmap=cmap)
        cb = plt.colorbar(hb, ax=ax)
        cb.set_label(cbar_label)            
        ax.set(**{'xlabel':xcol,'ylabel':ycol})
    elif scafmt=='sca':
        ax=d.plot.scatter(x=xcol,y=ycol,ax=ax,**params_scatter)    
    if axscale_log:
        ax.set_xscale("log", nonposx='clip')
        ax.set_yscale("log", nonposy='clip')
    if trendline:
        from roux.viz.colors import saturate_color
        coef = np.polyfit(d[xcol], d[ycol],1)
        poly1d_fn = np.poly1d(coef) 
        # poly1d_fn is now a function which takes in x and returns an estimate for y
        ax.plot(d[xcol], poly1d_fn(d[xcol]), linestyle='solid',lw=2,
               color=saturate_color(params_scatter['color']) if 'color' in params_scatter else None)
        
    if trendline_lowess:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        xys_lowess=lowess(d[ycol], d[xcol],frac=0.9,it=20)
        ax.plot(xys_lowess[:,0],xys_lowess[:,1], linestyle='--',
                color=params_scatter['color'] if 'color' in params_scatter else None)
    from roux.stat.corr import get_corr
    textstr=get_corr(d[xcol],d[ycol],method=method,bootstrapped=bootstrapped,outstr=True)#.replace('\n',', ')
    if title_stat:
        ax.set_title(textstr)            
    else:
        ax.text(ax.get_xlim()[0],ax.get_ylim()[1],textstr,
                ha='left',va='top',
               )
        ax.set_title(f"{'' if d.columns.name is None else d.columns.name+' '}",loc='left',
                     ha='left')    
    if plotsave:
        if plotp is None:
            plotp=f"plot/{scafmt}_{make_pathable_string(xcol)}_vs_{make_pathable_string(ycol)}.svg"
        print(plotp)
        plt.savefig(plotp)
    else:
        return ax

def plot_corr(dplot,x,y,ax=None,params_sns_regplot={},params_ax={}):
    if ax is None:ax=plt.subplot()
    ax=sns.regplot(data=dplot,x=x,y=y,fit_reg=True,
                lowess=True,
#                 scatter_kws={'color':'gray'},
#                 line_kws={'color':'r','label':'sc.stats.spearmanr(d[xcol], d[ycol])[0]'},
                ax=ax,
                **params_sns_regplot
               )
    ax.set(**params_ax)
    return ax

from roux.viz.ax_ import *    
def plot_scatterbysubsets(df,colx,coly,colannot,
                        ax=None,dfout=False,
                          kws_dfannot2color={'cmap':'spring'},
                        label_n=False,
                        kws_scatter={},
                         annot2color=None,equallim=True,
                         test=False):
    if test: 
        dfout=True
    if ax is None:ax=plt.subplot()
    if annot2color is None:
        from roux.viz.annot import dfannot2color
        df,annot2color=dfannot2color(df,colannot,renamecol=False,
                                     test=test,
                                     **kws_dfannot2color)
    else:
        colcolor=f"{colannot} color"
        df=df.loc[df[colannot].isin(annot2color.keys()),:]
        df[colcolor]=df[colannot].apply(lambda x : annot2color[x] if not pd.isnull(x) else x)
    colc=f"{colannot} color"    
    annot2len=df.groupby(colannot).agg(len).iloc[:,0].sort_values(ascending=False).to_dict()
    for annot in annot2len:
        df_=df.loc[df[colannot]==annot,[colx,coly,colc]].dropna()
        ax.scatter(x=df_[colx],y=df_[coly],c=df_[colc],
        label=f"{annot} (n={len(df_)})" if label_n else annot,
                  **kws_scatter)
    ax.set_xlabel(colx)
    ax.set_ylabel(coly)
    if equallim:
        ax=set_equallim(ax,diagonal=True)    
    ax=sort_legends(ax,params={'loc':'best'}) 
    ax.grid(True)
    if dfout:
        return df
    else:
        return ax    

def plot_scatter_agg(dplot,colgroupby,colx,coly,
                     params_errorbar=dict(),
                     params_legend=dict(loc=2,bbox_to_anchor=[1,1]),
                     ax=None):
    ax=plt.subplot() if ax is None else ax
    from roux.stat.variance import confidence_interval_95
    dplot2=dplot.groupby(colgroupby).agg({k:[np.median,confidence_interval_95] for k in [colx,coly]})
    dplot2.columns=coltuples2str(dplot2.columns)
    dplot2=dplot2.reset_index()
    dplot2.apply(lambda x: ax.errorbar(x=colx,y=coly,
                                       **params_errorbar,
                ),axis=1)
#     ax.set_xlabel(params_errorbar['x'])
#     ax.set_ylabel(params_errorbar['y'])
    ax.legend(**params_legend)
    return ax
        
def plot_circlify(dplot,circvar2col,threshold_side=0,ax=None,cmap_parent='binary',cmap_child='Reds'):
#     from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from roux.viz.ax_ import set_legend_custom
    import circlify as circ
    
    dplot=dplot.rename(columns={circvar2col[k]:k for k in circvar2col})
    if not 'child datum' in dplot:
        dplot['child datum']=1
    data=dplot.groupby(['parent id','parent datum']).apply(lambda df: {'id': df.loc[:,'parent id'].unique()[0],
                                                                      'datum': df.loc[:,'parent datum'].unique()[0],
                                                              'children':list(df.loc[:,['child id','child datum']].rename(columns={c:c.split(' ')[1] for c in ['child id','child datum']}).T.to_dict().values())}).tolist()
    from roux.viz.colors import get_val2color,get_cmap_subset
    type2params_legend={k:{'title':circvar2col[f"{k} color"]} for k in ['child','parent'] if f"{k} color" in circvar2col}
    if "child color" in circvar2col:
        dplot['child color'],type2params_legend['child']['data']=get_val2color(dplot['child color'],cmap=cmap_child)
    if not cmap_parent is None:
        dplot['parent color'],type2params_legend['parent']['data']=get_val2color(dplot['parent color'],cmap=cmap_parent) 
    id2color={**(dplot.set_index('child id')['child color'].to_dict() if "child color" in circvar2col else {}),
              **(dplot.set_index('parent id')['parent color'].to_dict() if not cmap_parent is None else {})}
    l = circ.circlify(data, show_enclosure=True)
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8))
    lineside2params={k:{} for k in ['-','+']}
    for circlei,circle in enumerate(l):
        c=plt.Circle((circle.circle[0],circle.circle[1]),circle.circle[2],
                     alpha=1,
                     fc='w' if circle.ex is None else id2color[circle.ex['id']] if circle.ex['id'] in id2color else 'w',
                     ec='darkgray' if circle.ex is None else '#dc9f4e' if circle.ex['id'].startswith('Scer') else '#6cacc5' if circle.ex['id'].startswith('Suva') else 'lightgray',
                    lw='1' if circle.ex is None else '1' if circle.ex['id'] in id2color else '3',
                    )
        ax.add_patch(c)
        if not circle.ex is None:
            if 'children' in circle.ex:
                lineside2params['-' if circle.circle[0]<threshold_side else '+'][circle.ex['id']]={
                    'x':[circle.circle[0]-circle.circle[2] if circle.circle[0]<threshold_side else circle.circle[0]+circle.circle[2],
                         -1 if circle.circle[0]<threshold_side else 1],
                    'y':[circle.circle[1],circle.circle[1]],
                }
#                 ax.plot()
#         if not circle.ex is None:
#             plt.text(circle.circle[0],circle.circle[1],'' if circle.ex is None else circle.ex['id'],
#                      ha='center',
#                     color='r' if 'child' in circle.ex else 'b')
    ax.plot()
#     from roux.lib.str import linebreaker
    lw=2
    color='k'
    for side in lineside2params:
        df=pd.DataFrame({k:np.ravel(list(lineside2params[side][k].values())) for k in lineside2params[side]}).T.sort_values(by=[3,0],
                                        ascending=[True,True])
        df[3]=np.linspace(-0.9,0.9,len(df))
        
        # resolve overlapping annotation lines
        line_ys=[]
        i_=None
        for i in df[2]:
            if not i_ is None:
#                 if test:
#                     print(i-i_)
                if abs(i-i_)<0.01:
                    i+=0.05
            line_ys.append(i)
            i_=i
        df[2]=line_ys
        
        df.apply(lambda x: ax.plot([x[0],x[1],x[1]+(-0.1 if side=='-' else 0.1)],
                                   [x[2],x[2],x[3]],color=color,
                                   alpha=0.5,
                                   lw=lw),axis=1)
        df.apply(lambda x: ax.text(x[1]+(-0.11 if side=='-' else 0.11),x[3],
                                   x.name.split(' [')[0].split(' -')[0].split(', ')[0],
                                   color=color,
                                   ha='right' if side=='-' else 'left',
                                   va='center'),axis=1)
#     return lineside2params
    ax.set_axis_off()    
    if 'child color' in circvar2col:
        set_legend_custom(ax,
                         param='color',lw=1,color='k',
                        legend2param={np.round(k,decimals=2):type2params_legend['child']['data'][k] for k in type2params_legend['child']['data']},
                        params_legend={'title':type2params_legend['child']['title'],
                                      'ncol':3,
                                       'loc':2,'bbox_to_anchor':[1,1.05]
    #                                    'loc':4,'bbox_to_anchor':[1.75,-0.1]
                                      },
                         )

#     print(type2params_legend)
#     for ki,k in enumerate(type2params_legend.keys()):
#         if 'data' in type2params_legend[k]:
#             axin = inset_axes(ax, width="100%", height="100%",
#                            bbox_to_anchor=(1.1+ki*0.2, 0.1, 0.2, 0.1),
#                            bbox_transform=ax.transAxes, 
# #                               loc=2, 
#                               borderpad=0
#                                     )
#             for x,y,c,s in zip(np.repeat(0.3,3),np.linspace(0.2,0.8,3),
#                                type2params_legend[k]['data'].values(),
#                                type2params_legend[k]['data'].keys()):
#                 axin.scatter(x,y,color=c,s=250)
#                 axin.text(x+0.2,y,s=f"{s:.2f}",ha='left',va='center')
#             axin.set_xlim(0,1)
#             axin.set_ylim(0,1)
#             axin.set_title(type2params_legend[k]['title'])
#             axin.set_axis_off()
#             ax.axis('equal')
    return ax

def plot_qq(x):    
    import statsmodels.api as sm
    fig = plt.figure(figsize = [3, 3])
    ax = plt.subplot()
    sm.qqplot(x, dist = sc.stats.norm, 
              line = 's', 
              ax=ax)
    ax.set_title("SW test "+pval2annot(sc.stats.shapiro(x)[1],alpha=0.05,fmt='<',linebreak=False))
    from roux.viz.ax_ import set_equallim
    ax=set_equallim(ax)
    return ax
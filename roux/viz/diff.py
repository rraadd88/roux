from roux.global_imports import *

def plot_stats_diff(df2,
                    coly=None,
#                     colcomparison=None,  
                    cols_subset=None,
                    colsorty=None,#'difference between mean (subset1-subset2)',
#                     testn='MWU test, FDR corrected',
                    tests=['MWU test, FDR corrected','FE test, FDR corrected'],
                    show_q=True,
                    show_ns=True,
                    ascending=False,
                    palette=None,
                    ax=None, fig=None,
                    params_ax={},
                    alpha=None,
                    legend_title=None,
                  loc='upper left', 
                  bbox_to_anchor=(1, 0),                    
                   **kws_pointplot):
    """
    :param df2: output of `get_stats`
    :param coly: unique
    
    TODOs:
    1. plot raw data
    2. ns on the right
    """
    if 'sorted' in df2:
        assert(df2['sorted'].nunique()==1)
    if cols_subset is None:
        cols_subset=['subset1','subset2']
#     if colcomparison_ is None:
    colcomparison_='comparison\n(n1,n2)'
    df2.loc[:,colcomparison_]=df2.apply(lambda x: f"{x[cols_subset[0]]} vs {x[cols_subset[1]]}\n({int(x['len subset1'])},{int(x['len subset2'])})",axis=1)
    if not coly is None or df2.rd.check_duplicated([colcomparison_]):
        if coly is None:
            coly=(df2.select_dtypes([object]).nunique()==len(df2)).loc[lambda x:x].index.tolist()[0]
        if show_ns:
            colcomparison=f'{coly}\n(n1,n2)'
            df2.loc[:,colcomparison]=df2.apply(lambda x: f"{x[coly]}\n({int(x['len subset1'])},{int(x['len subset2'])})",axis=1)
        else:
            colcomparison=coly
    else:
        colcomparison=colcomparison_
    info(f"colcomparison={colcomparison}")
    assert(not df2.rd.check_duplicated([colcomparison]))
    if colsorty is None:
        colsorty='sorty'
        df2[colsorty]=df2['difference between mean (subset1-subset2)'].abs()
    df2=df2.sort_values(by=colsorty,
                        ascending=ascending)
    df2=df2.drop([c for c in df2 if 'subset1-subset2' in c],axis=1)
    if palette is None:
        palette=get_colors_default()[:2]
    df3=melt_paired(df=df2,
                suffixes=cols_subset,
                ).rename(columns={'suffix':'subset'})
    params=dict(x='mean',
                y=colcomparison,
                hue='id',
                palette=palette,
               )
    if fig is None: plt.figure(figsize=[2.5,(len(df3)*0.175)+1])
    ax=plt.subplot() if ax is None else ax
    ax=sns.pointplot(data=df3,
                 **params,
                  join=False,
                  dodge=0.2,
                 ax=ax,
                     zorder=2,
                    **kws_pointplot)
    from roux.viz.ax_ import color_ticklabels
    df2.loc[(df2[f'significant change ({tests[0]})' if f'significant change ({tests[0]})' in df2 else 'change']=='ns'),'color yticklabel']='lightgray'
    df2.loc[(df2[f'significant change ({tests[0]})' if f'significant change ({tests[0]})' in df2 else 'change']!='ns'),'color yticklabel']='gray'
    logging.warning(f"yticklabels shaded by {tests[0]}")
    ax=color_ticklabels(ax, ticklabel2color=df2.loc[:,[params['y'],'color yticklabel']].drop_duplicates().rd.to_dict([params['y'],'color yticklabel']),
                        axis='y')
    
    from roux.viz.ax_ import get_ticklabel2position
    df3['y']=df3[params['y']].map(get_ticklabel2position(ax, axis='y'))
    ## apply dodge
    df3['y+off']=df3.apply(lambda x: x['y']+(-0.1 if x['subset']=='subset1' else 0.1),axis=1)
    df3['xerr color']=df3.apply(lambda x: palette[0] if x['subset']=='subset1' else palette[1],axis=1)
    
    df3['std']=df3['var'].astype(float).apply(np.sqrt)
    df3[f"{params['x']}+std"]=df3[params['x']]+df3['std']
    df3[f"{params['x']}-std"]=df3[params['x']]-df3['std']
    ## TODO make scatter using apply
#   def apply(ax,x,params):
#         ax.scatter()
# #         ax.plot([min([x[f"{params['x']}-std"],x[f"{params['x']}+std"]]),
# #                                    max([x[f"{params['x']}-std"],x[f"{params['x']}+std"]])],
# #                                 [x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1),x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1)],
# #                                 color=palette[0] if x[params['y']].startswith(x['subset']) else palette[1],
# #                                  )
#         ax.plot([min([x[f"{params['x']}-std"],
#                       x[f"{params['x']}+std"]]),
#                 max([x[f"{params['x']}-std"],
#                     x[f"{params['x']}+std"]])],
#                 [x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1),
#                  x['y']+(-0.1 if x[params['y']].startswith(x['subset']) else 0.1)],
#                  color=palette[0] if x[params['y']].startswith(x['subset']) else palette[1],
#                  )
#         return ax
    _=df3.apply(lambda x: ax.plot([min([x[f"{params['x']}-std"],x[f"{params['x']}+std"]]),
                                   max([x[f"{params['x']}-std"],x[f"{params['x']}+std"]])],
                                  [x['y+off'],x['y+off']],
#                                 [x['y']+(-0.1 if x[colcomparison_].startswith(x['subset']) else 0.1),
#                                  x['y']+(-0.1 if x[colcomparison_].startswith(x['subset']) else 0.1)],
                                color=x['xerr color'],
                                  zorder=1,
                                 ),axis=1)
    ax.set(**params_ax)
    ## annot p-vals
    w=(ax.get_xlim()[1]-ax.get_xlim()[0])
#     cols_pvalues=[]
#     for k in tests:
#         if f'P ({k test, FDR corrected})' in df3 and show_q:
#             cols_pvalues.append(f'P ({k} test, FDR corrected)')
#         elif f'P ({k} test)' in df3:
#             cols_pvalues.append(f'P ({k} test)')
#         else:
#             continue
    for i,c in enumerate(tests):
        if df3[f'P ({c})'].isnull().all():
            logging.error(f"all null for {c}")
            continue
        posx=ax.get_xlim()[0]+w+(w*(i*0.3))
        ax.text(posx,
                ax.get_ylim()[1],
                f" {'P' if not ('corrected' in c or not c.startswith('Q')) else 'Q'} ({c.split(',')[0].split(' ')[0]})",
                va='bottom',
                color='gray',
               )
        df3.drop_duplicates(subset=['y',f'P ({c})']).apply(lambda x: ax.text(posx,x['y'],
                                                                   pval2annot(x[f'P ({c})'], 
                                                                               alternative='two-sided', 
                                                                               alpha=alpha, 
                                                                               fmt='<', 
                                                                               linebreak=False).replace('P',''),
                                                                    va='center',
                                                                   color='gray'),
                                                  axis=1)
    o1=ax.legend(
              loc=loc,#'upper left', 
              bbox_to_anchor=bbox_to_anchor,#(1, 0),
              title=legend_title,
              frameon=True,
    )
    o1.get_frame().set_edgecolor((0.95,0.95,0.95))    
    ax.tick_params(axis='y', colors='k')
    from roux.viz.ax_ import format_ticklabels
    ax=format_ticklabels(ax=ax,axes=['y'])
    ax.set(ylim=(len(ax.get_yticklabels())-0.5,-0.5),
          )
    set_ylabel(ax,y=1.05)
    return ax

## volcano
def plot_volcano(dplot,
                 colindex,#='gene name test',
                 colgroup,#='condition',
                 x='difference between mean (subset1-subset2)',
                 y='P (MWU test, FDR corrected)',
                 coffs=[0.01,0.05,0.2],
                 colns=None,#f'not mean+-{2}*std',
                 ax=None,
                 filter_rows=None,
                 ylabel='significance\n(-log10(P))',
                 kws_binby_pvalue_coffs={},
                 out_df=False,
                 title=None,
                 **kws_ax):
    """
    1. create dplot
    """
    # 1
    from roux.stat.diff import binby_pvalue_coffs
    df1,df_=binby_pvalue_coffs(dplot,coffs=coffs,
                                colindex=colindex,#'gene name test',
                                 colgroup=colgroup,#'condition',
                                 colns=colns,#None,#f'not mean+-{2}*std',
                              color=True,
                              **kws_binby_pvalue_coffs)
    assert(df1[y].isnull().sum()==0)
    df1[y]=df1[y].apply(lambda x : -1*(np.log10(x)))
#     df1=df1.rename(columns={'value difference between mean (subset1-subset2)':x})
    # 2
    from roux.viz.colors import saturate_color
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
    df1.plot.scatter(x=x,y=y,c='c',
                       s=1,ax=ax)
    ax.set(ylabel=ylabel,)
    set_(ax,**kws_ax)
    df_.apply(lambda x: ax.hlines(x['y'],x['x'],ax.get_xlim()[0 if x['change']=='decrease' else 1],
                                                 colors=x['color'],
                                                 linestyles="solid",lw=1,
                                              ),axis=1)
    df_.apply(lambda x: ax.vlines(x['x'],x['y'],ax.get_ylim()[1],
                                                 colors=x['color'],
                                                 linestyles="solid",lw=1,
                                              ),axis=1)
    df_.apply(lambda x: ax.text(ax.get_xlim()[0 if x['change']=='decrease' else 1],x['y'],x['text'],
    #                                 color=saturate_color(x['color'],3),
                                    color='k',alpha=x['y alpha'],
                                    ha='left' if x['change']=='decrease' else 'right',
                                              ),axis=1)
    df_.loc[:,['y','y text','y alpha']].drop_duplicates().apply(lambda x: ax.text(ax.get_xlim()[1],x['y'],x['y text'],
                                                                   color='k',alpha=x['y alpha']),
                                                  axis=1)
    if (filter_rows is not None):
        if not all([isinstance(s,str) for s in filter_rows.values()]):
            filter_rows={k:df1.sort_values(by=x).iloc[0,:][colindex] if v==min else df1.sort_values(by=x).iloc[0,:][colindex] if v==max else None for k,v in filter_rows.items()}
        ## TODOS more than one color
#         assert(len(filter_rows.keys())==1)
        assert(all([isinstance(s,str) for s in filter_rows.values()]))
        for k in filter_rows:
            df_=df1.rd.filter_rows({k:filter_rows[k]})
            df_.groupby(k).apply(lambda df: ax.scatter(x=df_[x],
                                                                y=df_[y],
                                                                marker='o', 
                                                                facecolors='none',
                                                                edgecolors='k',
                                                                label=f"{df.name}\n(n={len(df)})",
                                                                ))
    ax.legend(loc='upper left',
             bbox_to_anchor=[1,1])
    ax.set_title(label=title,loc='left')
    if out_df:
        return ax,df1
    else:
        return ax

from roux.lib.str import linebreaker

def plot_ranking(dplot,
             x,y,colgroup,
             estimator='min',
            **kws_ax):
    dplot=dplot.rd.groupby_sort_values(
        col_groupby=y,
        col_sortby=x,
        subset=None,
        col_subset=None,
        func=estimator,
        ascending=True,)
    df2=dplot.loc[(dplot[x]==dplot[f"{x} per {y}"]),:]
    fig,ax=plt.subplots(figsize=[1.5,dplot[y].nunique()*0.3])
    ax=sns.violinplot(data=dplot,
                    x=x,y=y,
                      color='#999999',
#                     color=[0.99,0.99,0.99],
#                     alpha=0.5,
#                     jitter=False,
#                     dodge=False, 
                      width=0,cut=0,
                    ax=ax,
                    zorder=1,
                     )
    ax.grid()    
    ax_ = ax.twinx()
    ax_=sns.pointplot(data=df2,
                    x=x,y=y,
                    estimator=getattr(np,estimator),
                    join=False,
                    errwidth=0,
                     palette=df2['c'],
                    ax=ax_,
#                     zorder=100,
                    )
    ax_.set(yticklabels=[],
           xlabel=None,
           ylabel=None)
    ax_.grid(False)
    ax.set_xlim([dplot[f'{x} per {y}'].min()-0.5,0.5])    
    if estimator in ['min','max','median']:
        from roux.viz.ax_ import get_ticklabel2position
        df_=pd.Series(get_ticklabel2position(ax,'y')).to_frame('y').reset_index()
        df3=df_.merge(df2,
                 how='left',
                 left_on='index',
                 right_on=y,
                 validate="1:1",
                 )
#         df_['label']=df_['index'].map(df2.rd.to_dict([y,'condition']))
        assert(df3[colgroup].isnull().sum()==0)
        _=df3.apply(lambda x: ax.text(0.5,x['y'],'- '+x[colgroup],
                                      color=[0.3,0.3,0.3],
#                                       color=x['c'],
                                      va='center',),axis=1)
    else:
        ValueError(estimator)
    set_(ax,**kws_ax,)
    return ax
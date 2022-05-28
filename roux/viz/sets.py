from roux.global_imports import *
from roux.viz.ax_ import *
from roux.viz.annot import *

def plot_venn(
    ds1: pd.Series,
    ax: plt.Axes=None,
    figsize: tuple=[2.5,2.5],
    show_n: bool=True
    ) -> plt.Axes:
    """Plot Venn diagram.

    Args:
        ds1 (pd.Series): input vector. Subsets in the index levels, mapped to counts. 
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to [2.5,2.5].
        show_n (bool, optional): show sample sizes. Defaults to True.

    Returns:
        plt.Axes: `plt.Axes` object.
        
    Notes:
        1. Create the input pd.Series from dict.
        
            df_=to_map_binary(dict2df(d_).explode('value'),colgroupby='key',colvalue='value')
            ds_=df_.groupby(df_.columns.tolist()).size()
    """
    assert isinstance(ds1,pd.Series)
    assert ds1.dtypes=='int'
    assert len(ds1.index.names)>=2
    if ax is None:
        fig,ax=plt.subplots(figsize=figsize)
    if show_n:
        from roux.lib.df import get_totals
        d1=get_totals(ds1)
        ds1.index.names=[f"{k}\n({d1[k]})" for k in ds1.index.names]
    set_labels=list(ds1.index.names)
    if len(set_labels)==1 or len(set_labels)>3:
        logging.warning("need 2 or 3 sets")
        return 
    ds1.index=[''.join([str(int(i)) for i in list(t)]) for t in ds1.index]
    import matplotlib_venn as mv
    _=getattr(mv,f"venn{len(set_labels)}")(subsets=ds1.to_dict(),
                                            set_labels=set_labels,
                                            ax=ax
         )
    return ax

def plot_intersections(
    ds1: pd.Series,
    item_name: str=None,
    figsize: tuple=[4,4],
    text_width: float=2,
    yorder: list=None,
    sort_by: str='cardinality',
    sort_categories_by: str=None,#'cardinality',
    element_size: int=40,
    facecolor: str='gray',
    bari_annot: int=None, # 0, 'max_intersections'
    totals_bar: bool=False,
    totals_text: bool=True,                           
    intersections_ylabel: float=None,
    intersections_min: float=None,
    test: bool=False,
    annot_text: bool=False,
    set_ylabelx: float=-0.25,
    set_ylabely: float=0.5,
    **kws,
    ) -> plt.Axes:
    """Plot upset plot.

    Args:
        ds1 (pd.Series): input vector.
        item_name (str, optional): name of items. Defaults to None.
        figsize (tuple, optional): figure size. Defaults to [4,4].
        text_width (float, optional): max. width of the text. Defaults to 2.
        yorder (list, optional): order of y elements. Defaults to None.
        sort_by (str, optional): sorting method. Defaults to 'cardinality'.
        sort_categories_by (str, optional): sorting method. Defaults to None.
        element_size (int, optional): size of elements. Defaults to 40.
        facecolor (str, optional): facecolor. Defaults to 'gray'.
        bari_annot (int, optional): annotate nth bar. Defaults to None.
        totals_text (bool, optional): show totals. Defaults to True.
        intersections_ylabel (float, optional): y-label of the intersections. Defaults to None.
        intersections_min (float, optional): intersection minimum to show. Defaults to None.
        test (bool, optional): test mode. Defaults to False.
        annot_text (bool, optional): annotate text. Defaults to False.
        set_ylabelx (float, optional): x position of the ylabel. Defaults to -0.25.
        set_ylabely (float, optional): y position of the ylabel. Defaults to 0.5.

    Keyword Args:
        kws: parameters provided to the `upset.plot` function. 

    Returns:
        plt.Axes: `plt.Axes` object.

    Notes:
        sort_by:{‘cardinality’, ‘degree’}
        If ‘cardinality’, subset are listed from largest to smallest. If ‘degree’, they are listed in order of the number of categories intersected.
        sort_categories_by:{‘cardinality’, None}
        Whether to sort the categories by total cardinality, or leave them in the provided order.

    References: 
        https://upsetplot.readthedocs.io/en/stable/api.html
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


def plot_enrichment(
    dplot: pd.DataFrame,
    x: str,
    y: str,
    s: str,
    size: int=None,
    color: str=None,
    annots_side: int=5,
    coff_fdr: float=None,
    xlim: tuple=None,
    xlim_off: float=0.2,
    ylim: tuple=None,
    ax: plt.Axes=None,
    break_pt: int=25,
    annot_coff_fdr: bool=False,
    kws_annot: dict=dict(
                    loc='right',
                    # annot_count_max=5,
                    offx3=0.15,
                    ),
    **kwargs
    ) -> plt.Axes:
    """Plot enrichment stats.

    Args:
        dplot (pd.DataFrame): input data.
        x (str): x column.
        y (str): y column.
        s (str): size column.
        size (int, optional): size of the points. Defaults to None.
        color (str, optional): color of the points. Defaults to None.
        annots_side (int, optional): how many labels to show on side. Defaults to 5.
        coff_fdr (float, optional): FDR cutoff. Defaults to None.
        xlim (tuple, optional): x-axis limits. Defaults to None.
        xlim_off (float, optional): x-offset on limits. Defaults to 0.2.
        ylim (tuple, optional): y-axis limits. Defaults to None.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        break_pt (int, optional): break point ('\n') for the labels. Defaults to 25.
        annot_coff_fdr (bool, optional): show FDR cutoff. Defaults to False.
        kws_annot (dict, optional): parameters provided to the `annot_side` function. Defaults to dict( loc='right', annot_count_max=5, offx3=0.15, ).

    Keyword Args:
        kwargs: parameters provided to the `sns.scatterplot` function. 

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if coff_fdr is None: 
        coff_fdr=1
    from roux.stat.transform import log_pval
    # if y.startswith('P '):
    dplot['significance\n(-log10(Q))']=dplot[y].apply(log_pval)
    dplot['Q']=pd.cut(x=dplot[y],
                        bins=[
                            # dplot['P (FE test, FDR corrected)'].min(),
                            0,0.01,0.05,coff_fdr],
                        right=False,
                      )
    y='significance\n(-log10(Q))'
    if not size is None:
        if not dplot[size].dtype == 'category':
            dplot[size]=pd.qcut(dplot[size],
                            q=3,
                            duplicates='drop')
        dplot=dplot.sort_values(size,ascending=False)
        dplot[size]=dplot[size].apply(lambda x: f"({x.left:.0f}, {x.right:.0f}]")
    if ax is None:
        fig,ax=plt.subplots()#(figsize=[1.5,4])
    sns.scatterplot(
                    data=dplot,
                    x=x,y=y,
                    size=size if not size is None else None,
                    size_order=dplot[size].unique() if not size is None else None,
                    hue='Q',
                    # color=color,
                    zorder=2,
                    ax=ax,
                    **kwargs,
    )
    if not size is None:
        ax.legend(loc='upper left',
                  bbox_to_anchor=(1.1, 0.1),
                 # title=size,
                  frameon=True,
        #          nrow=3,
                  ncol=2,)
    if xlim is None:
        ax=set_axlims(ax,off=xlim_off,axes=['x'])
    else:
        ax.set(xlim=xlim)
    # if ylim is None:
    #     ax.set(ylim=(log_pval(coff_fdr),ax.get_ylim()[1]),
    # #               xlim=(dplot[x].min(),dplot[x].max()),
    #           )
    # else:
    #     ax.set(ylim=ylim)        
    if annot_coff_fdr:
        ax.annotate(f"Q={coff_fdr}",
            xy=(ax.get_xlim()[0],log_pval(coff_fdr)), xycoords='data',
#             xytext=(-10,log_pval(coff_fdr)), textcoords='data',
            xytext=(0.01,0.1), textcoords='figure fraction',
            ha='left',va='top',
            arrowprops=dict(arrowstyle="->", color="0.5",
#                             shrinkA=5, shrinkB=5,
#                             patchA=None, patchB=None,
                            connectionstyle="arc3,rad=-0.3",
                            ),
            )
    from roux.viz.annot import annot_side
    ax=annot_side(
        ax=ax,
        df1=dplot.sort_values(y,ascending=False).head(annots_side),
        colx=x,
        coly=y,
        cols=s,
        break_pt=break_pt,
        offymin=0.1 if not size is None else 0,
        zorder=3,
        **kws_annot,
        )
    return ax


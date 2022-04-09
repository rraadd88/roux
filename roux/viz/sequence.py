from roux.global_imports import *   

# sequence
def plot_domain(
    d1: dict,
    x: float=1,
    xoff: float=0,
    y: float=1,
    height: float=0.8,
    ax: plt.Axes=None,
    **kws,
    ) -> plt.Axes:
    """Plot protein domain.

    Args:
        d1 (dict): plotting data including intervals.
        x (float, optional): x position. Defaults to 1.
        xoff (float, optional): x-offset. Defaults to 0.
        y (float, optional): y position. Defaults to 1.
        height (float, optional): height. Defaults to 0.8.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
    ## plot (start -> domain start) and (domain end -> end) separately
#     ax.plot([x+xoff,d1['start']+xoff],[y,y],color='k',zorder=1)
#     ax.plot([d1['end']+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
#     print(d1)
#     print([x+xoff,d1['start']+xoff])
#     print([d1['end']+xoff,d1['protein length']+xoff])

    ax.plot([x+xoff,d1['protein length']+xoff],[y,y],color='k',zorder=1)
    if pd.isnull(d1['type']):
        return ax
    patches=[]
    import matplotlib.patches as mpatches
    width=d1['end']-d1['start']
    patches.append(mpatches.Rectangle(
        xy=(d1['start']+xoff,
            y-(height*0.5)),
        width=width,
        height=height,
#         color=color,
#         mutation_scale=10,
#     #     mutation_aspect=1.5,
        joinstyle='round',
#         fc='none',
        zorder=2,
        **kws,
    ))
    _=[ax.add_patch(p) for p in patches]
    return ax

def plot_protein(
    df: pd.DataFrame,
    ax: plt.Axes=None,
    label: str=None,
    alignby: str=None,
    test: bool=False,
    **kws
    ) -> plt.Axes:
    """Plot protein.

    Args:
        df (pd.DataFrame): input data.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        label (str, optional): proein name. Defaults to None.
        alignby (str, optional): align proteins by this domain. Defaults to None.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        plt.Axes: `plt.Axes` object.
    """
    if ax is None:
        fig,ax=plt.subplots(figsize=[3,3])
    if len(df)==1 and pd.isnull(df.iloc[0,:]['type']):
        alignby=None
    if not alignby is None:
        if alignby in df['description'].tolist():
            xoff=df.loc[(df['description']==alignby),:].iloc[0,:]['start']*-1
        else:
            xoff=0
    else:
        xoff=df.sort_values(by=['start']).iloc[0,:]['start']*-1
    _=df.apply(lambda x: plot_domain(d1=x,y=x['y'],
                                        ax=ax,
                                        xoff=xoff,
                                        color=None if not 'color' in x else x['color'],
                                     label=x['description'],
                                        **kws),axis=1)        
    if not label is None:
        ax.text(-10+xoff,df['y'].tolist()[0],label,ha='right',va='center')
#         ax.text(0,df['y'].tolist()[0],label,ha='left',va='bottom')
    return ax

def plot_gene(
    df1: pd.DataFrame,
    label: str=None,
    kws_plot: dict={},
    test: bool=False,
    outd: str=None,
    ax: plt.Axes=None,
    off_figw: float=1,
    off_figh: float=1,
    #**kws_plot_protein
    ) -> plt.Axes:
    """Plot genes.

    Args:
        df1 (pd.DataFrame): input data.
        label (str, optional): label to show. Defaults to None.
        kws_plot (dict, optional): parameters provided to the `plot` function. Defaults to {}.
        test (bool, optional): test mode. Defaults to False.
        outd (str, optional): output directory. Defaults to None.
        ax (plt.Axes, optional): `plt.Axes` object. Defaults to None.
        off_figw (float, optional): width offset. Defaults to 1.
        off_figh (float, optional): height offset. Defaults to 1.

    Returns:
        plt.Axes: `plt.Axes` object.
    """

    if hasattr(df1,'name'): 
        geneid=df1.name
    else:
        geneid=df1.iloc[0,:]['gene id']
    if 'title' in df1:
        kws_plot['title']=df1.iloc[0,:]['title']
    elif 'gene symbol' in df1:
        kws_plot['title']=df1.iloc[0,:]['gene symbol']
    else:
        kws_plot['title']=geneid
    if ax is None:
        figsize=[((df1['protein length'].max()-75)/250)*off_figw,
                                 ((df1['protein id'].nunique()+2)*0.3)*off_figh]
        fig,ax=plt.subplots(figsize=figsize)
    if label=='index':
        df1['yticklabel']=df1['y']*-1
    if df1['description'].isnull().all():
        alignby=None
    else:
        alignby=df1['description'].value_counts().index[0]
    _=df1.groupby('protein id',
                 sort=False).apply(lambda df1: plot_protein(df1,
                                                           label=df1.iloc[0,:]['yticklabel'] if (('yticklabel' in df1) and (not label is None)) else df1.name if (not label is None) else label,
                                                           alignby=alignby,
                                                           ax=ax,
                                                           test=False))
    if not test:ax.axis('off')
    ax.set(**{k:v for k,v in kws_plot.items() if not k=='title'},)
    set_label(0.05,1,f"${kws_plot['title']}$",ax=ax,va='bottom')
    return ax

def plot_genes_legend(df: pd.DataFrame,d1: dict):
    """Make the legends for the genes.

    Args:
        df (pd.DataFrame): input data.
        d1 (dict): plotting data.
    """
    fig,ax=plt.subplots()
#     d1=df.set_index('description')['color'].dropna().drop_duplicates().to_dict()
    import matplotlib.patches as mpatches
    l1=[mpatches.Patch(color=d1[k], label=k) for k in d1]        
    savelegend(f"plot/schem_gene_{' '.join(df['gene id'].unique()[:5]).replace('.',' ')}_legend.png",
           legend=plt.legend(handles=l1,frameon=True,title='domains/regions'),
           )
#     df.loc[df['color'].isnull(),'color']=(0,0,0,1)
    df['color']=df['color'].fillna('k').apply(str)
    # to_table(df,f"plot/schem_gene_{' '.join(df['gene id'].unique()[:5]).replace('.',' ')}_legend.pqt")

from roux.query.ensembl import to_protein_seq,to_domains
def plot_genes_data(
    df1: pd.DataFrame,
    release: int,
    species: str,
    custom: bool=False,
    colsort: str=None,
    cmap: str='Spectral',
    fast: bool=False
    ) -> tuple:
    """Plot gene-wise data.

    Args:
        df1 (pd.DataFrame): input data.
        release (int): Ensembl release.
        species (str): species name.
        custom (bool, optional): customised. Defaults to False.
        colsort (str, optional): column to sort by. Defaults to None.
        cmap (str, optional): colormap. Defaults to 'Spectral'.
        fast (bool, optional): parallel processing. Defaults to False.

    Returns:
        tuple: (dataframe, dictionary)
    """
    species=species.lower().replace(' ','_')
    if not custom:
        from pyensembl import EnsemblRelease
        ensembl=EnsemblRelease(release=release,species=species)
        if not 'protein length' in df1:
            df1['protein length']=df1['protein id'].progress_apply(lambda x: to_prtseq(x,ensembl,length=True))
        from roux.query.ensembl import gid2gname
        df1['gene symbol']=df1['gene id'].apply(lambda x: gid2gname(x,ensembl))
        # getattr(df1,'parallel_apply' if fast else "apply")
        df2=getattr(df1.groupby(['gene id','protein id']),("progress" if not (fast or len(df1<20)) else 'parallel')+"_apply")(lambda df: proteinid2domains(get_name(df,'protein id'),
        species=species,release=release)).reset_index().rd.clean()
        df2=df2.log.dropna(subset=['description'])
        df2['description']=df2['description'].replace('',np.nan).fillna(df2['id'])
        if len(df2)!=0:
            df2=df1.merge(df2,
                         on=['gene id','protein id'],
                         how='left',
                         )
        else:
            df2=df1.copy()
            for c in ['description','type']: 
                df2[c]=np.nan
#     df2=df2.log.dropna(subset=['type'])
    else:
        df2=df1.copy()

    df2['start']=df2.apply(lambda x: 1 if pd.isnull(x['type']) else x['start'],axis=1)
    df2['end']=df2.apply(lambda x: x['protein length'] if pd.isnull(x['type']) else x['end'],axis=1)
    #     print(df2.columns)
    df2['domains']=df2['protein id'].map(df2.dropna(subset=['description']).groupby('protein id').size().to_dict())
    df2['domains']=df2['domains'].fillna(0)
    if colsort is None:
        df2=df2.sort_values(['gene id','domains','protein length','start'],ascending=[True,False,False,True])
    else:
        df2=df2.sort_values(['gene id',colsort],ascending=[True,True])        
    def gety(df):
        df['y']=(df.groupby('protein id',sort=False).ngroup()+1)*-1
        return df
    df2=df2.groupby('gene id',as_index=False).apply(gety)
#     df=df.sort_values(by=['domains','protein length'],ascending=[False,False])        
    
    from roux.viz.colors import get_ncolors
    cs=get_ncolors(n=df2['description'].nunique(),
                   cmap=cmap, ceil=False,
                   vmin=0.2,vmax=1)
    d1=dict(zip(df2['description'].dropna().unique(),
                cs
               ))
    df2['color']=df2['description'].map(d1)
    return df2,d1
def plot_genes(df1,
    custom=False,
    colsort=None,
    release=100,
    cmap='Spectral',
    **kws_plot_gene
    ):
    """Plot many genes.

    Args:
        df1 (pd.DataFrame): input data.
        release (int): Ensembl release.
        custom (bool, optional): customised. Defaults to False.
        colsort (str, optional): column to sort by. Defaults to None.
        cmap (str, optional): colormap. Defaults to 'Spectral'.
    
    Keyword Args:
        kws_plot_gene: parameters provided to the `plot_genes_data` function. 
    
    Returns:
        tuple: (dataframe, dictionary)
    """
    df2,d1=plot_genes_data(df1,custom=custom,colsort=colsort,
                           release=release,
                           cmap=cmap)
    axs=df2.groupby('gene id').apply(plot_gene,**kws_plot_gene)
    plot_genes_legend(df2,d1)
    return axs,df2    
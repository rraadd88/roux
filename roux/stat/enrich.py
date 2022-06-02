from roux.global_imports import *

# set enrichment
def get_enrichment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    background: int,
    colid: str='gene id',
    colref: str='gene set id',
    colrefname: str='gene set name',
    colreftype: str='gene set type',
    colrank: str='rank',
    outd: str=None,
    name: str=None,
    # enrichr
    cutoff: float=0.05, # only used for plotting. 
    # prerank
    permutation_num: int=1000,
    # etc
    verbose: bool=False,
    no_plot: bool=True,
    **kws_prerank,
    ):
    """Get enrichments between sets.

    Args:
        df1 (pd.DataFrame): test data.
        df2 (pd.DataFrame): reference set data.
        background (int): background size.
        colid (str, optional): column containing unique ids of the elements. Defaults to 'gene id'.
        colref (str, optional): column containing the unique ids of the sets. Defaults to 'gene set id'.
        colrefname (str, optional): column containing names of the sets. Defaults to 'gene set name'.
        colreftype (str, optional): column containing the type/group name of the sets. Defaults to 'gene set type'.
        colrank (str, optional): column containing the ranks. Defaults to 'rank'.
        outd (str, optional): output directory path. Defaults to None.
        name (str, optional): name of the result. Defaults to None.
        cutoff (float, optional): p-value cutoff. Defaults to 0.05.
        verbose (bool, optional): verbose. Defaults to False.
        no_plot (bool, optional): do no plot. Defaults to True.

    Returns:
        pd.DataFrame: if rank -> high rank first within the leading edge gene ids.
        
    Notes:
        1. Unique ids are provided as inputs.
    """
    import gseapy as gp
    name=get_name(df2,colreftype)
    
    import tempfile
    with tempfile.TemporaryDirectory() as p:
        outd=outd if not outd is None else p
        o1 = gp.enrichr(
            gene_list=df1[colid].unique().tolist(),
             # or gene_list=glist
             description=name,
             gene_sets=df2.rd.to_dict([colref,colid]),
             background=background, # or the number of genes, e.g 20000
             outdir=to_path(f'{outd}/{name}'),
             cutoff=cutoff, # only used for plotting.
             verbose=verbose,
             no_plot=no_plot,
             # **kws,
             )
        df3=o1.results
    if len(df3)==0:
        logging.error('aborting because no enrichments found.')
        return
    df3=df3.rename(columns={'Term':colref,
                             'P-value':'P (FE test)',
                             'Adjusted P-value':'P (FE test, FDR corrected)',
                             'Genes':f'{colid}s overlap'},errors='raise')
    df_=df3['Overlap'].str.split('/',expand=True).rename(columns={0:f"{colid}s overlap size",
                                                         1:f"{colid}s per {colref}"}).applymap(int)
    df3=df3.join(df_)
#     df3['overlap %']=df3['Overlap'].apply(eval)*100    
    df3['overlap %']=df3.apply(lambda x: (x[f"{colid}s overlap size"]/x[f"{colid}s per {colref}"])*100,axis=1)
    df3=df3.drop(['Gene_set','Overlap',],axis=1)
    info('enrichr: '+perc_label(sum(df3['P (FE test, FDR corrected)']<=0.05),len(df3)))
    if not colrank in df1:
        if colrefname in df2:
            df3[colrefname]=df3[colref].map(df2.set_index(colref)[colrefname].drop_duplicates())
        return df3
    if df3[f"{colid}s overlap size"].max()<2:
        logging.error("df3[f'{colid}s overlap size'].max()<2 # can not run prerank")
        return df3
    with tempfile.TemporaryDirectory() as p:
        outd=outd if not outd is None else p
        o2 = gp.prerank(rnk=df1.loc[:,[colid,colrank]],
                         gene_sets=df2.rd.to_dict([colref,colid]),
                         min_size=2,
                         max_size=df3[f"{colid}s overlap size"].max(),
                         processes=1,
                         permutation_num=permutation_num, # reduce number to speed up testing
                         ascending=False, # look for high number 
                         outdir=to_path(f'{outd}/{name}'),
                         pheno_pos='high',
                         pheno_neg='low',
                         format='png',
                        no_plot=no_plot,
                        seed=1,
                        graph_num=1,
                        **kws_prerank,
                       )
        to_dict({'prerank':o2},
               f'{outd}/{name}/prerank.joblib')
        df4=o2.res2d.reset_index()
#     es	nes	pval	fdr	geneset_size	matched_size	genes	ledge_genes
    df4=df4.rename(columns={
        'Term':colref,
        'es':'enrichment score',
        'nes':'normalized enrichment score', 
        'pval':'P (GSEA test)',
        'fdr':'FDR (GSEA test)',
        'matched_size':f"{colid}s overlap size",
        'geneset_size':f"{colid}s per {colref}",
        'ledge_genes':f'{colid}s leading edge',
        },
        errors='raise')
    df4['overlap %']=df4.apply(lambda x: (x[f"{colid}s overlap size"]/x[f"{colid}s per {colref}"])*100,axis=1)
    df4=df4.drop(['genes',],axis=1)
    info('preraked: '+perc_label(sum(df4['P (GSEA test)']<=0.05),len(df4)))
    
    df5=df3.merge(df4,
              on='gene set id',
              how='left',
             validate="1:1",
                 suffixes=['',' (prerank)'])
    df5=df5.drop([
#                 f'{colid}s',
                  f'{colid}s per {colref} (prerank)',f'{colid}s overlap size (prerank)','overlap % (prerank)'],axis=1)
    return df5

def get_enrichments(df1: pd.DataFrame,
                    df2: pd.DataFrame,
                    background: int,
                    coltest: str='subset',
                    colid: str='gene id',
                    colref: str='gene set id',
                    colreftype: str='gene set type',
                    fast: bool=False,
                    **kws) -> pd.DataFrame:
    """Get enrichments between sets, iterate over types/groups of test elements e.g. upregulated and downregulated genes.

    Args:
        df1 (pd.DataFrame): test data.
        df2 (pd.DataFrame): reference set data.
        background (int): background size.
        colid (str, optional): column containing unique ids of the elements. Defaults to 'gene id'.
        colref (str, optional): column containing the unique ids of the sets. Defaults to 'gene set id'.
        colrefname (str, optional): column containing names of the sets. Defaults to 'gene set name'.
        colreftype (str, optional): column containing the type/group name of the sets. Defaults to 'gene set type'.
        fast (bool, optional): parallel processing. Defaults to False.

    Returns:
        pd.DataFrame: output.
    """

    return getattr(df1.groupby(coltest),'progress_apply' if not fast else 'parallel_apply')(lambda df_: df2.groupby(colreftype).apply(lambda df: get_enrichment(df_,df,background=background,**kws)).reset_index(0)).reset_index(0)
    
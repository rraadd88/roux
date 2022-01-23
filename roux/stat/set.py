from roux.global_imports import *

# set enrichment
def get_intersection_stats(df,coltest,colset,background_size=None):
    from roux.stat.binary import compare_bools_jaccard
    from scipy.stats import hypergeom,fisher_exact
    """
    :param background: size of the union (int)
    """
    hypergeom_p=hypergeom.sf(sum(df[coltest] & df[colset])-1,
                             len(df) if background_size is None else background_size,
                             df[colset].sum(),
                             df[coltest].sum(),)
    contigency=[[sum(df[coltest] & df[colset]),sum(df[coltest] & ~df[colset])],[sum(~df[coltest] & df[colset]),sum(~df[coltest] & ~df[colset])],]
    odds_ratio,fisher_exactp=fisher_exact(contigency,alternative='two-sided')
    jaccard=compare_bools_jaccard(df[coltest],df[colset])
    return hypergeom_p,fisher_exactp if jaccard!=0 else 1,odds_ratio,jaccard

def get_set_enrichment_stats(test,sets,background,fdr_correct=True):
    """
    test:
        get_set_enrichment_stats(background=range(120),
                        test=range(100),
                        sets={f"set {i}":list(np.unique(np.random.randint(low=100,size=i+1))) for i in range(100)})
        # background is int
        get_set_enrichment_stats(background=110,
                        test=unique(range(100)),
                        sets={f"set {i}":unique(np.random.randint(low=140,size=i+1)) for i in range(0,140,10)})                        
    """
    if isinstance(background,list):
        background_elements=np.unique(background)
        background_size=None
    elif isinstance(background,(int,float)):
        background_elements=list2union([test]+list(sets.values()))
        background_size=background
        if len(background_elements)>background_size:
            logging.error(f"invalid data type of background {type(background)}")
    else:
        logging.error(f"invalid data type of background {type(background)}")
    delement=pd.DataFrame(index=background_elements)
    delement.loc[np.unique(test),'test']=True
    for k in sets:
        delement.loc[np.unique(sets[k]),k]=True
    delement=delement.fillna(False)
    dmetric=pd.DataFrame({colset:get_intersection_stats(delement,'test',colset,background_size=background_size) for colset in delement if colset!='test'}).T.rename(columns=dict(zip([0,1,2,3],['hypergeom p-val','fisher_exact p-val','fisher_exact odds-ratio','jaccard index'])))
    if fdr_correct:
        from statsmodels.stats.multitest import multipletests
        for c in dmetric:
            if c.endswith(' p-val'):
                dmetric[f"{c} corrected"]=multipletests(dmetric[c], alpha=0.05, method='fdr_bh',
                                                        is_sorted=False,returnsorted=False)[1]
    return dmetric

def test_set_enrichment(tests_set2elements,test2_set2elements,background_size):
    from tqdm import tqdm
    from roux.lib.set import list2union
    dn2df={}
    for test1n in tqdm(tests_set2elements):
        for test2n in test2_set2elements:
            if len(tests_set2elements[test1n])!=0:
                dn2df[(test1n,test2n)]=get_set_enrichment_stats(test=tests_set2elements[test1n],
                                         sets={test2n:test2_set2elements[test2n]},
                                         background=background_size,
                                        fdr_correct=True,
                                        )
    denrich=pd.concat(dn2df,axis=0,names=['difference','test2 set'])
    return denrich

def get_paired_sets_stats(l1,l2):
    """
    overlap, intersection, union, ratio
    """
    if all([isinstance(l, list) for l  in [l1,l2]]):
        l=list(jaccard_index(l1,l2))
        l.append(get_ratio_sorted(len(l1),len(l2)))
        return l

def get_enrichment(df1,df2,
                   background,
           colid='gene id',
           colref='gene set id',
           colrefname='gene set name',
           colreftype='gene set type',
           colrank='rank',
           outd=None,
           name=None,
            # enrichr
           cutoff=0.05, # only used for plotting. 
            # prerank
           permutation_num=1000,
            # etc
            verbose=False,no_plot=True,
           **kws_prerank,
                  ):
    """
    
    :return leading edge gene ids: high rank first
    """
    
    import gseapy as gp
    name=get_name(df2,colreftype)
    
    import tempfile
    with tempfile.TemporaryDirectory() as p:
        outd=outd if not outd is None else p
        o1 = gp.enrichr(gene_list=df1[colid].unique().tolist(),
                         # or gene_list=glist
                         description=name,
                         gene_sets=df2.rd.to_dict([colref,colid]),
                         background=background, # or the number of genes, e.g 20000
                         outdir=f'{outd}/{name}',
                         cutoff=cutoff, # only used for plotting.
                         verbose=verbose,
                         no_plot=no_plot,
    #                      **kws,
                         )
        df3=o1.results
    df3=df3.rename(columns={'Term':colref,
                             'P-value':'P (FE test)',
                             'Adjusted P-value':'P (FE test, FDR corrected)',
                             'Genes':f'{colid}s'},errors='raise')
    df_=df3['Overlap'].str.split('/',expand=True).rename(columns={0:f"{colid}s overlap",
                                                         1:f"{colid}s per {colref}"}).applymap(int)
    df3=df3.join(df_)
#     df3['overlap %']=df3['Overlap'].apply(eval)*100    
    df3['overlap %']=df3.apply(lambda x: (x[f"{colid}s overlap"]/x[f"{colid}s per {colref}"])*100,axis=1)
    df3=df3.drop(['Gene_set','Overlap',],axis=1)
    info('enrichr: '+perc_label(sum(df3['P (FE test, FDR corrected)']<=0.05),len(df3)))
    if not colrank in df1:
        if colrefname in df2:
            df3[colrefname]=df3[colref].map(df2.set_index(colref)[colrefname].drop_duplicates())
        return df3
    if df3[f"{colid}s overlap"].max()<2:
        logging.error("df3[f'{colid}s overlap'].max()<2 # can not run prerank")
        return df3
    with tempfile.TemporaryDirectory() as p:
        outd=outd if not outd is None else p    
        o2 = gp.prerank(rnk=df1.loc[:,[colid,colrank]],
                         gene_sets=df2.rd.to_dict([colref,colid]),
                         min_size=2,
                         max_size=df3[f"{colid}s overlap"].max(),
                         processes=1,
                         permutation_num=permutation_num, # reduce number to speed up testing
                         ascending=False, # look for high number 
                         outdir=f'{outd}/{name}',
                         pheno_pos='high',
                         pheno_neg='low',
                         format='png',
                        no_plot=no_plot,
                        seed=1,
                        graph_num=1,
                        **kws_prerank,
                       )
        df4=o2.res2d.reset_index()
#     es	nes	pval	fdr	geneset_size	matched_size	genes	ledge_genes
    df4=df4.rename(columns={'Term':colref,
                            'es':'enrichment score',
                            'nes':'normalized enrichment score', 
                             'pval':'P (GSEA test)',
                            'fdr':'FDR (GSEA test)',
                            'matched_size':f"{colid}s overlap",
                            'geneset_size':f"{colid}s per {colref}",
                             'ledge_genes':f'{colid}s leading edge',
                           },
                   errors='raise')
    df4['overlap %']=df4.apply(lambda x: (x[f"{colid}s overlap"]/x[f"{colid}s per {colref}"])*100,axis=1)
    df4=df4.drop(['genes',],axis=1)
    info('preraked: '+perc_label(sum(df4['P (GSEA test)']<=0.05),len(df4)))
    
    df5=df3.merge(df4,
              on='gene set id',
              how='left',
             validate="1:1",
                 suffixes=['',' (prerank)'])
    df5=df5.drop([
#                 f'{colid}s',
                  f'{colid}s per {colref} (prerank)',f'{colid}s overlap (prerank)','overlap % (prerank)'],axis=1)
    return df5

def get_enrichments(df1,
                    df2,
                    background,
                    coltest='subset',
                    colid='gene id',
                    colref='gene set id',
                    colreftype='gene set type',
                    fast=False,
                    **kws):
    """
    :param df1: test sets
    :param df2: reference sets
    """
    return getattr(df1.groupby(coltest),'progress_apply' if not fast else 'parallel_apply')(lambda df_: df2.groupby(colreftype).apply(lambda df: get_enrichment(df_,df,background=background,**kws)).reset_index(0)).reset_index(0)
    
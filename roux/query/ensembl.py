### Ensembl
# for human genome
# release 77 uses human reference genome GRCh38
# from pyensembl import EnsemblRelease
# EnsemblRelease(release=100)
# for many other species
# ensembl = pyensembl.EnsemblRelease(species=pyensembl.species.Species.register(
# latin_name='saccharomyces_cerevisiae',
# synonyms=['saccharomyces_cerevisiae'],
# reference_assemblies={
#     'R64-1-1': (92, 92),
# }),release=92)
from roux.global_imports import *
import numpy as np
import pandas as pd
import logging
import requests, sys

#pyensembl faster
def gid2gname(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        return ensembl.gene_name_of_gene_id(id)
    except:
        return np.nan

def gname2gid(name,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        names=ensembl.gene_ids_of_gene_name(name)
        if len(names)>1:
            logging.warning('more than one ids')
            return '; '.join(names)
        else:
            return names[0]
    except:
        return np.nan
    
def tid2pid(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_id
    except:
        return np.nan    
    
def tid2gid(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(id)
        return t.gene_id
    except:
        return np.nan 
    
def pid2tid(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        return ensembl.transcript_id_of_protein_id(id)
    except:
        return np.nan    
    
def gid2dnaseq(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        g=ensembl.gene_by_id(id)
        ts=g.transcripts
        lens=[len(t.protein_sequence) if not t.protein_sequence is None else 0 for t in ts]
        return ts[lens.index(max(lens))].id, ts[lens.index(max(lens))].protein_sequence
    except:
        return np.nan,np.nan    
    
def gid2pid_longest(gid,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    g=ensembl.gene_by_id(gid)
    return pd.Series({t.protein_id:t.protein_sequence for t in g.transcripts}).dropna().apply(len).sort_values().tail(1).index.to_list()[0]

def tid2prtseq(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(id)
        return t.protein_sequence
    except:
        return np.nan
def pid2prtseq(id,ensembl,
               length=False):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.protein_sequence(id)
        if not length:
            return t
        else:
            return len(t)            
    except:
        return np.nan    
    
def tid2cdsseq(id,ensembl):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(id)
        return t.coding_sequence
    except:
        return np.nan 
    
def get_utr_sequence(ensembl,x,loc='five'):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_protein_id(x)
        return getattr(t,f'{loc}_prime_utr_sequence')
    except: 
        logging.warning(f"{x}: no sequence found")
        return     
    
def pid2tid(protein_id,ensembl):    
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    if (protein_id in ensembl.protein_ids() and (not pd.isnull(protein_id))):
        return ensembl.transcript_by_protein_id(protein_id).transcript_id
    else:
        return np.nan    

def is_protein_coding(x,ensembl,geneid=True):
    """
        from pyensembl import EnsemblRelease
        ensembl EnsemblRelease(release=100)    
    """
    try:
        if geneid:
            g=ensembl.gene_by_id(x)
        else:
            g=ensembl.transcript_by_id(x)
    except:
        logging.error('gene id not found')
        return 
    return g.is_protein_coding

#restful api    
def rest(ids,function='lookup',
                 target_taxon='9606',
                 release='100',
                 format_='full',
                 test=False,
                 **kws):
    import requests, sys

    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/{function}/id"
    headers={ "Content-Type" : "application/json", "Accept" : "application/json"}
    
    headers['target_taxon']=target_taxon
    headers['release']=release
    headers['format']=format_
    headers.update(kws)
    if test: print(headers)
        
    if isinstance(ids,str):
        r = requests.get(server+ext+f'{ids}?', headers=headers)
    elif isinstance(ids,list):
        r = requests.post(server+ext, headers=headers, 
                          data='{ "ids" : ['+', '.join([f'"{s}"' for s in ids])+' ] }')
    else:
        raise ValueError(f"ids should be str or list")
    if not r.ok:
        r.raise_for_status()
    else:
        decoded = r.json()
    #     print(repr(decoded))
        return decoded

def geneid2homology(x='ENSG00000148584',
                    release=100,
                    homologytype='orthologues',
                   outd='data/database',
                   force=False):
    """
    Gene id to homology.
    
        outp='data/database/'+replacemany(p.split(';content-type')[0],{'https://':'','?':'/',';':'/'})+'.json'
        
    Ref: f"https://e{release}.rest.ensembl.org/documentation/info/homology_ensemblgene
    """
    p=f"https://e{release}.rest.ensembl.org/homology/id/{x}?type={homologytype};compara=vertebrates;sequence=none;cigar_line=0;content-type=application/json;format=full"
    outp=outp=f"{outd}/{p.replace('https://','')}.json"
    if exists(outp) and not force:
        return read_dict(outp)
    else:
        d1=read_dict(p)
        to_dict(d1,outp)
    return d1

def proteinid2domains(x,
                    release,
                     outd='data/database',
                     force=False):
    """
    """
    p=f'https://e{release}.rest.ensembl.org/overlap/translation/{x}?content-type=application/json;species=homo_sapiens;feature=protein_feature;type=pfam'
    outp=outp=f"{outd}/{p.replace('https://','')}.json"
    if exists(outp) and not force:
        d1=read_dict(outp)
    else:
        d1=read_dict(p)
        to_dict(d1,outp)
    if d1 is None: return
    if len(d1)==0:
        logging.error(f"{x}: no domains/regions found")
        return
    #d1 is a list
    return pd.concat([pd.DataFrame(pd.Series(d)).T for d in d1],
                     axis=0)
pid2domains=proteinid2domains

## species
def taxid2name(k):
    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/taxonomy/id/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    decoded = r.json()
    return decoded['scientific_name']

def taxname2id(k):
    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/taxonomy/name/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok or r.status_code==400:
        logging.warning(f'no tax id found for {k}')
        return 
    decoded = r.json()
    if len(decoded)!=0:
        return decoded[0]['id']
    else:
        logging.warning(f'no tax id found for {k}')
        return
    
## convert between assemblies    
def convert_coords_human_assemblies(release,chrom,start,end,
                                    frm=38,to=37,
                                    test=False,
                                   force=False):
    import requests, sys,yaml 
    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/map/human/GRCh{frm}/{chrom}:{start}..{end}:1/GRCh{to}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if test: logging.info(r.url)
    if not r.ok:
        if not force:
            r.raise_for_status()
        else:
            logging.info(f"not ok: {r.url}")
    return yaml.safe_load('\n'.join(r.text.split('\n')[3:-1]))
#     decoded = r.json()
#     d=eval(repr(decoded))
#     if 'mappings' in d:
#         for d_ in d['mappings']:
#             if 'mapped' in d_:
# #                 return d_['mapped']['seq_region_name'],d_['mapped']['start'],d_['mapped']['end']
#                 return pd.Series(d_['mapped'])#['seq_region_name'],d_['mapped']['start'],d_['mapped']['end']
    
## convert coords 
def coords2geneid(x,
                 biotype='protein_coding'):
    # x=df02.iloc[0,:]
    from pyensembl import EnsemblRelease
    ensembl=EnsemblRelease(release=100)
    contig,pos=x['Genome Location'].split(':')

    start,end=[int(s) for s in pos.split('-')]

    # l1=ensembl.gene_ids_at_locus
    l1=ensembl.genes_at_locus(contig=contig,
                              position=start, 
                              end=end, 
                              strand=None)

    # def range_overlap(l1,l2):
    #     return set.intersection(set(range(l1[0],l1[1]+1,1)),
    #                             set(range(l2[0],l2[1]+1,1)))
#     ds1=pd.Series({ 
    d1={}
    for g in l1:
        if g.biotype==biotype:
            d1[g.gene_id]=len(range_overlap([g.start,g.end],[start,end]))
    ds1=pd.Series(d1).sort_values(ascending=False)
    print(ds1)
    return ds1.index[0]


def map_id(df1,gene_id,release,release_to,out='df',
            test=False):
    """
    gene_id='ENSG00000187990',release=75,release_to=100,
    
    ## m:m mappings are possible
    ## e.g. https://useast.ensembl.org/Homo_sapiens/Gene/Idhistory?db=core;g=ENSG00000276410;r=6:26043227-26043713;t=ENST00000615966
    """        
    def get_release(df2,release,which,
                    col='Release',
                    test=False):
        # try:
        if test: print(df2)
        df1_=df2.loc[(df2[col]<=release),:]
        if test: print(df1_)
        if len(df1_)!=0:
            df_=df1_.tail(1)
        elif which=='old':
            df_=df2.head(1)
        elif which=='new':
            df_=df2.tail(1)
        if test: print(df_)
        return df_.iloc[-1,:][col]
        
    df2=df1.loc[df1['Old stable ID'].str.startswith(gene_id),:].sort_values('Release')
    if len(df2)==0:
        # mostly new genes that are uncharacterzed in the old release 
        # e.g. https://uswest.ensembl.org/Homo_sapiens/Gene/Idhistory?db=core;g=ENSG00000272104;r=3:50350892-50367923;t=ENST00000606589
        return [gene_id]
    df2_=df2.loc[(df2['Release']==get_release(df2,release=release,which='old',test=test)),['Old stable ID','Release']].drop_duplicates()
    if test: print(df2_)
    # try:
    assert(len(df2_)==1)
    df3=df2.loc[(df2['Release']==get_release(df2,release=release_to,which='new',test=test)),:].drop(['Old stable ID'],axis=1)
    df3['id']=df3['New stable ID'].str.split('.',expand=True)[0]
    df3['id']=df3['id'].replace('<retired>',np.nan)
    if out=='list':
        return df3['id'].dropna().unique().tolist()
    elif out=='df':
        ## more info
        df3['old id.version']=df2_.iloc[0,:]['Old stable ID']
        df3['old id updated on release']=df2_.iloc[0,:]['Release']
        df4=df3.rename(columns={'New stable ID':'id.version',
                           'Release':'id updated on release',
                           },errors='raise').rd.lower_columns()
        return df4
    # except:
    #     print(df2)
    #     return 'error'

def read_idmapper_output(outp):
    from pathlib import Path
    file = Path(outp)
    file.write_text(file.read_text().replace('.Old stable ID', 'Old stable ID'))
    df01=pd.read_csv(outp+'_',sep=', ')
    df1=df01.log('Old stable ID').loc[(df01['Old stable ID']!='Old stable ID'),:].log('Old stable ID')
    assert(df1['Old stable ID'].str.startswith('ENSG').all())
    df1['Release']=df1['Release'].astype(float)
    return df1

def map_ids_(ids,df00,release,release_to):
    df0=pd.DataFrame(ids,columns=[f'id {release}'])
    df0['id']=df0[f'id {release}'].parallel_apply(lambda x: map_id(df00,gene_id=x,
                                                   release=release,
                                                   release_to=release_to,
                                                   out='list'))
    assert(not df0['id'].isnull().any())
    # reomve retired ids
    df2=df0.log().explode('id').log.dropna(subset=['id'])
    return df2

def map_ids(srcp,
        dbp,
        ids,
        release=75,
        release_to=100,
           species='human',
           test=False):
    """
    srcp='deps/ensembl-tools/scripts/id_history_converter/IDmapper.pl',
    dbp='data/database/ensembl_id_history_converter/db.pqt',
    ids=ensembl.gene_ids(),    
    """
    from roux.query.ensembl import map_ids_,read_idmapper_output
    if exists(dbp):
        df00=read_table(dbp)
        df1=map_ids_(ids,df00,release,release_to)
        if test:print(df1.columns.tolist())
        ids_=list(set(ids) - set(df1[f'id {release}'].tolist()))
    else:
        logging.warning(f"database not found: {dbp}")
        ids_=ids
    if len(ids_)==0:
        logging.warning(f"ids not found in database, will convert: {len(ids_)}")
        return df1
    #run idmapper on the remaining
    outp=abspath('test/ids.out')
    import os
    if exists(outp):os.remove(outp)
    runbash_tmp(f"cd {dirname(srcp)};./{basename(srcp)} -s {species} -f INPUT > OUTPUT",
                env='ensembl',
                df1=ids_,
                input_type='list',
                inp='INPUT',outp=outp,
                force=True,
                test=test,
                # **kws,
               )
    if exists(outp):
        if test:info(outp)
        df2=read_idmapper_output(outp)
        if exists(dbp):
            df4=df00.log().append(df2).log()
            logging.warning(f"database updated: {dbp}")
        else:
            df4=df2.copy()
        to_table(df4,dbp)
        df3=map_ids_(ids_,df2,release,release_to)
    else:
        logging.warning("new ids could not be converted")
        if 'df1' in locals():
            df3=df1.copy()
        else:
            return
    return df3

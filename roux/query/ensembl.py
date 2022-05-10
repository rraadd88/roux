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
from roux.lib.df import *
import numpy as np
import pandas as pd
import logging
import requests, sys

release2prefix={100:'apr2020.archive',
                101:'aug2020.archive',
                102:'nov2020.archive',
                103:'feb2021.archive',
                104:'may2021.archive',
                 93:'jul2018.archive',
                 75:'feb2014.archive',
                 'grch37':'grch37',
                }
## Ref: https://m.ensembl.org/info/website/archives/index.html 


#pyensembl faster
def to_gene_name(k: str ,ensembl: object) -> str:
    """Gene id to gene name.

    Args:
        k (str): gene id.
        ensembl (object): ensembl object.

    Returns:
        str: gene name.

    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """
    try:
        return ensembl.gene_name_of_gene_id(k)
    except:
        return np.nan
    
def to_protein_id(k: str,ensembl: object) -> str:
    """Transcript id to protein id.

    Args:
        x (str): transcript id.
        ensembl (str): ensembl object.

    Returns:
        str: protein id.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(k)
        return t.protein_id
    except:
        return np.nan    
    
def to_gene_id(k: str ,ensembl: object) -> str:
    """Transcript id to gene id.

    Args:
        k (str): transcript id.
        ensembl (object): ensembl object.

    Returns:
        str: gene id.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """
    try:
        t=ensembl.transcript_by_id(k)
        return t.gene_id
    except:
        return np.nan 
    
def to_transcript_id(k: str ,ensembl: object) -> str:
    """Protein id to transcript id.

    Args:
        k (str): protein id.
        ensembl (object): ensembl object.

    Returns:
        str: transcript id.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """
    try:
        return ensembl.transcript_id_of_protein_id(k)
    except:
        return np.nan 
    
def to_dnaseq(k: str ,ensembl: object) -> str:
    """Gene id to DNA sequence.

    Args:
        k (str): gene id.
        ensembl (object): ensembl object.

    Returns:
        str: DNA sequence.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """
    try:
        g=ensembl.gene_by_id(k)
        ts=g.transcripts
        lens=[len(t.protein_sequence) if not t.protein_sequence is None else 0 for t in ts]
        return ts[lens.index(max(lens))].id, ts[lens.index(max(lens))].protein_sequence
    except:
        return np.nan,np.nan    
    
def to_protein_id_longest(k: str ,ensembl: object) -> str:
    """Gene id to protein id of the longest protein.

    Args:
        k (str): gene id.
        ensembl (object): ensembl object.

    Returns:
        str: protein id.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """    
    g=ensembl.gene_by_id(gid)
    return pd.Series({t.protein_id:t.protein_sequence for t in g.transcripts}).dropna().apply(len).sort_values().tail(1).index.to_list()[0]

def to_protein_seq(k: str ,ensembl: object,
                    transcript: bool=False) -> str:
    """Protein/transcript id to protein sequence.

    Args:
        k (str): protein id.
        ensembl (object): ensembl object.

    Returns:
        str: protein sequence.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """    
    try:
        # t=ensembl.transcript_by_id(k)
        # return t.protein_sequence
        t=ensembl.protein_sequence(k)
        if not length:
            return t
        else:
            return len(t)            
    except:
        return np.nan    
    
def to_cdsseq(k: str,
               ensembl: object) -> str:
    """Transcript id to coding sequence (CDS).

    Args:
        k (str): transcript id.
        ensembl (object): ensembl object.

    Returns:
        str: CDS sequence.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """    
    try:
        t=ensembl.transcript_by_id(k)
        return t.coding_sequence
    except:
        return np.nan 
    
def get_utr_sequence(k: str,ensembl: object,loc: str='five') -> str:
    """Protein id to UTR sequence.

    Args:
        k (str): transcript id.
        ensembl (object): ensembl object.
        loc (str): location of the UTR.

    Returns:
        str: UTR sequence.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """    
    try:
        t=ensembl.transcript_by_protein_id(k)
        return getattr(t,f'{loc}_prime_utr_sequence')
    except: 
        logging.warning(f"{k}: no sequence found")
        return     
    
def is_protein_coding(k: str,ensembl: object,geneid: bool=True) -> bool:
    """A gene or protein is protein coding or not.

    Args:
        k (str): protein/gene id.
        ensembl (object): ensembl object.
        geneid (bool): if gene id is provided.

    Returns:
        bool: is protein coding.
        
    Notes:
        1. `ensembl` object.
            from pyensembl import EnsemblRelease
            ensembl EnsemblRelease(release=100)    
    """    
    try:
        if geneid:
            g=ensembl.gene_by_id(k)
        else:
            g=ensembl.transcript_by_id(k)
    except:
        logging.error('gene id not found')
        return 
    return g.is_protein_coding

#restful api    
def rest(ids: list,
        function: str='lookup',
        target_taxon: str='9606',
        release: str='100',
        format_: str='full',
        test: bool=False,
        **kws):
    """Query Ensembl database using REST API.

    Args:
        ids (list): ids.
        function (str, optional): query function. Defaults to 'lookup'.
        target_taxon (str, optional): taxonomic id of the species. Defaults to '9606'.
        release (str, optional): ensembl release. Defaults to '100'.
        format_ (str, optional): format of the output. Defaults to 'full'.
        test (bool, optional): test mode. Defaults to False.

    Keyword Args:
        kws: additional queries.

    Raises:
        ValueError: ids should be str or list.

    Returns:
        dict: output.
    """
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

def to_homology(
                    x: str,
                    release: int=100,
                    homologytype: str='orthologues',
                    outd: str='data/database',
                    force: bool=False
                    ) -> dict:
    """
    Query homology of a gene using Ensembl REST API.
        
    Args:
        x (str): gene id.
        release (int, optional): Ensembl release number. Defaults to 100.
        homologytype (str, optional): type of the homology. Defaults to 'orthologues'.
        outd (str, optional): path of the output folder. Defaults to 'data/database'.
        force (bool, optional): overwrite output. Defaults to False.

    Returns:
        dict: output.

    References:
        1. Documentation: https://e{release}.rest.ensembl.org/documentation/info/homology_ensemblgene
    """
    p=f"https://e{release}.rest.ensembl.org/homology/id/{x}?type={homologytype};compara=vertebrates;sequence=none;cigar_line=0;content-type=application/json;format=full"
    outp=outp=f"{outd}/{p.replace('https://','')}.json"
    if exists(outp) and not force:
        return read_dict(outp)
    else:
        d1=read_dict(p)
        to_dict(d1,outp)
    return d1

def to_domains(x: str,
                    release: int,
                    species: str='homo_sapiens',
                    outd: str='data/database',
                    force: bool=False
                    ) -> pd.DataFrame:
    """Protein id to domains. 

    Args:
        x (str): protein id.
        release (int): Ensembl release.
        species (str, optional): species name. Defaults to 'homo_sapiens'.
        outd (str, optional): path of the output directory. Defaults to 'data/database'.
        force (bool, optional): overwrite output. Defaults to False.

    Returns:
        pd.DataFrame: output.
    """
    species=species.lower().replace(' ','_')
    p=f'https://e{release}.rest.ensembl.org/overlap/translation/{x}?content-type=application/json;species={species};feature=protein_feature;type=pfam'
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

## species
def to_species_name(k: str) -> str:
    """Convert to species name.

    Args:
        k (_type_): taxonomic id.

    Returns:
        str: species name.
    """
    server = f"https://e{release}.rest.ensembl.org"
    ext = f"/taxonomy/id/{k}?"
    r = requests.get(server+ext, headers={ "Content-Type" : "application/json"})
    if not r.ok:
        r.raise_for_status()
        sys.exit()
    decoded = r.json()
    return decoded['scientific_name']

def to_taxid(k: str) -> str:
    """Convert to taxonomic ids.  

    Args:
        k (str): species name.

    Returns:
        str: taxonomic id.
    """
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
def convert_coords_human_assemblies(release: int,
                                    chrom: str,
                                    start: int,
                                    end: int,
                                    frm: int=38,
                                    to: int=37,
                                    test: bool=False,
                                    force: bool=False) -> dict:
    """Convert coordinates between human assemblies.

    Args:
        release (int): Ensembl release.
        chrom (str): chromosome name.
        start (int): start position.
        end (int): end position.
        frm (int, optional): assembly to convert from. Defaults to 38.
        to (int, optional): assembly to convert to. Defaults to 37.
        test (bool, optional): test mode. Defaults to False.
        force (bool, optional): overwrite outputs. Defaults to False.

    Returns:
        dict: output.
    """
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

def map_id(df1: pd.DataFrame,
            gene_id: str,
            release: str,
            release_to: str,
            out: str='df',
            test: bool=False) -> pd.DataFrame:
    """Map ids between releases.

    Args:
        df1 (pd.DataFrame): input dataframe.
        gene_id (str): gene id.
        release (str): release to convert from.
        release_to (str): release to convert to.
        out (str, optional): output type. Defaults to 'df'.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        pd.DataFrame: output.

    Notes:     
        1. m:m mappings are possible. e.g. https://useast.ensembl.org/Homo_sapiens/Gene/Idhistory?db=core;g=ENSG00000276410;r=6:26043227-26043713;t=ENST00000615966
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

def read_idmapper_output(outp: str) -> pd.DataFrame:
    """Read the output of Ensembl's idmapper.

    Args:
        outp (str): path to the file.

    Returns:
        pd.DataFrame: output.
    """
    from pathlib import Path
    file = Path(outp)
    file.write_text(file.read_text().replace('.Old stable ID', 'Old stable ID'))
    df01=pd.read_csv(outp,#+'_',
                     sep=', ')
    df1=df01.log('Old stable ID').loc[(df01['Old stable ID']!='Old stable ID'),:].log('Old stable ID')
    assert(df1['Old stable ID'].str.startswith('ENSG').all())
    df1['Release']=df1['Release'].astype(float)
    return df1

def map_ids_(ids: list,df00: pd.DataFrame,release: int,release_to: int) -> pd.DataFrame:
    """Function for mapping many ids.

    Args:
        ids (list): list of ids.
        df00 (pd.DataFrame): source dataframe.
        release (str): release to convert from.
        release_to (str): release to convert to.

    Returns:
        pd.DataFrame: output.
    """
    df0=pd.DataFrame(ids,columns=[f'id {release}'])
    df0['id']=df0[f'id {release}'].parallel_apply(lambda x: map_id(df00,gene_id=x,
                                                   release=release,
                                                   release_to=release_to,
                                                   out='list'))
    assert(not df0['id'].isnull().any())
    # reomve retired ids
    df2=df0.log().explode('id').log.dropna(subset=['id'])
    return df2

def map_ids(srcp: str,
        dbp: str,
        ids: list,
        release: int=75,
        release_to: int=100,
        species: str='human',
        test: bool=False) -> pd.DataFrame:
    """Map many ids between Ensembl releases.

    Args:
        srcp (str): path to the IDmapper.pl file.
        dbp (str): path to the database.
        ids (list): list of ids.
        release (str): release to convert from.
        release_to (str): release to convert to.
        species (str, optional): species name. Defaults to 'human'.
        test (bool, optional): test mode. Defaults to False.

    Returns:
        pd.DataFrame: output.

    Examples:
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

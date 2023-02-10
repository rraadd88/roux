"""For processing biological sequence data."""
import pandas as pd
import numpy as np
from Bio import SeqIO,SeqRecord,Seq
from roux.lib.dfs import *
import logging

## for back-compatibility
from Bio.Seq import Seq as to_seq

##defs
def reverse_complement(
    s: str,
    ) -> str: 
    """Reverse complement.

    Args:
        s (str): sequence

    Returns:
        s (str): reverse complemented sequence
    """
    return str((str2seq(s) if isinstance(s,str) else s).reverse_complement())

def fa2df(
    alignedfastap: str,
    ids2cols=False,
    ) -> pd.DataFrame:
    """_summary_

    Args:
        alignedfastap (str): path.
        ids2cols (bool, optional): ids of the sequences to columns. Defaults to False.

    Returns:
        DataFrame: output dataframe.
    """
    dtmp=pd.read_csv(alignedfastap,names=["c"],
                    header=None,)
    dtmp=dtmp.iloc[::2].reset_index(drop=True).join(dtmp.iloc[1::2].reset_index(drop=True),rsuffix='r')
    dtmp.columns=['id','sequence']
    dtmp=dtmp.set_index('id')
    dtmp.index=[i[1:] for i in dtmp.index]
    dtmp.index.name='id'
    if ids2cols:
        for i in dtmp.index:
            seqid,contig,strand,start,end=i.split('|')
            dtmp.loc[i,'seqid']=seqid
            dtmp.loc[i,'contig']=contig
            dtmp.loc[i,'strand']=strand
            dtmp.loc[i,'start']=start
            dtmp.loc[i,'end']=end
    return dtmp

def to_genomeocoords(
    genomecoord: str,
    ) -> tuple:
    """String-formated genome co-ordinates to separated values.

    Args:
        genomecoord (str):

    Raises:
        ValueError: format of the genome co-ordinates.

    Returns:
        tuple: separated values i.e. chrom,start,end,strand
    """
    try:
        chrom=genomecoord.split(':')[0]
    except:
        raise ValueError(genomecoord)
    start=genomecoord.split(':')[1].split('-')[0]

    end=genomecoord.split(':')[1].split('-')[1].replace('+','').replace('-','')

    tail=genomecoord.split(':')[1].replace(start,'')
    if tail.endswith('+'):
        strand='+'
    elif tail.endswith('-'):
        strand='-'
    else:
        strand=''
#     print(tail,strand)
    return chrom,start,end,strand

def to_bed(
    df: pd.DataFrame,
    col_genomeocoord: str,
    bed_colns: list=['chromosome', 'start', 'end', 'id', 'NM', 'strand'],
    ) -> pd.DataFrame:
    """Genome co-ordinates to bed.

    Args:
        df (DataFrame): input dataframe.
        col_genomeocoord (str): column with the genome coordinates.

    Returns:
        DataFrame: output dataframe.
    """
    df=df.dropna(subset=[col_genomeocoord])
    dbed=df.apply(lambda x: to_genomeocoords(x[col_genomeocoord]),axis=1).apply(pd.Series)
    if len(dbed)!=0:
        dbed.columns=['chromosome', 'start', 'end','strand']
        dbed['id']=df[col_genomeocoord]
        dbed['NM']=np.nan
        return dbed[bed_colns]
    else:
        return pd.DataFrame(columns=['chromosome', 'start', 'end','strand','id','NM'])
    
## io file
### multiple seq fasta
def read_fasta(
    fap: str,
    key_type: str='id',
    duplicates: bool=False,
    ) -> dict:
    """Read fasta

    Args:
        fap (str): path
        key_type (str, optional): key type. Defaults to 'id'.
        duplicates (bool, optional): duplicates present. Defaults to False.

    Returns:
        dict: data.

    Notes:
        1. If `duplicates` key_type is set to `description` instead of `id`.
    """
    if (not duplicates) or key_type=='id':
        try:
            id2seq=SeqIO.to_dict(SeqIO.parse(fap,format='fasta'))
            id2seq={k:str(id2seq[k].seq) for k in id2seq}
            return id2seq
        except:
            duplicates=True
    if duplicates or key_type=='description':
        id2seq={}
        for seq_record in SeqIO.parse(fap, "fasta"):
            id2seq[getattr(seq_record,key_type)]=str(seq_record.seq)
        return id2seq

def to_fasta(
    sequences: dict,
    output_path: str,
    molecule_type: str,
    force: bool=True,
    **kws_SeqRecord,
    ) -> str:
    """Save fasta file.

    Args:
        sequences (dict): dictionary mapping the sequence name to the sequence.
        output_path (str): path of the fasta file.
        force (bool): overwrite if file exists.
        
    Returns:
        output_path (str): path of the fasta file
    """
    assert len(sequences)!=0
    from roux.lib.sys import makedirs,exists
    if exists(output_path) and not force: 
        logging.warning('file exists.')
        return
    makedirs(output_path)
    molecule_type=molecule_type.capitalize()
    assert molecule_type in ['Protein','RNA','DNA']
    seqs = (SeqRecord.SeqRecord(Seq.Seq(sequences[k]), id=k,annotations=dict(molecule_type=molecule_type),**kws_SeqRecord) for k in sequences)
    SeqIO.write(seqs, output_path, "fasta")
    return output_path

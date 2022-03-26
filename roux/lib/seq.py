import pandas as pd
import numpy as np
from Bio import SeqIO,SeqRecord,Seq
from roux.lib.dfs import *
import logging

##defs
def reverse_complement(s): 
    """Reverse complement.

    Args:
        s (str): sequence

    Returns:
        s (str): reverse complemented sequence
    """
    return str((str2seq(s) if isinstance(s,str) else s).reverse_complement())

def fa2df(alignedfastap: str,ids2cols=False) -> pd.DataFrame:
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

def to_genomeocoords(genomecoord: str) -> tuple:
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

bed_colns=['chromosome', 'start', 'end', 'id', 'NM', 'strand']
def to_bed(df, col_genomeocoord):
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

def to_seq(s : str,prt=False) -> Seq:
    """String to Seq object.

    Args:
        s (str): input string.
        prt (bool, optional): format. Defaults to False.

    Returns:
        Seq: Seq object.
    """
    if prt:
        alpha=Alphabet.ProteinAlphabet
    else:
        alpha=Alphabet.generic_dna
    return Seq.Seq(s,alpha)
    
def translate(dnaseq: str,fmtout=str,tax_id=None) -> str:
    """Translates a DNA sequence
    Args:
        dnaseq (str): DNA sequence
        fmtout (_type_, optional): format of output sequence. Defaults to str.
        tax_id (_type_, optional): _description_. Defaults to None.

    Returns:
        str: protein sequence.
    """
    if isinstance(dnaseq,str): 
        dnaseq=Seq.Seq(dnaseq)
    if tax_id is None:
        tax_id=1 # standard codon table. ref http://biopython.org/DIST/docs/tutorial/Tutorial.html#htoc25
    prtseq=dnaseq.translate(table=tax_id)
    if fmtout is str:
        return str(prtseq)
    else:
        return prtseq
    
## io file
### multiple seq fasta
def read_fasta(fap: str,key_type='id',duplicates=False) -> dict:
    """Read fasta

    Args:
        fap (str): path
        key_type (str, optional): key type. Defaults to 'id'.
        duplicates (bool, optional): duplicates present. Defaults to False.

    Returns:
        dict: data.

    TODOs:
        1. Check for duplicate keys.
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

def to_fasta(ids2seqs: dict,fastap: str,) -> str:
    """Save fasta file.

    Args:
        ids2seqs (dict): dictionary mapping the sequence name to the sequence.
        fastap (str): path of the fasta file

    Returns:
        fastap (str): path of the fasta file
    """
    from roux.lib.sys import makedirs,exists
    # if exists(fastap) and not force: return
    makedirs(fastap)
    seqs = (SeqRecord.SeqRecord(Seq.Seq(ids2seqs[id]), id) for id in ids2seqs)
    SeqIO.write(seqs, fastap, "fasta")
    return fastap

def dedup_fasta(fap: str,faoutp=None) -> str:
    """Deduplicate fasta file.

    Args:
        fap (str): path
        faoutp (_type_, optional): output path. Defaults to None.

    Returns:
        str: output path.
    """
    return to_fasta(read_fasta(fap,key_type='description'),
             fastap=f"{splitext(fap)[0]}_dedup.fasta" if faoutp is None else faoutp)    
    
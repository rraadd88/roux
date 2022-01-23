import pandas as pd
import numpy as np
from Bio import SeqIO,SeqRecord,Seq
from roux.lib.dfs import *
import logging

##vars
# common 
mol2codes={'amino acid':["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","X","Y","*"], #for indexing
'amino acid 3letter':['ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'],
'codons':["TTT",    "TTC",    "TTA",  "TTG",  "TCT",  "TCC",  "TCA",  "TCG",  "TAT",  "TAC",  "TAA",  "TAG",  "TGT",  "TGC",  "TGA",  "TGG",  "CTT",  "CTC",  "CTA",  "CTG",  "CCT",  "CCC",  "CCA",  "CCG",  "CAT",  "CAC",  "CAA",  "CAG",  "CGT",  "CGC",  "CGA",  "CGG",  "ATT",  "ATC",  "ATA",  "ATG",  "ACT",  "ACC",  "ACA",  "ACG",  "AAT",  "AAC",  "AAA",  "AAG",  "AGT",  "AGC",  "AGA",  "AGG",  "GTT",  "GTC",  "GTA",  "GTG",  "GCT",  "GCC",  "GCA",  "GCG",  "GAT",  "GAC",  "GAA",  "GAG",  "GGT",  "GGC",  "GGA",  "GGG"],}

def aathreeletters2one(s):
    from Bio.SeqUtils import IUPACData
    if s!='Ter':
        return IUPACData.protein_letters_3to1[s]
    else:
        return '*'

##defs
def reverse_complement(s): return str((str2seq(s) if isinstance(s,str) else s).reverse_complement())

def reverse_complement_multintseq(seq,nt2complement):
    complement=[]
    for s in list(seq):
        for ss in nt2complement:
            if ss==s:
#                 print(nt2complement[s],s)
                complement.append(nt2complement[s])
                break
    return "".join(complement[::-1]    )
def reverse_complement_multintseqreg(seq,multint2regcomplement,nt2complement):
    complement=[]
    for s in list(seq):
        if s in multint2regcomplement.keys():
            for ss in multint2regcomplement:
                if ss==s:
    #                 print(nt2complement[s],s)
                    complement.append(multint2regcomplement[s])
                    break
        elif s in nt2complement.keys():
            for ss in nt2complement:
                if ss==s:
                    complement.append(nt2complement[s])
                    break            
        else:
            logging.error(f'odd character {s} in seq {seq}')
        
    return "".join(complement[::-1]    )


def fa2df(alignedfastap,ids2cols=False):
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

def bedids2bed(df, col_genomeocoord):
    bed_colns=['chromosome', 'start', 'end', 'id', 'NM', 'strand']
    dbed=df.apply(lambda x: x[col_genomeocoord].split('|'),axis=1).apply(pd.Series)
    dbed.columns=['gene id','chromosome', 'strand', 'start', 'end']

    dbed['id']=df[col_genomeocoord]
    dbed['NM']=np.nan
    return dbed[bed_colns]

def genomeocoords2sections(genomecoord):
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
def genomeocoords2bed(df, col_genomeocoord):
    df=df.dropna(subset=[col_genomeocoord])
    dbed=df.apply(lambda x: genomeocoords2sections(x[col_genomeocoord]),axis=1).apply(pd.Series)
    if len(dbed)!=0:
        dbed.columns=['chromosome', 'start', 'end','strand']
        dbed['id']=df[col_genomeocoord]
        dbed['NM']=np.nan
        return dbed[bed_colns]
    else:
        return pd.DataFrame(columns=['chromosome', 'start', 'end','strand','id','NM'])

def str2seq(s,prt=False):
    if prt:
        alpha=Alphabet.ProteinAlphabet
    else:
        alpha=Alphabet.generic_dna
    return Seq.Seq(s,alpha)

def gffatributes2ids(s):
    """
    Deconvolutes ids from `attributes` column in GFF3 to seprate columns.
    :param s: attribute string.
    :returns: tuple of ids
    """
    Name,gene_id,transcript_id,protein_id,exon_id=np.nan,np.nan,np.nan,np.nan,np.nan
    if '=' in s:
        d=dict([i.split('=') for i in s.split(';')])
        if 'Parent' in d:
            d[d['Parent'].split(':')[0]+'_id']=d['Parent'].split(':')[1]
        Name,gene_id,transcript_id,protein_id,exon_id=np.nan,np.nan,np.nan,np.nan,np.nan
        if 'Name' in d:    
            Name=d['Name']
        if 'gene_id' in d:    
            gene_id=d['gene_id']
        if 'transcript_id' in d:    
            transcript_id=d['transcript_id']
        if 'protein_id' in d:    
            protein_id=d['protein_id']
        if 'exon_id' in d:    
            exon_id=d['exon_id']
    return Name,gene_id,transcript_id,protein_id,exon_id

def hamming_distance(s1, s2):
    """Return the Hamming distance between equal-length sequences"""
#     print(s1,s2)
    if len(s1) != len(s2):
        raise ValueError("Undefined for sequences of unequal length")
    return sum(el1 != el2 for el1, el2 in zip(s1.upper(), s2.upper()))


def get_align_metrics(alignments,
         outscore=False,test=False):
    import operator
    from Bio import pairwise2
    alignsymb=np.nan
    score=np.nan
    sorted_alignments = sorted(alignments, key=operator.itemgetter(2))
    for a in alignments:
        alignstr=pairwise2.format_alignment(*a)
        alignsymb=alignstr.split('\n')[1]
        score=a[2]
        if test:
            print(alignstr)
        break
    if not outscore:
        return alignsymb.replace(' ','-'),score
    else:
        return score
def align_global(seq1, seq2,test=False):
    # Import pairwise2 module
    from Bio import pairwise2
    # Import format_alignment method
    from Bio.pairwise2 import format_alignment
    # Get a list of the global alignments between the two sequences ACGGGT and ACG
    # No parameters. Identical characters have score of 1, else 0.
    # No gap penalties.
    alignments = pairwise2.align.globalxx(seq1, seq2)
    # Use format_alignment method to format the alignments in the list
    if test:
        for a in alignments:
            return(format_alignment(*a))
    return alignments

def align(s1,s2,test=False,seqfmt='dna',
         psm=None,pmm=None,pgo=None,pge=None,
         matrix=None,
         outscore=False):
    """
    Creates pairwise local alignment between seqeunces.
    Get the visualization and alignment scores.
    :param s1: seqeunce 1
    :param s2: seqeunce 2    
    
    REF: http://biopython.org/DIST/docs/api/Bio.pairwise2-module.html
    The match parameters are:

    CODE  DESCRIPTION
    x     No parameters. Identical characters have score of 1, otherwise 0.
    m     A match score is the score of identical chars, otherwise mismatch
          score.
    d     A dictionary returns the score of any pair of characters.
    c     A callback function returns scores.
    The gap penalty parameters are:

    CODE  DESCRIPTION
    x     No gap penalties.
    s     Same open and extend gap penalties for both sequences.
    d     The sequences have different open and extend gap penalties.
    c     A callback function returns the gap penalties.  
    --
    DNA: 
    localms: psm=2,pmm=0.5,pgo=-3,pge=-1):
    Protein:
    http://resources.qiagenbioinformatics.com/manuals/clcgenomicsworkbench/650/Use_scoring_matrices.html
    """
    import operator
    from Bio import pairwise2
    if seqfmt=='dna':
        if any([p is None for p in [psm,pmm,pgo,pge]]):
            alignments = pairwise2.align.localxx(s1.upper(),s2.upper())
        else:
            alignments = pairwise2.align.localms(s1.upper(),s2.upper(),psm,pmm,pgo,pge)
    elif seqfmt=='protein':
        from Bio.pairwise2 import format_alignment
        from Bio.SubsMat import MatrixInfo
        if matrix is None:
            matrix = MatrixInfo.blosum62
        alignments =pairwise2.align.globaldx(s1, s2, matrix)
#         print(format_alignment(*a))        
    if test:
        print(alignments)
    alignsymb=np.nan
    score=np.nan
    sorted_alignments = sorted(alignments, key=operator.itemgetter(2))
    for a in alignments:
        alignstr=pairwise2.format_alignment(*a)
        alignsymb=alignstr.split('\n')[1]
        score=a[2]
        if test:
            print(alignstr)
        break
    if not outscore:
        return alignsymb.replace(' ','-'),score
    else:
        return score
    
def translate(dnaseq,fmtout=str,tax_id=None):
    """
    Translates a DNA seqeunce
    :param dnaseq: DNA sequence
    :param fmtout: format of output sequence
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
def read_fasta(fap,key_type='id',duplicates=False):
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
def to_fasta(ids2seqs,fastap):
    from os.path import exists,dirname
    from roux.lib.sys import makedirs
    if not exists(dirname(fastap)) and dirname(fastap)!='':
        makedirs(dirname(fastap),exist_ok=True)    
    seqs = (SeqRecord.SeqRecord(Seq.Seq(ids2seqs[id]), id) for id in ids2seqs)
    SeqIO.write(seqs, fastap, "fasta")
    return fastap
def dedup_fasta(fap,faoutp=None):
    return to_fasta(read_fasta(fap,key_type='description'),
             fastap=f"{splitext(fap)[0]}_dedup.fasta" if faoutp is None else faoutp)    
# to be deprecated
def fap2id2seq(fap):
    return fap2id2seq(fap)
def ids2seqs2fasta(ids2seqs,fastap):
    return ids2seqs2fasta(ids2seqs,fastap)
    
## generate mutations
def seq_with_substitution(record,pos,sub,test=False):
    from roux.lib.str import replacebyposition    
    subfrom=sub[0]
    subto=sub[-1]
    seq=str(record.seq)
    if seq[pos]==subfrom:
        seq=replacebyposition(seq,pos,subto)
        return SeqRecord.SeqRecord(str2seq(seq),id=record.id)        
    else:
        logging.warning(f'indexing issue: {seq[pos-8:pos+7]} {seq[pos]}!={subfrom} {pos}')
#         return None

## lambda function
def process_fasta(infap,outfap,deff,deff_params):
    record=deff(SeqIO.read(infap,format='fasta'),**deff_params)
    record.description=outfap
    if not record is None:
        with open(outfap, "w") as handle:
            SeqIO.write(record, handle, "fasta")    
        return outfap

# indexing seqs
def aai2nti(i):
    """
    # nt index is 0-based
    # aa index is 1-based
    # test 
    # dict(zip(range(1,60),[aai2nti(i) for i in range(1,60)]))
    """
    return [(i-1)*3,(i-1)*3+1,(i-1)*3+2]

## modified from https://github.com/mrzResearchArena/PyFeat/blob/master/Codes/generateFeatures.py
import itertools
import numpy as np

def kmers(seq, k):
    v = []
    for i in range(len(seq) - k + 1):
        v.append(seq[i:i + k])
    return v

def pseudoKNC(x, kTuple,elements):
    ### k-mer ###
    ### A, AA, AAA
    out={}
    for i in range(1, kTuple + 1, 1):
        v = list(itertools.product(elements, repeat=i))
        # seqLength = len(x) - i + 1
        for j in v:
            # print(x.count(''.join(i)), end=',')
#             t.append(x.count(''.join(i)))
            out[f'{i} {j}']=x.count(''.join(j))
    return out
    ### --- ###

def zCurve(x, seqtype):
    ### Z-Curve ### total = 3

    if seqtype == 'DNA' or seqtype == 'RNA':

        if seqtype == 'DNA':
            TU = x.count('T')
        else:
            if seqtype == 'RNA':
                TU = x.count('U')
            else:
                None

        A = x.count('A'); C = x.count('C'); G = x.count('G');

        x_ = (A + G) - (C + TU)
        y_ = (A + C) - (G + TU)
        z_ = (A + TU) - (C + G)
#         return x_,y_,z_
        return {'x':x_,'y':y_,'z':z_}
        # print(x_, end=','); print(y_, end=','); print(z_, end=',')
#         t.append(x_); t.append(y_); t.append(z_)
        ### print('{},{},{}'.format(x_, y_, z_), end=',')
        ### --- ###
        # trackingFeatures.append('x_axis'); trackingFeatures.append('y_axis'); trackingFeatures.append('z_axis')

def gcContent(x, seqtype):

    if seqtype == 'DNA' or seqtype == 'RNA':

        if seqtype == 'DNA':
            TU = x.count('T')
        else:
            if seqtype == 'RNA':
                TU = x.count('U')
            else:
                None

        A = x.count('A');
        C = x.count('C');
        G = x.count('G');

        return {'%': (G + C) / (A + C + G + TU)  * 100.0}
#         t.append( (G + C) / (A + C + G + TU)  * 100.0 )


def cumulativeSkew(x, seqtype):

    if seqtype == 'DNA' or seqtype == 'RNA':

        if seqtype == 'DNA':
            TU = x.count('T')
        else:
            if seqtype == 'RNA':
                TU = x.count('U')
            else:
                None

        A = x.count('A');
        C = x.count('C');
        G = x.count('G');

        GCSkew = (G-C)/(G+C)
        ATSkew = (A-TU)/(A+TU)

#         t.append(GCSkew)
#         t.append(ATSkew)
        return {'GC skew': GCSkew,'AT skew':ATSkew}


def atgcRatio(x, seqtype):

    if seqtype == 'DNA' or seqtype == 'RNA':

        if seqtype == 'DNA':
            TU = x.count('T')
        else:
            if seqtype == 'RNA':
                TU = x.count('U')
            else:
                None

        A = x.count('A');
        C = x.count('C');
        G = x.count('G');

#         t.append( (A+TU)/(G+C) )
        return  {'ratio': (A+TU)/(G+C)}


def monoMonoKGap(x, g,m2):  # 1___1
    ### g-gap
    '''
    AA      0-gap (2-mer)
    A_A     1-gap
    A__A    2-gap
    A___A   3-gap
    A____A  4-gap
    '''
    out={}
    m = m2
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 2)
        # seqLength = len(x) - (i+2) + 1
        #
        for gGap in m:
            # print(gGap[0], end='')
            # print('-'*i, end='')
            # print(gGap[1])
            # trackingFeatures.append(gGap[0] + '-' * i + gGap[1])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-1] == gGap[1]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out
    ### --- ###

def monoDiKGap(x, g,m3):  # 1___2
    out={}
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        # seqLength = len(x) - (i+2) + 1
        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print('-' * i, end='')
            # print(gGap[1], end='')
            # print(gGap[2], end=' ')
            # trackingFeatures.append(gGap[0] + '-' * i + gGap[1] + gGap[2])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-2] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def diMonoKGap(x, g,m3):  # 2___1
    out={}
    m = m3
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 3)
        # seqLength = len(x) - (i+2) + 1

        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print(gGap[1], end='')
            # print('-'*i, end='')
            # print(gGap[2], end=' ')
            # trackingFeatures.append(gGap[0] + gGap[1] + '-' * i + gGap[2])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-1] == gGap[2]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def monoTriKGap(x, g,m4):  # 1___3

    # A_AAA       1-gap
    # A__AAA      2-gap
    # A___AAA     3-gap
    # A____AAA    4-gap
    # A_____AAA   5-gap upto g
    out={}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        # seqLength = len(x) - (i+2) + 1

        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print('-' * i, end='')
            # print(gGap[1], end='')
            # print(gGap[2], end=' ')
            # print(gGap[3], end=' ')
            # trackingFeatures.append(gGap[0] + '-' * i + gGap[1] + gGap[2] + gGap[3])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[-3] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def triMonoKGap(x, g,m4):  # 3___1

    # AAA_A       1-gap
    # AAA__A      2-gap
    # AAA___A     3-gap
    # AAA____A    4-gap
    # AAA_____A   5-gap upto g

    out={}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        # seqLength = len(x) - (i+2) + 1

        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print(gGap[1], end='')
            # print(gGap[2], end='')
            # print('-'*i, end='')
            # print(gGap[3], end=' ')
            # trackingFeatures.append(gGap[0] + gGap[1] + gGap[2] + '-' * i + gGap[3])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def diDiKGap(x, g,m4):

    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AA       1-gap
    # AA__AA      2-gap
    # AA___AA     3-gap
    # AA____AA    4-gap
    # AA_____AA   5-gap upto g
    out={}
    m = m4
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 4)
        # seqLength = len(x) - (i+2) + 1
        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print(gGap[1], end='')
            # print('-'*i, end='')
            # print(gGap[2], end='')
            # print(gGap[3], end='')
            # trackingFeatures.append(gGap[0] + gGap[1] + '-' * i + gGap[2] + gGap[3])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-2] == gGap[2] and v[-1] == gGap[3]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def diTriKGap(x, g,m5):  # 2___3

    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AA_AAA       1-gap
    # AA__AAA      2-gap
    # AA___AAA     3-gap
    # AA____AAA    4-gap
    # AA_____AAA   5-gap upto g
    out={}
    m = m5
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        # seqLength = len(x) - (i+2) + 1
        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print(gGap[1], end='')
            # print('-' * i, end='')
            # print(gGap[2], end='')
            # print(gGap[3], end='')
            # print(gGap[4], end='')
            # trackingFeatures.append(gGap[0] + gGap[1] + '-' * i + gGap[2]  + gGap[3] + gGap[4])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[-3] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out

    ### --- ###

def triDiKGap(x, g,m5):  # 3___2

    ### gapping ### total = [(64xg)] = 2,304 [g=9]
    # AAA_AA       1-gap
    # AAA__AA      2-gap
    # AAA___AA     3-gap
    # AAA____AA    4-gap
    # AAA_____AA   5-gap upto g
    out={}
    m = m5
    for i in range(1, g + 1, 1):
        V = kmers(x, i + 5)
        # seqLength = len(x) - (i+2) + 1
        # print(V)
        for gGap in m:
            # print(gGap[0], end='')
            # print(gGap[1], end='')
            # print(gGap[2], end='')
            # print('-'*i, end='')
            # print(gGap[3], end='')
            # print(gGap[4], end='')
            # trackingFeatures.append(gGap[0] + gGap[1] + gGap[2] + '-' * i + gGap[3] + gGap[4])

            C = 0
            for v in V:
                if v[0] == gGap[0] and v[1] == gGap[1] and v[2] == gGap[2] and v[-2] == gGap[3] and v[-1] == gGap[4]:
                    C += 1
            # print(C, end=',')
#             t.append(C)
            out[f"{i} {gGap}"]=C
    return out
            
    ### --- ###
    
## wrapper generated annotated df
#     %run features.py
def get_seq_feats(X,cfg):
    """
    cfg={'seqtype':'DNA',
    'kTuple':3,
     'g':5,} # kGap
    """

    seqtype2elements={'DNA' : 'ACGT',
                    'RNA' : 'ACGU',
                    'protein' : 'ACDEFGHIKLMNPQRSTVWY',
                    }
    fun2params={
        'zCurve':['seqtype'],              #3
        'gcContent':['seqtype'],           #1
        'cumulativeSkew':['seqtype'],      #2
        'atgcRatio':['seqtype'],         #1
        'pseudoKNC':['kTuple','elements'],            #k=2|(16), k=3|(64), k=4|(256), k=5|(1024);
        'monoMonoKGap':['g','m2'],      #4*(k)*4 = 240
        'monoDiKGap':['g','m3'],        #4*k*(4^2) = 960
        'monoTriKGap':['g','m4'],       #4*k*(4^3) = 3,840
        'diMonoKGap':['g','m3'],        #(4^2)*k*(4)    = 960
        'diDiKGap':['g','m4'],          #(4^2)*k*(4^2)  = 3,840
        'diTriKGap':['g','m5'],         #(4^2)*k*(4^3)  = 15,360
        'triMonoKGap':['g','m4'],       #(4^3)*k*(4)    = 3,840
        'triDiKGap':['g','m5'],}

#     X=list(read_fasta('PyFeat/Datasets/DNA/FASTA.txt').values())
    # params
    cfg['elements']=seqtype2elements[cfg['seqtype']]
    for i in list(range(2,6)):
        cfg[f'm{i}']=list(itertools.product(cfg['elements'], repeat=i))

    dfeat_=pd.DataFrame({'x':X})
    fun2df={}
    for fun in fun2params:
        fun2df[fun]=dfeat_['x'].apply(lambda x: globals()[fun](x, **{k:cfg[k] for k in fun2params[fun]})).apply(pd.Series)
    #     break
    dfeat=pd.concat(fun2df,axis=1)
    dfeat.columns=coltuples2str(dfeat)
    dfeat.index=X
    dfeat.index.name='x'
    return dfeat
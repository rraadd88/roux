"""
io_df -> io_dfs -> io_files
"""
# paths
from roux.lib.dfs import *
from roux.lib.dict import read_dict,to_dict # overwrite to_dict from df
from roux.lib.sys import * #is_interactive_notebook,basenamenoext,makedirs,get_all_subpaths
from roux.lib.str import replacemany
from shutil import copyfile
import logging
# def copy(src, dst):copyfile(src, dst)
# def cp(src, dst):copy(src, dst)

## paths
def read_ps(ps,test=True):
    if isinstance(ps,str): 
        if '*' in ps:
            ps=glob(ps)
        else:
            ps=[ps]
    ps=sorted(ps)
    if test:
        ds1=pd.Series({p:p2time(p) if exists(p) else np.nan for p in ps}).sort_values().dropna()
        if len(ds1)>1:
            from roux.lib.str import get_suffix
            d0=ds1.iloc[[0,-1]].to_dict()
            for k_,k,v  in zip(['oldest','latest'],get_suffix(*d0.keys(),common=False),d0.values()):
                logging.info(f"{k_}: {k}\t{v}")
        elif len(ds1)==0:
            logging.warning('paths do not exist.')
    return ps

def to_outp(ps,outd=None,outp=None,suffix=''):
    if not outp is None:
        return outp
    from roux.lib.str import get_prefix
    makedirs(outd)
    ps=read_ps(ps)
    pre=get_prefix(ps[0],ps[-1], common=True)
    if not outd is None:
        pre=outd+(basename(pre) if basename(pre)!='' else basename(dirname(pre)))
    outp=f"{pre}_{suffix}{splitext(ps[0])[1]}"
    return outp

## text files
def cat(ps,outp):
    """
    Concatenate text files.
    """
    makedirs(outp,exist_ok=True)
    with open(outp, 'w') as outfile:
        for p in ps:
            with open(p) as infile:
                outfile.write(infile.read())    

def get_encoding(p):
    """
    Get encoding of a file.
    """
    import chardet
    with open(p, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']                

import shutil
def _to_zip(source, destination=None,
          fmt='zip'):
    """
    Zip a folder.
    Ref:
    https://stackoverflow.com/a/50381250/3521099
    """
    if destination is None:
        destination=source.rsplit('/')+"."+fmt
    from os import sep
    base = basename(destination)
    fmt = base.split('.')[-1]
    name = base.replace('.'+fmt,'')
#     print(base,name,fmt)
    archive_from = dirname(source)
    if archive_from=='':
        archive_from='./'
#     archive_to = basename(source.strip(sep))
#     print(archive_from,archive_to)
    shutil.make_archive(name, fmt, archive_from, 
#                         archive_to
                       )
    shutil.move(f'{name}.{fmt}', destination)
    return destination

def to_zip(p, outp=None,fmt='zip'):
    if isinstance(p,str):
        if isdir(p):
            return _to_zip(p, destination=outp, fmt=fmt)        
    ps=read_ps(p)
    import tempfile
    with tempfile.TemporaryDirectory() as outd:
        _=[copyfile(p,f"{outd}/{basename(p)}") for p in ps]
        return _to_zip(outd+'/', destination=outp, fmt=fmt)
    
def read_zip(p,file_open=None, fun_read=None):
    from io import BytesIO
    from zipfile import ZipFile
    from urllib.request import urlopen
    if p.startswith('http') or p.startswith('ftp'):
        p=BytesIO(urlopen(p).read())
    file = ZipFile(p)
    if file_open is None:
        return file
    else:
        if fun_read is None:
            return file.open(file_open)
        else:
            return fun_read(file.open(file_open))
        
def get_version(suffix=''):
    return 'v'+get_time().replace('_','')+'_'+suffix

def version(p,outd=None,
            **kws):
    p=p.rstrip("/")
    if outd is None:
        outd=f"{dirname(p)}{'/' if dirname(p)!='' else ''}"
    if isdir(p):
        outp=f"{outd}.{get_version(basename(p),**kws)}"
    else:
        outp=f"{outd}.{get_version(basenamenoext(p),**kws)}{splitext(p)[1]}"    
    logging.info(p,outp)
    shutil.move(p,outp)
    return outp

def backup(p,outd,
#            timed=True,
           versioned=False,
           suffix='',
           zipped=False,
           test=True,
          no_test=False
          ):
    """    
    backup
        0. create version dir in outd
        1. move ps to version (time) dir with common parents till the level of the version dir
        2. zip or not

    TODOs:
    chain to if exists and force

    trash dirs
        find and move/zip
        "find -regex .*/_.*"
        "find -regex .*/test.*"
    """
    print(p)
    print(outd)    
    if no_test:
        test=False
    logging.warning(f"test={test}")
    from roux.lib.set import unique
    ps=read_ps(p)
    assert(len(ps)!=0)
#     if test:print(ps)
    ps=[ p+'/' if isdir(p) else p for p in ps]
#     if test:print(ps)
    from roux.lib.sys import get_time
#     if timed:
    outd2=outd+'/.'+get_version(suffix)#'/_v'+get_time()+'_'+(suffix+'_' if not suffix is None else '')
    # create directoried in outd
    outds=unique([dirname(dirname(p)) if isdir(p) else dirname(p) for p in ps])
#     if test:print(outds)
    outds=[replacemany(p,{outd:outd2})+'/' for p in outds]
#     if test:print(outds)
    l1=[(p,replacemany(p,{outd:outd2})) for p in ps]
#     if test:print(l1)
    l1=[(p1,dirname(dirname(p2)) if p2.endswith('/') else p2) for p1,p2 in l1]    
    if test:
        return l1
    assert(len(outds)!=0)
    _=[makedirs(p) for p in outds]
    l2=[shutil.move(*t) for t in l1]
    # zip
    if zipped:
        to_zip(outd2, f"{outd2}.zip")
        return f"{outd2}.zip"
    else:
        return l2

def read_url(url):
    """
    Read text from a url.
    """
    from urllib.request import urlopen
    f = urlopen(url)
    myfile = f.read()
    return str(myfile)

def get_download_date(path):
    import datetime
    t = os.path.getctime(path)
    return str(datetime.datetime.fromtimestamp(t))

def download(url,path=None,outd='data/database',
             force=False):
    if path is None:
        path=replacemany(url,
               {'https://':'',
                'http://':'',
               })
        path=f"{outd}/{path}"
    if not exists(path) or force:
        import urllib.request
        makedirs(path,exist_ok=True)
        urllib.request.urlretrieve(url, path)
    else:
        logging.info(f"exists: {path}")
    return path

## dfs
def post_read_table(df1,clean,tables,
                    verbose=True,
                    **kws_clean):
    if clean:
        df1=df1.rd.clean(**kws_clean)
    if tables==1 and verbose:
        df1=df1.log()
    return df1
    
from roux.lib.text import get_header
def read_table(p,
               params={},
               ext=None,
               test=False,
               filterby_time=None,
               clean=True,
               check_paths=True, # read files in the path column
               kws_clean={},
               kws_cloud={},
               tables=1,
               verbose=True,
               **kws_read_tables,):
    """
    :param replaces_index: 'basenamenoext' if path to basename
    :param params['columns']: columns to read

    #read many
    :param drop_index: (True)
    :param colindex: ('path')
    :param replaces_index: str|dict|fun
    'decimal':'.'
    
    examples:
    1. Specific columns only: params=dict(columns=list)
    
    2.
    s='.vcf|vcf.gz'
    read_table(p,
               params_read_csv=dict(
               #compression='gzip',
               sep='\t',comment='#',header=None,
               names=replacemany(get_header(path,comment='#',lineno=-1),['#','\n'],'').split('\t'))
               )
               
    TODOs:
    1. deprecate params_read_csv
     if len(params_read_csv.keys())!=0:
         params=params_read_csv.copy()
    """
    if isinstance(p,list) or (isinstance(p,str) and ('*' in p)):
        if (isinstance(p,str) and ('*' in p)):
            ps=read_ps(p,test=False)
            if exists(p.replace('/*','')):
                logging.warning(f"exists: {p.replace('/*','')}")
        elif isinstance(p,list):
            ps=p
        return read_tables(ps,params=params,
                               filterby_time=filterby_time,
                               tables=len(ps),
                               **kws_read_tables)
    elif isinstance(p,str):
        ## read paths
        if check_paths and isdir(splitext(p)[0]):
            # if len(read_ps(f"{splitext(p)[0]}/*{splitext(p)[1]}",test=False))>0:
            df_=read_table(p,check_paths=False)
            if df_.columns.tolist()[-1]=='path':
                logging.info(f"paths read {len(df_['path'].tolist())}paths from the file")
                return read_table(df_['path'].tolist())
            else:
                return df_
        elif p.startswith("https://docs.google.com/file/"):
            if not 'outd' in kws_cloud:
                logging.warning("outd not found in kws_cloud")
            from roux.lib.google import download_file
            return read_table(download_file(p,**kws_cloud))
    else:
        ValueError(p)
    assert exists(p), f"not found: {p}"
    if len(params.keys())!=0 and not 'columns' in params:
        return post_read_table(pd.read_csv(p,**params),clean=clean,tables=tables,verbose=verbose,**kws_clean)
    else:
        if len(params.keys())==0:
            params={}
        if ext is None:
            ext=basename(p).split('.',1)[1]
        if any([s==ext for s in ['pqt','parquet']]):#p.endswith('.pqt') or p.endswith('.parquet'):
            return post_read_table(pd.read_parquet(p,engine='fastparquet',**params),
                                   clean=clean,tables=tables,verbose=verbose,**kws_clean)        
        params['compression']='gzip' if ext.endswith('.gz') else 'zip' if ext.endswith('.zip') else None
        
        if not params['compression'] is None:
            ext=ext.split('.',1)[0]
            
        if any([s==ext for s in ['tsv','tab','txt']]):
            params['sep']='\t'
        elif any([s==ext for s in ['csv']]):
            params['sep']=','            
        elif ext=='vcf':
            from roux.lib.str import replacemany
            params.update(dict(sep='\t',
                               comment='#',
                               header=None,
                               names=replacemany(get_header(path=p,comment='#',lineno=-1),['#','\n'],'').split('\t'),
                              ))
        elif ext=='gpad':
            params.update(dict(
                  sep='\t',
                  names=['DB','DB Object ID','Qualifier','GO ID','DB:Reference(s) (|DB:Reference)','Evidence Code','With (or) From',
                         'Interacting taxon ID','Date','Assigned by','Annotation Extension','Annotation Properties'],
                 comment='!',
                 ))
        else: 
            logging.error(f'unknown extension {ext} in {p}')
        if test: print(params)
        return post_read_table(pd.read_table(p,**params,),clean=clean,tables=tables,verbose=verbose,**kws_clean)            

def get_logp(ps):
    from roux.lib.str import get_prefix
    p=get_prefix(min(ps),max(ps),common=True,clean=True)
    if not isdir(p):
        p=dirname(p)
    return f"{p}.log"

def apply_on_paths(ps,func,
                   replaces_outp=None,
                   path=None,
                   replaces_index=None,
                   drop_index=True,
                   colindex='path',
                   filter_rows=None,
                   fast=False, 
                   progress_bar=True,
                   params={},
#                    log=True,
                   dbug=False,
                   test1=False,
                   kws_read_table={},
                   **kws,
                  ):
    """
    :param func:
    def apply_(p,outd='data/data_analysed',force=False):
        outp=f"{outd}/{basenamenoext(p)}.pqt'
        if exists(outp) and not force:
            return
        df01=read_table(p)
    apply_on_paths(
    ps=glob("data/data_analysed/*"),
    func=apply_,
    outd="data/data_analysed/",
    force=True,
    fast=False,
    read_path=True,
    )
    :params path: tuple (colindex,replaces_index)
    :param replaces_outp: to genrate outp
    :param replaces_index: to replace (e.g. dirname) in p
    :param colindex: column containing path 
    """
    def read_table_(df,read_path=False,
                    save_table=False,
                    filter_rows=None,
                    replaces_outp=None,
                    params={},
                    dbug=False,
                    **kws_read_table,
                   ):
        p=df.iloc[0,:]['path']
        if read_path:
            if save_table:
                outp=replacemany(p, replaces=replaces_outp, replacewith='', ignore=False)
                if dbug: ic(outp)
#                 if exists(outp):
#                     if 'force' in kws:
#                         if kws['force']:
#                             return None,None
#                 else:
                return p,outp
            else:
                return p,
        else:
            df=read_table(p,params=params,**kws_read_table)
            if not filter_rows is None:
                df=df.rd.filter_rows(filter_rows)            
            return df,
    import inspect
    read_path=inspect.getfullargspec(func).args[0]=='p'
    save_table=(not replaces_outp is None) and ('outp' in inspect.getfullargspec(func).args)
    if not replaces_index is None: drop_index=False
    ps=read_ps(ps)
    if len(ps)==0:
        logging.error('no paths found')
        return
    if test1:
        ps=ps[:1]
        logging.warning(f"test1=True, {ps[0]}")
    if (not replaces_outp is None) and ('force' in kws):
        if not kws['force']:
            # p2outp
            p2outp={p:replacemany(p, replaces=replaces_outp, replacewith='', ignore=False) for p in ps}
            if dbug: print(p2outp)
            d_={}
            d_['from']=len(ps)
            ps=[p for p in p2outp if (not exists(p2outp[p])) or isdir(p2outp[p])]
            d_['  to']=len(ps)
            if d_['from']!=d_['  to']:
                logging.info(f"force=False, so len(ps) reduced from: {d_['from']}")
                logging.info(f"                                  to: {d_['  to']}")            
    df1=pd.DataFrame({'path':ps})
    if len(df1)==0:
        logging.info('no paths remained to be processed.')
        return df1
    if fast and not progress_bar: progress_bar=True
    df2=getattr(df1.groupby('path',as_index=True),
                            f"{'parallel' if fast else 'progress'}_apply" if progress_bar else "apply"
               )(lambda df: func(*(read_table_(df,read_path=read_path,
                                                 save_table=save_table,
                                                 replaces_outp=replaces_outp,
                                                 filter_rows=filter_rows,
                                                 params=params,
                                                 dbug=dbug,
                                              **kws_read_table,)),
                                 **kws))
    if isinstance(df2,pd.Series):
        return df2
    if save_table: 
        if len(df2)!=0 and not test1:
            # save log file
            from roux.lib.set import to_list
            logp=get_logp(df2.tolist())
            to_list(df2.tolist(),logp)
            logging.info(logp)
        return df2
    if not path is None:
        drop_index=False
        colindex,replaces_index=path
    if drop_index:
        df2=df2.rd.clean().reset_index(drop=drop_index).rd.clean()
    else:
        df2=df2.reset_index(drop=drop_index).rd.clean()
        if colindex!='path':
            df2=df2.rename(columns={'path':colindex},errors='raise')
    if not replaces_index is None:
        if isinstance(replaces_index,str):
            if replaces_index=='basenamenoext':
                replaces_index=basenamenoext
        df2[colindex]=df2[colindex].apply(lambda x: replacemany(x, replaces=replaces_index, replacewith='', ignore=False))
    return df2

def read_tables(ps,
                    fast=False,
                    filterby_time=None,
                    drop_index=True,
                    to_dict=False,
                    params={},
                    tables=None,
                    **kws_apply_on_paths,
                   ):
    """
    :param ps: list
    
    :TODO: info: creation dates of the newest and the oldest files.
    """
    
    if not filterby_time is None:
        from roux.lib.sys import ps2time
        df_=ps2time(ps)
        ps=df_.loc[df_['time'].str.contains(filterby_time),'p'].unique().tolist()
        drop_index=False # see which files are read
    if not to_dict:
        df2=apply_on_paths(ps,func=lambda df: df,
                           fast=fast,
                           drop_index=drop_index,
                           params=params,
                           kws_read_table=dict(tables=tables), 
#                            kws_read_table=dict(verb=False if len(ps)>5 else True),
                           **kws_apply_on_paths)
        return df2
    else:
        return {p:read_table(p,
                             params=params) for p in read_ps(ps)}
def get_path(p):
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    return p
## save table
def to_table(df,p,
             colgroupby=None,
             test=False,**kws):
    if is_interactive_notebook(): test=True
    p=get_path(p)
    if df is None:
        df=pd.DataFrame()
        logging.warning(f"empty dataframe saved: {p}")
#     if len(basename(p))>100:
#         p=f"{dirname(p)}/{basename(p)[:95]}_{basename(p)[-4:]}"
#         logging.warning(f"p shortened to {p}")
    if not df.index.name is None:
        df=df.reset_index()
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(p,exist_ok=True)
    if not colgroupby is None:
        to_manytables(df,p,colgroupby,**kws)
    elif p.endswith('.tsv') or p.endswith('.tab'):
        df.to_csv(p,sep='\t')
        if test:logging.info(p)
    elif p.endswith('.pqt'):
        to_table_pqt(df,p,**kws)
        if test: logging.info(p)
    else: 
        logging.error(f'unknown extension {p}')
    return p

def to_manytables(df,p,colgroupby,
                  fmt='',
                  ignore=False,
                  **kws_get_chunks):
    """
    :params colvalue: if colgroupby=='chunk'
    :params fmt: if '=' column names in the folder name e.g. col1=True
    
    TODOs:
    1. fmt='=' by default.
    """
    outd,ext=splitext(p)
    if isinstance(colgroupby,str):
        colgroupby=[colgroupby]
    if colgroupby=='chunk':
        if not ignore:
            if exists(outd):
                logging.error(f"can not overwrite existing chunks: {outd}/")
            assert not exists(outd), outd
        df[colgroupby]=get_chunks(df1=df,value='right',
                                  **kws_get_chunks)
#     print(df.groupby(colgroupby).progress_apply(lambda x: f"{outd}/{x.name if not isinstance(x.name, tuple) else '/'.join(x.name)}{ext}"))
    
    if (df.loc[:,colgroupby].dtypes=='float').any():        
        logging.error('columns can not be float')
        info(df.loc[:,colgroupby].dtypes)
        return 
    elif (df.loc[:,colgroupby].dtypes=='bool').any():
        fmt='='
        logging.warning('bool column detected, fmt changed to =')
    def to_outp(names,outd,colgroupby,fmt):
        if isinstance(names, str):
            names=[names]
        d1=dict(zip(colgroupby,names))
        s1='/'.join([(f"{k}{fmt}" if fmt!='' else fmt)+f"{str(v)}" for k,v in d1.items()])
        return get_path(f"{outd}/{s1}{ext}")                
    df2=df.groupby(colgroupby).progress_apply(lambda x: to_table(x,to_outp(x.name,outd,colgroupby,fmt))).to_frame('path').reset_index()
    to_table(df2,p)
    
def to_table_pqt(df,p,engine='fastparquet',compression='gzip',**kws_pqt):
    if len(df.index.names)>1:
        df=df.reset_index()    
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(p,exist_ok=True)
    df.to_parquet(p,engine=engine,compression=compression,**kws_pqt)

def to_pathtable(p):
    """
    temporary
    back_compatible
    TODOs:
    1. deprecate
    """
    to_table(pd.DataFrame({'path':glob(f"{splitext(p)[0]}/*{splitext(p)[1]}")}),p)    
    
def tsv2pqt(p):
    to_table_pqt(pd.read_csv(p,sep='\t',low_memory=False),f"{p}.pqt")
def pqt2tsv(p):
    to_table(read_table(p),f"{p}.tsv")
    
def read_excel(p,sheet_name=None,to_dict=False,kws_cloud={},**params):
#     if not 'xlrd' in sys.modules:
#         logging.error('need xlrd to work with excel; pip install xlrd')
    if p.startswith("https://docs.google.com/spreadsheets/"):
        if not 'outd' in kws_cloud:
            ValueError("outd not found in kws_cloud")
        from roux.lib.google import download_file
        return read_excel(download_file(p,**kws_cloud),**params)
    if not to_dict:
        if sheet_name is None:
            xl = pd.ExcelFile(p)
#             xl.sheet_names  # see all sheet names
            if sheet_name is None:
                sheet_name=input(', '.join(xl.sheet_names))
            return xl.parse(sheet_name) 
        else:
            return pd.read_excel(p, sheet_name, **params)
    else:
        xl = pd.ExcelFile(p)
        # see all sheet names
        sheetname2df={}
        for sheet_name in xl.sheet_names:
            sheetname2df[sheet_name]=xl.parse(sheet_name) 
        return sheetname2df
        
def to_excel(sheetname2df,datap,append=False,**kws):
#     if not 'xlrd' in sys.modules:
#         logging.error('need xlrd to work with excel; pip install xlrd')
    makedirs(datap)
    writer = pd.ExcelWriter(datap)
    startrow=0
    for sn in sheetname2df:
        if not append:
            sheetname2df[sn].to_excel(writer,sn,index=False,**kws)
        else:
            sheetname2df[sn].to_excel(writer,startrow=startrow,index=False,**kws)  
            startrow+=len(sheetname2df[sn])+2
    writer.save()
    
## to table: validate
def check_chunks(outd,col,plot=True):
    df1=pd.concat({p:read_table(f'{p}/*.pqt',params=dict(columns=[col])) for p in glob(outd)})
    df2=df1.reset_index(0).log.dropna()
    df3=df2.groupby('level_0')[col].nunique()
    df3.index=[s.replace(outd,'').replace('/','') for s in df3.index]
    logging.info(df3)
    if plot: 
        import seaborn as sns
        sns.swarmplot(df3)
    return df3

"""For input/output of data files."""
# io_df -> io_dfs -> io_files

# paths
from roux.lib.dfs import *
from roux.lib.sys import * #is_interactive_notebook,basenamenoext,makedirs,get_all_subpaths
from roux.lib.str import replace_many
import logging

## operate

def to_zip(p, outp=None,fmt='zip'):
    """Compress a file/directory.
    
    Parameters:
        p (str): path to the file/directory.
        outp (str): path to the output compressed file.
        fmt (str): format of the compressed file.
    
    Returns:
        outp (str): path of the compressed file.
    """
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
    
    if isinstance(p,str):
        if isdir(p):
            return _to_zip(p, destination=outp, fmt=fmt)        
    ps=read_ps(p)
    import tempfile
    with tempfile.TemporaryDirectory() as outd:
        _=[shutil.copyfile(p,f"{outd}/{basename(p)}") for p in ps]
        return _to_zip(outd+'/', destination=outp, fmt=fmt)
    
def read_zip(
    p: str,
    file_open: str=None,
    fun_read=None,
    test: bool=False,
    ):
    """Read the contents of a zip file.
    
    Parameters:
        p (str): path of the file.
        file_open (str): path of file within the zip file to open.
        fun_read (object): function to read the file.
    
    Examples:
        1. Setting `fun_read` parameter for reading tab-separated table from a zip file.

             fun_read=lambda x: pd.read_csv(io.StringIO(x.decode('utf-8')),sep='\t',header=None),

    """
    from io import BytesIO
    from zipfile import ZipFile,ZipExtFile
    from urllib.request import urlopen
    if isinstance(p,ZipExtFile):
        file=p
    else:
        if isinstance(p,str):
            if p.startswith('http') or p.startswith('ftp'):
                p=BytesIO(urlopen(p).read())
        file = ZipFile(p)
    if file_open is None:
        logging.info(f"list of files within the zip file: {[f.filename for f in file.filelist]}")
        return file
    else:
        if fun_read is None:
            return file.open(file_open).read()
        else:
            return fun_read(file.open(file_open).read())
        
def get_version(suffix=''):
    """Get the time-based version string.
    
    Parameters:
        suffix (string): suffix.
    
    Returns:
        version (string): version.
    """
    return 'v'+get_datetime()+'_'+suffix

def version(p,outd=None,
            **kws):
    """Get the version of the file/directory.
    
    Parameters:
        p (str): path.
        outd (str): output directory.
        
    Keyword parameters:
        kws (dict): provided to `get_version`.
    
    Returns: 
        version (string): version.        
    """
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
           versioned=False,
           suffix='',
           zipped=False,
           move_only=False,
           test=True,
          no_test=False
          ):
    """Backup a directory
    
    Steps:
        0. create version dir in outd
        1. move ps to version (time) dir with common parents till the level of the version dir
        2. zip or not
        
    Parameters:
        p (str): input path.
        outd (str): output directory path.
        versioned (bool): custom version for the backup (False).
        suffix (str): custom suffix for the backup ('').
        zipped (bool): whether to zip the backup (False).
        test (bool): testing (True).
        no_test (bool): no testing (False).
                
    TODO:
        1. Chain to if exists and force.
        2. Option to remove dirs
            find and move/zip
            "find -regex .*/_.*"
            "find -regex .*/test.*"
    """
    info(p,outd)
    if no_test:
        test=False
    logging.warning(f"test={test}")
    from roux.lib.set import unique
    ps=read_ps(p)
    assert(len(ps)!=0)
#     if test:print(ps)
    ps=[ p+'/' if isdir(p) else p for p in ps]
#     if test:print(ps)
    from roux.lib.sys import get_datetime
#     if timed:
    outd2=outd+'/.'+get_version(suffix)#'/_v'+get_datetime()+'_'+(suffix+'_' if not suffix is None else '')
    # create directoried in outd
    outds=unique([dirname(dirname(p)) if isdir(p) else dirname(p) for p in ps])
#     if test:print(outds)
    outds=[replace_many(p,{outd:outd2})+'/' if not move_only else outd2+p for p in outds]
#     if test:print(outds)
    l1=[(p,replace_many(p,{outd:outd2})) if not move_only else (p,outd2+p) for p in ps]
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
    """Read text from an URL.
    
    Parameters:
        url (str): URL link.
        
    Returns:
        s (string): text content of the URL.
    """
    from urllib.request import urlopen
    f = urlopen(url)
    myfile = f.read()
    return str(myfile)

def download(
    url: str,
    outd:str,
    path: str=None,
    force: bool=False,
    verbose: bool=True,
    )->str:
    """Download a file.
    
    Parameters:
        url (str): URL. 
        path (str): custom output path (None)
        outd (str): output directory ('data/database').
        force (bool): overwrite output (False).
        verbose (bool): verbose (True).
        
    Returns: 
        path (str): output path (None)
    """
    def get_download_date(path):
        import os
        import datetime
        t = os.path.getctime(path)
        return str(datetime.datetime.fromtimestamp(t))
    if path is None:
        path=replace_many(url,
               {'https://':'',
                'http://':'',
               })
        path=f"{outd}/{path}"
    if not exists(path) or force:
        import urllib.request
        makedirs(path,exist_ok=True)
        urllib.request.urlretrieve(url, path)
    elif verbose:
        logging.info(f"downloaded on: {get_download_date(path)}")
    return path

## text file
def read_text(p):
    """Read a file. 
    To be called by other functions  

    Args:
        p (str): path.

    Returns:
        s (str): contents.
    """
    with open(p,'r') as f:
        s=f.read()
    return s

## lists
## io
def to_list(l1,p):
    """Save list.
    
    Parameters:
        l1 (list): input list.
        p (str): path.
    
    Returns:
        p (str): path.        
    """
    from roux.lib.sys import makedirs
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    makedirs(p)    
    with open(p,'w') as f:
        f.write('\n'.join(l1))
    return p

def read_list(p):
    """Read the lines in the file.

    Args:
        p (str): path.

    Returns:
        l (list): list.
    """
    with open(p,'r') as f:
        s=f.readlines()
    return s
# alias to be deprecated
read_lines=read_list

## dict
def read_yaml(p):
    """Read `.yaml` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """
    import yaml    
    with open(p,'r') as f:
        return yaml.safe_load(f)
    
def to_yaml(d,p,**kws): 
    """Save `.yaml` file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
        
    Keyword Arguments:
        kws (d): parameters provided to `yaml.safe_dump`.
        
    Returns:
        p (str): path.
    """
    import yaml
    with open(p,'w') as f:
        yaml.safe_dump(d,f,**kws)
    return p

def read_json(path_to_file,encoding=None):    
    """Read `.json` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """
    import json    
    with open(path_to_file,encoding=encoding) as p:
        return json.load(p)
    
def to_json(data,p):
    """Save `.json` file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
                
    Returns:
        p (str): path.
    """
    import json    
    with open(p, 'w') as outfile:
        json.dump(data, outfile)
    return p

def read_pickle(p):
    """Read `.pickle` file.
    
    Parameters:
        p (str): path.
        
    Returns:
        d (dict): output dictionary.
    """    
    import pickle
    return pickle.load(open(p,
               'rb'))
def is_dict(p):
    return p.endswith(('.yml','.yaml','.json','.joblib','.pickle'))
    
def read_dict(
    p,
    fmt:str='',
    apply_on_keys=None,
    **kws,
    )-> dict:
    """Read dictionary file.
    
    Parameters:
        p (str): path.
        fmt (str): format of the file.
        
    Keyword Arguments:
        kws (d): parameters provided to reader function.
        
    Returns:
        d (dict): output dictionary.
    """    
    if '*' in p:
        d1={p:read_dict(p) for p in read_ps(p)}
        if not apply_on_keys is None:
            assert len(set([replace_many(k, replaces=apply_on_keys, replacewith='', ignore=False) for k in d1])) == len(d1.keys()), "apply_on_keys(keys)!=keys"
            d1={replace_many(k, replaces=apply_on_keys, replacewith='', ignore=False):v for k,v in d1.items()}
        return d1
    if p.endswith('.yml') or p.endswith('.yaml') or fmt=='yml' or fmt=='yaml':
        return read_yaml(p,**kws)
    elif p.endswith('.json') or fmt=='json':
        return read_json(p,**kws)
    elif p.startswith('https'):
        from urllib.request import urlopen
        try:
            return json.load(urlopen(p))
        except:
            print(logging.error(p))
#         return read_json(p,**kws)    
    elif p.endswith('.pickle'):
        return read_pickle(p,**kws)
    elif p.endswith('.joblib'):
        import joblib
        return joblib.load(p,**kws)
    else:
        logging.error(f'supported extensions: .yml .yaml .json .pickle .joblib')
        
def to_dict(d,p,**kws):
    """Save dictionary file.
    
    Parameters:
        d (dict): input dictionary.
        p (str): path.
                
    Keyword Arguments:
        kws (d): parameters provided to export function.
        
    Returns:
        p (str): path.
    """
    from roux.lib.sys import makedirs
    if not 'My Drive' in p:
        p=p.replace(' ','_')
    else:
        logging.warning('probably working on google drive; space/s left in the path.')
    makedirs(p)
    if p.endswith('.yml') or p.endswith('.yaml'):
        return to_yaml(d,p,**kws)
    elif p.endswith('.json'):
        return to_json(d,p,**kws)
    elif p.endswith('.pickle'):
        import pickle
        return pickle.dump(d, open(p, 'wb'),**kws)
    elif p.endswith('.joblib'):
        import joblib
        return joblib.dump(d, p,**kws)     
    else:
        raise ValueError(f'supported extensions: .yml .yaml .json .pickle .joblib')
        
## tables
def post_read_table(df1,clean,tables,
                    verbose=True,
                    **kws_clean):
    """Post-reading a table.
    
    Parameters: 
        df1 (DataFrame): input dataframe.
        clean (bool): whether to apply `clean` function. 
        tables ()
        verbose (bool): verbose.
    
    Keyword parameters:
        kws_clean (dict): paramters provided to the `clean` function. 
        
    Returns:
        df (DataFrame): output dataframe. 
    """
    if clean:
        df1=df1.rd.clean(**kws_clean)
    if tables==1 and verbose:
        df1=df1.log()
    return df1
    
from roux.lib.text import get_header
def read_table(
    p,
    ext=None,
    clean=True,
    filterby_time=None,
    params={},
    kws_clean={},
    kws_cloud={},
    check_paths=True, # read files in the path column
    tables=1,
    test=False,
    verbose=True,
    **kws_read_tables,
    ):
    """
    Table/s reader.
    
    Parameters:
        p (str): path of the file. It could be an input for `read_ps`, which would include strings with wildcards, list etc. 
        ext (str): extension of the file (default: None meaning infered from the path).
        clean=(default:True).
        filterby_time=None).
        check_paths (bool): read files in the path column (default:True). 
        test (bool): testing (default:False).
        params: parameters provided to the 'pd.read_csv' (default:{}). For example
            params['columns']: columns to read.
        kws_clean: parameters provided to 'rd.clean' (default:{}).
        kws_cloud: parameters for reading files from google-drive (default:{}).
        tables: how many tables to be read (default:1).
        verbose: verbose (default:True).
                
    Keyword parameters:
        kws_read_tables (dict): parameters provided to `read_tables` function. For example:
            drop_index (bool): whether to drop the index column e.g. `path` (default: True).
            replaces_index (object|dict|list|str): for example, 'basenamenoext' if path to basename.
            colindex (str): the name of the column containing the paths (default: 'path')
    
    Returns:
        df (DataFrame): output dataframe. 
        
    Examples:
        1. For reading specific columns only set `params=dict(columns=list)`.
        
        2. While reading many files, convert paths to a column with corresponding values:
        
                drop_index=False,
                colindex='parameter',   
                replaces_index=lambda x: Path(x).parent
        
        3. Reading a vcf file.
                p='*.vcf|vcf.gz'
                read_table(p,
                           params_read_csv=dict(
                           #compression='gzip',
                           sep='\t',comment='#',header=None,
                           names=replace_many(get_header(path,comment='#',lineno=-1),['#','\n'],'').split('\t'))
                           )
    """
    if isinstance(p,list) or (isinstance(p,str) and ('*' in p)):
        if (isinstance(p,str) and ('*' in p)):
            ps=read_ps(p,test=False)
            if exists(p.replace('/*','')):
                logging.warning(f"exists: {p.replace('/*','')}")
        elif isinstance(p,list):
            ps=p
        return read_tables(
                    ps,
                    params=params,
                    filterby_time=filterby_time,
                    tables=len(ps),
                    verbose=verbose,
                    **kws_read_tables,# is kws_apply_on_paths,
                    )
    elif isinstance(p,str):
        ## read paths
        if check_paths and isdir(splitext(p)[0]):
            # if len(read_ps(f"{splitext(p)[0]}/*{splitext(p)[1]}",test=False))>0:
            df_=read_table(p,check_paths=False)
            if df_.columns.tolist()[-1]=='path':
                logging.info(f"paths read {len(df_['path'].tolist())}paths from the file")
                ps=df_['path'].tolist()
                return read_tables(
                    ps,
                    params=params,
                    filterby_time=filterby_time,
                    tables=len(ps),
                    verbose=verbose,
                    **kws_read_tables,# is kws_apply_on_paths,
                    )
            else:
                return df_
        elif p.startswith("https://docs.google.com/file/"):
            if not 'outd' in kws_cloud:
                logging.warning("outd not found in kws_cloud")
            from roux.lib.google import download_file
            return read_table(download_file(p,**kws_cloud))
    else:
        raise ValueError(p)
    assert exists(p), f"not found: {p}"
    if len(params.keys())!=0 and not 'columns' in params:
        return post_read_table(pd.read_csv(p,**params),clean=clean,tables=tables,verbose=verbose,**kws_clean)
    else:
        if len(params.keys())==0:
            params={}
        if ext is None:
            ext=basename(p).rsplit('.',1)[1]
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
            from roux.lib.str import replace_many
            params.update(dict(sep='\t',
                               comment='#',
                               header=None,
                               names=replace_many(get_header(path=p,comment='#',lineno=-1),['#','\n'],'').split('\t'),
                              ))
        elif ext=='gpad':
            params.update(dict(
                  sep='\t',
                  names=['DB','DB Object ID','Qualifier','GO ID','DB:Reference(s) (|DB:Reference)','Evidence Code','With (or) From',
                         'Interacting taxon ID','Date','Assigned by','Annotation Extension','Annotation Properties'],
                 comment='!',
                 ))
        else: 
            raise ValueError(f'unknown extension {ext} in {p}')
        if test: print(params)
        return post_read_table(pd.read_table(p,**params,),clean=clean,tables=tables,verbose=verbose,**kws_clean)            

def get_logp(ps):
    """Infer the path of the log file.
    
    Parameters:
        ps (list): list of paths.     

    Returns:
        p (str): path of the output file.     
    """
    from roux.lib.str import get_prefix
    p=get_prefix(min(ps),max(ps),common=True,clean=True)
    if not isdir(p):
        p=dirname(p)
    return f"{p}.log"

def apply_on_paths(
    ps,
    func,
    replaces_outp=None,
    # path=None,
    replaces_index=None,
    drop_index=True, # keep path
    colindex='path',
    filter_rows=None,
    fast=False, 
    progress_bar=True,
    params={},
    #log=True,
    dbug=False,
    test1=False,
    verbose=True,
    kws_read_table={},
    **kws,
    ):
    """Apply a function on list of files.
    
    Parameters:
        ps (str|list): paths or string to infer paths using `read_ps`.
        func (function): function to be applied on each of the paths.
        replaces_outp (dict|function): infer the output path (`outp`) by replacing substrings in the input paths (`p`).
        filter_rows (dict): filter the rows based on dict, using `rd.filter_rows`.
        fast (bool): parallel processing (default:False). 
        progress_bar (bool): show progress bar(default:True).
        params (dict): parameters provided to the `pd.read_csv` function.
        dbug (bool): debug mode on (default:False).
        test1 (bool): test on one path (default:False).
        kws_read_table (dict): parameters provided to the `read_table` function (default:{}).
        replaces_index (object|dict|list|str): for example, 'basenamenoext' if path to basename.
        drop_index (bool): whether to drop the index column e.g. `path` (default: True).
        colindex (str): the name of the column containing the paths (default: 'path')
    
    Keyword parameters:
        kws (dict): parameters provided to the function.

    Example:
            1. Function: 
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
                
    TODOs:
        Move out of io. 
    """
    def read_table_(df,read_path=False,
                    save_table=False,
                    filter_rows=None,
                    replaces_outp=None,
                    params={},
                    dbug=False,
                    verbose=True,
                    **kws_read_table,
                   ):
        p=df.iloc[0,:]['path']
        if read_path:
            if save_table:
                outp=replace_many(p, replaces=replaces_outp, replacewith='', ignore=False)
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
            df=read_table(p,params=params,verbose=verbose,**kws_read_table)
            if not filter_rows is None:
                df=df.rd.filter_rows(filter_rows)            
            return df,
    import inspect
    read_path=inspect.getfullargspec(func).args[0]=='p'
    save_table=(not replaces_outp is None) and ('outp' in inspect.getfullargspec(func).args)
    if not replaces_index is None: drop_index=False
    ps=read_ps(ps,test=verbose)
    if len(ps)==0:
        logging.error('no paths found')
        return
    if test1:
        ps=ps[:1]
        logging.warning(f"test1=True, {ps[0]}")
    if (not replaces_outp is None) and ('force' in kws):
        if not kws['force']:
            # p2outp
            p2outp={p:replace_many(p, replaces=replaces_outp, replacewith='', ignore=False) for p in ps}
            if dbug: print(p2outp)
            d_={}
            d_['from']=len(ps)
            ps=[p for p in p2outp if (not exists(p2outp[p])) or isdir(p2outp[p])]
            d_['  to']=len(ps)
            if d_['from']!=d_['  to']:
                logging.info(f"force=False, so paths reduced from: {d_['from']}")
                logging.info(f"                                to: {d_['  to']}")
    if dbug: info(ps)
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
                                               verbose=verbose,
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
    # if not path is None:
    #     drop_index=False
    #     colindex,replaces_index=path
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
        df2[colindex]=df2[colindex].apply(lambda x: replace_many(x, replaces=replaces_index, replacewith='', ignore=False))
    return df2

def read_tables(
    ps,
    fast=False,
    filterby_time=None,
    to_dict=False,
    params={},
    tables=None,
    **kws_apply_on_paths,
    ):
    """Read multiple tables.
    
    Parameters:
        ps (list): list of paths.
        fast (bool): parallel processing (default:False)
        filterby_time (str): filter by time (default:None)
        drop_index (bool): drop index (default:True)
        to_dict (bool): output dictionary (default:False)
        params (dict): parameters provided to the `pd.read_csv` function (default:{})
        tables: number of tables (default:None).
        
    Keyword parameters:
        kws_apply_on_paths (dict): parameters provided to `apply_on_paths`.
        
    Returns:
        df (DataFrame): output dataframe. 
    
    TODOs:
        Parameter to report the creation dates of the newest and the oldest files.
    """
    if not filterby_time is None:
        from roux.lib.sys import ps2time
        df_=ps2time(ps)
        ps=df_.loc[df_['time'].str.contains(filterby_time),'p'].unique().tolist()
        kws_apply_on_paths['drop_index']=False # see which files are read
    if not to_dict:
        df2=apply_on_paths(ps,func=lambda df: df,
                           fast=fast,
                           # drop_index=drop_index,
                           params=params,
                           kws_read_table=dict(tables=tables), 
#                            kws_read_table=dict(verb=False if len(ps)>5 else True),
                           **kws_apply_on_paths)
        return df2
    else:
        return {p:read_table(p,
                             params=params) for p in read_ps(ps)}

## save table
def to_table(
    df,
    p,
    colgroupby=None,
    test=False,
    **kws
    ):
    """Save table.
    
    Parameters:
        df (DataFrame): the input dataframe. 
        p (str): output path.
        colgroupby (str|list): columns to groupby with to save the subsets of the data as separate files.
        test (bool): testing on (default:False).
        
    Keyword parameters:
        kws (dict): parameters provided to the `to_manytables` function.
    
    Returns:
        p (str): path of the output.    
    """
    if is_interactive_notebook(): test=True
    p=to_path(p)
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
        df.to_csv(p,sep='\t',**kws)
    elif p.endswith('.pqt'):
        to_table_pqt(df,p,**kws)
    else: 
        logging.error(f'unknown extension {p}')
    return p

def to_manytables(df,p,colgroupby,
                  fmt='',
                  ignore=False,
                  **kws_get_chunks):
    """
    Save many table.
    
    Parameters:
        df (DataFrame): the input dataframe. 
        p (str): output path.
        colgroupby (str|list): columns to groupby with to save the subsets of the data as separate files.
        fmt (str): if '=' column names in the folder name e.g. col1=True.
        ignore (bool): ignore the warnings (default:False).
        
    Keyword parameters:
        kws_get_chunks (dict): parameters provided to the `get_chunks` function.
    
    Returns:
        p (str): path of the output.   
    
    TODOs:
        1. Change in default parameter: `fmt='='`.
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
        """
        Get output path for each group.
        """
        if isinstance(names, str):
            names=[names]
        d1=dict(zip(colgroupby,names))
        s1='/'.join([(f"{k}{fmt}" if fmt!='' else fmt)+f"{str(v)}" for k,v in d1.items()])
        return to_path(f"{outd}/{s1}{ext}")                
    df2=df.groupby(colgroupby).progress_apply(lambda x: to_table(x,to_outp(x.name,outd,colgroupby,fmt))).to_frame('path').reset_index()
    to_table(df2,p)
    
def to_table_pqt(df,p,engine='fastparquet',compression='gzip',**kws_pqt):
    if len(df.index.names)>1:
        df=df.reset_index()    
    if not exists(dirname(p)) and dirname(p)!='':
        makedirs(p,exist_ok=True)
    df.to_parquet(p,engine=engine,compression=compression,**kws_pqt)
    
def tsv2pqt(p):
    """Convert tab-separated file to Apache parquet. 
    
    Parameters:
        p (str): path of the input.
        
    Returns: 
        p (str): path of the output.
    """
    to_table_pqt(pd.read_csv(p,sep='\t',low_memory=False),f"{p}.pqt")
    
def pqt2tsv(
    p: str,
    )->str:
    """Convert Apache parquet file to tab-separated. 
    
    Parameters:
        p (str): path of the input.
        
    Returns: 
        p (str): path of the output.
    """    
    to_table(read_table(p),f"{p}.tsv")

## tables: excel
def read_excel(
    p: str,
    sheet_name: str=None,
    kws_cloud: dict={},
    test: bool= False,
    **kws,
    ):
    """Read excel file
    
    Parameters:
        p (str): path of the file. 
        sheet_name (str|None): read 1st sheet if None (default:None)
        kws_cloud (dict): parameters provided to read the file from the google drive (default:{})
        test (bool): if False and sheet_name not provided, return all sheets as a dictionary, else if True, print list of sheets.
        
    Keyword parameters:
        kws: parameters provided to the excel reader.
    """
    #if not 'xlrd' in sys.modules:
    #   logging.error('need xlrd to work with excel; pip install xlrd')
    if isinstance(p,str):
        if p.startswith("https://docs.google.com/spreadsheets/"):
            if not 'outd' in kws_cloud:
                raise ValueError("outd not found in kws_cloud")
            from roux.lib.google import download_file
            return read_excel(download_file(p,**kws_cloud),**kws)
    if sheet_name is None:
        xl = pd.ExcelFile(p)
        if test: 
            logging.info(f"`sheet_name`s (to select from) the excel file : {xl.sheet_names}")
            return xl
        ## return all sheets
        sheetname2df={}
        for sheet_name in xl.sheet_names:
            sheetname2df[sheet_name]=xl.parse(sheet_name)
            logging.info(f"'{sheet_name}':{sheetname2df[sheet_name].shape}")
        return sheetname2df            
    else:
        return pd.read_excel(p, sheet_name, **kws)

def to_excel_commented(
    p: str,
    comments: dict,
    outp: str=None,
    author: str=None,
    ):
    """Add comments to the columns of excel file and save.

    Args:
        p (str): input path of excel file.
        comments (dict): map between column names and comment e.g. description of the column.
        outp (str): output path of excel file. Defaults to None.
        author (str): author of the comments. Defaults to 'Author'.
        
    TODOs:
        1. Increase the limit on comments can be added to number of columns. Currently it is 26 i.e. upto Z1.
    """
    if author is None:
        author='Author'
    if outp is None:
        outp=p
        logging.warning('overwritting the input file')
    from openpyxl import load_workbook
    from openpyxl.comments import Comment
    from string import ascii_uppercase
    wb = load_workbook(filename = outp)
    for sh in wb:
        for k in [s+'1' for s in list(ascii_uppercase)+['A'+s_ for s_ in ascii_uppercase]]:
            if (not sh[k].value is None):
                if (sh[k].value in comments):
                    sh[k].comment = Comment(comments[sh[k].value],author=author)
                else:
                    logging.warning(f"no comment for column: '{sh[k].value}'")
            else:
                break
    wb.save(outp)
    wb.close()
    return outp
    
def to_excel(
    sheetname2df: dict,
    outp: str,
    comments: dict=None,
    author: str=None,
    append: bool=False,
    **kws,
    ):
    """Save excel file.
    
    Parameters:
        sheetname2df (dict): dictionary mapping the sheetname to the dataframe.
        outp (str): output path. 
        append (bool): append the dataframes (default:False).
        comments (dict): map between column names and comment e.g. description of the column.

    Keyword parameters: 
        kws: parameters provided to the excel writer.
    """
#     if not 'xlrd' in sys.modules:
#         logging.error('need xlrd to work with excel; pip install xlrd')
    # makedirs(outp)
    if not comments is None:
        ## order the columns
        for k1 in sheetname2df:
            sheetname2df[k1]=sheetname2df[k1].loc[:,[k for k in comments if k in sheetname2df[k1]]+[k for k in sheetname2df[k1] if not k in comments]]

        ## insert a table with the description
        items = list(sheetname2df.items())
        items.insert(0, ('description', dict2df(comments,colkey='column name',colvalue='description')))
        sheetname2df = dict(items)
    
    outp=to_path(outp)
    writer = pd.ExcelWriter(outp)
    startrow=0
    for sn in sheetname2df:
        if not append:
            sheetname2df[sn].to_excel(writer,sn,index=False,**kws)
        else:
            sheetname2df[sn].to_excel(writer,startrow=startrow,index=False,**kws)  
            startrow+=len(sheetname2df[sn])+2
    writer.save()
    if not comments is None:
        to_excel_commented(
            outp,
            comments=comments,
            outp=outp,
            author=author,
            )
    return outp
    
## to table: validate
def check_chunks(outd,col,plot=True):
    """Create chunks of the tables.
    
    Parameters:
        outd (str): output directory.
        col (str): the column with values that are used for getting the chunks.
        plot (bool): plot the chunk sizes (default:True).
    
    Returns:
        df3 (DataFrame): output dataframe.
    """
    df1=pd.concat({p:read_table(f'{p}/*.pqt',params=dict(columns=[col])) for p in glob(outd)})
    df2=df1.reset_index(0).log.dropna()
    df3=df2.groupby('level_0')[col].nunique()
    df3.index=[s.replace(outd,'').replace('/','') for s in df3.index]
    logging.info(df3)
    if plot: 
        import seaborn as sns
        sns.swarmplot(df3)
    return df3

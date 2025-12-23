"""For input/output of data files."""

import pandas as pd
import logging
from tqdm import tqdm

# paths
from roux.lib.sys import (
    basename,
    basenamenoext,
    dirname,
    exists,
    get_datetime,
    glob,
    isdir,
    makedirs,
    read_ps,
    shutil,
    splitext,
    to_path,
)  # is_interactive_notebook,basenamenoext,makedirs,get_all_subpaths
from pathlib import Path
from roux.lib.str import replace_many

# import path: df -> dfs -> io
import roux.lib.dfs as rd  # noqa


## operate
def read_zip(
    p: str,
    file_open: str = None,
    fun_read=None,
    test: bool = False,
):
    """Read the contents of a zip file.

    Parameters:
        p (str): path of the file.
        file_open (str): path of file within the zip file to open.
        fun_read (object): function to read the file.

    Examples:
        1. Setting `fun_read` parameter for reading tab-separated table from a zip file.

            from io import StringIO
            ...
            fun_read=lambda x: pd.read_csv(io.StringIO(x.decode('utf-8')),sep='\t',header=None),

            or

            from io import BytesIO
            ...
            fun_read=lambda x: pd.read_table(BytesIO(x)),
    """
    from io import BytesIO
    from zipfile import ZipFile, ZipExtFile
    from urllib.request import urlopen

    if isinstance(p, ZipExtFile):
        file = p
    else:
        if isinstance(p, str):
            if p.startswith("http") or p.startswith("ftp"):
                p = BytesIO(urlopen(p).read())
        file = ZipFile(p)
    if file_open is None:
        logging.info(
            f"list of files within the zip file: {[f.filename for f in file.filelist]}"
        )
        return file
    else:
        if fun_read is None:
            return file.open(file_open).read()
        else:
            return fun_read(file.open(file_open).read())


def to_zip_dir(
    source,
    destination=None,
    fmt="zip"
    ):
    """
    Zip a folder.
    Ref:
    https://stackoverflow.com/a/50381250/3521099
    """
    if isinstance(source,str):
        if destination is None:
            destination = source.rsplit("/") + "." + fmt
        paths={source:destination}
    elif isinstance(source,dict):
        paths=source
        
    base = basename(destination)
    fmt = base.split(".")[-1]
    name = base.replace("." + fmt, "")
    #     print(base,name,fmt)
    archive_from = dirname(source)
    if archive_from == "":
        archive_from = "./"
    #     archive_to = basename(source.strip(sep))
    #     print(archive_from,archive_to)
    shutil.make_archive(
        name,
        fmt,
        archive_from,
        #                         archive_to
    )
    shutil.move(f"{name}.{fmt}", destination)
    return destination


def to_zip(
    p: str,
    outp: str = None,
    func_rename=None,
    fmt: str = "zip",
    test: bool = False,
):
    """Compress a file/directory.

    Parameters:
        p (str): path to the file/directory.
        outp (str): path to the output compressed file.
        fmt (str): format of the compressed file.

    Returns:
        outp (str): path of the compressed file.
    """
    if isinstance(p, str):
        if isdir(p):
            return to_zip_dir(p, destination=outp, fmt=fmt)
            
    ps = read_ps(p)
    import tempfile

    with tempfile.TemporaryDirectory() as outd:
        if test:
            return {
                p: f"{outd}/{basename(p) if func_rename is None else func_rename(p)}"
                for p in ps
            }
        _ = [
            shutil.copyfile(
                p, f"{outd}/{basename(p) if func_rename is None else func_rename(p)}"
            )
            for p in ps
        ]
        return to_zip_dir(outd + "/", destination=outp, fmt=fmt)

def to_arxv(
    ind,
    outp=None,
    verbose=True,
    force=False,
    wait=True,
    simulate=False,
    exclude=None,
    ):
    """
    Notes:
        # test reversibility
        tree tests/input/
        tar --remove-files -czf tests/input/roux_workflow_cfgs.tar.gz tests/input/roux_workflow_cfgs
        tree tests/input/
        tar -xzf tests/input/roux_workflow_cfgs.tar.gz && rm tests/input/roux_workflow_cfgs.tar.gz
        tree tests/input/    
    """
    if simulate:
        verbose=True

    if len(read_ps(ind,verbose=verbose))>1:
        logging.warning("running recursively ..")
        
        outps={}
        for p_ in tqdm(read_ps(ind,verbose=False)):
            outps[p_]=to_arxv(
                p_,
                
                verbose=verbose,
                force=force,
                wait=wait,
                simulate=simulate,   
                exclude=exclude,             
                )
        return pd.Series(outps) # to truncate the print

    if Path(ind).is_file() and Path(ind).exists():
        ind=Path(ind).with_suffix('').as_posix()
        if verbose:
            logging.warning(f"archiving sub.dir. instead: {ind}")        
    if not Path(ind).exists():
        logging.error(f"not found: {ind}")
        return 

    if outp is None:
        outp=f"{ind}.tar.gz"
    ## low priority
    com=''
    if not wait:
        com+="nice -n 10 ionice -c2 -n7 "

    arg_exclude=''
    if isinstance(exclude,str):
        exclude=[exclude]
    if isinstance(exclude,list):
        arg_exclude=' --exclude '.join(exclude)
    
    com+=f" tar --remove-files -czf {outp} {arg_exclude} {ind} &"
    if verbose:
        logging.info(com)

    if not simulate:   
        if not Path(outp).exists() or force:
            from roux.lib.sys import run_com
            run_com(
                com,
                verbose=verbose,
                wait=wait, # do not wait for the process to finish
                )
        else:
            logging.warning(f"exists: {outp}")
        if wait:
            assert Path(outp).exists(), outp
    return outp

def read_arxv(
    p, # path to the file in arxv
    tree_depth=3, ## search the root arxv upto

    ap=None,

    outd=None, ## base of output dir. path
    
    read_func=None, ## without savng the extracted file
    
    verbose=False,
    force=False,
    **kws_extract,
    ):
    """
    Read (from) archive. 

    Examples:
        read_func=lambda f: pd.read_csv(
            BytesIO(f.read()),
            sep='\t',
        )
    """
    if p.startswith('../'):
        assert p.count('../')==1, p
        
    p=Path(p).as_posix()
    if Path(p).exists() and not force:
        return p
    if ap is None:
        # find root arxv
        for i in range(1,tree_depth+1):
            ap=f"{p.rsplit('/',i)[0]}.tar.gz"
            if Path(ap).exists():
                if verbose:
                    logging.info(f"root arxv: {ap}")
                break
    # read from the arxv    
    import tarfile
    with tarfile.open(ap,'r') as tar:
        ps=tar.getnames()
        if verbose:
            logging.info(ps if len(ps)<10 else len(ps))

        p_inside=Path(p.split('../')[-1]).as_posix()
        # logging.warning(p_inside)

        aps_inside=sorted([p_ for p_ in ps if p_.endswith('.tar.gz')])
        if p_inside not in ps:
            
            logging.warning(f"searching among the {len(aps_inside)} arxives found inside ..")
            if verbose:
                logging.info(aps_inside)
                
            for ap_inside in aps_inside:
                ap_inside_=ap_inside.rstrip('.tar.gz')
                # print(ap_inside_)
                if p_inside.startswith(ap_inside_):
                    logging.warning(
                        f"found in {ap_inside}",
                    )
                    # return
                    ## extract ap_inside from ap
                    read_arxv(
                        p=f"{'../' if p.startswith('../') else ''}{ap_inside}",
                        ap=ap,
                    )
                    ## extract p_inside from ap_inside, by recursing
                    return read_arxv(
                        p=p,
                        ap=f"{'../' if p.startswith('../') else ''}{ap_inside}",
                    )
                    
        assert p_inside in ps, (p,ps)
        
        if not read_func:
            outp=tar.extractall(
                path=outd if outd is not None else '../' if p.startswith('../') else '.',
                members=[p_inside],
                **kws_extract
                )
            assert Path(p).exists(), p
            return p
        else: 
            return read_func(tar.extractfile(p))
            # Read the file-like object into a pandas DataFrame
            # Use io.BytesIO to wrap the file object for pandas if needed
            # if f:
            #     df = pd.read_csv(io.BytesIO(f.read()))     
            
def to_copy(
    paths: dict,
    
    replaces=None,
    
    flatten=False,
    flatten_rename_basename=None,
    flatten_outd: str=None,
    
    force=False,
    simulate=False,
    **kws_replace_many,
    ):
    """
    Copy with the dir. structure

    Notes:
        Similar to:
        
            rsync -a --remove-source-files --prune-empty-dirs --info=progress2  --include=$ind --include='*/'   --include=$pattern  --exclude='*'   $ind $outd
    """
    
    if isinstance(paths,str):
        assert replaces is not None, replaces
        paths={p: replace_many(p,replaces, **kws_replace_many) for p in read_ps(paths)}
        
    paths = {k: v for k, v in paths.items()}

    import shutil
    if not flatten:
        copy_paths=paths
    else:
        makedirs(flatten_outd)
        copy_paths = {}
        for d, ps in paths.items():
            for p in ps:
                if flatten_rename_basename is not None:
                    outb = flatten_rename_basename(p)
                else:
                    outb = basename(p)
                copy_paths[p] = f"{flatten_outd}/{d}/{outb}"
                
    assert len(list(set(copy_paths.keys()))) == len(
        list(set(copy_paths.values()))
    ), copy_paths
    
    if simulate:
        return copy_paths
        
    logging.info("copying ..")
    for p, outp in copy_paths.items():
        if not Path(outp).exists() or force:
            logging.info(f"{p} -> {outp}")
            if Path(p).is_file():
                shutil.copyfile(p, makedirs(outp))
            elif Path(p).is_dir():
                shutil.copytree(p, makedirs(outp))
    return copy_paths


def get_version(
    suffix: str = "",
) -> str:
    """Get the time-based version string.

    Parameters:
        suffix (string): suffix.

    Returns:
        version (string): version.
    """
    return "v" + get_datetime() + "_" + suffix


def to_version(
    p: str,
    outd: str = None,
    test: bool = False,
    label: str = "",
    **kws: dict,
) -> str:
    """Rename a file/directory to a version.

    Parameters:
        p (str): path.
        outd (str): output directory.

    Keyword parameters:
        kws (dict): provided to `get_version`.

    Returns:
        version (string): version.

    TODOs:
        1. Use `to_dir`.
    """
    p = p.rstrip("/")
    if outd is None:
        outd = f"{dirname(p)}{'/' if dirname(p)!='' else ''}"
    if isdir(p):
        outp = f"{outd}/.{get_version(basename(p)+' '+label,**kws)}"
    else:
        outp = (
            f"{outd}/.{get_version(basenamenoext(p)+' '+label,**kws)}{splitext(p)[1]}"
        )
    outp = to_path(outp)
    logging.info(f"-->{outp}")
    if not test:
        shutil.move(p, outp)
        to_dict(
            dict(
                src=p,
                dst=outp,
            ),
            outp + "/.params.json",
        )

    else:
        logging.warning("test mode.")
    return outp

def backup(
    p: str,
    outd: str = None,
    versioned: bool = False,
    suffix: str = "",
    zipped: bool = False,
    move_only: bool = False,
    test: bool = True,
    verbose: bool = False,
    no_test: bool = False,
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
        no_test (bool): no testing. Usage in command line (False).

    TODOs:
        1. Use `to_dir`.
        2. Option to remove dirs
            find and move/zip
            "find -regex .*/_.*"
            "find -regex .*/test.*"
    """
    logging.warning("deprecation: prefer to_version.")
    return to_version(p, outd)


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
    path: str = None,
    outd: str = None,
    force: bool = False,
    verbose: bool = True,
) -> str:
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
        assert outd is not None
        path = replace_many(
            url,
            {
                "https://": "",
                "http://": "",
            },
        )
        path = f"{outd}/{path}"
    if not exists(path) or force:
        import urllib.request

        makedirs(path, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    if verbose:
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
    with open(p, "r") as f:
        s = f.read()
    return s


## lists
## io
def to_list(l1, p):
    """Save list.

    Parameters:
        l1 (list): input list.
        p (str): path.

    Returns:
        p (str): path.
    """
    from roux.lib.sys import makedirs

    if "My Drive" not in p:
        p = p.replace(" ", "_")
    else:
        logging.warning("probably working on google drive; space/s left in the path.")
    makedirs(p)
    with open(p, "w") as f:
        f.write("\n".join(l1))
    return p


def read_list(p):
    """Read the lines in the file.

    Args:
        p (str): path.

    Returns:
        l (list): list.
    """
    with open(p, "r") as f:
        s = f.read().split("\n")
    return s


# alias to be deprecated
read_lines = read_list


def is_dict(p):
    return p.endswith((".yml", ".yaml", ".json", ".joblib", ".pickle"))
def is_table(p):
    return p.endswith((".tsv", ".csv", ".pqt", ".parquet",))

## meta
def is_data(p):
    return is_dict(p) or is_table(p)     
def read_data(p,**kws):
    if isinstance(p,dict):
        ## recurse
        ## placed here for uniformity
        datas={}
        for k,p_ in p.items():
            datas[k]=read_data(p_,**kws)
        return datas 
        
    if is_dict(p):
        return read_dict(p,**kws)
    elif is_table(p):
        return read_table(p,**kws)
    else:
        logging.error(f"not detected as data {p}")
        return 
def to_data(data,p,**kws):
    if is_dict(p):
        return to_dict(data,p,**kws)
    elif is_table(p):
        return to_table(data,p,**kws)
    else:
        logging.error(f"not detected as data {p}")
        return 
## dict

def read_dict(
    p,
    fmt: str = "",
    apply_on_keys=None,
    # encoding=None,
    sort_ps=True,
    verbose=False,
    **kws,
) -> dict:
    """Read dictionary file.

    Parameters:
        p (str): path.
        fmt (str): format of the file.

    Keyword Arguments:
        kws (d): parameters provided to reader function.

    Returns:
        d (dict): output dictionary.
    """
    assert isinstance(p, (str, list)), p
    if "*" in p or isinstance(p, list):
        if sort_ps:
            p=read_ps(p,verbose=verbose)
        else:
            assert isinstance(p,list)
        d1 = {p: read_dict(p) for p in p}
        if apply_on_keys is not None:
            assert len(
                set(
                    [
                        replace_many(
                            k, replaces=apply_on_keys, replacewith="", ignore=False
                        )
                        for k in d1
                    ]
                )
            ) == len(d1.keys()), "apply_on_keys(keys)!=keys"
            d1 = {
                replace_many(k, replaces=apply_on_keys, replacewith="", ignore=False): v
                for k, v in d1.items()
            }
        return d1
    if p.endswith(".yml") or p.endswith(".yaml") or fmt == "yml" or fmt == "yaml":
        import yaml

        with open(p, "r") as f:
            d1 = yaml.safe_load(f, **kws)
        return d1 if d1 is not None else {}

    elif p.endswith(".json") or fmt == "json":
        import json

        with open(p, "r") as p:
            return json.load(p, **kws)

    elif p.startswith("https"):
        from urllib.request import urlopen

        try:
            import json

            return json.load(urlopen(p))
        except:
            print(logging.error(p))
    #         return read_json(p,**kws)

    elif p.endswith(".pickle"):
        import pickle

        return pickle.load(open(p, "rb"))

    elif p.endswith(".joblib"):
        import joblib

        return joblib.load(p, **kws)

    else:
        logging.error("supported extensions: .yml .yaml .json .pickle .joblib")


def to_dict(d, p, **kws):
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

    if "My Drive" not in p:
        p = p.replace(" ", "_")
    else:
        logging.warning("probably working on google drive; space/s left in the path.")
    makedirs(p)
    if p.endswith(".yml") or p.endswith(".yaml"):
        import yaml

        with open(p, "w") as f:
            yaml.safe_dump(d, f, **{**dict(sort_keys=False),**kws})
        return p
    elif p.endswith(".json"):
        import json

        with open(p, "w") as outfile:
            json.dump(d, outfile, **kws)
        return p
    elif p.endswith(".pickle"):
        import pickle

        return pickle.dump(d, open(p, "wb"), **kws)
    elif p.endswith(".joblib"):
        import joblib

        return joblib.dump(d, p, **kws)
    else:
        raise ValueError("supported extensions: .yml .yaml .json .pickle .joblib")


## tables
def post_read_table(
    df1: pd.DataFrame,
    clean: bool,
    verbose: bool = True,
    **kws_clean: dict,
):
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
        # print(type(df1))
        df1 = df1.rd.clean(**kws_clean)
    if verbose:
        df1 = df1.log().log.head()
    return df1


from roux.lib.text import get_header

def read_table(
    p: str,
    ext: str = None,

    use_paths: bool = False,  # read files in the path column even if not available in the sub-dir
    use_dir_paths: bool = False,  # =use_paths, will be deprec.d
    filterby_time=None,

    params: dict = None,

    post=True,
    clean: bool = True,
    kws_clean: dict = {},

    kws_cloud: dict = {},

    ## internal
    engine: str = "pyarrow",
    tables: int = 1,     
    test: bool = False,
    verbose: bool = True,
    **kws_read_tables: dict,
):
    """
    Unified table/s reader.

    Parameters:
        p (str): path of the file. It could be an input for `read_ps`, which would include strings with wildcards, list etc.
        ext (str): extension of the file (default: None meaning infered from the path).
        clean=(default:True).
        filterby_time=None).
        use_paths (bool): forced read files in the path column (default:False).
        test (bool): testing (default:False).
        params: parameters provided to the 'pd.read_csv' (default:{}). For example
            params['columns']: columns to read.
        kws_clean: parameters provided to 'rd.clean' (default:{}).
        kws_cloud: parameters for reading files from google-drive (default:{}).
        tables: how many tables to be read (default:1).
        verbose: verbose (default:True).

    Keyword parameters:
        kws_read_tables (dict): parameters provided to `read_tables` function. For example:
            to_col={colindex: replaces_index}

    Returns:
        df (DataFrame): output dataframe.

    Examples:
        1. For reading specific columns only set `params=dict(columns=list)`.

        2. For reading many files, convert paths to a column with corresponding values:

                to_col={colindex: replaces_index}

        3. Reading a vcf file.
                p='*.vcf|vcf.gz'
                read_table(p,
                           params_read_csv=dict(
                           #compression='gzip',
                           sep='\t',comment='#',header=None,
                           names=replace_many(get_header(path,comment='#',lineno=-1),['#','\n'],'').split('\t'))
                           )
    TODOs:
        params -> kws_reader
            with back-compatibility
            kws_reader=params

        Standardised arg.s to be preferred
            kws_read_table=dict(
                path=
                cols= # query cols
                expr= # query rows
                )
            with back-compatibility
                path=p
                cols=kws_reader['columns] 
                expr=
    """
    if params is None:
        params={}
    # print(params,ext)
    
    # params={}
    if tables==1:
        ## check for read_tables
        if isinstance(p, (dict,list)) or ((isinstance(p, str) and ("*" in p))):
            if isinstance(p, str) and ("*" in p):
                _ps = read_ps(p, verbose=False)
                if exists(p.replace("/*", "")):
                    logging.warning(f"exists: {p.replace('/*','')}")
            elif isinstance(p, (list)):
                _ps = p
            elif isinstance(p, dict):
                _ps = p
                kws_read_tables={
                    **kws_read_tables,
                    **dict(
                        to_dict=True
                    ),
                }
            else:
                raise ValueError(p)
                
            return read_tables(
                p,
                params=params,
                filterby_time=filterby_time,
                tables=len(_ps),
                clean=clean,
                verbose=verbose,
                # **{
                **kws_read_tables,
                #     # **dict(
                #     #     verbose=False,
                #     # ),
                # },  # is kws_apply_on_paths,
            )            
        elif isinstance(p, str):
            ## read paths
            if use_paths or use_dir_paths:
                ## if tables in the subdir (common)
                ## 
                # if len(read_ps(f"{splitext(p)[0]}/*{splitext(p)[1]}",test=False))>0:
                df_ = read_table(
                    p,
                    post=False,
                    verbose=False,
                )
                if df_.empty:
                    logging.warning("empty table found")
                    return df_
                elif df_.columns.tolist()[-1] == "path":
                    if verbose:
                        logging.info(
                            f" {len(df_['path'].tolist())} paths from the file."
                        )
                    ps = df_["path"].tolist()
                    # print(ps)
                    return read_tables(
                        ps,
                        params=params,
                        filterby_time=filterby_time,
                        tables=len(ps),
                        clean=clean,
                        verbose=verbose,
                        **kws_read_tables,  # is kws_apply_on_paths,
                    )
                else:
                    return df_
        else:
            raise ValueError(p)
    # else:
        # running through read_tables

    ## 1 table
    # if p.startswith("https://docs.google.com/file/"):
    #     if "outd" not in kws_cloud:
    #         logging.warning("outd not found in kws_cloud")
    #     from roux.lib.google import download_file

    #     return read_table(download_file(p, **kws_cloud))

    assert exists(p), f"not found: {p}"
    if ext is None:
        # ext = basename(p).rsplit(".", 1)[1]
        ext=Path(p).suffix[1:]
        
    if ext in ["pqt", "parquet"]:  # p.endswith('.pqt') or p.endswith('.parquet'):
        # print(params,ext)
        df0=pd.read_parquet(
                p,
                engine=engine,
                **params
            )
    elif len(params.keys()) != 0 and "columns" not in params:        
        df0=pd.read_csv(p, **params)

    else:
        params["compression"] = (
            "gzip" if ext.endswith(".gz") else "zip" if ext.endswith(".zip") else None
        )

        if params["compression"] is not None:
            ext = ext.split(".", 1)[0]

        if any([s == ext for s in ["tsv", "tab", "txt"]]):
            params["sep"] = "\t"
        elif any([s == ext for s in ["csv"]]):
            params["sep"] = ","
        elif ext == "vcf":
            from roux.lib.str import replace_many

            params.update(
                dict(
                    sep="\t",
                    comment="#",
                    header=None,
                    names=replace_many(
                        get_header(path=p, comment="#", lineno=-1), ["#", "\n"], ""
                    ).split("\t"),
                )
            )
        elif ext == "gpad":
            params.update(
                dict(
                    sep="\t",
                    names=[
                        "DB",
                        "DB Object ID",
                        "Qualifier",
                        "GO ID",
                        "DB:Reference(s) (|DB:Reference)",
                        "Evidence Code",
                        "With (or) From",
                        "Interacting taxon ID",
                        "Date",
                        "Assigned by",
                        "Annotation Extension",
                        "Annotation Properties",
                    ],
                    comment="!",
                )
            )
        else:
            raise ValueError(f"unknown extension {ext} in {p}")
        if test:
            print(params)
        df0=pd.read_table(
                p,
                **params,
            )
        del params
    if verbose and 'path' in df0:        
        logging.info("use_paths=True to read the paths in the col 'path'.")
        
    if not post:
        return df0
    else:
        return post_read_table(
            df0,
            clean=clean,
            verbose=verbose,
            **kws_clean,
        )      
        
def get_logp(
    ps: list,
) -> str:
    """Infer the path of the log file.

    Parameters:
        ps (list): list of paths.

    Returns:
        p (str): path of the output file.
    """
    from roux.lib.str import get_prefix

    p = get_prefix(min(ps), max(ps), common=True, clean=True)
    if not isdir(p):
        p = dirname(p)
    return f"{p}.log"


def apply_on_paths(
    ps: list,
    func,
    replaces_outp: str = None,
    to_col: dict = None,
    replaces_index=None,
    drop_index: bool = True,  # keep path
    colindex: str = "path",
    filter_rows: dict = None,
    # progress_bar: bool = True,
    params: dict = {},

    cpus: int = 1,
    fast: bool = False, # to be deprecated

    dbug: bool = False,
    test1: bool = False,
    verbose: bool = True,
    kws_read_table: dict = {},
    **kws: dict,
):
    """Apply a function on list of files.

    Parameters:
        ps (str|list): paths or string to infer paths using `read_ps`.
        to_col (dict): convert the paths to a column e.g. {colindex: replaces_index}
        func (function): function to be applied on each of the paths.
        replaces_outp (dict|function): infer the output path (`outp`) by replacing substrings in the input paths (`p`).
        filter_rows (dict): filter the rows based on dict, using `rd.filter_rows`.
        fast (bool|int): parallel processing tasks (default:False).
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
    if isinstance(fast,int):
        cpus=fast
        del fast

    def read_table_(
        p,
        read_path=False,
        save_table=False,
        filter_rows=None,
        replaces_outp=None,
        params={},
        dbug=False,
        verbose=True,
        **kws_read_table,
    ):
        if isinstance(p,pd.DataFrame):
            p = p.iloc[0, :]["path"]
        if read_path:
            if save_table:
                outp = replace_many(
                    p, replaces=replaces_outp, replacewith="", ignore=False
                )
                if dbug:
                    logging.debug(outp)
                return p, outp
            else:
                return (p,)
        else:
            try:
                df = read_table(
                    p,
                    params=params,
                    verbose=verbose,
                    **kws_read_table
                )
            except Exception as e:
                raise ValueError(f"An error occurred for {p}: {e}.")
                
            if filter_rows is not None:
                df = df.rd.filter_rows(filter_rows)
            return (df,)
    
    import inspect

    read_path = inspect.getfullargspec(func).args[0] == "p"
    save_table = (replaces_outp is not None) and (
        "outp" in inspect.getfullargspec(func).args
    )

    if to_col is not None:
        colindex = list(to_col.keys())[0]
        replaces_index = list(to_col.values())[0]

    if replaces_index is not None:
        drop_index = False
        
        if replaces_index=='ids':
            # print(ps)
            from roux.lib.dict import flip_dict
            to_ids=flip_dict(
                    read_ps(
                    ps,
                    fmt=replaces_index,
                    verbose=verbose,
                )
            )
            # print(to_ids)
            replaces_index=lambda x: to_ids[x]

    ps = read_ps(ps, verbose=verbose)

    if len(ps) == 0:
        logging.error("no paths found")
        return
        
    if test1:
        ps = ps[:1]
        logging.warning(f"test1=True, {ps[0]}")
        
        
    if (replaces_outp is not None) and ("force" in kws):
        if not kws["force"]:
            # p2outp
            p2outp = {
                p: replace_many(p, replaces=replaces_outp, replacewith="", ignore=False)
                for p in ps
            }
            if dbug:
                print(p2outp)
            d_ = {}
            d_["from"] = len(ps)
            ps = [p for p in p2outp if (not exists(p2outp[p])) or isdir(p2outp[p])]
            d_["  to"] = len(ps)
            if d_["from"] != d_["  to"]:
                logging.info(f"force=False, so paths reduced from: {d_['from']}")
                logging.info(f"                                to: {d_['  to']}")
    
    if dbug:
        logging.info(ps)
        
    df1 = pd.DataFrame({"path": ps})
    
    if len(df1) == 0:
        logging.info("no paths remained to be processed.")
        return df1

    
    if cpus>1 and drop_index:
        logging.info(f"using {cpus} cpus ..")
        import roux.lib.df_apply as rd #noqa
        return df1.rd.apply_async(
            lambda x: read_table(
                x['path'],
                **{
                    **kws_read_table,
                    **dict(
                        verbose=False,
                    ),
                },
            ),
            cpus=cpus,
            unstack=False,
        )
        
    _groupby = df1.groupby("path", as_index=True,sort=False)

    df2 = getattr(
        _groupby,
        # "progress_apply"
        # else "progress_apply"
        # if hasattr(_groupby, "progress_apply")
        # else "apply",
        "apply",
    )(
        lambda df: func(
            *(
                read_table_(
                    df,
                    read_path=read_path,
                    save_table=save_table,
                    replaces_outp=replaces_outp,
                    filter_rows=filter_rows,
                    params=params,
                    dbug=dbug,
                    verbose=verbose,
                    **{
                        **kws_read_table,
                        ## only last
                        **dict(
                            post=False,#(df.name==next(iter(_groupby.groups), None)),
                        ),
                    },
                )
            ),
            **kws,
        )
    )

    if isinstance(df2, pd.Series):
        return df2
        
    if save_table:
        if len(df2) != 0 and not test1:
            # save log file
            from roux.lib.set import to_list

            logp = get_logp(df2.tolist())
            to_list(df2.tolist(), logp)
            logging.info(logp)
        return df2
        
    if drop_index:
        df2 = df2.rd.clean().reset_index(drop=drop_index).rd.clean()
    else:
        df2 = df2.rd.clean().reset_index(drop=drop_index).rd.clean()
        if colindex != "path":
            if colindex in df2:
                logging.warning(f"{colindex} found in the dataframe; hence dropped.")
                df2=df2.drop(
                    [colindex],
                    axis=1
                )
            df2 = df2.rename(columns={"path": colindex}, errors="raise")
    if replaces_index is not None:
        logging.debug(f"setting {colindex} column from the paths ..")
        # print(replaces_index)
        # print(df2[colindex].head())
        if isinstance(replaces_index, str):
            if replaces_index in ['stem',"basenamenoext"]:
                replaces_index = basenamenoext
        ## update: faster renaming
        to_value={
            x: replace_many(
                    x,
                    replaces=replaces_index,
                    replacewith="",
                    ignore=False
                    )  for x in df2[colindex].unique()
        }
        df2 = (
            df2
            .assign(
                **{
                    colindex:lambda df: df[colindex].map(to_value)
                   }
                )
            )
    return df2


def read_tables(
    ps: list,
    filterby_time=None,
    to_dict: bool = False,
    params: dict = {},
    tables: int = None,

    post=True,
    clean=True,
    
    verbose=True,
    **kws_apply_on_paths: dict, # cpus
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
    if filterby_time is not None:
        if isinstance(ps, str) and ("*" in p):
            ps = read_ps(ps, verbose=False)
            if exists(ps.replace("/*", "")):
                logging.warning(f"exists: {p.replace('/*','')}")
        elif isinstance(p, list):
            ps = ps
            
        from roux.lib.sys import ps2time
        df_ = ps2time(ps)
        ps = df_.loc[df_["time"].str.contains(filterby_time), "p"].unique().tolist()
        kws_apply_on_paths["drop_index"] = False  # see which files are read

    if not to_dict and not isinstance(ps,dict):
        df0 = apply_on_paths(
            ps,
            func=lambda df: df,
            # cpus=cpus,
            # drop_index=drop_index,
            params=params,
            kws_read_table=dict(tables=tables),
            #                            kws_read_table=dict(verb=False if len(ps)>5 else True),
            **kws_apply_on_paths,
        )
        if not post:
            return df0
        else:
            ## only at the end
            return post_read_table(
                df0,
                clean=clean,
                verbose=verbose,
                # **kws_clean,
            )
    else:
        if not isinstance(ps,dict):
            ps=read_ps(ps,verbose=False)
            ps=dict(zip(ps,ps))
        return {k: read_table(p, params=params,verbose=False) for k,p in ps.items()}

## save table
def to_table(
    df: pd.DataFrame, p: str, colgroupby: str = None, test: bool = False,
    # cpus = 1, #TODO
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
    # if is_interactive_notebook():
    #     test = True
    p = to_path(p)
    if df is None:
        df = pd.DataFrame()
        logging.warning(f"empty dataframe saved: {p}")
    #     if len(basename(p))>100:
    #         p=f"{dirname(p)}/{basename(p)[:95]}_{basename(p)[-4:]}"
    #         logging.warning(f"p shortened to {p}")
    if df.index.name is not None:
        df = df.reset_index()
    if not exists(dirname(p)) and dirname(p) != "":
        makedirs(p, exist_ok=True)
    if colgroupby is not None:
        to_manytables(df, p, colgroupby, **kws)
    elif p.endswith(".tsv") or p.endswith(".tab"):
        df.to_csv(
            p,
            sep="\t",
            **{**dict(index=False), **kws},
        )
    elif p.endswith(".pqt"):
        to_table_pqt(df, p, **kws)
    else:
        raise ValueError(f"unknown extension {p}")
    return p


def to_manytables(
    df: pd.DataFrame,
    p: str,
    colgroupby: str,
    fmt: str = "",
    ignore: bool = False,
    kws_get_chunks={},
    # cpus = 1, #TODO
    **kws_to_table,
):
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
    outd, ext = splitext(p)
    if isinstance(colgroupby, str):
        colgroupby = [colgroupby]
    if colgroupby == "chunk":
        from roux.lib.df import get_chunks

        if not ignore:
            if exists(outd):
                logging.error(f"can not overwrite existing chunks: {outd}/")
            assert not exists(outd), outd
        df = get_chunks(df1=df, **kws_get_chunks)
    #
    if (df.loc[:, colgroupby].dtypes == "float").any():
        logging.error("columns can not be float")
        logging.info(df.loc[:, colgroupby].dtypes)
        return
    elif (df.loc[:, colgroupby].dtypes == "bool").any():
        fmt = "="
        logging.warning("bool column detected, fmt changed to =")

    def to_outp(names, outd, colgroupby, fmt):
        """
        Get output path for each group.
        """
        if isinstance(names, str):
            names = [names]
        d1 = dict(zip(colgroupby, names))
        s1 = "/".join(
            [(f"{k}{fmt}" if fmt != "" else fmt) + f"{str(v)}" for k, v in d1.items()]
        )
        return to_path(f"{outd}/{s1}{ext}")

    _groupby = df.groupby(colgroupby)
    df2 = (
        getattr(
            _groupby,
            "progress_apply" if hasattr(_groupby, "progress_apply") else "apply", #TODO rd.apply(cpus=cpus)
        )(
            lambda x: to_table(
                x,
                to_outp(
                    names=x.name,
                    outd=outd,
                    colgroupby=colgroupby,
                    fmt=fmt,
                ),
                **kws_to_table,
            )
        )
        .to_frame("path")
        .reset_index()
    )
    to_table(
        df2,
        p,
    )


def to_table_pqt(
    df: pd.DataFrame,
    p: str,
    engine: str = "pyarrow",
    compression: str = "gzip",
    **kws_pqt: dict,
) -> str:
    """Save a parquet file.

    Parameters:
        df (pd.DataFrame): table.
        p (str): path.

    Keyword parameters:
        Parameters provided to `pd.DataFrame.to_parquet`.

    Returns:

    """
    if len(df.index.names) > 1:
        df = df.reset_index()
    if not exists(dirname(p)) and dirname(p) != "":
        makedirs(p, exist_ok=True)
    df.to_parquet(p, engine=engine, compression=compression, **kws_pqt)
    return p

def to_tables(
    dfs,
    outp,
    ):
    mdata_raw={}
    ps={}
    for k,df in dfs.items():
        if isinstance(df,dict):
            ## recurse
            outp_=f"{Path(outp).with_suffix('').as_posix()}/{k}.yaml"
            ps[k]=to_tables(
                df,
                outp_,
            )
            mdata_raw[k]=read_dict(outp_)
        else:      
            mdata_raw[k]=dict(shape=df.shape,cols=df.columns.tolist())
            ps[k]=to_table(
                df,
                f"{Path(outp).with_suffix('').as_posix()}/{k}.pqt"
            )
        
    logging.info({k:d if 'shape' in d else {k_:d_['shape'] for k_,d_ in d.items()} for k,d in mdata_raw.items()})
    to_dict(
        mdata_raw,
        outp,
    )
    return ps

# import pandas as pd
# from roux.lib.io import to_table
# from glob import glob
def to_tables_flt(
    dir_path,
    outp,
    ids: list,
    force=False,
    test1=False,
    check=False, # not save
    ):
    """
    Links the file input file paths. Thereby avoids copying of files.
    """
    from roux.lib.set import flatten
    if  isinstance(dir_path,dict): 
        dir_paths=dir_path
        ## recursive
        to_var={}
        for var,dir_path in dir_paths.items():
            to_var[var]=to_tables_flt(
                dir_path=dir_path,
                outp=f"{outp}/{var}.tsv",
                ids=ids,
            )
        return to_var    
    if not Path(outp).exists() or force:
        logging.info("writing: "+outp)
        # ps=[read_ps(f"{dir_path}/{k}.*",verbose=False) for k in tqdm(ids)]
        # if '*' in dir_path and exists()
        ps=flatten(
            [glob(f"{dir_path}/{k}.pqt") for k in ids]
            )
        # ps=[f"{dir_path}/{k}.pqt" for k in ids]
        # logging.info(f"len(ps)={len(ps)}")
        # ps=[p for p in ps if Path(p).exists()]
        if len(ps)==0:
            logging.warning(f"not found  = {dir_path} ..")
            return
        logging.info(f"exist  = {len(ps)}")
        assert len(ps)<=len(ids),(len(ps), len(ids))
        if len(ps)<len(ids):
            logging.warning(f"{len(ps), len(ids)}")
        df_=pd.DataFrame(
            {
                'path':ps
            }
            ).drop_duplicates().log()
        if not check:
            to_table(
                df_,
                outp,
            )
        if test1:
            print(read_table(df_['path'].tolist()[0]).head(1))
    return outp
    
def tsv2pqt(
    p: str,
) -> str:
    """Convert tab-separated file to Apache parquet.

    Parameters:
        p (str): path of the input.

    Returns:
        p (str): path of the output.
    """
    to_table_pqt(pd.read_csv(p, sep="\t", low_memory=False), f"{p}.pqt")


def pqt2tsv(
    p: str,
) -> str:
    """Convert Apache parquet file to tab-separated.

    Parameters:
        p (str): path of the input.

    Returns:
        p (str): path of the output.
    """
    ps=read_ps(p)
    if len(ps)>1:
        ## recursive
        logging.info("converting files recursively")    
        for p_ in read_ps(p):
            pqt2tsv(
                p_,
            )
    to_table(
        read_table(p),
        Path(p).with_suffix(".tsv").as_posix(),
        )


## tables: excel
def read_excel(
    p: str,
    sheet_name: str = None,
    kws_cloud: dict = {},
    test: bool = False,
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
    # if not 'xlrd' in sys.modules:
    #   logging.error('need xlrd to work with excel; pip install xlrd')
    if isinstance(p, str):
        if p.startswith("https://docs.google.com/spreadsheets/"):
            if "outd" not in kws_cloud:
                raise ValueError("outd not found in kws_cloud")
            from roux.lib.google import download_file

            return read_excel(download_file(p, **kws_cloud), **kws)
    if sheet_name is None:
        xl = pd.ExcelFile(p)
        if test:
            logging.info(
                f"`sheet_name`s (to select from) the excel file : {xl.sheet_names}"
            )
            return xl
        ## return all sheets
        sheetname2df = {}
        for sheet_name in xl.sheet_names:
            sheetname2df[sheet_name] = xl.parse(sheet_name)
            logging.info(f"'{sheet_name}':{sheetname2df[sheet_name].shape}")
        return sheetname2df
    else:
        return pd.read_excel(p, sheet_name, **kws)


def to_excel_commented(
    p: str,
    comments: dict,
    outp: str = None,
    author: str = None,
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
        author = "Author"
    if outp is None:
        outp = p
        logging.warning("overwritting the input file")
    from openpyxl import load_workbook
    from openpyxl.comments import Comment
    from string import ascii_uppercase

    wb = load_workbook(filename=outp)
    for sh in wb:
        for k in [
            s + "1"
            for s in list(ascii_uppercase) + ["A" + s_ for s_ in ascii_uppercase]
        ]:
            if sh[k].value is not None:
                if sh[k].value in comments:
                    sh[k].comment = Comment(comments[sh[k].value], author=author)
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
    comments: dict = None,
    save_input: bool = False,
    author: str = None,
    append: bool = False,
    adjust_column_width: bool = True,
    **kws,
):
    """Save excel file.

    Parameters:
        sheetname2df (dict): dictionary mapping the sheetname to the dataframe.
        outp (str): output path.
        append (bool): append the dataframes (default:False).
        comments (dict): map between column names and comment e.g. description of the column.
        save_input (bool): additionally save the input tables in text format.

    Keyword parameters:
        kws: parameters provided to the excel writer.
    """
    #     if not 'xlrd' in sys.modules:
    #         logging.error('need xlrd to work with excel; pip install xlrd')
    # makedirs(outp)
    if comments is not None:
        ## order the columns
        cols=[]
        for k1 in sheetname2df:
            sheetname2df[k1] = sheetname2df[k1].loc[
                :,
                [k for k in comments if k in sheetname2df[k1]]
                + [k for k in sheetname2df[k1] if k not in comments],
            ]
            cols+=sheetname2df[k1].columns.tolist()
        cols=list(set(cols))
        if not any([k.lower().startswith("descr") for k in sheetname2df]):
            ## insert a table with the description
            items = list(sheetname2df.items())
            from roux.lib.df import dict2df

            items.insert(
                0,
                (
                    "description",
                    dict2df({k:v for k,v in comments.items() if k in cols}, colkey="column name", colvalue="description"),
                ),
            )
            sheetname2df = dict(items)

    outp = makedirs(to_path(outp))
    writer = pd.ExcelWriter(outp)
    startrow = 0
    for k, df_ in sheetname2df.items():
        if not append:
            df_.to_excel(writer, k, index=False, **kws)
        else:
            df_.to_excel(writer, startrow=startrow, index=False, **kws)
            startrow += len(df_) + 2
        if adjust_column_width:
            # auto-adjust column widths
            for c in df_:
                col_idx = df_.columns.get_loc(c)
                try:
                    writer.sheets[k].set_column(
                        col_idx,
                        col_idx,
                        max(df_[c].astype(str).map(len).max(), len(c)),
                        # 30,
                    )
                except:
                    logging.error(
                        "to adjust column width use xlsxwriter: pip install xlsxwriter"
                    )
    try:
        writer.close()
    except:
        # old pandas version
        writer.save()
    ## save in tsv format
    if save_input:
        for k in sheetname2df:
            to_table(sheetname2df[k], f"{splitext(outp)[0]}/{k}.tsv")

    if comments is not None:
        to_excel_commented(
            outp,
            comments=comments,
            outp=outp,
            author=author,
        )
    return outp


## to table: validate
def check_chunks(outd, col, plot=True):
    """Create chunks of the tables.

    Parameters:
        outd (str): output directory.
        col (str): the column with values that are used for getting the chunks.
        plot (bool): plot the chunk sizes (default:True).

    Returns:
        df3 (DataFrame): output dataframe.
    """
    df1 = pd.concat(
        {p: read_table(f"{p}/*.pqt", params=dict(columns=[col])) for p in glob(outd)}
    )
    df2 = df1.reset_index(0).log.dropna()
    df3 = df2.groupby("level_0")[col].nunique()
    df3.index = [s.replace(outd, "").replace("/", "") for s in df3.index]
    logging.info(df3)
    if plot:
        import seaborn as sns

        sns.swarmplot(df3)
    return df3

def is_table_empty(
    p,
    ):
    """
    Check for empty table without opening the file.
    
    Tests:
    
        p=to_table(
        pd.DataFrame(
            ),
            # 'test/test.tsv'
            'test/test.pqt'
        )
        
        Path(p).stat().st_size
    """
    if (
        Path(p).suffix in ['.tsv'] and Path(p).stat().st_size<2
        or 
        Path(p).suffix in ['.pqt','.parquet'] and Path(p).stat().st_size<977
    ):
        return True
    else:
        logging.info(Path(p).stat().st_size)
        return False
"""For task executions."""

## logging
import logging
from tqdm import tqdm 
## paths
from pathlib import Path

import pandas as pd

import roux.lib.dfs as rd #noqa
from roux.lib import to_rd

from roux.lib.io import read_table, to_table

@to_rd
def apply_async(
    df: pd.DataFrame,
    func, # lambda x: 
    cpus: int,
    unstack: bool=True, ## to cols
    axis=1, #noqa ## unused, for swappability with .apply
    by=None, ## groupby
    verbose=False,
    **kws, ## to by
    ) -> list:
    """
    Notes:
        post:
            expand assigned to cols:
                df1.join(pd.DataFrame(df1['res'].tolist())).drop(columns='res')
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = {}
    with ThreadPoolExecutor(max_workers=cpus) as executor:
        if by is None:
            ## row/item wise
            idx=df.index.tolist()        
            if len(idx)==0:
                return
            else:
                assert idx==list(range(len(df))), "before apply_async, need: .reset_index(drop=True)"#, {idx}, {list(range(len(df)))}"
                
            futures = {
                executor.submit(func, row): name
                for name, row in df.iterrows()
            }             
        else:
            ## groupby
            futures = {
                executor.submit(func, group): name
                for name, group in df.groupby(by,**kws)
            }
        for future in tqdm(as_completed(futures)):
            name = futures[future]
            try:
                results[name] = future.result()
            except:
                raise ValueError(name)
    
    if by is None:
        # # Since Python 3.7 (and officially in the language spec from 3.8), the built-in dict preserves insertion order.
        # results=list(results.values())
        ## but the llel proc. can reorder the insertion order
        results=[results[i] for i in idx]
        if isinstance(results[0],(pd.Series)):
            return pd.concat([ds.to_frame().T for ds in results],axis=0)
        elif isinstance(results[0],(pd.DataFrame)):
            df1 = pd.concat(results,axis=0)
            if unstack:
                df1=df1.unstack(1)
            return df1
        else: 
            return pd.Series(results)
    else:
        # # Combine all group results into a single DataFrame
        return  pd.concat(
            results,
            names=by,
            # ignore_index=True
            ) 

@to_rd
def apply_async_chunks(
    data : pd.DataFrame,
    func,
    col_id : str = None,
    func_to_df=None, ## to make df compatible to save
    chunk_size : int = 1000,
    fn_len : int = None,
    kws_get_chunks : dict = {},
    outd : str = None,
    out_df=False, ## for assign
    clean : bool = True,
    verbose : bool = False,
    force : bool = False,
    test1: bool=False,
    **kws_apply_async,
    ) -> pd.DataFrame:
    """
    Notes:
        - [x] Parallel.
        - [x] Resumable.
    """
    assert 'out' not in data
    
    if test1:
        force=True
        
    if col_id==None:
        col_id='temp_col_id'
        data=(
            data.assign(
                **{
                    col_id:range(len(data)),
                }
            )
        )
        temp_col_id=True
    else:
        temp_col_id=False
        
    logging.info("chunking ..")
    data=(
        data
        .rd.assert_no_dups(
            subset=[col_id],
        )
        .sort_values(
            [col_id]
        )
        .rd.get_chunks(
            size=chunk_size,
            **kws_get_chunks,
        )
    )    
    if fn_len is None:
        fn_len=len(str(len(data)))+1    
    else:
        assert fn_len>=len(str(len(data)))+1
    
    if outd is None:
        import tempfile
        outd=tempfile.TemporaryDirectory().name
        logging.info(f"temp. dir path: {outd}")
        temp_outd=True #todo delete temp dir in post.
    else:
        temp_outd=False

    collect_from_paths=False
    outps=[]
    dfs_out={}
    for k,df_ in tqdm(data.groupby('chunk')):
        outp=f"{outd}/{k:0{fn_len}d}.pqt"
        outps.append(outp)
        if (not Path(outp).exists() or force) or temp_outd:
            if verbose:
                logging.info(f"processsing {outp} ..")
            df_out=(
                df_
                .reset_index(drop=True)
                .rd.apply_async(
                    func=func,
                    **kws_apply_async,
                )
                )
            
            if isinstance(df_out,pd.Series):
                df_out=df_out.to_frame()
                
            df_out=(
                df_out
                .assign(
                    **{
                        col_id:df_[col_id].tolist(),
                    },
                )
                .rename(columns={0:'out'})
                )
            if func_to_df is not None:
                df_out=func_to_df(df_out)
            # print(df_out)
            to_table(
                df_out,
                outp,
            )
            if not collect_from_paths:
                dfs_out[k]=df_out
            if test1:
                logging.info("returning becaue test1=True")
                return df_out
        else:
            collect_from_paths=True
                
    logging.info("collecting processed chunks ..")
    from roux.lib.sys import read_ps
    assert data['chunk'].nunique()==len(read_ps(outps,errors='raise',verbose=False))
    
    if not collect_from_paths:
        if verbose:
            logging.info(f"{collect_from_paths:=}")
        df1=(
            pd.concat(
                dfs_out,
                axis=0,
                names=['chunk']
            )
            .reset_index(0)
        )
    else:
        if verbose:
            logging.info(f"{collect_from_paths:=}")
        df1=(
            read_table(
                outps,
                to_col=dict(chunk=lambda x: int(Path(x).stem)),
            )
        )
    
    df1=(
        df1 
        ## back to the original order
        .sort_values(
            [col_id]
        )
        .reset_index(drop=True)
       )
    
    if not out_df:
        assert len(df1) == len(data), (len(data), len(df1))
        return df1['out']
        
    df2=(
        data
        .log.merge(
            right=df1,
            how='left',
            on=[col_id,'chunk'],
            validate="1:1",
        )
    )
    
    if clean:
        df2=df2.drop(
            ['chunk'],
            axis=1,
        )
        
    if temp_col_id:
        df2=df2.drop([col_id],axis=1)
        
    return df2

## wrapper
@to_rd
def apply(
    ## apply async
    data: pd.DataFrame,
    func, # lambda x: 
    cpus: int,
    unstack: bool=True, ## to cols
    axis=1, #noqa ## unused, for swappability with .apply
    by=None, ## groupby
    verbose=False,
    ## chunks
    kws_chunks={},
    
    **kws, ## to by   
):
    if len(kws_chunks)==0:
        return apply_async(
            data,  #: pd.DataFrame,
            func=func,  #, # lambda x: 
            cpus=cpus,  #: int,
            unstack=unstack,  #: bool=True, ## to cols
            axis=axis,  #=1, #noqa ## unused, for swappability with .apply
            by=by,  #=None, ## groupby
            verbose=verbose,  #=False,
            **kws, ## to by
        )
    else:
        return apply_async_chunks(
            data,  #: pd.DataFrame,
            func=func,  #, # lambda x: 
            cpus=cpus,  #: int,
            unstack=unstack,  #: bool=True, ## to cols
            axis=axis,  #=1, #noqa ## unused, for swappability with .apply
            by=by,  #=None, ## groupby
            verbose=verbose,  #=False,
            **kws, ## to by
            
            **kws_chunks,
        )
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
    func,
    cpus: int,
    unstack: bool=True,
    ) -> list:
    
    idx=df.index.tolist()
    
    if len(idx)==0:
        return
    else:    
        assert idx==list(range(len(df))), f"before apply_async, need: .reset_index(drop=True), {idx}, {list(range(len(df)))}"
    
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=cpus
    ) as executor:
        results = list(
            executor.map(
                func,
                [x for _, x in df.iterrows()],
            )
        )
    if isinstance(results[0],(pd.Series)):
        return pd.concat([ds.to_frame().T for ds in results],axis=0)
    elif isinstance(results[0],(pd.DataFrame)):
        df1 = pd.concat(results,axis=0)
        if unstack:
            df1=df1.unstack(1)
        return df1
    else: 
        return pd.Series(results)

@to_rd
def apply_async_chunks(
    df : pd.DataFrame,
    func,
    cpus : int,
    col_id : str = None,
    func_to_df=None, ## to make df compatible to save
    chunk_size : int = 1000,
    fn_len : int = None,
    kws_get_chunks : dict = {},
    outd : str = None,
    clean : bool = True,
    verbose : bool = False,
    force : bool = False,
    test1: bool=False,
    **kws_apply_async,
    ) -> pd.DataFrame:
    if test1:
        force=True
        
    if col_id==None:
        col_id='temp_col_id'
        df=(
            df.assign(
                **{
                    col_id:range(len(df)),
                }
            )
        )
        temp_col_id=True
    else:
        temp_col_id=False
        
    logging.info("chunking ..")
    df=(
        df
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
        fn_len=len(str(len(df)))+1    
    else:
        assert fn_len>=len(str(len(df)))+1
    
    if outd is None:
        import tempfile
        outd=tempfile.TemporaryDirectory().name
        logging.info(f"temp. dir path: {outd}")
        temp_outd=True #todo delete temp dir in post.
    else:
        temp_outd=False
    # func_=lambda x: (x[col_id],func(x))
    outps=[]
    for k,df_ in tqdm(df.groupby('chunk')):
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
                    cpus=cpus,
                    **kws_apply_async,
                )
                )
            # return df_out
            if isinstance(df_out,pd.Series):
                df_out=df_out.to_frame()
            df_out=(
                df_out
                .assign(
                    **{
                        col_id:df_[col_id].tolist(),
                    },
                )
                )
            if func_to_df is not None:
                df_out=func_to_df(df_out)
            # print(df_out)
            to_table(
                df_out,
                outp,
            )
            if test1:
                logging.info("returning becaue test1=True")
                return df_out
                
    logging.info("collecting processed chunks ..")
    assert df['chunk'].nunique()==len(outps)
    
    df1=read_table(
        outps,
        to_col=dict(chunk=lambda x: int(Path(x).stem)),
    )
    
    df2=(
        df
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


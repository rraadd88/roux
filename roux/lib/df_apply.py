"""For task executions."""

## logging
import logging
from tqdm import tqdm 
## os
import os

## paths
from pathlib import Path
import pandas as pd
import tempfile
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import roux.lib.dfs as rd #noqa
from roux.lib import to_rd

from roux.lib.io import read_table, to_table

@to_rd
def apply_async(
    df: pd.DataFrame,
    func, # lambda x: 
    cpus: int,
    executor_type: str = 'thread', # Choose 'thread' or 'process'
    unstack: bool=True, ## to cols
    axis=1, #noqa ## unused, for swappability with .apply
    by=None, ## groupby
    verbose=False,
    **kws, ## to by
) -> pd.DataFrame:
    """
    Applies a function to a DataFrame in parallel.

    Notes:
        - For CPU-bound functions, use executor_type='process' to bypass the GIL.
        - For I/O-bound functions, executor_type='thread' is sufficient.
    """
    executor_class = {'thread': ThreadPoolExecutor, 'process': ProcessPoolExecutor}.get(executor_type)
    if executor_class is None:
        raise ValueError("executor_type must be 'thread' or 'process'")

    results = {}
    with executor_class(max_workers=cpus) as executor:
        if by is None:
            idx = df.index.tolist()
            if not idx:
                return pd.DataFrame()
            assert idx == list(range(len(df))), "DataFrame index must be reset before using apply_async without 'by'."
            futures = {executor.submit(func, row): name for name, row in df.iterrows()}
        else:
            if isinstance(by, str):
                by = [by]
            futures = {executor.submit(func, group): name for name, group in df.groupby(by, **kws)}

        iterable = as_completed(futures)
        if verbose:
            iterable = tqdm(iterable, total=len(futures))

        for future in iterable:
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                logging.error(f"An error occurred while processing group/row: {name}")
                raise e

    if by is None:
        results = [results[i] for i in idx]
        if not results:
            return pd.Series(dtype='object')
        if isinstance(results[0], pd.Series):
            return pd.concat([ds.to_frame().T for ds in results], axis=0).reset_index(drop=True)
        elif isinstance(results[0], pd.DataFrame):
            df1 = pd.concat(results, axis=0)
            if unstack:
                df1 = df1.unstack(1)
            return df1
        else:
            return pd.Series(results)
    else:
        return pd.concat(results, names=by)

def _process_and_save_chunk(df_chunk, chunk_path, func, kws_apply_async, func_to_df, col_id, test1):
    """Helper to process one chunk and save it to a file."""
    if Path(chunk_path).exists() and not test1: # Preserve test1 logic
        logging.info(f"Skipping existing chunk: {chunk_path}")
        return None

    logging.info(f"Processing chunk for {chunk_path}...")
    df_out = df_chunk.reset_index(drop=True).rd.apply_async(
        func=func,
        **kws_apply_async,
    )

    if isinstance(df_out, pd.Series):
        df_out = df_out.to_frame(name='out')
    
    df_out[col_id] = df_chunk[col_id].tolist()

    if func_to_df is not None:
        df_out = func_to_df(df_out)

    to_table(df_out, chunk_path)
    return df_out


@to_rd
def apply_async_chunks(
    data: pd.DataFrame,
    func,
    col_id: str = None,
    func_to_df=None, ## to make df compatible to save
    chunk_size: int = 1000,
    fn_len : int = None,
    kws_get_chunks: dict = {},
    outd: str = None,
    out_df=False, ## for assign
    out_ps=False, ## for i/o tables
    clean: bool = True,
    verbose: bool = False,
    force: bool = False,
    test1: bool=False, # PRESERVED ARGUMENT
    **kws_apply_async,
) -> pd.DataFrame:
    """
    Applies a function to a DataFrame in parallel, saving results in chunks.
    This is resumable if an `outd` is provided.
    """
    if test1: # PRESERVED LOGIC
        force = True

    temp_col_id = False
    if col_id is None:
        col_id = '_temp_col_id'
        data = data.assign(**{col_id: range(len(data))})
        temp_col_id = True

    data = data.rd.assert_no_dups(subset=[col_id]).sort_values([col_id])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        output_directory = outd if outd is not None else temp_dir
        if outd is None:
            logging.info(f"Using temporary directory for chunks: {output_directory}")

        chunked_data = data.rd.get_chunks(size=chunk_size, **kws_get_chunks)
        
        if fn_len is None:
            fn_len=len(str(len(data)))+1
        else:
            assert fn_len>=len(str(len(data)))+1

        chunk_paths = []
        dfs_out = {}
        collect_from_paths = False

        for k, df_chunk in tqdm(chunked_data.groupby('chunk'), disable=not verbose):
            chunk_path = Path(output_directory) / f"{k:0{fn_len}d}.pqt"
            chunk_paths.append(str(chunk_path))
            if force:
                chunk_path.unlink(missing_ok=True)
            
            df_out = _process_and_save_chunk(df_chunk, chunk_path, func, kws_apply_async, func_to_df, col_id, test1)

            if df_out is not None:
                 if not collect_from_paths:
                    dfs_out[k] = df_out
                 if test1: # PRESERVED LOGIC
                    logging.info("returning because test1=True")
                    return df_out
            else:
                collect_from_paths = True


        logging.info("Collecting processed chunks...")
        if not collect_from_paths:
            df_results = pd.concat(dfs_out, axis=0, names=['chunk']).reset_index(0)
        else:
            df_results = read_table(chunk_paths, to_col=dict(chunk=lambda x: int(Path(x).stem)))

    if out_ps:
        return data

    df_final = data.merge(df_results, how='left', on=[col_id,'chunk'] if 'chunk' in df_results else col_id, validate="1:1")
    
    if out_df:
        if clean:
            df_final = df_final.drop(columns=['chunk'])
        if temp_col_id:
            df_final = df_final.drop(columns=[col_id])
        return df_final
    else:
        if not out_df and not out_ps:
            assert len(df_final) == len(data), (len(data), len(df_final))
            return df_final['out'] if 'out' in df_final else df_final


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
    logging.info(f"using {cpus}/{os.cpu_count()} cpus/threads ..")
    
    if not kws_chunks:
        return apply_async(
            data,
            func=func,
            cpus=cpus,
            unstack=unstack,
            axis=axis,
            by=by,
            verbose=verbose,
            **kws,
        )
    else:
        return apply_async_chunks(
            data,
            func=func,
            cpus=cpus,
            unstack=unstack,
            axis=axis,
            by=by,
            verbose=verbose,
            kws_chunks=kws_chunks,
            **kws,
        )
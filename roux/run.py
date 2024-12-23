"""For access to a few functions from the terminal."""

import logging

logging.getLogger().setLevel(logging.INFO)

import argh
from roux.lib.io import backup, to_version, to_zip
from roux.lib.io import pqt2tsv
from roux.lib.sys import read_ps
from roux.workflow.io import read_config, read_metadata, replacestar, to_clean_nb, to_html
from roux.workflow.task import (
    # run_task, 
    run_tasks ## preferred because it infers setup for the outputs
)
from roux.workflow.nb import to_clear_unused_cells, to_clear_outputs

def head_table(
    p : str,
    n : int = 5,
    use_dir_paths : bool = True,
    not_use_dir_paths : bool = False,
    use_paths : bool = True,
    not_use_paths : bool = False,
    **kws,
    ):
    if not_use_dir_paths: 
        use_dir_paths=False
    if not_use_paths: 
        use_paths=False
        
    from roux.lib.io import read_table
    return read_table(
        p,
        use_dir_paths=use_dir_paths,
        use_paths=use_paths,
        **kws,
    ).head(n)
    
## begin
parser = argh.ArghParser()
parser.add_commands(
    [
        ## io
        read_ps,
        ## checks
        head_table,
        ## backup
        backup,
        to_version,
        to_zip,
        pqt2tsv,
        ## workflow io
        read_config,
        read_metadata,
        ## workflow execution
        # run_task,
        run_tasks,
        ## notebook post-processing
        replacestar,
        to_clear_unused_cells,
        to_clear_outputs,
        to_clean_nb,  ## wrapper for above
        ## notebook rendering
        to_html,
    ]
)

if __name__ == "__main__":
    parser.dispatch()

"""For access to a few functions from the terminal."""
import warnings
warnings.simplefilter("ignore", SyntaxWarning)

import logging
logging.getLogger().setLevel(logging.INFO)

# from roux.lib.log import Logger
# logging=Logger()

import argh

from roux.lib.sys import read_ps, mv_ln_s
from roux.lib.log import to_diff

from roux.lib.io import (
    pqt2tsv,
    # backup,
    # to_version,
    # to_zip,
    read_arxv,
    to_arxv,
    )

from roux.workflow.log import test_params
from roux.workflow.io import replacestar, to_clean_nb, to_html, to_src, to_nb_kernel

from roux.workflow.task import (
    # run_task, 
    run_tasks ## preferred because it infers setup for the outputs
)
from roux.workflow.task import check_tasks, post_tasks
from roux.workflow.nb import to_clear_unused_cells, to_clear_outputs
from roux.workflow.cfgs import read_config, read_metadata, to_cfg_run_arc

def peek_table(
    p : str,
    n : int = 5,
    use_dir_paths : bool = True,
    not_use_dir_paths : bool = False,
    use_paths : bool = True,
    not_use_paths : bool = False,
    
    cols_desc=None,
    **kws,
    ):
    if not_use_dir_paths: 
        use_dir_paths=False
    if not_use_paths: 
        use_paths=False
    
    from roux.lib.io import read_table
    df_=read_table(
        p,
        use_dir_paths=use_dir_paths,
        use_paths=use_paths,
        **kws,
    )
    from roux.lib.df import _get_preview_log_str
    # logging.info(df_.shape)
    logging.info(
        _get_preview_log_str(df_.head(1))
    )
    if cols_desc:
        return df_[cols_desc].describe()
    else:        
        return df_.head(n)

def query_table(
    p : str,
    expr: str,
    # n : int = 5,
    # use_dir_paths : bool = True,
    # not_use_dir_paths : bool = False,
    # use_paths : bool = True,
    # not_use_paths : bool = False,
    **kws,
    ):
    """
    Examples: 
        "\`col\` == value"
    """
    # if not_use_dir_paths: 
    #     use_dir_paths=False
    # if not_use_paths: 
    #     use_paths=False
    ps=read_ps(p)
    # print(ps)
    # return
    
    if len(ps)>1:
        ## recurse
        for p_ in ps:
            logging.info("\n")
            logging.info(p_)
            try:
                _=query_table(
                    p_,
                    expr=expr,
                    **kws,
                    )
                print(_)
            except Exception as e:
                logging.error(e)
        return 
        
    from roux.lib.io import read_table
    return (
        read_table(
            p,
            # use_dir_paths=use_dir_paths,
            # use_paths=use_paths,
            **kws,
        )
        .query(
            expr=expr,    
        )
    )
    
## begin
parser = argh.ArghParser()
parser.add_commands(
    [
        ## io
            read_ps,
            mv_ln_s,
        ## checks
            peek_table,
            query_table,
        ### logs
            to_diff,
        ## backup
            read_arxv,
            to_arxv,
            # backup,
            # to_version,
            # to_zip,
            pqt2tsv,
        ## workflow io
            ## cfgs
            read_config,
            read_metadata,
            to_cfg_run_arc,
        ## workflow execution
            test_params,
            run_tasks,
            ## slurm
            check_tasks,
            post_tasks,
        ## notebook
        ### pre-processing        
            to_nb_kernel,
            ### post-processing
            replacestar,
            to_clear_unused_cells,
            to_clear_outputs,
            to_clean_nb,  ## wrapper for above
            ### convert
            to_html,
            ### rendering
            to_src,
    ]
)

if __name__ == "__main__":
    parser.dispatch()

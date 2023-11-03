"""For task management."""
## logging
import logging
## internal
from roux.lib.io import to_dict
from roux.lib.sys import (basenamenoext, dirname, exists, get_datetime, makedirs, splitext)

try:
    from tqdm import tqdm
    from roux.lib.sys import is_interactive_notebook
    if not is_interactive_notebook():
        ## progress bar
        tqdm.pandas()
    else:
        from tqdm import notebook
        notebook.tqdm().pandas()
except ImportError:
    logging.warning('ImportError: IProgress not found. Please update jupyter and ipywidgets. See https://ipywidgets.readthedocs.io/en/stable/user_install.html')
    
## execution
def run_task(
    parameters: dict,
    input_notebook_path: str,
    kernel: str,
    output_notebook_path: str= None,
    # force=False,
    test=False,
    verbose=False,
    force=False,
    **kws_papermill,
    ) -> str:
    """
    Run a single task.
    
    Prameters:
        parameters (dict): parameters including `output_path`s.
        input_notebook_path (dict): path to the input notebook which is parameterized.
        kernel (str): kernel to be used.
        output_notebook_path: path to the output notebook which is used as a report. 
        test (bool): test-mode.
        verbose (bool): verbose.
        
    Keyword parameters:
        kws_papermill: parameters provided to the `pm.execute_notebook` function.
    
    Returns:
        Output path.
    """
    if exists(parameters['output_path']) and not force:
        return parameters['output_path']
    if not output_notebook_path:
        ## save report i.e. output notebook
        output_notebook_path=f"{splitext(parameters['output_path'])[0]}_reports/{get_datetime()}_{basenamenoext(input_notebook_path)}.ipynb"
        makedirs(output_notebook_path)
    if test:
        logging.info(parameters['output_path'],output_notebook_path)
    ## save parameters
    to_dict(parameters,f"{dirname(output_notebook_path)}/parameters.yaml")

    if verbose: logging.info(parameters)
    import papermill as pm
    pm.execute_notebook(
        input_path=input_notebook_path,
        output_path=output_notebook_path,
        parameters=parameters,
        kernel=kernel,
        report=True,
        start_timeout=480,
        **kws_papermill,
    )
    return parameters['output_path']

def run_tasks(
    input_notebook_path: str,
    kernel: str,
    inputs: list= None,
    output_path_base: str=None,
    parameters_list: list=None,
    fast: bool=False,
    fast_workers:int=6,
    test1: bool=False,
    force: bool=False,
    test: bool=False,
    verbose: bool=False,
    **kws_papermill,
    )-> list:
    """
    Run a list of tasks.
    
    Prameters:
        input_notebook_path (dict): path to the input notebook which is parameterized.
        kernel (str): kernel to be used.
        inputs (list): list of parameters without the output paths, which would be inferred by encoding.  
        output_path_base (str): output path with a placeholder e.g. 'path/to/{KEY}/file'.  
        parameters_list (list): list of parameters including the output paths.  
        fast (bool): enable parallel-processing.
        fast_workers (bool): number of parallel-processes.
        force (bool): overwrite the outputs.
        test (bool): test-mode.
        verbose (bool): verbose.
        
    Keyword parameters:
        kws_papermill: parameters provided to the `pm.execute_notebook` function.
    
    Returns:
        parameters_list (list): list of parameters including the output paths, inferred if not provided.
        
    TODOs: 
        0. Ignore temporary parameters e.g test, verbose etc while encoding inputs. 
        1. Integrate with apply_on_paths for parallel processing etc.
    """
    ## save task in unique directories
    if parameters_list is None:
        from roux.lib.sys import to_output_paths
        parameters_list=to_output_paths(
            inputs = inputs,
            output_path_base = output_path_base,
            encode_short = True,
            key_output_path='output_path',
            verbose=verbose,
            force=force,
            )
        ## save all parameters
        for k,parameters in parameters_list.items():
            ## save parameters
            output_dir_path=output_path_base.split('{KEY}')[0]
            to_dict(parameters,
                    f"{output_dir_path}/{k.split(output_dir_path)[1].split('/')[0]}/.parameters.yaml")
    elif isinstance(parameters_list,list):
        before=len(parameters_list)
        ## TODO: use `to_outp`?
        parameters_list=[d for d in parameters_list if (force if force else not exists(d['output_path']))]
        if not force:
            logging.info(f"parameters_list reduced because force=False: {before} -> {len(parameters_list)}")
    else:
        raise ValueError(parameters_list)
    ## chech for duplicate output paths
    assert len(set([d['output_path'] for d in parameters_list]))==len(parameters_list), (len(set([d['output_path'] for d in parameters_list])),len(parameters_list))
    
    ## run tasks
    import pandas as pd
    ds1=pd.Series(parameters_list)
    
    if len(ds1)!=0:
        if test1:
            ds1=ds1.head(1)
            logging.warning("testing only the first input.")
        if not fast: 
            _=getattr(ds1,'progress_apply' if hasattr(ds1,'progress_apply') else 'apply')(
                lambda x: run_task(
                    x,
                    input_notebook_path=input_notebook_path,
                    kernel=kernel,
                    **kws_papermill,
                    force=force,
                    ))            
        else:
            from pandarallel import pandarallel
            pandarallel.initialize(nb_workers=fast_workers,progress_bar=True,use_memory_fs=False)            
            _=ds1.parallel_apply(
                lambda x: run_task(
                    x,
                    input_notebook_path=input_notebook_path,
                    kernel=kernel,
                    **kws_papermill,
                    force=force,
                    ))
    return parameters_list
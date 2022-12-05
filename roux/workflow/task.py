"""For task management."""

from roux.lib.io import *

def run_experiment(
    parameters: dict,
    input_notebook_path: str,
    kernel: str,
    output_notebook_path: str= None,
    # force=False,
    test=False,
    verbose=False,
    **kws_papermill,
    ):
    """
    [UNDER DEVELOPMENT] Execute a single notebook.    
    """
    if not output_notebook_path:
        ## save report i.e. output notebook
        output_notebook_path=f"{splitext(parameters['output_path'])[0]}_reports/{get_datetime()}_{basenamenoext(input_notebook_path)}.ipynb"
        makedirs(output_notebook_path)
    if test:
        info(parameters['output_path'],output_notebook_path)
    ## save parameters
    to_dict(parameters,f"{dirname(output_notebook_path)}/parameters.yaml")

    if verbose: info(d1)
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

def run_experiments(
    input_notebook_path: str,
    inputs: list,
    output_path: str,
    kernel: str,
    fast: bool=False,
    test1: bool=False,
    force: bool=False,
    test: bool=False,
    verbose: bool=False,
    **kws_papermill,
    ):
    """
    [UNDER DEVELOPMENT] Execute a list of notebooks.
    
    TODOs: 
        1. Integrate with apply_on_paths for parallel processing etc.
        2. Reporting by quarto?
    """
    ## save experiments in unique directories
    from roux.lib.sys import to_output_paths
    parameters_list=to_output_paths(
        inputs = inputs,
        output_path = output_path,
        encode_short = True,
        key_output_path='output_path',
        verbose=verbose,
        force=force,
        )
    ## save all parameters
    for k,parameters in parameters_list.items():
        ## save parameters
        output_dir_path=output_path.split('{KEY}')[0]
        to_dict(parameters,
                f"{output_dir_path}/{k.split(output_dir_path)[1].split('/')[0]}/.parameters.yaml")
    ## run experiments
    ds1=pd.Series(parameters_list)
    if test1:
        ds1=ds1.head(1)
        logging.warning("testing only 1st input.")
    
    return getattr(ds1,'parallel_apply' if fast else 'progress_apply')(lambda x: run_notebook(x,
                                                                                                input_notebook_path=input_notebook_path,
                                                                                                kernel=kernel,
                                                                                                **kws_papermill,
                                                                                                ))
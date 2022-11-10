from roux.global_imports import *

def run_notebooks(
    input_notebook_path: str,
    input_table_paths: list,
    input_path_replace: str,
    output_path_replace: str,
    env_name: str,
    parameters={},
    force=True,
    test=False,
    verbose=False,
    **kws_papermill,
    ):
    """
    Execute a list of notebooks.
    
    TODOs: 
        1. Integrate with apply_on_paths for parallel processing etc.
        2. Reporting by quarto?
    """
    assert not ('output_table_path' in parameters or 'input_table_path' in parameters)
    
    if test:
        verbose=True
    output_notebook_paths=[]
    ## iterate over inputs
    for input_table_path in input_table_paths:
        ## infer paths
        output_table_path=replace_many(input_table_path,{input_path_replace:output_path_replace,'*':'combined'})
        output_notebook_path=f"{splitext(output_table_path)[0]}_reports/{get_datetime()}_{basenamenoext(input_notebook_path)}.ipynb"
        
        makedirs(output_notebook_path)
        output_notebook_paths.append(output_notebook_path)
        
        if test:
            info(output_table_path,output_notebook_path)
            continue
        if exists(output_table_path) and not force:
            continue
        ## set parameters
        d1=dict(
            input_table_path=input_table_path,
            output_table_path=output_table_path,
            )
        d1.update(parameters)
        to_dict(d1,f"{splitext(output_notebook_path)[0]}/parameters.yaml")
        
        if verbose: info(d1)
        import papermill as pm
        pm.execute_notebook(
            input_path=input_notebook_path,
            output_path=output_notebook_path,
            parameters=d1,
            kernel=env_name,
            report=True,
            start_timeout=480,
            **kws_papermill,
        )
        # break
    to_diff_notebooks(output_notebook_paths)
    return output_notebook_paths
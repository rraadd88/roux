import logging
from pathlib import Path

def print_parameters(d: dict):
    """
    Print a directory with parameters as lines of code

    Parameters:
        d (dict): directory with parameters
    """
    assert isinstance(d, dict)
    
    if logging.root.level <= logging.INFO:
        logger=logging.info
    else:
        logger=print
    logger(
        f"## for testing\nimport os\nos.chdir('{Path.cwd().as_posix()}')\n\n## parameters\n"+(
            "\n".join(
                [
                    k + "=" + ('"' + v + '"' if isinstance(v, str) else str(v))
                    for k, v in d.items()
                ]
            )
        )
    )

def test_params(
    params,
    i=0, #index
    ):
    if isinstance(params, str):
        from roux.lib.io import read_dict
        params=read_dict(params)
        
    if isinstance(params, dict):
        from roux.workflow.task import validate_params
        if validate_params(
            params[
                list(params.keys())[0]
            ]
            ):
            params = list(params.values())
        else:  
            params=[params]
        
    logging.info(f"total params: {len(params)}")
    print_parameters(params[i])

    ## tests
    if 'input_path' in params[i] and Path(params[i]['input_path']).is_file():
        if not Path(params[i]['input_path']).exists():
            logging.warning(f"not found: {params[i]['input_path']}")
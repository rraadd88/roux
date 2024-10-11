import logging
from pathlib import Path

def print_parameters(d: dict):
    """
    Print a directory with parameters as lines of code

    Parameters:
        d (dict): directory with parameters
    """
    assert isinstance(d, dict)
    print(
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
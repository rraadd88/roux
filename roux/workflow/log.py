import logging

def print_parameters(d: dict):
    """
    Print a directory with parameters as lines of code

    Parameters:
        d (dict): directory with parameters
    """
    assert isinstance(d, dict)
    print(
        "\n".join(
            [
                k + "=" + ('"' + v + '"' if isinstance(v, str) else str(v))
                for k, v in d.items()
            ]
        )
    )

def test_params(
    params,
    i=0, #index
    ):
    if isinstance(params, dict):
        params=[params]
    logging.info(f"total params: {len(params)}")
    print_parameters(params[i])
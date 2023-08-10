def print_parameters(d: dict):
    """
    Print a directory with parameters as lines of code
    
    Parameters:
        d (dict): directory with parameters
    """
    assert isinstance(d,dict)
    print('\n'.join([k+'='+('"'+v+'"' if isinstance(v,str) else str(v)) for k,v in d.items()]))
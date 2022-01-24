import pandas as pd

def to_class(cls):
    """
    Ref: https://gist.github.com/mgarod/09aa9c3d8a52a980bd4d738e52e5b97a
    """
    def decorator(func):
#         @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(self._obj,*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator

@pd.api.extensions.register_dataframe_accessor("rd")
class rd:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

to_rd=to_class(rd)

# @pd.api.extensions.register_dataframe_accessor("stat")
# class stat:
#     def __init__(self, pandas_obj):
#         self._obj = pandas_obj



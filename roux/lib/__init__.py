import pandas as pd

def to_class(cls):
    """Get the decorator to attach functions.

    Parameters:
        cls (class): class object.

    Returns:
        decorator (decorator): decorator object.

    References:
        https://gist.github.com/mgarod/09aa9c3d8a52a980bd4d738e52e5b97a
    """
    from functools import wraps

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self._obj, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


@pd.api.extensions.register_dataframe_accessor("rd")
class rd:
    """`roux-dataframe` (`.rd`) extension."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj


# create the `roux-dataframe` (`.rd`) decorator
to_rd = to_class(rd)

@pd.api.extensions.register_series_accessor("rs")
class rs:
    """`roux-series` (`.rs`) extension."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj


# create the `roux-series` (`.rs`) decorator
to_rs = to_class(rs)

# @pd.api.extensions.register_dataframe_accessor("stat")
# class stat:
#     def __init__(self, pandas_obj):
#         self._obj = pandas_obj

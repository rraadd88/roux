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

from pandas.plotting._core import PlotAccessor

# 1. The helper class that dynamically creates plotting methods
class _PiperPlotter:
    def __init__(self, pandas_obj):
        """Initializes with the DataFrame and dynamically wraps all plot methods."""
        self._obj = pandas_obj
        
        # Dynamically find and wrap all plotting methods
        for plot_method_name in dir(PlotAccessor):
            if not plot_method_name.startswith('_'):
                plot_method = getattr(PlotAccessor, plot_method_name)
                if callable(plot_method):
                    # Use a function factory to correctly capture the method and its name
                    wrapped_method = self._make_piper_plot_method(plot_method_name)
                    setattr(self, plot_method_name, wrapped_method)
                    
        # import seaborn as sns
        # setattr(self, 'joint', self._make_piper_plot_method(real_plot_method=sns.jointplot))
        
    def _make_piper_plot_method(
        self,
        name=None,
        real_plot_method=None,
        ):
        """A factory to create a wrapped plotting method."""
        
        if real_plot_method is None:
            # Get the real plotting method from the DataFrame's .plot accessor
            real_plot_method = getattr(
                (
                    self._obj.plot #if name!='hist' else self._obj
                ), 
                name
            )

        def piper_plot_method(
            func_ax=None, #lambda
            **kwargs,
            ):
            """This is the wrapped method that will be called.
            It calls the real plot function and then returns the DataFrame.
            """
            # print(kwargs)
            ax=real_plot_method(**kwargs)
            if func_ax is not None:
                func_ax(ax)
            return self._obj # Return the DataFrame for chaining
            
        return piper_plot_method

@pd.api.extensions.register_dataframe_accessor("rd")
class rd:
    """`roux-dataframe` (`.rd`) extension."""

    def __init__(self, pandas_obj):
        self._obj = pandas_obj
        self.plot = _PiperPlotter(self._obj)

    ## exceptions
    def hist(
        self,
        func_ax=None,
        **kws,
    ):
        ax=self._obj.hist(**kws)
        if func_ax is not None:
            func_ax(ax)
        return self._obj


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

import pandas as pd
import altair as alt
## data function from roux

## settings
alt.data_transformers.disable_max_rows()


def plot_scatters_grouped(
    data: pd.DataFrame,
    cols_groupby: list,
    aggfunc: dict,
    orient="h",
    **kws_encode,
):
    """
    Scatters grouped by categories.

    Args:
        data (pd.DataFrame): input data,
        cols_groupby (list): list of colummns to groupby,
        aggfunc (dict): columns mapped to the aggregation function,

    Keyword Args:
        kws_encode: parameters provided to the `encode` attribute

    Returns:
        Altair figure
    """
    kws_encode = {
        **dict(
            x=list(aggfunc.keys())[0],
            y=list(aggfunc.keys())[1],
        ),
        **kws_encode,
    }

    selection = alt.selection_point(fields=cols_groupby[:1])

    plots = {}
    _c = None
    for i, c in enumerate(cols_groupby):
        if isinstance(c, str):
            c = [c]
        if _c is not None:
            c += _c
        data1 = (
            data.log(c[0], c[1] if len(c) > 1 else None)
            .groupby(c)
            .agg(aggfunc)
            .reset_index()
        )
        if i == 0:
            plots[i] = (
                alt.Chart(data1)
                .mark_point()
                .encode(
                    **kws_encode,
                    tooltip=c,
                    opacity=alt.condition(selection, alt.value(1.0), alt.value(0.5)),
                )
                .properties(
                    title=c[0],
                )
                .interactive()
                .add_params(selection)
            )
        else:
            plots[i] = (
                alt.Chart(data1)
                .mark_circle()
                .encode(
                    **kws_encode,
                    tooltip=c,
                    color=alt.Color(c[0], legend=None),
                )
                .properties(
                    title=c[0],
                )
                .transform_filter(selection)
                .interactive()
            )
        _c = c
        del data1, c
    return getattr(alt, f"{orient}concat")(*plots.values())

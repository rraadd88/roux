"""For set related stats."""

import logging
import pandas as pd

# attributes
import roux.lib.dfs as rd  # noqa


def get_overlap(
    items_set: list,
    items_test: list,
    output_format: str = "list",
) -> list:
    """Get overlapping items as a string.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        output_format (str, optional): format of the output. Defaults to 'list'.

    Raises:
        ValueError: output_format can be list or str
    """
    overlap = sorted(list(set(items_set) & set(items_test)))
    if output_format == "list":
        return overlap
    elif output_format == "str":
        return ";".join(overlap)
    else:
        raise ValueError(output_format)


def get_overlap_size(
    items_set: list,
    items_test: list,
    fraction: bool = False,
    perc: bool = False,
    by: str = None,
) -> float:
    """Percentage Jaccard index.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        fraction (bool, optional): output fraction. Defaults to False.
        perc (bool, optional): output percentage. Defaults to False.
        by (str, optional): fraction by. Defaults to None.

    Returns:
        float: overlap size.
    """
    if perc or by != None:  # noqa
        fraction = True
    overlap_size = len(
        get_overlap(
            items_set=items_set,
            items_test=items_test,
            output_format="list",
        )
    )
    if not fraction:
        return overlap_size
    if by == "test":
        return (100 if perc else 1) * overlap_size / len(set(items_test))
    else:
        return (100 if perc else 1) * (
            overlap_size / len(set(items_set) | set(items_test))
        )


def get_item_set_size_by_background(
    items_set: list,
    background: int,
) -> float:
    """Item set size by background

    Args:
        items_set (list): items in the reference set
        background (int): background size

    Returns:
        float: Item set size by background

    Notes:
        Denominator of the fold change.
    """
    return len(set(items_set)) / background


def get_fold_change(
    items_set: list,
    items_test: list,
    background: int,
) -> float:
    """Get fold change.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        background (int): background size

    Returns:
        float: fold change

    Notes:

        fc = (intersection/(test items))/((items in the item set)/background)
    """
    return get_overlap_size(
        items_set=items_set, items_test=items_test, by="test"
    ) / get_item_set_size_by_background(items_set, background)


def get_hypergeom_pval(
    items_set: list,
    items_test: list,
    background: int,
) -> float:
    """Calculate hypergeometric P-value.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        background (int): background size

    Returns:
        float: hypergeometric P-value
    """
    from scipy.stats import hypergeom

    return hypergeom.sf(
        k=get_overlap_size(
            items_test=items_test,
            items_set=items_set,
        )
        - 1,  # len(set(items_test) & set(items_set))-1, # size of the intersection
        M=background,  # background
        n=len(set(items_test)),  # size of set1
        N=len(set(items_set)),  # size of set2
    )


def get_contigency_table(
    items_set: list,
    items_test: list,
    background: int,
) -> list:
    """Get a contingency table required for the Fisher's test.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        background (int): background size

    Returns:
        list: contingency table

    Notes:

                                within item (/referenece) set:
                                True            False
        within test item: True  intersection    True False
                        False   False False     total-size of union

    """
    items_set, items_test = set(items_set), set(items_test)
    table = [
        [
            len((items_set) & (items_test)),  # intersection size
            len((items_test) - (items_set)),
        ],  # in one set not the other (items in test but not set )
        [
            len(
                (items_set) - (items_test)
            ),  # in one set not the other (items in set and not test)
            background - len((items_set) | (items_test)),
        ],  # items that are not in my background or the test or set
        # not in either set
    ]
    assert sum(table[0]) + sum(table[1]) == background
    return table


def get_odds_ratio(
    items_set: list,
    items_test: list,
    background: int,
) -> float:
    """Calculate Odds ratio and P-values using Fisher's exact test.

    Args:
        items_set (list): items in the reference set
        items_test (list): items to test
        background (int): background size

    Returns:
        float: Odds ratio
    """
    from scipy.stats import fisher_exact

    return fisher_exact(
        get_contigency_table(items_set, items_test, background), alternative="two-sided"
    )


def get_enrichment(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    colid: str,
    colset: str,
    background: int,
    coltest: str = None,
    test_type: list = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Calculate the enrichments.

    Args:
        df1 (pd.DataFrame): table containing items to test
        df2 (pd.DataFrame): table containing refence sets and items
        colid (str): column with IDs of items
        colset (str): column sets
        coltest (str): column tests
        background (int): background size.
        test_type (list): hypergeom or Fisher. Defaults to both.
        verbose (bool): verbose

    Returns:
        pd.DataFrame: output table
    """
    assert isinstance(background, int)
    if test_type is None:
        test_type = []
    ## calculate the background for the Fisher's test that is compatible with the contigency tables
    background_fisher_test = len(set(df1[colid].tolist() + df2[colid].tolist()))
    if coltest is None:
        coltest = "Unnamed"
        df1 = df1.assign(**{coltest: 1})
    ## statistics
    df3 = (
        df1.groupby(by=coltest).apply(  # iterate over the groups of items to test
            lambda df1_: (
                df2.groupby(colset).apply(  # iterate over the groups of item sets
                    lambda df2_: pd.Series(
                        {
                            "overlap size": get_overlap_size(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                            ),
                            "overlap %": get_overlap_size(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                                perc=True,
                            ),
                            "overlap items": get_overlap(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                            ),
                            "contingency table": get_contigency_table(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                                background=background_fisher_test,
                            ),
                            "Odds ratio": get_odds_ratio(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                                background=background_fisher_test,
                            )[0],
                            "fold change": get_fold_change(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                                background=background,
                            ),
                            "overlap/test %": get_overlap_size(
                                items_test=df1_[colid].tolist(),
                                items_set=df2_[colid].tolist(),
                                by="test",
                                perc=True,
                            ),
                            "set/background": get_item_set_size_by_background(
                                items_set=df2_[colid].tolist(),
                                background=background,
                            ),
                        },
                    )
                )
            )
        )
    ).reset_index()
    if "hypergeom" in test_type:
        df_ = (
            (
                df1.groupby(
                    by=coltest
                ).apply(  # iterate over the groups of items to test
                    lambda df1_: (
                        df2.groupby(
                            colset
                        ).apply(  # iterate over the groups of item sets
                            lambda df2_: pd.Series(
                                {
                                    "P (hypergeom. test)": get_hypergeom_pval(
                                        items_test=df1_[colid].tolist(),
                                        items_set=df2_[colid].tolist(),
                                        background=background,
                                    ),
                                },
                            )
                        )
                    )
                )
            )
            .reset_index()
            .rd.clean()
        )
        ## concat with the output
        # df3=pd.concat([df3,df_],axis=1)
        df3 = df3.merge(
            right=df_,
            how="outer",
            on=[coltest, colset],
            validate="1:1",
        )
    if "Fisher" in test_type:
        df_ = (
            (
                df1.groupby(
                    by=coltest
                ).apply(  # iterate over the groups of items to test
                    lambda df1_: (
                        df2.groupby(
                            colset
                        ).apply(  # iterate over the groups of item sets
                            lambda df2_: pd.Series(
                                {
                                    "P (Fisher's exact)": get_odds_ratio(
                                        items_test=df1_[colid].tolist(),
                                        items_set=df2_[colid].tolist(),
                                        background=background_fisher_test,
                                    )[1],
                                },
                            )
                        )
                    )
                )
            )
            .reset_index()
            .rd.clean()
        )
        ## concat with the output
        # df3=pd.concat([df3,df_],axis=1)
        df3 = df3.merge(
            right=df_,
            how="outer",
            on=[coltest, colset],
            validate="1:1",
        )

    def get_qs(df):
        from roux.stat.transform import get_q

        ## Multiple test correction. Calculate adjusted P values i.e. Q values
        # from statsmodels.stats.multitest import fdrcorrection
        for col in df.filter(regex="^P.*"):
            logging.info(col)
            df[col.replace("P", "Q")] = get_q(ds1=df[col], verb=verbose)
        return df

    df4 = (
        df3.query(expr="`overlap size` != 0")
        .groupby(by=coltest)  # iterate over tests
        .apply(get_qs)
        .rd.clean()
    )
    return df4.sort_values(df4.filter(regex="^Q.*").columns.tolist())

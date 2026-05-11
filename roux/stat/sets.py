"""For set related stats."""

import logging
import pandas as pd
import numpy as np

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
    background: int,
    colset: str = None,
    coltest: str = None,
    test_type: list = None,
    verbose: bool = False,
    df_covars: pd.DataFrame = None,
) -> pd.DataFrame:
    """Calculate the enrichments.

    Args:
        df1 (pd.DataFrame): table containing items to test
        df2 (pd.DataFrame): table containing refence sets and items
        colid (str): column with IDs of items
        colset (str): column sets
        coltest (str): column tests
        background (int): background size for hypergeom. test. Ignored for logistic.
        test_type (list): 'hypergeom', 'Fisher', or 'logistic'. Defaults to ['hypergeom', 'Fisher'].
        verbose (bool): verbose
        df_covars (pd.DataFrame, optional): A DataFrame with items as index and
            covariate values in columns. Required if 'logistic' is in test_type.

    Returns:
        pd.DataFrame: output table

    Notes:
        The 'logistic' test type uses logistic regression to model the probability
        that an item belongs to a reference set, while accounting for covariates.
        The model is specified as:

        P(item ∈ Reference Set) ~ β₀ + β₁*(item ∈ Test Set) + β₂*Covariate₁ + ...

        A positive and significant β₁ coefficient indicates enrichment after
        adjusting for the provided covariates.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> # Test items (e.g., differentially expressed genes)
        >>> df1 = pd.DataFrame({'gene': ['A', 'B', 'C']})
        >>> # Reference sets (e.g., pathways)
        >>> df2 = pd.DataFrame({
        ...     'pathway': ['P1', 'P1', 'P2', 'P2'],
        ...     'gene': ['A', 'B', 'D', 'E']
        ... })
        >>> # Covariates for all genes in the background
        >>> df_covars = pd.DataFrame({
        ...     'gene': ['A', 'B', 'C', 'D', 'E', 'F'],
        ...     'length': [100, 150, 80, 200, 120, 90],
        ...     'gc_content': [0.4, 0.5, 0.6, 0.4, 0.5, 0.5]
        ... }).set_index('gene')
        >>> # Run enrichment with logistic regression
        >>> get_enrichment(
        ...     df1=df1,
        ...     df2=df2,
        ...     colid='gene',
        ...     colset='pathway',
        ...     background=6,
        ...     test_type=['logistic'],
        ...     df_covars=df_covars
        ... )
    """
    assert isinstance(background, int)

    if colset is None:
        colset = df2.set_index(colid).columns.tolist()
    elif isinstance(colset, str):
        colset = [colset]

    if test_type is None:
        test_type = ["hypergeom", "Fisher"]
    ## calculate the background for the Fisher's test that is compatible with the contigency tables
    background_fisher_test = len(set(df1[colid].tolist() + df2[colid].tolist()))
    if coltest is None:
        coltest = "tmp"
        df1 = df1.assign(**{coltest: 1})
    ## statistics
    df3 = (
        df1.groupby(by=coltest)
        .apply(  # iterate over the groups of items to test
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
        .reset_index()
    )
    if "hypergeom" in test_type:
        df_ = (
            (
                df1.groupby(by=coltest)
                .apply(  # iterate over the groups of items to test
                    lambda df1_: (
                        df2.groupby(colset).apply(  # iterate over the groups of item sets
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
            on=[coltest] + colset,
            validate="1:1",
        )
    if "Fisher" in test_type:
        df_ = (
            (
                df1.groupby(by=coltest)
                .apply(  # iterate over the groups of items to test
                    lambda df1_: (
                        df2.groupby(colset).apply(  # iterate over the groups of item sets
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
            on=[coltest] + colset,
            validate="1:1",
        )
    if "logistic" in test_type:
        if df_covars is None:
            raise ValueError(
                "df_covars must be provided for logistic regression test."
            )
        try:
            import statsmodels.api as sm
        except ImportError:
            raise ImportError(
                "statsmodels is required for logistic regression. "
                "Please install it with 'pip install statsmodels'"
            )

        def _run_logistic(df1_, df2_, df_covars):
            items_in_test = df1_[colid].tolist()
            items_in_set = df2_[colid].tolist()

            # 1. Construct the model DataFrame from covariates
            model_df = df_covars.copy()
            if colid in model_df:
                model_df=model_df.set_index(colid)
            if model_df.index.name != colid:
                model_df.index.name = colid

            # 2. Create the dependent and predictor variables
            model_df["is_in_set"] = model_df.index.isin(items_in_set).astype(int)
            model_df["is_in_test"] = model_df.index.isin(items_in_test).astype(int)

            if model_df["is_in_test"].sum() == 0 or model_df["is_in_set"].sum() == 0:
                return pd.Series(
                    {
                        "P (logistic)": np.nan,
                        "Odds ratio (logistic)": np.nan,
                    }
                )

            # 3. Fit the model
            covariate_cols = [c for c in df_covars.columns if c != colid]
            X = model_df[["is_in_test"] + covariate_cols]
            X = sm.add_constant(X)
            y = model_df["is_in_set"]

            try:
                model = sm.Logit(y, X).fit(disp=0)
                # 4. Extract results
                return pd.Series(
                    {
                        "P (logistic)": model.pvalues["is_in_test"],
                        "Odds ratio (logistic)": np.exp(model.params["is_in_test"]),
                    }
                )
            except Exception:  # Catches perfect separation and other errors
                return pd.Series(
                    {
                        "P (logistic)": np.nan,
                        "Odds ratio (logistic)": np.nan,
                    }
                )

        df_ = (
            df1.groupby(by=coltest)
            .apply(
                lambda df1_: df2.groupby(colset).apply(
                    lambda df2_: _run_logistic(df1_, df2_, df_covars)
                )
            )
            .reset_index()
            .rd.clean()
        )
        df3 = df3.merge(
            right=df_,
            how="outer",
            on=[coltest] + colset,
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
    if "tmp" in df4:
        df4 = df4.drop(["tmp"], axis=1)
    return df4.sort_values(df4.filter(regex="^Q.*").columns.tolist())


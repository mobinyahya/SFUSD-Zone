import pandas as pd


def clean_pd_lists(df: pd.DataFrame, column_names: list[str]):
    """Clean a pandas dataframe that has columns that have list as strings.
    If we have a pandas dataframe column with values like
    ["value_1", "value_2", ...] but with datatype str,
    we can convert it back into a proper list.
    Ref: https://towardsdatascience.com/dealing-with-list-values-in-pandas-dataframes-a177e534f173
    Args:
        df (pd.DataFrame): the pandas dataframe to clean.
        column_names (List[str]): the list of column names to clean.

    Returns:
        pd.DataFrame: the cleaned versino of pandas dataframe.
    """
    for col in column_names:
        # NOTE: use loc because
        # #https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
        df.loc[:, col] = df[col].fillna("[]")
        df.loc[:, col] = df[col].apply(clean_alt_list).apply(eval)
    return df


def clean_alt_list(list_):
    list_ = list_.replace("'", "")
    list_ = list_.replace('"', "")
    list_ = list_.replace(", ", '","')
    list_ = list_.replace("[", '["')
    list_ = list_.replace("]", '"]')
    return list_

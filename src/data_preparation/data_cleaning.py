import pandas as pd
from typing import List


def drop_columns(df: pd.DataFrame, columns_to_drop: List[str]) -> pd.DataFrame:
    """
    Removes the specified columns from a DataFrame.

    Args:
        df: The DataFrame to remove columns from.
        columns_to_drop: A list of strings specifying the names of the columns to remove.

    Returns:
        A new DataFrame with the specified columns removed.
    """
    return df.drop(columns=columns_to_drop)


def drop_null_rows(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Removes any rows in the DataFrame that contain null values.

    Args:
        df: The DataFrame to remove null rows from.
        subset: A list of column names to check for null values in. If None, all columns are checked.

    Returns:
        A new DataFrame with the null rows removed.
    """
    return df.dropna(subset=subset)


def replace_null_values(df: pd.DataFrame, replacement_dict: dict) -> pd.DataFrame:
    """
    Replaces null values in the DataFrame with specified values.

    Args:
        df: The DataFrame to replace null values in.
        replacement_dict: A dictionary where the keys are the names of columns to replace null values in, and the
                          values are the values to replace null values with.

    Returns:
        A new DataFrame with null values replaced.
    """
    return df.fillna(value=replacement_dict)


def replace_values(df: pd.DataFrame, replacement_dict: dict) -> pd.DataFrame:
    """
    Replaces specified values in the DataFrame with other specified values.

    Args:
        df: The DataFrame to replace values in.
        replacement_dict: A dictionary where the keys are the values to replace, and the values are the values to
                          replace them with.

    Returns:
        A new DataFrame with values replaced.
    """
    return df.replace(to_replace=replacement_dict)


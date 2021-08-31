import pandas as pd
import re
from dateutil.parser import parse

"""df = pd.read_csv("logs.csv")  # read data

df = df.set_index("d")  # to re-index with a column 'd'
df = df.sort_index()  # to sort with respect to the index """


def check_extension(data):
    """check if the extension of data is within CSV, JSON or SQL

    Args:
        data (): data set

    Returns:
        type of allowed extension or none.
    """
    if re.search("\.csv$", data, flags=re.IGNORECASE):
        return "csv"
    if re.search("\.json$", data, flags=re.IGNORECASE):
        return "json"
    if re.search("\.sql$", data, flags=re.IGNORECASE):
        return "sql"
    if re.search("\.xlsx$", data, flag=re.IGNORECASE):
        return "xlsx"
    return "none"


def _to_DataFrame(data):
    """read data and transform it to DataFrame

    Args:
        data (csv, json, sql, xlsx): data

    Returns:
        Dataframe
    """

    ext = check_extension(data)
    assert ext != "none"
    f_dict = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "sql": pd.read_sql,
        "xlsx": pd.read_excel,
    }
    df = f_dict[ext](data)
    return df


def get_metadata(df):
    """read a dataframe and generate relevant metadata such as columns types etc

    Args:
        df (DataFrame): data

    Returns:
        dict: {name_of_column: metadata_associated}
    """
    metadata = []
    for column in df:
        metadata.append(check_data_type(df[column]))
    return metadata


def check_data_type(column):
    """check type in a column which type is it using a voting method from all the non na data

    Args:
        column (pandas.core.series.Series): column from a dataframe

    Returns:
        [type]: [description]
    """
    types_dict = {}
    for e in column[column.notna()]:
        try:
            e = parse(e, False)
        except:
            pass
        if type(e) not in types_dict:
            types_dict[type(e)] = 1
        else:
            types_dict[type(e)] += 1
    if len(types_dict) != 0:
        return max(types_dict, key=types_dict.get)
    else:
        return


def is_date(string, fuzzy=False):
    """check if a given string is a date and return the date if true and raise a ValueError if false

    Args:
        string (string): string to check
        fuzzy (bool, optional): Enable a more lenient search in the string. Defaults to False.

    Raises:
        ValueError: raised when string is not likely to be a date

    Returns:
        string: datetime as a string
    """
    try:
        pd.to_datetime(string)
        return pd.to_datetime(string)

    except ValueError:
        raise ValueError


def _is_duplicated(df):
    """Find duplicated row and return dataframe without the duplication

    Args:
        df (pandas.DataFrame): data frame

    Returns:
        duplicated_row: the duplicated rows
        df_clean: the DataFrame without the duplicated rows
    """
    df_new = df.drop(["Unnamed: 0"], axis=1)
    duplicated_row = df[df_new.duplicated()]  # duplicated row
    df_clean = df[~df_new.duplicated()]  # without duplication row
    return df_clean, duplicated_row


def _is_unique(df, col_name=""):
    """verify uniqueness over a specified column, and find the uniqueness coefficient

    Args:
        df (pandas.DataFrame): Data Frame.
        col_name (str, optional): Column name. Defaults to "".

    Returns:
        ratio : 1 - (number of repeated data in a column)/card(the column)
                if 1 means all values are unique
    """
    df_clean, _ = _is_duplicated(df)
    if df_clean[col_name].is_unique:
        ratio = 1
    else:
        repeated = pd.concat(
            g for _, g in df_clean.groupby(col_name) if len(g) > 1
        )  # repeated columns

        ratio = 1 - len(repeated) / df_clean.shape[0]
    return ratio


def _is_none(df, col_name=""):
    """find none ration in a specific columns

    Args:
        df ([type]): [description]
        col_name (str, optional): [description]. Defaults to "".

    Returns:
        ratio: none ration in the columns
                1 means all the columns is none
                0 means non none
    """
    df_clean, _ = _is_duplicated(df)
    none_element = df_clean[col_name][df_clean[col_name].isnull()]
    ratio = len(none_element) / df_clean.shape[0]
    return ratio

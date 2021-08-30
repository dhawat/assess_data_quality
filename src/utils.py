import pandas as pd
import re

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
        data (CSV, JSON, SQL): data

    Returns:
        Dataframe
    """

    ext = check_extension(data)
    f_dict = {'csv': pd.read_csv, 'json': pd.read_json, 'sql': pd.read_sq, 'xlsx': pd.read_excel}
    df = f_dict[ext]()
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
    """ check type in a column which type is it using a voting method from all the non na data

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
    """check if a given string is a date and return the date if true and raise a Valueerror if false

    Args:
        string (string): string to check 
        fuzzy (bool, optional): Enable a more leniant search in the string. Defaults to False.

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
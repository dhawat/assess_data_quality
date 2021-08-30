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

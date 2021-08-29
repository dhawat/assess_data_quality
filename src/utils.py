import pandas as pd

"""df = pd.read_csv("logs.csv")  # read data

df = df.set_index("d")  # to re-index with a column 'd'
df = df.sort_index()  # to sort with respect to the index """

import re


def check_extension(data):
    """check if the expension of data is within CSV, JSON or SQL

    Args:
        data (): data set

    Returns:
        type of allowed extension or none.
    """
    if re.search("\.csv$", data, flags=re.IGNORECASE):
        return "csv"
    if re.search("\.json$", fname, flags=re.IGNORECASE):
        return "json"
    if re.search("\.sql$", fname, flags=re.IGNORECASE):
        return "sql"
    return "none"


def _to_DataFrame(data):
    """read data and transformit to DataFrame

    Args:
        data (CSV, JSON, SQL): data

    Returns:
        Dataframe
    """
    return data

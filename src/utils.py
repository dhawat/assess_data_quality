import pandas as pd
import re
from dateutil.parser import parse

from sklearn.cluster import AffinityPropagation
from difflib import SequenceMatcher


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
        if type(e) not in types_dict:
            types_dict[type(e)] = 1
        else:
            types_dict[type(e)] += 1
    if len(types_dict) != 0:
        return max(types_dict, key=types_dict.get)
    else:
        return


def _is_date(string, fuzzy=False):
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
    return df_clean[col_name].nunique() / df_clean[col_name].shape[0]


def _is_none(df, col_name=""):
    """find none ratio in a specific columns

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


def proba_model(col, mean, std, tresh=3):
    """cutting distribution between mean-3*std and mean+3*std

    Args:
        df ([type]): [description]
        col_name ([type]): [description]
        mean ([type]): [description]
        std ([type]): [description]
        tresh (int, optional): [description]. Defaults to 6.

    Returns:
        [type]: [description]
    """
    upper_bound = mean + tresh * std
    lower_bound = mean - tresh * std
    idx = col[
        ~((col > lower_bound) * (col < upper_bound))
    ].index  # trancate values from the column
    # clean dataframe
    return idx


# Possibilité d'améliorer
# Threshold for anomalie is fixed at Q_1 = round(np.percentile(unique_counts, 5)), could be improved.
# DBSCAN for example on the number of occurences on words.


def uncorrect_grammar(df_names, cluster):
    """index of element

    Args:
        df_names ([type]): [description]
        cluster ([type]): [description]

    Returns:
        [type]: [description]
    """
    words = np.asarray(df_names)
    unique_words, unique_counts = np.unique(df_names, return_counts=True)
    Q_1 = round(np.percentile(unique_counts, 5))
    Low_risk_words = unique_words[np.where(unique_counts > Q_1)[0]]
    index_In_words = []
    for w in cluster:
        if not (w in list(Low_risk_words)):
            index_In_words = index_In_words + np.ndarray.tolist(np.where(words == w)[0])
    return index_In_words


def Index_Uncorrect_grammar(df_State):
    df_State = df["state"]
    df_State_unique = np.unique(df_State)
    words = np.asarray(df_State_unique)  # So that indexing with a list will work
    lev_similarity = np.array(
        [[SequenceMatcher(None, w1, w2).ratio() for w1 in words] for w2 in words]
    )
    affprop = AffinityPropagation(affinity="precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    list_uncorrect = []
    if len(np.unique(affprop.labels_)) == 1:
        return list_uncorrect
    else:
        for cluster_id in np.unique(affprop.labels_):
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            list_uncorrect = list_uncorrect + uncorrect_grammar(df_State, cluster)
    return list_uncorrect

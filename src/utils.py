import pandas as pd
import re
import numpy as np

from dateutil.parser import parse
from difflib import SequenceMatcher
from sklearn import preprocessing
from sklearn.cluster import AffinityPropagation

# from gensim.models import Word2Vec
#! add insignifante column test for non in column

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
    if re.search("\.xlsx$", data, flags=re.IGNORECASE):
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
    if column.dropna().values.size == 0:
        return type(None)

    for e in column[column.notna()]:
        if type(e) not in types_dict:
            types_dict[type(e)] = 1
        else:
            types_dict[type(e)] += 1
    if max(types_dict, key=types_dict.get) not in [type(int()), type(float())]:
        try:
            column = pd.to_datetime(column.dropna())
            return column.dtype
        except ValueError:
            pass
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

    df_new = df.drop([df.columns[0]], axis=1)
    duplicated_row = df[df_new.duplicated()]  # duplicated row
    df_clean = df[~df_new.duplicated()]  # without duplication row

    return df_clean, duplicated_row


def _duplicated_idx(df):
    df_new = df.drop([df.columns[0]], axis=1)
    return df_new.duplicated()


def _summery_duplication(df, col_name):
    """summery of duplications in a specific column
    Args:
        df ([type]): Data frame
        col_name ([type]): column name
    """
    df.pivot_table(columns=[col_name], aggfunc="size")  # summery of duplication


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
    df_clean, _ = _is_duplicated(df)  # remove duplicated lines
    none_element = df_clean[col_name][df_clean[col_name].isnull()]
    ratio = len(none_element) / df_clean.shape[0]
    return ratio


def _z_score(col, mean, std, tresh=3):
    """cutting distribution between mean-6*std and mean+6*std
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
        ~((col > lower_bound) & (col < upper_bound))
    ].index  # trancate values from the column
    return idx


# todo: Possibilité d'améliorer: Threshold for anomalie is fixed at Q_1 = round(np.percentile(unique_counts, 5)), could be improved. DBSCAN for example on the number of occurences on words.

# todo: or the repeated words also could be detected by this method, for each word detected as outlier we can divide the score by  the number of repetition of the word
def uncorrect_grammar(df_names, cluster, min_occurence):
    """index of element
    Args:
        df_names ([type]): [description]
        cluster ([type]): [description]
        min_occurence (int): [min # of répétition of a label to be considered an error]
    Returns:
        [type]: [description]
    """
    words = np.asarray(df_names)
    unique_words, unique_counts = np.unique(df_names, return_counts=True)
    index_In_words = []
    for w in cluster:
        count = unique_counts[np.where(unique_words == w)[0]][0]
        if count <= min_occurence:
            index_In_words = index_In_words + np.ndarray.tolist(np.where(words == w)[0])
    return index_In_words


def index_uncorrect_grammar(df_State):
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
            if len(cluster) > 1:
                list_uncorrect = list_uncorrect + uncorrect_grammar(
                    df_State, cluster, 10
                )
    return list_uncorrect


def _row_is_none(df, thresh_row_1=0.7, thresh_row_2=0.5, thresh_col=0.8):
    """check row having mean number of none > thresh_row_1 in the data frame cleaned and > thresh_row_2 in data frame cleaned from columns having mean number of none > threshcol.
    Args:
        df ([DataFrame]): [description]
        thresh_row_1 (float, optional): [description]. Defaults to 0.75.
        thresh_row_2 (float, optional): [description]. Defaults to 0.5.
        thresh_col (float, optional): [description]. Defaults to 0.8.
    Returns:
        [type]: [description]
    """
    df_clean, _ = _is_duplicated(df)

    mean_none_row_1 = df_clean.isnull().mean(axis=1)  # mean(none) in each row of df_c
    list_drop_row_1 = mean_none_row_1[mean_none_row_1 >= thresh_row_1]
    index_1 = list_drop_row_1.index  # index of drop rows

    mean_none_col = df_clean.isnull().mean(axis=0)  # mean(none) in each column of df_c
    index_col_drop = mean_none_col[
        mean_none_col >= thresh_col
    ].index  # list of names of the column with none mean>thresh_col
    df_drop_col = df_clean.drop(
        labels=index_col_drop, axis=1
    )  # drop column with mean(none)>thresh_col from df_clean
    mean_none_row_2 = df_drop_col.isnull().mean(
        axis=1
    )  # mean number of none in each row
    list_drop_row_2 = mean_none_row_2[
        mean_none_row_2 >= thresh_row_2
    ]  # drop row having mean(none)>thresh_row over df_drop_col
    index_2 = list_drop_row_2.index

    ind = index_1 & index_2

    return ind


def _string_to_nbr(df):
    """Convert a DataFrame (which may have multiple columns) of strings into a return df
    with vectors inside.
    Args:
        df ([DataFrame]): [DataFrame containing strings]
    Returns:
        [DataFrame]: [DataFrame converted into vectors]
    """
    df = df.fillna("Nan")
    document = []
    for col in df.columns:
        document.append(df[col].fillna("Nan").tolist())
    tokenized_sentences = document
    model = Word2Vec(
        tokenized_sentences, vector_size=100, window=2, min_count=0, workers=6
    )
    df = df.applymap(lambda x: model.wv[x])
    return df


def _year(x):
    return x.year + x.month / 12 + x.day / 365


def _to_date_and_float(df):
    pd.options.mode.chained_assignment = None
    for col in df.columns:
        if df[col].dtype == "datetime64[ns]":
            df[col] = df[col].apply(lambda x: _year(x))
        elif df[col].dtype == "int64":
            df[col] = df[col].apply(lambda x: float(x))
        else:
            pass
    return df


def _tendancy_detection(df, thresh=0.999):
    """
    df = dataframe
    thresh = threshold for the tendancy detection
    return:
            dictionnaire with the pair of columns and the anomaly index detected.
    """
    dictionnaire_anomalie_tendance = {}

    for w1 in df:
        for w2 in df:
            range_anomalie = 0
            proportion = np.shape(np.where(df[w1] < df[w2])[0])[0] / len(df)
            if proportion > thresh:
                range_anomalie = np.shape(np.where(df[w1] > df[w2])[0])[0]
                if range_anomalie > 0:
                    dictionnaire_anomalie_tendance[(w1, w2)] = np.ndarray.tolist(
                        np.where(df[w1] > df[w2])[0]
                    )
    return dictionnaire_anomalie_tendance

from math import nan
import pandas as pd
import re
import numpy as np

from dateutil.parser import parse
from difflib import SequenceMatcher
from sklearn.cluster import AffinityPropagation

# from gensim.models import Word2Vec
#! add insignifante column test for non in column

"""df = pd.read_csv("logs.csv")  # read data
df = df.set_index("d")  # to re-index with a column 'd'
df = df.sort_index()  # to sort with respect to the index """


def check_extension(path):
    """Check if the extension of the data set belongs to {csv, json, sql, xlsx}.

    Args:

        path (path): path of the set of data.

    Returns:

        type of allowed extension or none.
    """
    if re.search("\.csv$", path, flags=re.IGNORECASE):
        return "csv"
    if re.search("\.json$", path, flags=re.IGNORECASE):
        return "json"
    if re.search("\.sql$", path, flags=re.IGNORECASE):
        return "sql"
    if re.search("\.xlsx$", path, flags=re.IGNORECASE):
        return "xlsx"
    return "none"


def _to_DataFrame(path):
    """Read data and transform it to DataFrame.

    Args:

        path (path): path to the corresponding directory of the data.

    Returns:

        df (pandas.DataFrame): DataFrame containing of the data.
    """

    ext = check_extension(path)  # check if the format of the data is acceptable
    assert ext != "none"
    f_dict = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "sql": pd.read_sql,
        "xlsx": pd.read_excel,
    }
    df = f_dict[ext](path)  # DataFrame containing the data
    return df


def get_metadata(df):
    """Read a dataframe and generate relevant metadata such as columns types etc.

    Args:

        df (pandas.DataFrame): DataFrame of data.

    Returns:

        dict: {name_of_column: metadata_associated}.
    """
    metadata = []
    for column in df:
        metadata.append(check_data_type(df[column]))
    return metadata


def check_data_type(column):
    !# what does this mean?
    """check the type used in a column by voting method from all the non nan data.

    Args:

        column (pandas.core.series.Series): column from a dataframe.

    Returns:

        [type]: [description]
    """
    !# return what
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
    """Check if a given string is a date, return the date if true and raise  ValueError if false.

    Args:

        string (str): string to check.

        fuzzy (bool, optional): Enable a more lenient search in the string. Defaults to False.

    Returns:

        string: datetime as a string.

    Raises:

        ValueError: raised when string is not likely to be a date.
    """
    try:
        pd.to_datetime(string)
        return pd.to_datetime(string)

    except ValueError:
        raise ValueError


def _is_duplicated(df):
    """Find duplicated row and return dataframe without the duplication.

    Args:

        df (pandas.DataFrame): data frame.

    Returns:

        duplicated_row: the duplicated rows.

        df_clean: the DataFrame without the duplicated rows.
    """

    df_new = df.drop([df.columns[0]], axis=1)
    duplicated_row = df[df_new.duplicated()]  # duplicated row
    df_clean = df[~df_new.duplicated()]  # without duplication row

    return df_clean, duplicated_row


def _duplicated_idx(df):
    """Find index of duplicated row in the DataFrame

    Args:

        df (pandas.DataFrame): DataFrame of input data

    Returns:

        list of index of duplicated row
    """
    df_new = df.drop([df.columns[0]], axis=1)
    return df_new.duplicated()

#! remove this function
def _summery_duplication(df, col_name):
    """summery of duplications in a specific column.

    Args:

        df ([type]): Data frame

        col_name ([type]): column name
    """
    df.pivot_table(columns=[col_name], aggfunc="size")  # summery of duplication


def _is_unique(df, col_name=""):
    """Uniqueness ratio of specified column of the DataFrame

    Args:

        df pandas.DataFrame): DataFrame of input data.

        col_name (str, optional): name of the column. Defaults to "".

    Returns:

        ratio : 1 - (number of repeated data in a column without nan)/card(the column without nan)

    .. note::

            A ratio 1 means all values are unique.
            A ratio 0 means all values in the columns are repeated or empty column.
    """
    if col_name is not nan:
        df = df[col_name]
    else:
        df = df
    if df.dropna().shape[0] == 0:
        return 0
    else:
        return df.dropna().nunique() / df.dropna().shape[0]


def _is_none(df, col_name):
    """Find none ratio in a specific column.

    Args:

        df (pandas.DataFrame): DataFrame of input data.

        col_name (str): name of the column. Defaults to "".

    Returns:

        ratio: none ration in the columns
                1 means all the columns is none
                0 means non none
    """
    df_clean, _ = _is_duplicated(df)  # remove duplicated lines
    none_element = df_clean[col_name][df_clean[col_name].isnull()]
    ratio = len(none_element) / df_clean.shape[0]
    return ratio


def _z_score(col, thresh=6, thresh_unique1=0.99, thresh_unique2=0.0001):
    r"""Result of the Z score test defined by

    .. math::

        Z = \frac{x-\nu}{\sigma}

    where :math:`x`is the observation data (the column DataFrame), :math:`\nu` is the mean of the sample and :math:`\sigma` is the standard deviation of the sample.
    The Z score test is calculated for the input column DataFrame and return the index of elements lying outside the interval :math:`[\nu - 6{\sigma}, \nu + 6{\sigma}]`. The test is only applied if the input satisfy the condition that the uniqueness ratio (see :py:meth:`_is_unique`), lies between the input uniqueness thresholds,This prevent false positive.

    .. seealso::

        `Z score <https://en.wikipedia.org/wiki/Standard_score>`_

    Args:
        col (pandas.DataFrame): Input column DataFrame

        thresh (int, optional): Z score cut threshold. Defaults to 6.

        thresh_unique1 (float, optional): threshold to skip the test for DataFrame with high uniqueness. Defaults to 0.99.

        thresh_unique2 (float, optional): threshold to skip the test for discret DataFrame i.e. high clustering sample. Defaults to 0.0001.

    Returns:

        list of indices of row rejected by  the Z score test.
    """
    ratio_uniqueness = _is_unique(df=col)
    if (ratio_uniqueness > thresh_unique1) or  (ratio_uniqueness < thresh_unique2):
        return []
    else :
        mean = col.mean()
        std = col.std()

        upper_bound = mean + thresh * std
        lower_bound = mean - thresh * std
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


def _drop_non_unique(df, col_name, thresh_1=0.99, thresh_2=0.001, clean_dup=True):

    if clean_dup:
        df, _ = _is_duplicated(df)  # cleaning duplications in the dataframe
    ratio_unique = _is_unique(df, col_name)
    print(ratio_unique)
    if ratio_unique > thresh_1:
        bad_idx = df[col_name][df[col_name].duplicated(keep=False)].index
    elif ratio_unique < thresh_2:
        bad_idx = df[col_name][~df[col_name].duplicated(keep=False)].index
    else:
        bad_idx = []
    return bad_idx


def _col_is_none(df, thresh_col=0.99, clean_dup=False):
    """check column having mean number of nan > thresh

    Args:
        df ([type]): [description]
        thresh_col (float, optional): [description]. Defaults to 0.99.
        clean_dup (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: [description]
    """

    if clean_dup:
        df, _ = _is_duplicated(df)  # cleaning duplications in the dataframe

    mean_none_col = df.isnull().mean(axis=0)  # mean(none) in each column of df_c
    name_col_drop = mean_none_col[
        mean_none_col >= thresh_col
    ].index  # list of names of the column with none mean>thresh_col

    return name_col_drop


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

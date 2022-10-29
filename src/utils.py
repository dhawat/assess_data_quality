from math import nan
import pandas as pd
import re
import numpy as np

from dateutil.parser import parse
from difflib import SequenceMatcher
from sklearn.cluster import AffinityPropagation
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from markov_clustering import MarkovClustering

# from gensim.models import Word2Vec
#! add insignifante column test for non in column

"""df = pd.read_csv("logs.csv")  # read data
df = df.set_index("d")  # to re-index with a column 'd'
df = df.sort_index()  # to sort with respect to the index """


def check_extension(path):
    """Checks if the extension of the data set belongs to {csv, json, sql, xlsx}.

    Args:

        path (str): data set name. The allowed formats are {csv, json, sql, xlsx}.

    Returns:

        string specifying the type of the data set if allowed, else none.
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


def _to_DataFrame(path, **kwargs):
    """Transforms the input data into a DataFrame.

    Args:

        path (str): data set name. The allowed formats are {csv, json, sql, xlsx}.

    Keyword Args:

        kwargs(dict): keyword arguments of ``pandas.read``.

    Returns:

        df (pandas.DataFrame): DataFrame containing the input data.
    """

    ext = check_extension(path)  #Check if the data format is acceptable
    assert ext != "none"
    f_dict = {
        "csv": pd.read_csv,
        "json": pd.read_json,
        "sql": pd.read_sql,
        "xlsx": pd.read_excel,
    }
    df = f_dict[ext](path, **kwargs)
    return df


def get_metadata(df):
    """Read the input DataFrame and generate relevant metadata.

    Args:

        df (pandas.DataFrame): input DataFrame.

    Returns:

        dict: {name_of_column: metadata_associated}.
    """
    metadata = []
    for column in df:
        metadata.append(check_data_type(df[column]))
    return metadata


#todo check the last return after else
def check_data_type(column):

    """Returns the dominant type (non-NAN) of the elements in the input column.

    Args:

        column (pandas.core.series.Series): input DataFrame's column.

    Returns:

        str: dominant type (non-NAN) of the elements in the input column.
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


def _is_duplicated(df):
    """Return a copy of the input DataFrame without duplicate rows.

    Args:

        df (pandas.DataFrame): input DataFrame.

    Returns:

        duplicated_row (pandas.DataFrame): DataFrame containing the duplicate rows.

        df_clean (pandas.DataFrame): DataFrame without duplicate rows.
    """
    duplicated_row = df[df.duplicated()]
    df_clean = df[~df.duplicated()]
    return df_clean, duplicated_row


def _duplicated_idx(df, keep='first'):
    """Return boolean Series denoting duplicate rows.
    pandas.DataFrame.duplicated function `pandas.DataFrame.duplicated <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html>`_.

    Args:

        df (pandas.DataFrame): input DataFrame.
        keep (str, bool): "first", "last" or bool. Determines which duplicates (if any) to mark.
        See `pandas.DataFrame.duplicated <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.duplicated.html#:~:text=keep%7B%E2%80%98first%E2%80%99%2C%20%E2%80%98last%E2%80%99%2C%20False%7D%2C%20default%20%E2%80%98first%E2%80%99>`_.

    Returns:

        Series: Boolean series for each duplicated rows.
    """
    return df.duplicated(keep=keep)



def _is_unique(df, column_name=""):
    """Returns the value of the uniqueness ratio for the input DataFrame's column.
    The ratio is equal to:
        1 - (number of repeated elements)/(total number of elements).
    NAN elements are excluded.

    Args:

        df (pandas.DataFrame): input DataFrame.

        column_name (str, optional): name of the column to be used. Defaults to "".

    Returns:

        float : 1 - (number of repeated elements in the column)/(total number of elements in the column).

    .. note::

            A ratio 1 means all the elements are unique.
            A ratio 0 means all the elements are repeated or the column is empty (NAN).
    """
    if column_name is not nan:
        df = df[column_name]
    else:
        df = df
    if df.dropna().shape[0] == 0:
        return 0
    else:
        return df.dropna().nunique() / df.dropna().shape[0]


def _z_score(column, uniqueness_ratio, thresh_std=6, thresh_unique1=0.99, thresh_unique2=0.0001):
    r"""Z-score test.
    The value of Z is computed as follows

    .. math::

        Z = \frac{x-\nu}{\sigma}

    where :math:`x` is the input column, :math:`\nu` and :math:`\sigma` are the corresponding sample mean and sample standard deviation.
    The test is applied to the input column if the value of the corresponding uniqueness ratio belongs to :math:`[thresh_unique1, thresh_unique2]`, see :py:meth:`_is_unique`.
    It returns the indexes of the elements lying outside the interval :math:`[\nu - 6 \sigma, \nu + 6 \sigma]`.

    .. seealso::

        `Z score <https://en.wikipedia.org/wiki/Standard_score>`_.

    Args:
        column (pandas.core.series.Series): input DataFrame.

        uniqueness_ratio (float): value of the uniqueness ratio of the input column. Obtained using :py:func:`_is_unique`.

        thresh_std (int, optional): Z-score threshold. The indexes of the elements lying outside the interval :math:`[\nu - thresh_std \sigma, \nu + thresh_std \sigma]` will be returned. Defaults to 6.

        thresh_unique1 (float, optional): upper bound of the uniqueness threshold. The test is performed if the uniqueness ratio of the input column is lower than ``thresh_unique1``. Defaults to 0.99.

        thresh_unique2 (float, optional): lower bound of the uniqueness threshold. The test is performed if the uniqueness ratio of the input column is greater than ``thresh_unique1``. Defaults to 0.0001.

    Returns:

        list: Indexes of the rejected elements of the input columns by the Z-score tests.
    """
    if (uniqueness_ratio > thresh_unique1) or (uniqueness_ratio < thresh_unique2):
        return []
    else:
        mean = column.mean()
        std = column.std()

        upper_bound = mean + thresh_std * std
        lower_bound = mean - thresh_std * std
        idx = column[
            ~((column > lower_bound) & (column < upper_bound))
        ].index  # trancate values from the column
        return idx


# todo: Suggestion: Currently the threshold for the anomaly is fixed at Q_1 = round(np.percentile(unique_counts, 5)), which could be improved. Using for example the number of occurrences of words. Also, duplicate words can be detected using this method. For each word detected as an outlier, we can divide the obtained score by the frequency of occurrence of the word.

def incorrect_grammar(col, cluster, thresh):
    """Return the indexes (in ``col``) of the elements of ``clusters`` identified as bad data.
    The bad data are those having an occurrence number less than the input threshold.

    Args:

        col (pandas.core.series.Series): input DataFrame column.

        cluster (numpy.array): cluster of words to be checked.

        thresh (int): lower bound for the number of repetitions of each word of ``cluster`` in the input column. Bad words are the ones with a repetition number less than this threshold.

    Returns:

        list: the list of the bad indexes.

    .. seealso::

        :py:meth:`index_uncorrect_grammar`.
    """
    words = np.asarray(col)
    unique_words, unique_counts = np.unique(col, return_counts=True)
    bad_idx = []
    for w in cluster:
        count = unique_counts[np.where(unique_words == w)[0]][
            0
        ]  # cardinality of occurrences  of w in col
        if count <= thresh:
            bad_idx = bad_idx + np.ndarray.tolist(np.where(words == w)[0])  # bad index
    return bad_idx


# todo change the name of the function
# todo check spell
def index_incorrect_grammar(col, thresh=10, method='affinity_propagation', affinity="precomputed", damping=0.5, random_state=None, **kwargs):
    r"""Returns a list of the detected miss-spelled words in the input column.
        First, the words are clustered using affinity propagation, or a Markov clustering method.
        Finally, the function :py:func:`incorrect_grammar` is used to identify the errors.

    Args:

        col (pandas.core.series.Series): input DataFrame column.


        thresh (int, optional): lower bound for the number of repetitions of each word of ``cluster`` in the input column. Errors are the ones with a repetition number less than this threshold. See :py:func:`incorrect_grammar`. Default to 10.

        method(str, optional): "affinity_propagation" or "markov_clustering". The available clustering methods. The words of the input column are clustered using the specified method, then the function :py:func:`incorrect_grammar` is applied to identify the errors.

        affinity(str, optional): used when method="affinity_propagation". Specify which affinity to use. At the moment 'precomputed' and 'euclidean' are supported. 'euclidean' uses the negative squared euclidean distance between points. See ` sklearn.cluster.AffinityPropagation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html#:~:text=the%20input%20similarities.-,affinity,-%7B%E2%80%98euclidean%E2%80%99%2C%20%E2%80%98precomputed%E2%80%99%7D%2C%20default>`_. Default to "precomputed".

        damping(float, optional): used when method="affinity_propagation". Damping factor in the range [0.5, 1.0) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). This in order to avoid numerical oscillations when updating these values. See ` sklearn.cluster.AffinityPropagation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html>`_. Default to 0.5.

        random_state (int, optional): Pseudo-random number generator to control the starting state. Use an int for reproducible results across function calls.See ` sklearn.cluster.AffinityPropagation <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html>`_. Default to None.

    Keyword arguments:

        kwargs: keyword arguments of the input method.

    Returns:

        list: The list of indexes of the detected miss-spelled or error words.

    .. seealso::

            :py:meth:`incorrect_grammar`.
    """
    list_incorrect = []
    words = np.unique(np.asarray(col))
    if method == 'affinity_propagation':
        lev_similarity = np.array(
            [[SequenceMatcher(None, w1, w2).ratio() for w1 in words] for w2 in words]
        )  #similarity matrix of `words`

        #fitting the model
        affprop = AffinityPropagation(affinity=affinity, damping=damping, random_state=random_state, **kwargs)
        affprop.fit(lev_similarity)

        if len(np.unique(affprop.labels_)) == 1:
            return list_incorrect
        else:
            for cluster_id in np.unique(affprop.labels_):
                cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
                if len(cluster) > 1:
                    list_incorrect = list_incorrect + incorrect_grammar(
                        col, cluster, thresh
                    )
        return list_incorrect
    elif method == 'markov_clustering':
        X = CountVectorizer().fit_transform(words)
        X = TfidfTransformer(use_idf=False).fit_transform(X)
        Model = MarkovClustering(X, **kwargs)
        dict_cluster = Model.fit().clusters()

        if len(dict_cluster) == 1:
            return list_incorrect
        else:
            for val in dict_cluster.values():
                cluster = np.unique(words[list(val)])
                if len(cluster) > 1:
                    list_incorrect = list_incorrect + incorrect_grammar(
                            col, cluster, thresh
                        )
            return list_incorrect
    else:
        raise ValueError("The available methods are 'affinity_propagation' and 'markov_clustering'.")


#! I'm here
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
            elements = df.loc[df[[w1, w2]].dropna().index]
            range_anomalie = 0
            proportion = np.shape(np.where(elements[w1] < elements[w2])[0])[0] / len(df)
            if proportion > thresh:
                range_anomalie = np.shape(np.where(elements[w1] > elements[w2])[0])[0]
                if range_anomalie > 0:
                    dictionnaire_anomalie_tendance[(w1, w2)] = np.ndarray.tolist(
                        np.where(elements[w1] > elements[w2])[0]
                    )
    return dictionnaire_anomalie_tendance

def outlier_detection(array_classe, q_1=0.25, q_3=0.75):
        QQ_1 = np.quantile(array_classe, 0.05)
        Q_1 = np.quantile(array_classe, q_1)
        Q_3 = np.quantile(array_classe, q_3)
        IQR = Q_3 - Q_1
        upper_bound = Q_3 + (IQR * 1.25)
        v_outlier = np.where(((array_classe <= QQ_1)
                                        | (array_classe >= upper_bound)))[0]
        if len(v_outlier) > 0:
            return v_outlier
        else:
            return []

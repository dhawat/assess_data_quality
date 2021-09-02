import numpy as np
from sklearn.cluster import AffinityPropagation
from difflib import SequenceMatcher
import ipdb

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
    index_in_words = []
    for w in cluster:
        count = unique_counts[np.where(unique_words == w)[0]][0]
        if count <= min_occurence:
            index_in_words += np.ndarray.tolist(np.where(words == w)[0])

    return index_in_words


def index_uncorrect_grammar(df):
    """[function for finding typo in a array like structure using clustering methods]

    Args:
        df ([DataFrame column, array_like]): [Array like variable containing strings to check for typos.]

    Returns:
        [list]: [returns a list of location index of found typos.]
    """

    df_unique = np.unique(df)
    words = np.asarray(df_unique) #So that indexing with a list will work
    lev_similarity = np.array([[SequenceMatcher(None, w1, w2).ratio() for w1 in words] for w2 in words])
    affprop = AffinityPropagation(affinity = "precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    list_uncorrect = []
    if len(np.unique(affprop.labels_)) == 1:
        return list_uncorrect
    else:
        for cluster_id in np.unique(affprop.labels_):
            ipdb.set_trace()
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            if len(cluster) > 1:
                list_uncorrect = list_uncorrect + uncorrect_grammar(df, cluster, 10)

    return list_uncorrect



def uncorrect_grammar_suggestion(df_names, cluster, min_occurence, nb_suggestion):
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
    cluster_count = []
    index_in_words = []
    suggestions = []
    for w in cluster:
        count = unique_counts[np.where(unique_words == w)[0]][0]
        cluster_count.append(count)
        if count <= min_occurence:
            index_in_words += np.ndarray.tolist(np.where(words == w)[0])
    for typo in index_in_words:
        for i in range(nb_suggestion):
            suggestions.append(cluster[np.argsort(cluster_count)][-(i + 1)])
    return index_in_words, suggestions


def index_uncorrect_grammar_suggestion(df, nb_suggestion=1):
    """[function for finding typo in a array like structure using clustering methods]

    Args:
        df ([DataFrame column, array_like]): [Array like variable containing strings to check for typos.]
        nb_suggestion ([int]): [Number of suggestion added with the returned index]

    Returns:
        [list]: [returns a list of location index of found typos.]
    """

    df_unique = np.unique(df)
    words = np.asarray(df_unique) #So that indexing with a list will work
    lev_similarity = np.array([[SequenceMatcher(None, w1, w2).ratio() for w1 in words] for w2 in words])
    affprop = AffinityPropagation(affinity = "precomputed", damping=0.5)
    affprop.fit(lev_similarity)
    list_uncorrect = []
    list_suggestion = []
    if len(np.unique(affprop.labels_)) == 1:
        return list_uncorrect, list_suggestion
    else:
        for cluster_id in np.unique(affprop.labels_):
            cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
            if len(cluster) > 1:
                guess = uncorrect_grammar_suggestion(df, cluster, 10, nb_suggestion)
                list_uncorrect += guess[0]
                list_suggestion +=guess[1]
    return list_uncorrect, list_suggestion



from improve_data_quality import Data 
data = Data('..\data_avec_erreurs_wasserstein.csv')
idx, suggestion = index_uncorrect_grammar_suggestion(data.data['state'], 1)
#print(data.data['state'].iloc[idx], suggestion)
print(list(zip(data.data['state'].iloc[idx], suggestion)))
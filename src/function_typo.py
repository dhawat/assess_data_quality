from markov_clustering import MarkovClustering
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

"""Data_file = "~/Downloads/echantillon.csv"

df = pd.read_csv(Data_file)

df_names = np.asarray(df["MARQUE"])"""


def Typo_correction(df_names):
    words = np.asarray(df_names)
    df_unique = np.unique(df_names)
    unique_words = np.asarray(df_unique)
    X = CountVectorizer().fit_transform(unique_words)
    X = TfidfTransformer(use_idf=False).fit_transform(X)
    Model = MarkovClustering(X)
    list_cluster = list(Model.fit().clusters().items())
    index_incorrect_words = []
    correct_words = []
    i = 0
    if len(list_cluster) == len(words):
        print("Rien n'a été détecter")
        return 0
    else:
        for cluster in list_cluster:
            list_clust = list(cluster)[1]
            cluster_count = []
            if len(unique_words[list(list_clust)]) > 1:
                _, unique_counts = np.unique(df_names, return_counts=True)
                count_cluster = [
                    unique_counts[np.where(unique_words == w)[0]][0]
                    for w in unique_words[list(list_clust)]
                ]
                for w in unique_words[list(list_clust)]:
                    count_w = unique_counts[np.where(unique_words == w)[0]][0]
                    if count_w < max(count_cluster):
                        index_incorrect_words += np.ndarray.tolist(
                            np.where(words == w)[0]
                        )
                        correct_words += [
                            unique_words[list(list_clust)][np.argmax(count_cluster)]
                        ] * len(np.ndarray.tolist(np.where(words == w)[0]))
    return (words[index_incorrect_words], correct_words)

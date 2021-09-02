import numpy as np
from scipy.sparse import linalg, eye, csr_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict

class MarkovClustering:
    def __init__(self, matrix, metric="cosine", bias=1):
        """
        Initializing similarity matrix
        Either by setting it (metric = None)
        Or by deriving it from a distance function (similarity = bias-distance)
        """
        self.labels_ = None
        if metric is None:
            self.T = matrix
        else:
            self.T = csr_matrix(bias-pairwise_distances(matrix, metric=metric))

    def normalize(self):
        self.T = normalize(self.T, norm='l1', axis=1)

    def self_loops(self, weight=0.01):
        self.T = eye(self.T.shape[0]) * weight + self.T

    def expansion(self, p=2):
        ret = self.T
        for _ in range(1,p):
            ret = ret * self.T
        self.T = ret

    def inflation(self, p=2, th=1e-10):
        for i in range(len(self.T.data)):
            if self.T.data[i] < th:
                self.T.data[i] = 0
            else:
                self.T.data[i] = self.T.data[i] ** p

    def fit(self, inflation_power=2, inflation_threshold=1e-10, self_loops_weight=0.01, expansion_power=2, iteration_limit=100,verbose=False):
        iterations = 0
        prev_T = csr_matrix(self.T.shape)
        self.self_loops(self_loops_weight)
        while (iterations < iteration_limit) and ((prev_T - self.T).nnz != 0):
            prev_T = self.T
            iterations += 1
            self.normalize()
            self.expansion(expansion_power)
            self.inflation(inflation_power, inflation_threshold)
            if verbose:
                print ("========Iteration #{i}=======".format(i=iterations))
                print(self.T.toarray())
        self.labels_ = self.extract_labels()
        return self

    def extract_labels(self):
        M = self.T.tocoo()
        rows = defaultdict(set)
        for i, d in enumerate(M.data):
            if d == 0:
                continue
            rows[M.row[i]].add(M.col[i])
        hash_row = lambda l: ",".join(map(str,sorted(l)))
        row_hashes = [hash_row(rows[i]) for i in range(M.shape[0])]
        d = dict([(l,i) for i,l in enumerate(set(row_hashes))])
        labels = [d[row_hashes[i]] for i in range(M.shape[0])]
        return labels

    def clusters(self, labels=None):
        ret = defaultdict(set)
        for i,c in enumerate(self.labels_):
            if labels is None:
                ret[c].add(i)
            else:
                ret[c].add(labels[i])
        return ret
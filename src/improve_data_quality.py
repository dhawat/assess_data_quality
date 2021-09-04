import utils as utils
import ipdb
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class Data:
    """Data class holding a column by column profile and index flagged as low quality data"""

    def __init__(self, path=""):
        """
        Args:
            data (CSV, JSON, SQL): data set.
        """
        if utils.check_extension(path) == "none":
            raise TypeError("data should be of provided as .csv or .json or .sql file")

        self.data = utils._to_DataFrame(path)
        self._good_index = list(range(self.data.shape[0]))
        self._bad_index = pd.DataFrame(columns=["idx", "column", "errtype"])
        self._nbr_col = []
        self._str_col = []

    @property
    def good_index(self):
        """getter for private attribute _good_index

        Returns:
            list: list of good indexes to use for ML training purposes
        """
        return self._good_index

    @property
    def bad_index(self):
        """getter for private attribute for _bad_index

        Returns:
            dataFrame: dataFrame containing error indexes and if applicable column and an explaination of the error
        """
        return self._bad_index

    @bad_index.setter
    def bad_index(self, list_idx):
        """setter if private attribute bad_index

        Args:
            list_idx (dataFrame): used as bad_index = bad_index.append(df)
        """
        self._bad_index = list_idx

    @good_index.setter
    def good_index(self, list_idx):
        """setter for private attribute _good_index

        Args:
            list_idx (list): list of good index to replace the previous one

        Raises:
            ValueError: if length is greater than the initial dataFrame raises Valueerror
        """
        if len(list_idx) > self.data.shape[0]:
            raise ValueError(
                "Index length must be smaller than the length of the dataframe"
            )
        self._good_index = list_idx

    @property
    def str_col(self):
        """getter for private attribute _str_col

        Returns:
            [list]: list of string columns
        """
        if not self._str_col:
            for column in self.data.columns:
                col_type = utils.check_data_type(self.data[column])
                if col_type == type(str()):
                    self._str_col.append(column)
                elif col_type in [type(int()), type(float())]:
                    self._nbr_col.append(column)
                elif str(col_type) == "datetime64[ns]":
                    self.data[column] = pd.to_datetime(self.data[column]).apply(
                        lambda x: utils._year(x)
                    )
                    self._nbr_col.append(column)

        return self._str_col

    @str_col.setter
    def str_col(self, value):
        self._str_col = value

    @property
    def nbr_col(self):
        """getter for private attribute _nbr_col

        Returns:
            [list]: list of number columns
        """
        if not self._nbr_col:
            for column in self.data.columns:
                col_type = utils.check_data_type(self.data[column])
                if col_type == type(str()):
                    self._str_col.append(column)
                elif col_type in [type(int()), type(float())]:
                    self._nbr_col.append(column)
                elif str(col_type) == "datetime64[ns]":
                    self.data[column] = pd.to_datetime(self.data[column]).apply(
                        lambda x: utils._year(x)
                    )
                    self._nbr_col.append(column)

        return self._nbr_col

    @nbr_col.setter
    def nbr_col(self, value):
        self._nbr_col = value

    def push_bad_index(self, list_idx):  # Find a better method name
        """decrepated, not sure if will be used or not.

        Args:
            list_idx ([type]): [description]
        """
        for elem in list_idx:
            try:
                self.bad_index.append(elem)
            except:
                pass

    def firstpass(self):
        """Push into self.bad_index the indexes and error types of data.
        This first pass detects duplicated data, typo, extreme values and incompleteness by row.
        """
        # Deterministic pass
        n_duped_idx = ~utils._duplicated_idx(self.data)

        for index in n_duped_idx[~n_duped_idx].index.values.tolist():
            self.bad_index = self.bad_index.append(
                {"idx": index, "column": "All", "errtype": "duplication"},
                ignore_index=True,
            )

        # Probabilistic pass
        # Columns of strings only
        for column in self.str_col:
            if (
                utils._is_unique(self.data, column) <= 0.005
            ):  # Filter column with too many different words
                clean_df = self.data[n_duped_idx]
                clean_df = clean_df[column][clean_df[column].notna().values]
                idx = utils.index_uncorrect_grammar(
                    clean_df
                )  # get the non duped indexes and not na from a column
                idx = clean_df.iloc[idx].index

                for index in idx:
                    row = {"idx": index, "column": column, "errtype": "typo"}
                    self.bad_index = self.bad_index.append(row, ignore_index=True)

        for column in self.nbr_col:  # Columns of numbers only
            clean_df = self.data[n_duped_idx]
            clean_df = clean_df[column][clean_df[column].notna().values]
            idx = utils._z_score(clean_df, clean_df.mean(), clean_df.std())
            idx = clean_df[idx].index

            for index in idx:
                row = {"idx": index, "column": column, "errtype": "extreme value"}
                self.bad_index = self.bad_index.append(row, ignore_index=True)

        # Completeness pass on each row
        idx = utils._row_is_none(self.data)
        for index in idx:
            row = {"idx": index, "column": "All", "errtype": "too much nan"}
            self.bad_index = self.bad_index.append(row, ignore_index=True)

    def secondpass(self):
        """Push into self.bad_index the indexes.
        This second pass detects only idx where outliers may lie.
        """

        # Tendency pass, 2 column explicable
        idx_dict = utils._tendancy_detection(
            utils._to_date_and_float(self.data[self.nbr_col])
        )
        for key, value in idx_dict.items():
            for idx in value:
                row = {"idx": idx, "column": key, "errtype": "Logic error"}
                self.bad_index = self.bad_index.append(row, ignore_index=True)

        # Outlier pass, less explicable.
        # idx = self.outlier_lof()[0]
        # for index in idx:
        #    row = {"idx": index, "column": "NA", "errtype": "Outlier"}
        #    self.bad_index = self.bad_index.append(row, ignore_index=True)

        idxes, col_names = self.bad_logical_index()
        for idex, cols in zip(idxes, col_names):
            for idx, col in zip(idex, cols):
                row = {"idx": idx, "column": [col[0], col[1]], "errtype": "Logic error"}
                self.bad_index = self.bad_index.append(row, ignore_index=True)

    def imputation_method(self, **params):
        params.setdefault("n_neighbors", 10)
        params.setdefault("weights", "uniform")
        self.data, _ = utils._is_duplicated(self.data)
        list_numeric_col_name = self.nbr_col  # name of numerical column
        numeric_df = self.data[list_numeric_col_name]  # numeric dataframe
        numeric_df = numeric_df.fillna(np.nan)  # fill none with np.nan
        imputer = KNNImputer(**params)  # initialize imputation
        numeric_df_imputation = imputer.fit_transform(numeric_df)  # imputation if df
        numeric_df_imputation = pd.DataFrame(numeric_df_imputation)
        numeric_df_imputation.columns = list_numeric_col_name
        return numeric_df_imputation

    def outlier_lof(self, drop_id=True, **params):
        """outlier detection over rows, from numerical columns
        .. warning::
                automatic imputation on the numerical columns
        Returns:
            [type]: [description]
        """
        # set default params
        params.setdefault("n_neighbors", 20)
        params.setdefault("contamination", 0.0005)
        params.setdefault("metric", "chebyshev")
        params.setdefault("n_jobs", -1)

        # drop unique column
        if drop_id:
            df_drop_col0 = self.data.drop(
                labels=self.data.columns[0], axis=1
            )  # drop unique column
        else:
            df_drop_col0 = self.data

        if (df_drop_col0.isnull()).sum().all() > 0:
            # imputation
            df_drop_col0 = self.imputation_method()

        # lof phase
        clf = LocalOutlierFactor(**params)
        y_pred = clf.fit_predict(np.asarray(df_drop_col0))
        ind = self.data.loc[y_pred == -1].index  # index of outlier dected
        neg_lof_score = clf.negative_outlier_factor_
        normalized_lof_score = np.abs(neg_lof_score[y_pred == -1]) / np.max(
            abs(neg_lof_score[y_pred == -1])
        )  # normalized lof score

        # adding lof score column to the data frame
        df_with_score = self.data
        df_with_score["lof"] = ""
        df_with_score["lof"].loc[ind] = normalized_lof_score

        return ind, normalized_lof_score, df_with_score

    def col_combined_result(self, col1_name, col2_name, first_pass=False):
        # todo add if condition for column where we do not detect error
        """Combine good result after first path of two columns, the output is a data frame combining good result from 2 column after first path with good index


        Args:
            col1_name (str): name of the first column
            col2_name (str): name of the second column

        Returns:
            [type]: Data frame combining good result from 2 column after first path


        """

        if first_pass:
            self.firstpass()

        df_bad = self.bad_index
        # df of summery of bad bata found during forst pass
        bad_idx_col1 = list(
            df_bad[df_bad["column"] == "col1_name"]["idx"]
        )  # index of bad data in col1
        bad_idx_col2 = list(
            df_bad[df_bad["column"] == "col2_name"]["idx"]
        )  # index of bad data in col1
        dup_idx = list(df_bad[df_bad["errtype"] == "duplication"]["idx"])
        # combining bad index

        bad_idx_all = (
            dup_idx + bad_idx_col1 + list(set(bad_idx_col2) - set(bad_idx_col1))
        )  # union of bad index
        bad_idx_all.sort()

        good_cols = (self.data.drop(bad_idx_all)[[col1_name, col2_name]]).dropna()
        return good_cols

    def dual_hist(self, col1_name, col2_name, unique_tresh=0.7, first_pass=False):
        if first_pass:
            self.firstpass()

        """for col1_name in self.str_col:
            for col2_name in self.str_col:
                # todo extract big uniqueness ratio
                if (
                    col1_name != col2_name
                    and (utils._is_unique(self.data, col1_name) < unique_tresh)
                    and (utils._is_unique(self.data, col2_name) < unique_tresh)
                ):
                    df_clean = self.col_combined_result(
                        col1_name=col1_name, col2_name=col2_name
                    )
                    df_combined = df_clean.apply(lambda row: tuple(row.values), axis=1)
                    # ipdb.set_trace()
                    summery_tuple = np.unique(df_combined, return_counts=True)
                    ipdb.set_trace()"""
        if (
            col1_name != col2_name
            and (utils._is_unique(self.data, col1_name) < unique_tresh)
            and (utils._is_unique(self.data, col2_name) < unique_tresh)
        ):
            df_clean = self.col_combined_result(
                col1_name=col1_name, col2_name=col2_name
            )
            df_combined = df_clean.apply(lambda row: tuple(row.values), axis=1)
            # ipdb.set_trace()
            summery_tuple = np.unique(df_combined, return_counts=True)
        else:
            print("non unique")
        return summery_tuple

    def bad_logical_index(self):
        df = self.data.iloc[self.good_index]
        col_names = []
        idxes = []
        for col1 in self.str_col:
            for col2 in self.str_col:
                if col1 != col2 and utils._is_unique(df, col1) and utils._is_unique(df, col2) < 0.001:
                    freq = df.groupby([col1, col2]).size()
                    elements = df.loc[df[[col1, col2]].dropna().index]
                    for elem in elements[col1].unique():
                        #
                        for e, index_serie in zip(freq[elem], freq[elem].index.tolist()):
                            if e < 10:
                                idxes.append(df[col1].index[df[col1] == index_serie].tolist() 
                                + df[col2].index[df[col2] == index_serie].tolist())
                                col_names.append([col1, col2])
        return idxes, col_names


#! please use our commun directory
data = Data('..\data\data_avec_erreurs_wasserstein.csv')
#data.firstpass()
data.secondpass()
#data.bad_index.to_csv('exemple.csv')

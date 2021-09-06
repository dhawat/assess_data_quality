from typing import KeysView
import utils as utils
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder


class Data:
    """Implementation of many test for detecting bad data in a dataframe, and return the index of bad data there type and the column if possible"""

    def __init__(self, path, drop_first_col=True, **kwargs):
        """
        Args:

            path (path): path to the data file.

            drop_first_col (bool, optional): If true drop first column which most of the time is unique and so there it is not interresting to use it for finding bad data. Defaults to True.

            kwargs(dic): dictionary of additional argument passed to the read method of the data. See :py:meth:`utils._to_DataFrame`

        Raises:

            TypeError: if the format of the data does not belong to {.csv, .json, .sql, .xlsx}
        """
        if utils.check_extension(path) == "none":
            raise TypeError(
                "data should be of provided as .csv or .json or .sql or .xlsx file"
            )

        self.data = utils._to_DataFrame(path, **kwargs)  # DataFrame of the data

        # drop first column
        if drop_first_col:
            self.data = self.data.drop(self.data.columns[0], axis=1)
        self._good_index = list(range(self.data.shape[0]))
        self._bad_index = pd.DataFrame(
            columns=["idx", "column", "errtype", "value1", "value2"]
        )
        self._nbr_col = []  # numeric columns in data
        self._str_col = []  # string columns in data
        self._corr_col = {}  # matrix of correlation between column
        self._uniq_col = {}  # ration of uniqueness of each column

    @property
    def good_index(self):
        """getter for private attribute _good_index

        Returns:

            list: list of good indexes to use for ML training purposes
        """
        return self._good_index

    @property
    def bad_index(self):
        """getter for private attribute for _bad_index.

        Returns:

            dataFrame: dataFrame containing error indexes and if applicable column and an explanation of the error
        """
        return self._bad_index

    @bad_index.setter
    def bad_index(self, list_idx):
        """setter if private attribute bad_index.

        Args:

            list_idx (dataFrame): used as bad_index = bad_index.append(df).
        """
        self._bad_index = list_idx

    @good_index.setter
    def good_index(self, list_idx):
        """setter for private attribute _good_index.

        Args:

            list_idx (list): list of good index to replace the previous one.

        Raises:

            ValueError: if length is greater than the initial dataFrame raises Valueerror.
        """
        if len(list_idx) > self.data.shape[0]:
            raise ValueError(
                "Index length must be smaller than the length of the dataframe"
            )
        self._good_index = list_idx

    @property
    def str_col(self):
        """getter for private attribute :py:meth:`_str_col`.

        Returns:

            [list]: list of string columns.
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
        """getter for private attribute :py:meth:`_nbr_col`.

        Returns:

            [list]: list of number columns.
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
        """setter method for setting :py:meth:`_nbr_col` private attribute.

        Args:

            value ([list of strings]): [list of column names].
        """
        self._nbr_col = value

    # todo check if used
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

    @property
    def corr_col(self):
        """getter for private attribute py:meth:`_corr_col`.

        Returns:

            [list]: dict of {col: [correlated_cols]}.
        """
        if not self._corr_col:
            self._corr_col = self.compute_corr_str()
            for col in self.str_col:
                if col not in self._corr_col:
                    self._corr_col[col] = []
            return self._corr_col
        else:
            return self._corr_col

    @corr_col.setter
    def corr_col(self, value):
        """setter for private attribute :py:meth:`_corr_col`.

        Args:

            value ([dict]): dict of {col: [correlated_cols]}
        """
        self._corr_col = value

    @property
    def uniq_col(self):
        """getter for private attribute :py:meth:`_uniq_col`.

        Returns:

            [dict]: [dict containing {name_of_col: ratio_of_uniqueness}]
        """
        if not self._uniq_col:
            self._uniq_col = {
                col: utils._is_unique(self.data, col) for col in self.data.columns
            }
            return self._uniq_col
        else:
            return self._uniq_col

    @uniq_col.setter
    def uniq_col(self, value):
        """setter for private attribute :py:meth:`_uniq_col`.

        Args:

            value ([dict]): [dict containing {name_of_col: ratio_of_uniqueness}]
        """
        self._uniq_col = value

    def firstpass(self, *methods):
        """First pass of bad data detection by searching for duplicated data, typo, extreme values and incompleteness by row.  This tests are done using respectively the methods :py:meth:`check_duplication`, :py:meth:`typo`, :py:meth:`extreme_value` and :py:meth:completeness. This method find the bed indices and send them to :py:meth:`bad_index`.
        By default all previous method will run. Specific methods could be set using the parameter `method`.

        """

        f_dict = {
            "duplication": self.check_duplication,
            "typo": self.check_typo,
            "extreme_value": self.check_extreme_value,
            "completeness": self.check_completeness,
        }

        for method in methods:
            f_dict[method]()
        if not methods:
            self.check_duplication()
            self.check_typo()
            self.check_extreme_value()
            self.check_completeness()

        # Eliminate the obvious errors from the good index
        for idx in self.bad_index["idx"]:
            try:
                self.good_index.remove(idx)
            except:
                pass

    def secondpass(self, *methods):
        """First pass of bad data detection by searching for tendency, outlier, logic and mixed_logic error.
        This tests are done using respectively the methods :py:meth:`check_tendency`, :py:meth:`check_outlier`, :py:meth:`check_logic` and :py:meth:`check_mixed_logic`.
        This method find the bed indices and send them to :py:meth:`bad_index`.
        By default all previous method will run. Specific methods could be set using the parameter `method`.
        """

        f_dict = {
            "tendency": self.check_tendency,
            "outlier": self.check_outlier,
            "logic": self.check_logic,
            "mixed_logic": self.check_mixed_logic,
        }

        for method in methods:
            f_dict[method]()
        if not methods:
            self.check_tendency()
            self.check_outlier()
            self.check_logic()
            self.check_mixed_logic()

    def check_duplication(self):
        """Find duplicated rows and return there indices."""
        n_duped_idx = ~utils._duplicated_idx(self.data)
        n_true_idx = ~utils._duplicated_idx(self.data) & utils._duplicated_idx(
            self.data, False
        )
        for index, true_index in zip(
            n_duped_idx[~n_duped_idx].index.values.tolist(),
            n_true_idx[n_true_idx].index.values.tolist(),
        ):
            self.add_to_bad_idx(
                [index, true_index], col="ALL", col_type="duplication", VALUE_FLAG=False
            )

    def check_typo(
        self,
        tresh_unique=0.005,
        tresh_typo_frequency=10,
        method="affinity_propagation",
        affinity="precomputed",
        damping=0.5,
        random_state=None,
        **kwargs
    ):
        """Find spelling

        Args:

            tresh_unique (float, optional): uniqueness threshold. Defaults to 0.005.

            tresh_typo_frequency (int, optional): frequency threshold. Defaults to 10.

            method (str, optional): name of the chosen method. Defaults to "affinity_propagation".

            affinity (str, optional): which affinity to use. At the moment ‘precomputed’ and euclidean are supported.
            ‘euclidean’ uses the negative squared euclidean distance between points. Defaults to "precomputed".

            damping (float, optional): Damping factor (between 0.5 and 1) is the extent to which the current value is maintained relative to incoming values (weighted 1 - damping). This in order to avoid numerical oscillations when updating these values (messages). Defaults to 0.5.

            random_state ([type], optional): [description]. Defaults to None.
        """
        n_duped_idx = ~utils._duplicated_idx(self.data)

        for column in self.str_col:
            if (
                self.uniq_col[column] <= tresh_unique
            ):  # Filter column with too many different words
                clean_df = self.data[n_duped_idx]
                clean_df = clean_df[column][clean_df[column].notna().values]
                idx = utils.index_incorrect_grammar(
                    clean_df,
                    tresh_typo_frequency,
                    method,
                    affinity,
                    damping,
                    random_state,
                    **kwargs
                )  # get the non duped indexes and not na from a column
                idx = clean_df.iloc[idx].index

                for index in idx:
                    self.add_to_bad_idx(
                        index, col=column, col_type="typo", VALUE_FLAG=True
                    )

    def check_extreme_value(
        self, thresh_std=6, thresh_unique1=0.99, thresh_unique2=0.0001
    ):
        """Find indices of extrem values using the z-score test

        Args:

            thresh_std (int, optional): coefficient of the standard deviation. Defaults to 6.

            thresh_unique1 (float, optional): thresholding for rejecting test on uniqueness. Defaults to 0.99.

            thresh_unique2 (float, optional): threshhold for rejecting discret. Defaults to 0.0001.
        """
        n_duped_idx = ~utils._duplicated_idx(self.data)

        for column in self.nbr_col:
            clean_df = self.data[n_duped_idx]
            clean_df = clean_df[column][clean_df[column].notna().values]
            idx = utils._z_score(
                clean_df,
                self.uniq_col[column],
                thresh_std,
                thresh_unique1,
                thresh_unique2,
            )
            idx = clean_df[idx].index

            for index in idx:
                self.add_to_bad_idx(
                    index, col=column, col_type="extreme", VALUE_FLAG=True
                )

    def check_completeness(self, thresh_row_1=0.7, thresh_row_2=0.5, thresh_col=0.8):
        """Find index of incomplete rows, i.e. rows containing too much nan.

        Args:

            thresh_row_1 (float, optional): threshold for row incompleteness. Defaults to 0.7.
            thresh_row_2 (float, optional): threshold for row incompleteness combined with column incompleteness. Defaults to 0.5.
            thresh_col (float, optional): threshold for column incompleteness. Defaults to 0.8.
        """
        index = utils._row_is_none(self.data, thresh_row_1, thresh_row_2, thresh_col)
        for idx in index:
            self.add_to_bad_idx(idx, col="All", col_type="empty", VALUE_FLAG=False)

    def check_tendency(self, tresh_order=0.999):
        """Detecting order between numerical columns

        Args:

            tresh_order (float, optional): threshold for accepting order relation hypothesis. Defaults to 0.999.
        """
        idx_dict = utils._tendancy_detection(
            utils._to_date_and_float(self.data[self.nbr_col]), tresh_order
        )
        for key, value in idx_dict.items():
            for idx in value:
                self.add_to_bad_idx(
                    idx, col=key, col_type="Logic error", VALUE_FLAG=True
                )

    def check_outlier(self, **params):
        """Outlier row detection by LOF algorithm, preceeded by imputation method"""
        idx = self.outlier_lof(**params)[0]
        for index in idx:
            self.add_to_bad_idx(index, col="NA", col_type="Outlier", VALUE_FLAG=False)

    def check_logic(self, thres_uniqueness=0.001, freq_error=10):
        """logic error on the numerical columns using frequency method

        Args:

            thres_uniqueness (float, optional): uniquenesess thresh. Defaults to 0.001.

            freq_error (int, optional): threshold for frequency detected as error. Defaults to 10.
        """
        idxes, col_names = self.bad_logical_index(thres_uniqueness, freq_error)
        for idex, cols in zip(idxes, col_names):
            for idx in idex:
                self.add_to_bad_idx(
                    idx, col=cols, col_type="Logic error", VALUE_FLAG=True
                )

    def check_mixed_logic(
        self, thres_unique_str=0.0002, thres_unique_nbr=0.04, thresh_std=6
    ):
        """Find index for logical error.

        Args:

            thres_unique_str (float, optional): uniqueness . Defaults to 0.0002.

            thres_unique_nbr (float, optional): [description]. Defaults to 0.04.

            thresh_std (int, optional): [description]. Defaults to 6.
        """
        idxes, col_names = self.bad_float_index(
            thres_unique_str, thres_unique_nbr, thresh_std
        )
        for idex, cols in zip(idxes, col_names):
            for idx in idex:
                self.add_to_bad_idx(
                    idx, col=cols, col_type="Logic error", VALUE_FLAG=True
                )

    def imputation_method(self, **params):
        # params.setdefault("n_neighbors", 10)
        # params.setdefault("weights", "uniform")
        parameters = {}
        # TODO: Change the fix, it would previously take the parameter from **params and results
        # In a failure because KNNimputer doesn't have an self.contamination attribute.
        parameters.setdefault("weights", "uniform")
        parameters.setdefault("n_neighbors", 10)
        df, _ = utils._is_duplicated(self.data)
        list_numeric_col_name = self.nbr_col
        numeric_df = df[list_numeric_col_name]  # numeric dataframe
        numeric_df = numeric_df.fillna(np.nan)  # fill none with np.nan
        imputer = KNNImputer(**parameters)  # initialize imputation

        numeric_df_imputation = imputer.fit_transform(numeric_df)  # imputation if df
        numeric_df_imputation = pd.DataFrame(numeric_df_imputation)
        numeric_df_imputation.columns = list_numeric_col_name
        return numeric_df_imputation

    def outlier_lof(self, **params):
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

        df_drop_col0 = self.data[self.nbr_col]

        if (df_drop_col0.isnull()).sum().any() > 0:
            # imputation
            df_drop_col0 = self.imputation_method(**params)

        # lof phase
        clf = LocalOutlierFactor(**params)
        y_pred = clf.fit_predict(np.asarray(df_drop_col0))
        ind = df_drop_col0.loc[y_pred == -1].index  # index of outlier dected
        neg_lof_score = clf.negative_outlier_factor_
        normalized_lof_score = np.abs(neg_lof_score[y_pred == -1]) / np.max(
            abs(neg_lof_score[y_pred == -1])
        )  # normalized lof score

        # adding lof score column to the data frame
        df_with_score = df_drop_col0
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
            and (self.uniq_col[col1_name] < unique_tresh)
            and (self.uniq_col[col2_name] < unique_tresh)
        ):
            df_clean = self.col_combined_result(
                col1_name=col1_name, col2_name=col2_name
            )
            df_combined = df_clean.groupby([col1_name, col2_name]).size
            df_combined = df_clean.apply(lambda row: tuple(row.values), axis=1)
            summery_tuple = np.unique(df_combined, return_counts=True)
        else:
            print("non unique")
        return summery_tuple

    def bad_logical_index(self, thres_uniqueness=0.001, freq_error=10):
        """Operates on data attribute directly. For each column which contains strings and doesn't have a uniqueness ratio too high.
        On theses columns compute the frequency between each unique data in columns.
        Returns:
            [idxes, col_names]: [list of list of bad indexes and associated columns names]
        """
        # TODO : Take good index from 2 columns only
        df = self.data.iloc[self.good_index]
        col_names = []
        idxes = []
        for col1 in self.str_col:
            for col2 in self.corr_col[col1]:
                if (
                    col1 != col2
                    and self.uniq_col[col1] < thres_uniqueness
                    and self.uniq_col[col2] < thres_uniqueness
                ):
                    freq = df.groupby([col1, col2]).size()
                    elements = df.loc[df[[col1, col2]].dropna().index]
                    for elem in elements[col1].unique():
                        for e, index_serie in zip(
                            freq[elem], freq[elem].index.tolist()
                        ):
                            if e < freq_error:
                                idxes.append(
                                    elements[col2]
                                    .index[
                                        (elements[col2] == index_serie)
                                        & (elements[col1] == elem)
                                    ]
                                    .tolist()
                                )
                                col_names.append([col1, col2])
        return idxes, col_names

    def compute_corr_str(self, threshold=0.5):
        """transform the strings column into categorical number and then compute the correlation matrix
        between the now number dataframe. Return for the columns the columns which have more than threshold of
        correlation.
        Args:
            threshold (float, optional): [minimum correlation for a column to be considered correlated]. Defaults to 0.5.
        Returns:
            [dict]: [contains for each column a list of possibly empty correlated columns]
        """
        list_col = []
        for col in self.str_col:
            if self.uniq_col[col] <= 0.001:
                list_col.append(col)
        df = self.data[list_col]

        le = LabelEncoder()
        le.fit(df.stack(dropna=False).reset_index(drop=True))
        for col in df.columns:
            df[col] = le.transform(df[col])

        corr_dict = {}
        df_corr = df.corr().abs() >= threshold
        for col in df_corr.columns:
            corr_col = df_corr.index[df_corr[col]].tolist()
            try:
                corr_col.remove(col)
            except:
                pass
            corr_dict[col] = corr_col

        return corr_dict

    def add_to_bad_idx(self, idx, col, col_type, VALUE_FLAG=True):
        if VALUE_FLAG:
            if type(col) == type(str()):
                row = {
                    "idx": idx,
                    "column": col,
                    "errtype": col_type,
                    "value1": self.data[col].loc[idx],
                    "value2": "",
                }
                self.bad_index = self.bad_index.append(row, ignore_index=True)
            else:
                row = {
                    "idx": idx,
                    "column": col,
                    "errtype": col_type,
                    "value1": self.data[col[0]].loc[idx],
                    "value2": self.data[col[1]].loc[idx],
                }
                self.bad_index = self.bad_index.append(row, ignore_index=True)
        else:
            if type(idx) == type(int()):
                row = {
                    "idx": idx,
                    "column": col,
                    "errtype": col_type,
                    "value1": "",
                    "value2": "",
                }
            else:
                row = {
                    "idx": idx[0],
                    "column": col,
                    "errtype": col_type,
                    "value1": idx[0],
                    "value2": idx[1],
                }
            self.bad_index = self.bad_index.append(row, ignore_index=True)

    def bad_float_index(
        self, thres_unique_str=0.0002, thres_unique_nbr=0.04, thresh_std=7
    ):
        """Same idea as in bad_logical_index function, execept it's between columns
        of string and columns of floats, using quantiles to determine interclasses extremes.
        Args :
            function_name : quantiles or z-scores.
        Returns:
        [idxes, col_names]: [list of list of bad indexes and associated columns names]
        """
        df = self.data.iloc[self.good_index]
        col_names = []
        idxes = []
        list_vue = []
        for col1 in self.str_col:
            if self.uniq_col[col1] < thres_unique_str:
                for col2 in self.nbr_col:
                    if (col1, col2) not in list_vue:
                        if self.uniq_col[col1] < 0.9 and self.uniq_col[col2] < 0.9:
                            if self.uniq_col[col2] > thres_unique_nbr:
                                elements = df.loc[df[[col1, col2]].dropna().index]
                                for elem in elements[col1].unique():
                                    ar_classe = elements[col2][elements[col1] == elem]
                                    if len(ar_classe) > 0:
                                        indx_outliers = utils._z_score(
                                            ar_classe, 0.2, thresh_std
                                        )  # 0,2 value is here to keep consistency when calling z_score and is for uniq_col
                                        if len(indx_outliers) > 0:
                                            val_outliers = ar_classe.loc[indx_outliers]
                                            idxes.append(
                                                elements[col2][
                                                    elements[col2].isin(val_outliers)
                                                ].index
                                            )
                                            col_names.append([col1, col2])
                            list_vue.append((col1, col2))

        return idxes, col_names

    def save_result(self, path, **kwargs):
        self.bad_index.to_csv(path, **kwargs)


#! please use our commun directory
"""

data = Data("..\data\data_avec_erreurs_wasserstein.csv")
data.firstpass('completeness')
#data.secondpass('mixed_logic')
data.bad_index.to_csv("exemple.csv")"""

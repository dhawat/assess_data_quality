import utils as utils
import ipdb
import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.neighbors import LocalOutlierFactor


class Data:
    """Data class holding a column by column profile and index flagged as low quality data"""

    def __init__(self, path):
        """
        Args:
            data (CSV, JSON, SQL): data set.
        """
        if utils.check_extension(path) == "none":
            raise TypeError("data should be of provided as .csv or .json or .sql file")

        self.data = utils._to_DataFrame(path)
        self._good_index = [range(self.data.shape[0])]
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
                col_type = check_data_type(column)
                if  col_type == type(str()):
                    _str_col.append(column)
                elif col_type in [type(float()), type(int())]:
                    _nbr_col.append(column)
        return _str_col

    @setter._str_col
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
                col_type = check_data_type(column)
                if  col_type == type(str()):
                    _str_col.append(column)
                elif col_type in [type(float()), type(int())]:
                    _nbr_col.append(column)
        return _nbr_col

    @setter._nbr_col
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
        for column in self.get_str_col():
            if (
                self.profile[column]._uniqueness <= 0.005
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

        for column in self.get_nbr_col():  # Columns of numbers only
            clean_df = self.data[n_duped_idx]
            clean_df = clean_df[column][clean_df[column].notna().values]
            idx = utils.proba_model(
                clean_df, self.profile[column]._mean, self.profile[column]._std
            )
            idx = clean_df[idx].index

            for index in idx:
                row = {"idx": index, "column": column, "errtype": "extreme value"}
                self.bad_index = self.bad_index.append(row, ignore_index=True)

        # Completeness pass on each row
        idx = utils._row_is_none(self.data)
        for index in idx:
            row = {"idx": index, "column": "All", "errtype": "too much nan"}
            self.bad_index = self.bad_index.append(row, ignore_index=True)

    def imputation_method(self, **params):
        params.setdefault("n_neighbors", 10)
        params.setdefault("weights", "uniform")
        self.data, _ = utils._is_duplicated(self.data)
        list_numeric_col_name = self.get_nbr_col()  # name of numerical column
        numeric_df = self.data[list_numeric_col_name]  # numeric dataframe
        numeric_df = numeric_df.fillna(np.nan)  # fill none with np.nan
        imputer = KNNImputer(**params)  # initialize imputation
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
        params.setdefault("contamination", 0.001)
        params.setdefault("metric", "correlation")
        params.setdefault("n_jobs", -1)

        # drop unique column
        df_drop_col0 = self.drop(
            labels=self.data.columns[0], axis=1
        )  # drop unique column

        # imputation
        df_drop_col0 = imputation_method(self)

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



# data = Data('..\data_avec_erreurs_wasserstein.csv')
# data.set_profile()
# data.firstpass()
# data.bad_index.to_csv('exemple.csv')

import utils as utils


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
        self._profile = None
        self._good_index = [range(self.data.shape[0])]
        self._bad_index = []

    @property
    def profile(self):
        if self._profile is None:
            raise Exception("profile is None")
        return self._profile

    def set_profile(self):
        """ """
        profile = {column: Profile(self, column) for column in self.data.columns}
        self._profile = profile

    @property
    def good_index(self):
        return self._good_index

    @property
    def bad_index(self):
        return self._bad_index

    @bad_index.setter
    def bad_index(self, list_idx):
        self._bad_index = list_idx

    @good_index.setter
    def good_index(self, list_idx):
        if len(list_idx) > self.data.shape[0]:
            raise ValueError(
                "Index length must be smaller than the length of the dataframe"
            )
        self._good_index = list_idx

    def get_str_col(self):
        col_list = []
        for column in self.data.columns:
            if self.profile[column]._col_type == type(str()):
                col_list.append(column)
        return col_list

    def get_nbr_col(self):
        col_list = []
        for column in self.data.columns:
            if self.profile[column]._col_type in [type(int()), type(float())]:
                col_list.append(column)
        return col_list

    def push_bad_index(self, list_idx):  # Find a better method name
        for elem in list_idx:
            try:
                self.bad_index.append(elem)
            except:
                pass

    def firstpass(self):
        # Deterministic pass
        n_duped_idx = self.data[~good_index_is_duplicated(self.data)[1]].index

        # Probabilistic pass
        for column in get_str_col():
            idx = index_uncorrect_grammar(
                self.data[column][n_duped_idx][self.data[column][n_duped_idx].notna()]
            )  # get the non duped indexes and not na from a column
            idx = (
                self.data[column][n_duped_idx][self.data[column][n_duped_idx].notna()]
                .iloc[idx]
                .index
            )
            self.bad_index(idx)
        for column in get_nbr_col():
            idx = utils.proba_model(
                self.data[column][n_duped_idx][self.data[column][n_duped_idx].notna()],
                self.profile[column]._mean,
                self.profile[column]._std,
            )
            idx = (
                self.data[column][n_duped_idx][self.data[column][n_duped_idx].notna()]
                .iloc[idx]
                .index
            )
            self.bad_index(idx)


class Profile:
    """A profile for a dataframe column."""

    def __init__(self, Data, column):
        self._emptiness = utils._is_none(Data.data, column)
        self._size = Data.data[column].shape[0]
        self._uniqueness = utils._is_unique(Data.data, column)
        self._col_type = utils.check_data_type(Data.data[column])
        if self._col_type == type(str()):
            pass
        if self._col_type in [type(int()), type(float())]:
            self._min = Data.data[column].min()
            self._max = Data.data[column].max()
            self._mean = Data.data[column].mean()
            self._std = Data.data[column].std()

    @property
    def emptiness(self):
        return self._emptiness

    @emptiness.setter
    def emptiness(self, value):
        self._emptiness = value

    @property
    def size(self):
        return self._size

    @size.setter
    def size(self, value):
        self._size = value

    @property
    def uniqueness(self):
        return self._uniqueness

    @uniqueness.setter
    def uniqueness(self, value):
        self._uniqueness = value

    @property
    def col_type(self):
        return self._col_type

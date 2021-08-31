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
    self._good_index = [range(data.shape[0])]
    self._bad_index = []

   @property
    def profile(self):
        if self._profile is None:
            raise Exception('profile is None')
        return self._profile
    
    def set_profile(self, df_results):
         profile = Profiler(self.data)
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
            raise ValueError('Index length must be smaller than the length of the dataframe')
        self._good_index = list_idx 
    
    def push_bad_index(self, list_idx): #Find a better method name
        for elem in list_idx:
            try:
                self.bad_index.append(elem)
                self.good_index.remove(elem)
            except:
                pass
            

class Profile:
    """A profile for a dataframe column.
    """

    def __init__(self, Data, column):
        self._completeness = 
        self._size = 
        self._uniqueness =
        self._col_type = utils.check_data_types(column)
        if _col_type == type(str()):
            pass
        if _col_type == type(int()) or == type(float()):
            self._min = Data[column].min()
            self._max = Data[column].max()
            self._mean = Data[column].mean()
            self._std = Data[column].std()
        self

        
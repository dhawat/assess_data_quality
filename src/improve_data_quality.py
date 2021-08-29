import utils as utils


class ImproveDataQuality:
    """Implementation of various method for improving data quality of a given data"""


def __init__(self, data):
    """
    Args:
        data (CSV, JSON, SQL): data set.
    """
    if utils.check_extension(data) == "none":
        raise TypeError("data should be of provided as .csv or .json or .sql file")

    self.data = utils._to_DataFrame(data)

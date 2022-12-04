import os
import sys
sys.path.insert(0, os.path.abspath('../src/'))
import improve_data_quality as idq #Import our package





# To start the user specify the path of the file, extensions can be csv, xlsx, sql ...
data = idq.Data("data.csv")

# Once loaded we will try to compute the first pass of the data without any parameters
data.firstpass() #Since no arguments were given, the method default to all the algorithms in the first pass
                 # i.e Typographical checking, duplicate checks, extreme values check and row completeness check.

# if the user wishes to only compute a single metric and the associated outliers she/he can pass as argument the types
# of error he wants to check.
data.secondpass()

# For a more advanced user she/he can specify which takes to undergo and specify the parameters

data.check_extreme_value(thresh_std=4.5, thresh_unique1=0.99)

# Finaly the user can export the bad index to a csv file by specifying the path
data.save_result('test_output.csv')

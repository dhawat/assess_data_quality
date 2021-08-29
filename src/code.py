import pandas as pd

df = pd.read_csv("logs.csv")  # read data

df = df.set_index("d")  # to re-index with a column 'd'
df = df.sort_index()  # to sort with respect to the index

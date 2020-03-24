import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection
from sklearn.cluster import MeanShift

style.use('ggplot')

df = pd.read_excel('dataset/titanic.xls')
original_df = df.copy()
df.drop(['body', 'name', 'ticket', 'boat'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())

def handle_non_numeric_data(df):
  columns = df.columns.values

  # iterate on all columns of data frame
  for column in columns:
    text_digit_vals = {} # new dict

    # utility method to convert non-numeric value to numeric value
    def convert_to_int(val):
      return text_digit_vals[val]

    # check if column type is not numeric
    if df[column].dtype != np.int64 and df[column].dtype != np.float64:
      column_contents = df[column].values.tolist() # collect all values
      unique_elements = set(column_contents) # make them unique
      x = 0
      for unique in unique_elements:
        if unique not in text_digit_vals:
          text_digit_vals[unique] = x
          x += 1
      
      df[column] = list(map(convert_to_int, df[column]))
    
  return df

df = handle_non_numeric_data(df)

x = np.array(df.drop(['survived'], 1).astype(float))
x = preprocessing.scale(x)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(x, y)

labels = clf.labels_
cluster_centres = clf.cluster_centers_

original_df['cluster_group'] = np.nan

print(len(original_df))
print(len(labels))

for i in range(len(original_df)):
  original_df['cluster_group'].iloc[i] = labels[i]

n_clsuters_ = len(np.unique(labels))

survival_rates = {}
for i in range(n_clsuters_):
  temp_df = original_df[original_df['cluster_group'] == float(i)]
  survival_cluster = temp_df[temp_df['survived'] == 1]
  survival_rate = len(survival_cluster)/len(temp_df)
  survival_rates[i] = survival_rate

print(survival_rates)

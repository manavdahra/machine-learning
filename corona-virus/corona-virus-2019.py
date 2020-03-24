import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import datetime

style.use('ggplot')

cases_df = pd.read_csv(
    'dataset/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
cases_df.drop(cases_df.columns[[1, 3, 21, 22, 23, 24, 25, 26]], 1, inplace=True)
cases_df.drop(['id', 'summary', 'If_onset_approximated', 'source', 'link', 'symptom', 'symptom_onset', 'hosp_visit_date', 'exposure_start', 'exposure_end'], 1, inplace=True)
cases_df['gender'].replace({'female': 0, 'male': 1}, inplace=True)

loc_map = {}
country_map = {}

for index, row in cases_df.iterrows():
  location = cases_df.iloc[index]['location']
  country = cases_df.iloc[index]['country']
  if location not in loc_map.keys():
    loc_map[location] = len(loc_map) + 1
  if country not in country_map.keys():
    country_map[country] = len(country_map) + 1

for index, row in cases_df.iterrows():
  try:
    timestamp = datetime.datetime.strptime(cases_df.iloc[index]['reporting date'], "%m/%d/%Y").timestamp()
    cases_df.at[index, 'reporting date'] = timestamp
  except:
    try:
      timestamp = datetime.datetime.strptime(cases_df.iloc[index]['reporting date'], "%m/%d/%y").timestamp()
      cases_df.at[index, 'reporting date'] = timestamp
    except:
      cases_df.at[index, 'reporting date'] = 0
  cases_df.at[index, 'location'] = loc_map.get(cases_df.iloc[index]['location'], 0)
  cases_df.at[index, 'country'] = country_map.get(cases_df.iloc[index]['country'], 0)
  if cases_df.iloc[index]['death'] != '0':
    cases_df.at[index, 'state'] = 2
  elif cases_df.iloc[index]['recovered'] != '0':
    cases_df.at[index, 'state'] = 0
  else:
    cases_df.at[index, 'state'] = 1

cases_df.drop(['death', 'recovered', 'reporting date'], 1, inplace=True)
cases_df.fillna(cases_df.mean(), inplace=True)
print(cases_df.head())

y = np.array(cases_df['country'])
x = np.array(cases_df['state'])
plt.scatter(x, y, color='r')
plt.show()
# y = np.array(cases_df['state']).astype(int)
# x = np.array(cases_df)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(
#     x, y, test_size=0.2)
# clf = neighbors.KNeighborsClassifier()
# clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print('Accuracy:', accuracy)

import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn import preprocessing, model_selection, svm

style.use('ggplot')

df = pd.read_excel('dataset/titanic.xls')
df.drop([
  'name',
  'home.dest',
  'ticket',
  'body',
  'sibsp',
], 1, inplace=True)
df['boat'].fillna(0, inplace=True)
df['embarked'].fillna('NA', inplace=True)
df['sex'].replace({'female': 0, 'male': 1}, inplace=True)
df['embarked'].replace({'S': 1, 'C': 2, 'Q': 3, 'NA': 0}, inplace=True)

cabin_map = {}
boat_map = {}

for index, row in df.iterrows():
  cabin = df.iloc[index]['cabin']
  boat = df.iloc[index]['boat']
  if cabin not in cabin_map.keys():
    cabin_map[cabin] = len(cabin_map) + 1
  if boat not in boat_map.keys():
    boat_map[boat] = len(boat_map) + 1

for index, row in df.iterrows():
  df.at[index, 'cabin'] = cabin_map.get(df.iloc[index]['cabin'], 0)
  df.at[index, 'boat'] = boat_map.get(df.iloc[index]['boat'], 0)

df.fillna(df.mean(), inplace=True)

print(df.describe())

y = np.array(df['survived'])
x = np.array(df.drop(['survived'], 1))

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2)
clf = svm.SVC(kernel='linear') #neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print('Accuracy:', accuracy)

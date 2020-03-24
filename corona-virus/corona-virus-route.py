import datetime

import pandas as pd

df = pd.read_csv('dataset/coronavirusdataset/route.csv')

for index, row in df.iterrows():
    try:
        timestamp = datetime.datetime.strptime(df.iloc[index]['date'], "%Y-%m-%d").timestamp()
        df.at[index, 'date'] = timestamp
    except ValueError as e:
        print(e)

cols = ['lat', 'long']
geo_df = pd.DataFrame(columns=cols)
for lat in range(360):
    for long in range(180):
        geo_df = geo_df.append(pd.DataFrame([[lat, long]]))

print(geo_df.head())

# x = np.array(df['date'])
# x = x.reshape(1, -1)
# y = np.array(df[['latitude', 'longitude']])

# print(x)
# print(y)

# x_train, x_test, y_train, y_test = model_selection.train_test_split(
#     x, y, test_size=0.2)
# clf = svm.SVC(kernel='poly')
# clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
# print('Accuracy:', accuracy)

# plt.scatter(x, y)
# plt.show()

import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors

df = pd.read_csv('dataset/creditcard.csv')
x = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
# accuracy = clf.score(x_test, y_test)
predictions = clf.predict(x_test)
print(predictions)

# print('accuracy: ', predictions)

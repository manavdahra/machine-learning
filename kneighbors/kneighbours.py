import numpy as np
import pandas as pd
from sklearn import model_selection, neighbors

df = pd.read_csv('../dataset/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

x = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

accuracies = []
for i in range(1):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)

    # with open('dump/breast_cancer_trained_clf', 'wb') as f:
    #   pickle.dump(clf, f)

    # with open('dump/breast_cancer_trained_clf', 'rb') as F:
    #   clf = pickle.load(F)

    accuracy = clf.score(x_test, y_test)
    print('Accuracy:', accuracy)
    accuracies.append(accuracy)

    # example_measures = np.array([
    #   [4,2,1,1,2,1,3,1,1],
    #   [4,2,2,1,2,2,3,1,1]
    # ])
    # example_measures = example_measures.reshape(len(example_measures), -1)

    # prediction = clf.predict(example_measures)
    # print(prediction)

# print(sum(accuracies)/len(accuracies))

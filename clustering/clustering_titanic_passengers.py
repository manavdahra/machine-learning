import numpy as np
import pandas as pd
from matplotlib import style
from sklearn import preprocessing
from sklearn.cluster import KMeans

style.use('ggplot')

df = pd.read_excel('dataset/titanic.xls')
df.drop(['body', 'name', 'ticket', 'boat'], 1, inplace=True)
# df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)
print(df.head())


def handle_non_numeric_data(df):
    columns = df.columns.values

    # iterate on all columns of data frame
    for column in columns:
        text_digit_vals = {}  # new dict

        # utility method to convert non-numeric value to numeric value
        def convert_to_int(val):
            return text_digit_vals[val]

        # check if column type is not numeric
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()  # collect all values
            unique_elements = set(column_contents)  # make them unique
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

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0
for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    print(predict_me)
    predict_me = predict_me.reshape(-1, len(predict_me))
    print(predict_me)
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct / len(x))

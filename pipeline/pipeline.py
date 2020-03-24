import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error
import datetime

style.use('ggplot')

cases_df = pd.read_csv(
    'dataset/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv')
cases_df.fillna(cases_df.mean(), inplace=True)

X_train, x_test, y_train, y_test = model_selection.train_test_split(
    cases_df, cases_df['gender'], test_size=0.2)

categorical_cols = [cname for cname in cases_df.columns if
                    cases_df[cname].nunique() < 10 and 
                    cases_df[cname].dtype == "object"]

numerical_cols = [cname for cname in cases_df.columns if 
                cases_df[cname].dtype in ['int64', 'float64']]

my_cols = categorical_cols + numerical_cols
X_train = cases_df[my_cols].copy()

numerical_xmer = SimpleImputer(strategy='constant')

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
  transformers=[
    ('num', numerical_xmer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
  ]
)

model = neighbors.KNeighborsClassifier()

clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

print(X_train.head())
clf.fit(X_train, y_train)

preds = clf.predict(x_test)

print('MAE:', mean_absolute_error(y_test, preds))

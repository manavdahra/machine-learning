import pandas as pd
from sklearn import model_selection
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

train_df = pd.read_csv('dataset/housing/train.csv')
test_data = pd.read_csv('dataset/housing/test.csv')

numerical_cols = train_df.select_dtypes(include='number')
categorical_cols = train_df.select_dtypes(include='object')

# Pre-processing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Pre-processing for categorical data
# categorical_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='most_frequent')),
#     ('onehot', OneHotEncoder(handle_unknown='ignore'))
# ])

X = train_df.drop(['SalePrice'], 1)
y = train_df['SalePrice']

for col in numerical_cols:
    if col == 'SalePrice':
        continue
    test_data[col] = numerical_transformer.fit_transform(test_data[[col]])
    X[col] = numerical_transformer.fit_transform(X[[col]])

label_encoder = LabelEncoder()
for col in categorical_cols:
    test_data[col] = test_data[col].fillna('N/A')
    test_data[col] = label_encoder.fit_transform(test_data[col])
    X[col] = X[col].fillna('N/A')
    X[col] = label_encoder.fit_transform(X[col])

# Bundle preprocessing for numerical and categorical data
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numerical_transformer, numerical_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])

print(X.head())
print(test_data.head())

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
model = XGBRegressor(n_estimators=500, learning_rate=0.05, n_jobs=4)
model.fit(X_train, y_train, early_stopping_rounds=5,
          eval_set=[(X_test, y_test)],
          verbose=False)

# Bundle preprocessing and modeling code in a pipeline
# pipeline = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('model', model)
# ])

# pipeline.fit(X_train, y_train)

# predictions = pipeline.predict(X_test)
predictions = model.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_test)))

test_preds = model.predict(test_data)
# print("Mean Absolute Error: " + str(mean_absolute_error(test_preds, y_test)))

output = pd.DataFrame({'Id': test_data.Id,
                       'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)

# print(df.describe())

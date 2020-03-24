import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn import preprocessing, model_selection, neighbors
from datetime import date

style.use('ggplot')

case_df = pd.read_csv('dataset/coronavirusdataset/case.csv')
patient_df = pd.read_csv('dataset/coronavirusdataset/patient.csv')
patient_df['region'].replace({
    'filtered at airport': 0,
    'capital area': 1,
    'Jeollabuk-do': 2,
    'Gwangju': 3,
    'Daegu': 4,
    'Gyeongsangbuk-do': 5,
    'Jeju-do': 6,
    'Busan': 7,
    'Daejeon': 8,
    'Chungcheongbuk-do': 9,
    'Chungcheongnam-do': 10,
    'Ulsan': 11,
    'Gangwon-do': 12,
    'Jeollanam-do': 13
}, inplace=True)
patient_df['sex'].replace({'female': 0, 'male': 1}, inplace=True)
patient_df = patient_df[patient_df['state'] != 'isolated']
patient_df['state'].replace(
    {
        'released': 0,
        'isolated': 1,
        'deceased': 2
    },
    inplace=True)
patient_df['country'].replace(
    {'China': 0, 'Korea': 1, 'Mongolia': 2}, inplace=True)

patient_df.drop(['patient_id', 'group', 'disease', 'contact_number', 'infection_reason',
                 'released_date', 'confirmed_date', 'deceased_date', 'infected_by'], 1, inplace=True)

for index in range(len(patient_df)):
    patient_df.at[index, 'birth_year'] = date.today(
    ).year - patient_df.iloc[index]['birth_year']

patient_df.fillna(patient_df.mean(), inplace=True)
y = np.array(patient_df['state']).astype(int)
x = np.array(patient_df)

x_train, x_test, y_train, y_test = model_selection.train_test_split(
    x, y, test_size=0.2)
clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print('Accuracy:', accuracy)

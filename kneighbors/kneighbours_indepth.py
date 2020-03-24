from math import sqrt
import numpy as np
import pandas as pd
import warnings
from collections import Counter
import random

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


def kNearestNeighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k cannot be less than total groups/neighbors')
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_dist = np.linalg.norm(
                np.array(features) - np.array(predict))
            distances.append([euclidean_dist, group])
    votes = [i[1] for i in sorted(distances)[:k]]
    result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return result, confidence

result = kNearestNeighbors(dataset, new_features, k=3)

df = pd.read_csv('../dataset/breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)
full_data = df.astype(float).values.tolist()

accuracies = []
for i in range(5):
  random.shuffle(full_data)

  test_size = 0.2
  train_set = {2: [], 4: []}
  test_set = {2: [], 4: []}

  limit = -int(test_size*len(full_data))

  train_data = full_data[:limit]
  test_data = full_data[limit:]

  for i in train_data:
    train_set[i[-1]].append(i[:-1])

  for i in test_data:
    test_set[i[-1]].append(i[:-1])

  correct = 0
  total = 0

  for group in test_set:
    for data in test_set[group]:
      vote, confidence = kNearestNeighbors(train_set, data, 5)
      if group == vote:
        correct += 1
      total += 1
  
  accuracies.append(correct/total)
  print('Accuracy: ', correct/total)

print(sum(accuracies)/len(accuracies))

import numpy as np

X = np.array([
  [1, 0, 0],
  [1, 0, 1],
  [1, 1, 0],
  [0, 0, 1],
  [0, 1, 0],
  [0, 1, 1],
])

Y = np.array([
  [1],
  [1],
  [1],
  [0],
  [0],
  [0],
])

np.random.seed(1)
W = 2 * np.random.random((3, 1)) - 1

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
  return x * (1 - x)

for _ in range(1000000):
  outputs = sigmoid(np.dot(X, W))
  error = Y - outputs
  adjustments = error * sigmoid_derivative(outputs)
  W += np.dot(X.T, adjustments)

print(outputs)

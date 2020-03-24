import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use('fivethirtyeight')


# xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

def createDataSet(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    xs = [i for i in range(len(ys))]

    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def sqError(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)


def coefficientOfDetermination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    sqErrorRegression = sqError(ys_orig, ys_line)
    sqErrorMean = sqError(ys_orig, y_mean_line)
    return 1 - (sqErrorRegression / sqErrorMean)


def bestFitSlope(xs, ys):
    return ((mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs)))


def bestFitIntercept(xs, ys, bestFitSlope):
    return mean(ys) - bestFitSlope * mean(xs)


def bestFitSlopeAndIntercept(xs, ys):
    m = bestFitSlope(xs, ys)
    b = bestFitIntercept(xs, ys, m)
    return m, b


xs, ys = createDataSet(40, 40, 2, False)

m, b = bestFitSlopeAndIntercept(xs, ys)
print(m, b)

regression_line = [(m * x + b) for x in xs]

perdict_x = 8
predict_y = m * perdict_x + b

rSquared = coefficientOfDetermination(ys, regression_line)
print(rSquared)

plt.scatter(xs, ys)
plt.scatter(perdict_x, predict_y, s=100, color='g')
plt.plot(xs, regression_line)
plt.show()

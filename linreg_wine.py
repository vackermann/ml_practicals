import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt

X, y = cp.load(open('winequality-white.pickle', 'rb')) # rb stands for read binary

N, D = X.shape
N_train = int(0.8 * N)
N_test = N - N_train

X_train = X[:N_train]
y_train = y[:N_train]
X_test = X[N_train:]
y_test = y[N_train:]

# HANDIN 1
def plotYDistribution():
    unique, counts = np.unique(y_train, return_counts=True)
    frequency = np.array(list(zip(unique, counts)))

    x_axis = np.arange(1, len(frequency)+1)
    y_freq = [num for (x, num) in frequency]
    labels = [num for (num, y) in frequency]

    width = 0.5
    bar1 = plt.bar(x_axis, y_freq, width, color="y")
    plt.ylabel("Quality")
    plt.xticks(x_axis + width/20.0, labels)
    plt.show()

# plotYDistribution()

# Handin 2
def zero_R():
    mean = np.mean(y_train)


def getSetMSEofZeroR(y_set):
    mean = np.mean(y_train)
    sum_mse = 0
    for y_val in y_set:
        sum_mse += getMSE(y_val, mean)
    return sum_mse / len(y_set)

def getMSE(x, y):
    return (x-y)*(x-y)

print(getSetMSEofZeroR(y_train))
print(getSetMSEofZeroR(y_test))

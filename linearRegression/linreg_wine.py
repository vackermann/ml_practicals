import _pickle as cp
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model
from sklearn.metrics import mean_squared_error

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
    plt.xlabel("Wine Quality (3-9)")
    plt.xticks(x_axis + width/20.0, labels)
    plt.title("Distribution of wine quality (= y values) in White wine data set.")
    plt.ylabel("Number of data points")
    plt.show()

plotYDistribution()

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

print('MSE of ZeroR on trainig set: %.4f' % getSetMSEofZeroR(y_train))
print('MSE of ZeroR on trainig set: %.4f' % getSetMSEofZeroR(y_test))


# HANDIN 3

# Scale training set to have mean=0 and std=1, scale testset with same values

scaler = preprocessing.StandardScaler().fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)

# Train linear classifier

regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)
y_pred_train = regr.predict(X_train)
y_pred_test = regr.predict(X_test)

# Get MSE of LinReg

mse_linreg_train = mean_squared_error(y_train, y_pred_train)
mse_linreg_test = mean_squared_error(y_test, y_pred_test)
print('MSE of LinReg on trainig set: %.4f' % mse_linreg_train)
print('MSE of LinReg on test set: %.4f' % mse_linreg_test)

# HANDIN 4

print('Hello')
training_sizes = range(20, 600, 20)
MSE_training_list = []
MSE_test_list = []
for training_size in range(20, 600, 20):
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X_train[:training_size], y_train[:training_size])
    y_pred_train = lin_reg.predict(X_train[:training_size])
    y_pred_test = lin_reg.predict(X_test)
    MSE_train = mean_squared_error(y_train[:training_size], y_pred_train)
    MSE_test = mean_squared_error(y_test, y_pred_test)
    MSE_training_list.append(MSE_train)
    MSE_test_list.append(MSE_test)


# Plot learning curve

plt.plot(training_sizes, MSE_training_list,'--', label="Training error")
plt.plot(training_sizes, MSE_test_list, label="Test error")
plt.title("Learning Curve for LinearRegression")
plt.xlabel("Training Set Size"), plt.ylabel("Mean Squared Error"), plt.legend(loc="best")
plt.tight_layout()
plt.show()

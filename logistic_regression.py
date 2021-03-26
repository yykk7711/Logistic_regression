import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_x_file = "iris_X_train.csv"
train_x = pd.read_csv("data/" + train_x_file)

train_y_file = "iris_y_train.csv"
train_y = pd.read_csv("data/" + train_y_file)

test_x_file = "iris_X_test.csv"
test_x = pd.read_csv("data/" + test_x_file)

test_y_file = "iris_y_test.csv"
test_y = pd.read_csv("data/" + test_y_file)

# Unique class labels
uniqueLabels = np.unique(train_y)

"""
sc = StandardScaler()
sc.fit(train_x)

# feature scaling
train_x_std = sc.transform(train_x)
print(train_x_std)
"""
class LogisticRegression(object):
    def __init__(self, eta = 0.05, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        # initialize the weight
        rgen = np.random.RandomState(self.random_state)
        # shape of (5,) where 1 + 4 features
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size=1 + X.shape[1])

        # cost
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input) #.reshape(100,1)
            Y_array = np.array(Y['Species'].to_list())

            errors = Y_array - output
            self.w_[1:] += self.eta * X.T.dot(errors)  # shape of (4,1)
            self.w_[0] += self.eta * errors.sum()
            # update cost
            cost = (-Y_array.dot(np.log(output)) - ((1 - Y_array).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self,X):
        return np.dot(X, self.w_[1:]) + self.w_[0]
    def activation(self, input):
        return 1. / (1. +np.exp(-np.clip(input, -250, 250)))
    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

# if the species is 0, set to 1, else 0
train_y_0 = train_y
for i in range(train_y_0.shape[0]):
    if train_y.loc[i][0] == 0:
        train_y_0.loc[i][0] = 1
    else:
        train_y_0.loc[i][0] = 0

test_y_0 = test_y
for i in range(test_y_0.shape[0]):
    if test_y.loc[i][0] == 0:
        test_y_0.loc[i][0] = 1
    else:
        test_y_0.loc[i][0] = 0

lr = LogisticRegression()
lr.fit(train_x, train_y_0)
print(lr.predict(test_x))

print(np.array(test_y_0['Species'].to_list()))
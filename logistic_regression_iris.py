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

sc = StandardScaler()
sc.fit(train_x)

# feature scaling
train_x = sc.transform(train_x)

sc = StandardScaler()
sc.fit(test_x)

# feature scaling
test_x = sc.transform(test_x)

class LogisticRegression(object):
    def __init__(self, eta = 0.1, n_iter = 1000, random_state = 1): # best performance: eta = 0.01, n_iter = 1000, random_state = 1
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

        Y_array = np.array(Y['Species'].to_list())
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input) #.reshape(100,1)
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

    def final_net_input(self, X):
        final_X = self.net_input(X)
        return(final_X)

# if the species is 0, set to 1, else 0

train_y_0 = train_y.copy()
train_y_1 = train_y.copy()
train_y_2 = train_y.copy()

for i in range(train_y.shape[0]):
    if train_y.loc[i][0] == 0:
        train_y_0.loc[i][0] = 1
    else:
        train_y_0.loc[i][0] = 0

for i in range(train_y.shape[0]):
    if train_y.loc[i][0] == 1:
        train_y_1.loc[i][0] = 1
    else:
        train_y_1.loc[i][0] = 0

for i in range(train_y.shape[0]):
    if train_y.loc[i][0] == 2:
        train_y_2.loc[i][0] = 1
    else:
        train_y_2.loc[i][0] = 0

test_y_0 = test_y.copy()
test_y_1 = test_y.copy()
test_y_2 = test_y.copy()

for i in range(test_y.shape[0]):
    if test_y.loc[i][0] == 0:
        test_y_0.loc[i][0] = 1
    else:
        test_y_0.loc[i][0] = 0

for i in range(test_y.shape[0]):
    if test_y.loc[i][0] == 1:
        test_y_1.loc[i][0] = 1
    else:
        test_y_1.loc[i][0] = 0

for i in range(test_y.shape[0]):
    if test_y.loc[i][0] == 2:
        test_y_2.loc[i][0] = 1
    else:
        test_y_2.loc[i][0] = 0

final = []
lr0 = LogisticRegression()
lr0.fit(train_x, train_y_0)
NI0 = lr0.final_net_input(test_x)

lr1 = LogisticRegression()
lr1.fit(train_x, train_y_1)
NI1 = lr1.final_net_input(test_x)

lr2 = LogisticRegression()
lr2.fit(train_x, train_y_2)
NI2 = lr2.final_net_input(test_x)

for i in range(len(NI0)):
    # select max from the three
    if NI0[i] > NI1[i] and NI0[i] > NI2[i]:
        final.append(0)
    elif NI1[i] > NI2[i]:
        final.append(1)
    else:
        final.append(2)

# uncommand the following 2 lines to show the final prediction class and it true class
#print(np.array(final))
#print(np.array(test_y['Species'].to_list()))
total = np.array(final).size
correct = np.count_nonzero(np.array(final) == np.array(test_y['Species'].to_list()))
print("result: %d / %d"%(correct, total))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

train_x_file = "car_X_train.csv"
train_x = pd.read_csv("data/" + train_x_file)

train_y_file = "car_y_train.csv"
train_y = pd.read_csv("data/" + train_y_file)

test_x_file = "car_X_test.csv"
test_x = pd.read_csv("data/" + test_x_file)

test_y_file = "car_y_test.csv"
test_y = pd.read_csv("data/" + test_y_file)

reference = ['acc', 'good', 'unacc', 'vgood']

def replacing_index(data):
    feature = data.columns
    count = 0
    for features in list(feature):
        uniq_feature = list(np.unique(data[features]))
        uniq_feature.sort() # ensure the index are the same (also as the reference)
        for row in range(data.shape[0]):
            data.loc[row][list(feature)[count]] = uniq_feature.index((data.loc[row][list(feature)[count]]))
        count += 1

replacing_index(train_x)
replacing_index(train_y)
replacing_index(test_x)
replacing_index(test_y)

train_x = train_x.values.tolist()
train_y = train_y["class"].values.tolist()
test_x = test_x.values.tolist()
test_y = test_y["class"].values.tolist()

sc = StandardScaler()
sc.fit(train_x)

# feature scaling
train_x = sc.transform(train_x)

sc = StandardScaler()
sc.fit(test_x)

# feature scaling
test_x = sc.transform(test_x)


class LogisticRegression(object):
    def __init__(self, eta = 10, n_iter = 5, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, Y):
        X = np.array(X)
        # initialize the weight
        rgen = np.random.RandomState(self.random_state)
        # shape of (7,) where 1 + 6 features
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size=1 + X.shape[1])

        # cost
        self.cost_ = []
        Y_array = np.array(Y)
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
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

final_net_input = []
final = []
for n_classes in range(len(reference)):
    Y = train_y.copy()
    for n, i in enumerate(Y):
        if i == n_classes:
            Y[n] = 1
        else:
            Y[n] = 0
    lr = LogisticRegression()
    lr.fit(train_x, Y)
    final_net_input.append(lr.final_net_input(test_x))

for i in range(len(test_y)):
    max = -99999
    for _class in range(len(final_net_input)):
        if final_net_input[_class][i] > max:
            max = final_net_input[_class][i]
            c = _class
    final.append(c)

print(final)

# uncommand the following 2 lines to show the final prediction class and it true class
#print(np.array(final))
#print(np.array(test_y['Species'].to_list()))

#total = np.array(final).size
#correct = np.count_nonzero(np.array(final) == np.array(test_y['Species'].to_list()))
#print("result: %d / %d"%(correct, total))
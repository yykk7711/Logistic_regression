# Logistic_regression
To run the program, go to terminal, cd to where python file is, type following commands

python3 logistic_regression_iris.py

python3 logistic_regression_car.py

parameter can be change directly in the class of LogisticRegression(object)

-------------------------------------
class LogisticRegression(object)

parameters:

- eta, the learning rate of logistic regression
- n_iter, number of iteration in learning
- random state, a number to make the sudo-random number is the same to ensure each execution of logistic Regression are the same

Test on 2 datasets:
=

IRIS:
-
The features are scaled in order to compare. 

As there are 3 unique label in iris, therefore, 3 Logistic regression classes are created.

The correctness of the classification will be shown in terminal after execute the py file

Result:

eta = 0.1, n_iter = 1000: 49/50 / 98%

eta = 0.05, n_iter = 50: 44/50 / 88%

eta = 0.0000000000001, n_iter = 50: 41/50 / 82%

eta = 10, n_iter = 50: 38/50 / 76%
 
------------------------------------------------------
CAR:
-
Similar to Iris data, 4 logistic regression classes are created.

Result:

eta = 0.001, n_iter = 500 : 346/519 / 66%

eta = 0.01, n_iter = 500 : 320/519 / 61.66%

eta = 0.1, n_iter = 500 :  292/519 / 56.26%

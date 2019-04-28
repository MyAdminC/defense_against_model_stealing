# Created by jikangwang at 3/17/19

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from Model import LogisticRegressionUsingGD
from sklearn.metrics import accuracy_score

def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df

data = load_data("data/marks.txt", None)

parameters=[-2.85831439,0.05214733,0.04531467]

X = data.iloc[:, :-1]

X = np.c_[np.ones((X.shape[0], 1)), X]

x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]


print(y_values)
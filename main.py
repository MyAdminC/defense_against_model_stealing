# imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from Model import LogisticRegressionUsingGD
from sklearn.metrics import accuracy_score


def load_data(path, header):
    marks_df = pd.read_csv(path, header=header)
    return marks_df


if __name__ == "__main__":
    # load the data from the file
    data = load_data("data/marks.txt", None)

    # X = feature values, all the columns except the last column
    X = data.iloc[:, :-1]

    # y = target values, last column of the data frame
    y = data.iloc[:, -1]

    # filter out the applicants that got admitted
    admitted = data.loc[y == 1]

    # filter out the applicants that din't get admission
    not_admitted = data.loc[y == 0]

    # plots
    plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
    plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10,
                label='Not Admitted')

    # preparing the data for building the model

    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))

    # Logistic Regression from scratch using Gradient Descent
    model = LogisticRegressionUsingGD()
    model.fit(X, y, theta)
    accuracy = model.accuracy(X, y.flatten())
    parameters = model.w_
    print("The accuracy of the model is {}".format(accuracy))
    print("The model parameters using Gradient descent")
    print("\n")
    print(parameters)

    tot_accuracy=[]
    tot_diff=[]

    # plotting the decision boundary
    # As there are two features
    # wo + w1x1 + w2x2 = 0
    # x2 = - (wo + w1x1)/(w2)


    x_values = [np.min(X[:, 1] - 2), np.max(X[:, 2] + 2)]
    # _values=X
    y_values = - (parameters[0] + np.dot(parameters[1], x_values)) / parameters[2]



    plt.plot(x_values, y_values, label='Decision Boundary')

    a = X
    b = model.predict(a)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[1:4]
    b = b[1:4]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_diff.append(diff)
    print(x)

    tot_accuracy.append(accuracy)

    print("The accuracy of the model with equation attack no round is {}".format(accuracy))
    plt.plot(x_values, y_values, label='no rounding')



    a = X
    b = model.predict_with_round(a, 0)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation label only attack is {}".format(accuracy))
    plt.plot(x_values, y_values, label='labels only')


    a=X
    b=model.predict_with_round(a,1)
    b=model.inverse_sigmoid(b)
    #a=a[:, 1:len(a[0])]
    b=b.flatten()
    a=a[:3]
    b=b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff= model.accuracy_with_eqution(X, y.flatten(),x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation round 1 attack is {}".format(accuracy))
    plt.plot(x_values, y_values, label='1 decimals')






    a = X
    b = model.predict_with_round(a, 2)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack round 2 is {}".format(accuracy))
    plt.plot(x_values, y_values, label='2 decimals')

    a = X
    b = model.predict_with_round(a, 3)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack round 3 is {}".format(accuracy))
    plt.plot(x_values, y_values, label='3 decimals')

    a = X
    b = model.predict_with_round(a, 4)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack round 4 is {}".format(accuracy))
    plt.plot(x_values, y_values, label='4 decimals')

    a = X
    b = model.predict_with_round(a, 5)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack round 5 is {}".format(accuracy))
    plt.plot(x_values, y_values, label='5 decimals')

    a = X
    b = model.predict_with_dp(a)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack dp is {}".format(accuracy))
    plt.plot(x_values, y_values, label='laplace')

    a = X
    b = model.predict_with_de(a)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack de is {}".format(accuracy))
    plt.plot(x_values, y_values, label='twist')

    a = X
    b = model.predict_with_gauss(a)
    b = model.inverse_sigmoid(b)
    # a=a[:, 1:len(a[0])]
    b = b.flatten()
    a = a[:3]
    b = b[:3]
    x = np.linalg.solve(a[:3], b[:3])
    y_values = - (x[0] + np.dot(x[1], x_values)) / x[2]
    accuracy,diff = model.accuracy_with_eqution(X, y.flatten(), x)
    tot_accuracy.append(accuracy)
    tot_diff.append(diff)

    print("The accuracy of the model with equation attack gauss is {}".format(accuracy))
    plt.plot(x_values, y_values, label='gauss')



    plt.xlabel('Marks in 1st Exam')
    plt.ylabel('Marks in 2nd Exam')
    plt.legend()
    plt.show()


    plt.figure()

    men_means = tot_accuracy

    ind = np.arange(len(men_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width,
                    color='SkyBlue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('1-Rtest')
    ax.set_xticks(ind)
    ax.set_xticklabels(('No R', 'Label Only', '1 D', '2 D', '3 D', '4 D', '5 D', 'Laplace', 'Twist',
                        'gauss'))
    ax.legend()

    plt.figure()

    men_means = tot_diff

    ind = np.arange(len(men_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, men_means, width,
                    color='SkyBlue')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Average Distance')
    ax.set_xticks(ind)
    ax.set_xticklabels(('No R', 'Label Only', '1 D', '2 D', '3 D', '4 D', '5 D', 'Laplace', 'Twist',
                        'gauss'))
    ax.legend()


    plt.show()

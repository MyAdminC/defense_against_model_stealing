# Created by jikangwang at 4/25/1
import numpy as np
from scipy.optimize import fmin_tnc
import random


class LogisticRegressionUsingGD:

    tot=0

    def noisyCount(self, sensitivety, epsilon):
        beta = sensitivety / epsilon
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        # print(n_value)
        return n_value/100

    def laplace_mech(self, data, sensitivety, epsilon):
        for i in range(len(data)):
            data[i] += self.noisyCount(sensitivety, epsilon)
        return data


    @staticmethod
    def sigmoid(x):
        # Activation function used to map any real value between 0 and 1
        # print(x)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def inverse_sigmoid(x):
        return np.log(x / (1 - x))


    @staticmethod
    def net_input(theta, x):
        # Computes the weighted sum of inputs Similar to Linear Regression

        return np.dot(x, theta)

    @staticmethod
    def eps_round(x,v, epsilon=0.999999999):
        return np.round(x / epsilon,v) * epsilon


    def probability(self, theta, x):
        # Calculates the probability that an instance belongs to a particular class
        return self.sigmoid(self.net_input(theta, x))

    def cost_function(self, theta, x, y):
        # Computes the cost function for all the training samples
        m = x.shape[0]
        total_cost = -(1 / m) * np.sum(
            y * np.log(self.probability(theta, x)) + (1 - y) * np.log(
                1 - self.probability(theta, x)))
        return total_cost

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, self.sigmoid(self.net_input(theta, x)) - y)

    def fit(self, x, y, theta):
        opt_weights = fmin_tnc(func=self.cost_function, x0=theta, fprime=self.gradient,
                               args=(x, y.flatten()))
        self.w_ = opt_weights[0]
        return self

    def predict(self, x):
        theta = self.w_[:, np.newaxis]
        return self.probability(theta, x)

    def predict_with_gauss(self, x):
        theta = self.w_[:, np.newaxis]
        return np.clip(random.gauss(0,0.001)+self.probability(theta, x), 0.0000001, 0.99999999)

    def predict_with_de(self, x):
        sensitivety = 1
        epsilon = 1
        theta = self.w_[:, np.newaxis]
        a=np.clip(self.probability(theta, x), 0.0000001, 0.99999999)
        a1=abs(a[1]-0.9999999)
        a2=abs(a[1]-0.0000001)

        if(a1>a2):
            a[1]=0.99999999
        else:
            a[1]=0.00000001
        a[1]=np.clip(a[0]-1, 0.0000001, 0.99999999)
        return a
        #return np.clip(self.laplace_mech(self.probability(theta, x), sensitivety, epsilon), 0.0000001, 0.99999999)

    def predict_with_dp(self, x):
        sensitivety = 1
        epsilon = 1
        theta = self.w_[:, np.newaxis]
        return np.clip(self.laplace_mech(self.probability(theta, x), sensitivety, epsilon), 0.0000001, 0.99999999)

    def predict_with_round(self, x,r):
        theta = self.w_[:, np.newaxis]
        return np.clip(np.round(self.probability(theta, x), r), 0.0000001, 0.99999999)

    def accuracy(self, x, actual_classes, probab_threshold=0.5):
        predicted_classes = (self.predict(x) >= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)
        return accuracy * 100

    def accuracy_with_eqution(self, x, actual_classes, para, probab_threshold=0.5):
        para=para.reshape(-1,1)
        p1=self.predict(x)
        p2=self.sigmoid(self.net_input(para,x))
        dif=np.sum(abs(p1-p2))/len(p1)

        actual_classes = (p1 >= probab_threshold).astype(int)
        actual_classes = actual_classes.flatten()

        predicted_classes = (p2>= probab_threshold).astype(int)
        predicted_classes = predicted_classes.flatten()
        accuracy = np.mean(predicted_classes == actual_classes)

        return accuracy * 100,dif

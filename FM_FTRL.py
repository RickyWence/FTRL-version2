#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-05 13:44:54
# @Author  : Ricky (liqiwang@corp.netease.com)
# @Link    : https://mail.163.com/
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from LR_FTRL import cal_loss, logistic, retrieve_nonzero, generate_samples, evaluate_model, get_auc
'''factorization machine'''


class FM:

    def __init__(self, dim, dim_map, sigma,
                 alpha_w, alpha_v, beta_w, beta_v,
                 lambda_w1, lambda_w2, lambda_v1, lambda_v2):
        """
        the constructor of FM class
        :param dim: the dimension of feature vector
        :param dim_map: the mapped dimension of intersected feature vector
        :param sigma: the scale for the initialization of V
        :param alpha_w: the alpha parameters for learning rate in the update of weights
        :param alpha_v: the alpha parameters for learning rate in the update of V
        :param beta_w: the beta parameters for learning rate in the update of weights
        :param beta_v: the beta parameters for learning rate in the update of V
        :param lambda_w1: the L1 regularization for weights
        :param lambda_w2: the L2 regularization for weights
        :param lambda_v1: the L1 regularization for V
        :param lambda_v2: the L2 regularization for V
        """

        self.dim = dim
        self.dim_map = dim_map
        self.alpha_w = alpha_w
        self.alpha_v = alpha_v
        self.beta_w = beta_w
        self.beta_v = beta_v
        self.lambda_w1 = lambda_w1
        self.lambda_w2 = lambda_w2
        self.lambda_v1 = lambda_v1
        self.lambda_v2 = lambda_v2

        # initialize the zws, nws, gradient, weights
        self._zws = np.zeros(self.dim + 1)
        self._nws = np.zeros(self.dim + 1)
        self.weights = np.zeros(self.dim + 1)

        # initialize the v, zvs, nvs
        self._zvs = np.zeros((self.dim, self.dim_map))
        self._nvs = np.zeros((self.dim, self.dim_map))
        self.V = np.random.normal(scale=sigma, size=(self.dim, self.dim_map))

    def update_param(self, sample, label, nonzero_index):
        """
        update the parameters: weights, bias, zws, nws, zvs, nvs
        :param sample: the feature vector                -array vector
        :param label: the ground truth label             -value
        :param nonzero_index: the nonzero index list     -list
        """
        # update bias
        if np.abs(self._zws[-1]) > self.lambda_w1:
            fore = (self.beta_w + np.sqrt(self._nws[-1])) / self.alpha_w + self.lambda_w2
            sign_zws = -1. if self._zws[-1] < 0. else 1.
            self.weights[-1] = -1. / fore * (self._zws[-1] - sign_zws * self.lambda_w1)
        else:
            self.weights[-1] = 0.0

        # update weights and V
        for i in nonzero_index:

            # update weights
            if np.abs(self._zws[i]) > self.lambda_w1:
                fore = (self.beta_w + np.sqrt(self._nws[i])) / self.alpha_w + self.lambda_w2
                sign_zws = -1. if self._zws[i] < 0. else 1.
                self.weights[i] = -1. / fore * (self._zws[i] - sign_zws * self.lambda_w1)
            else:
                self.weights[i] = 0.0

            # update V
            for f in range(self.dim_map):
                if np.abs(self._zvs[i, f]) <= self.lambda_v1:
                    fore = (self.beta_v + np.sqrt(self._nvs[i, f])) / self.alpha_v + self.lambda_v2
                    sign_zvs = -1. if self._zvs[i, f] < 0. else 1.
                    self.V[i, f] = -1. / fore * (self._zvs[i, f] - sign_zvs * self.lambda_v1)
                else:
                    self.V[i, f] = 0.0

        # predict the sample
        prediction = self.predict(sample)

        # update the zws, nws
        loss = prediction - label
        for i in nonzero_index:
            gradient = loss * sample[i]
            sigma = (np.sqrt(self._nws[i] + gradient ** 2) - np.sqrt(self._nws[i])) / self.alpha_w
            self._zws[i] += gradient - sigma * self.weights[i]
            self._nws[i] += gradient ** 2

            # update the zvs, nvs
            for f in range(self.dim_map):
                gradient_v = loss * sample[i] * sum(self.V[:, f] * sample[f]) - self.V[i, f] * (sample[i] ** 2)
                sigma = (np.sqrt(self._nvs[i, f] + gradient_v ** 2) - np.sqrt(self._nvs[i, f])) / self.alpha_v
                self._zvs[i, f] += gradient_v - sigma * self.V[i, f]
                self._nvs[i, f] += gradient_v ** 2

        # update the zws, nws related to bias
        sigma = (np.sqrt(self._nws[-1] + loss ** 2) - np.sqrt(self._nws[-1])) / self.alpha_w
        self._zws[-1] += loss - sigma * self.weights[-1]
        self._nws[-1] += loss ** 2

        return prediction

    def predict(self, samples):
        """
        :param samples: the unseen sample  		        -array(n_samples, dimension)
        :return: prediction                             -array(n_samples, )
        """
        raw_output1 = np.dot(samples, self.weights[:-1]) + self.weights[-1]
        raw_output2 = (np.square(np.dot(samples, self.V)) - np.dot(np.square(samples), np.square(self.V))).sum()
        return logistic(raw_output1 + raw_output2 / 2)

    def train_ftrl(self, samples, labels, iteration, is_print=False):
        """
        train the LR model using the ftrl-proximal optimization algorithm
        :param samples: the feature matrix	            -n_sample * dimension
        :param labels: the label vector  	            -n_sample * 1
        :param iteration: the stooping criterion        -int
        :param is_print: whether to print               -boolean
        :return:
        """
        n_samples, dim = np.shape(samples)
        i = 0
        preds = np.zeros(n_samples)
        while i < iteration:
            log_loss = 0.0
            for t in range(n_samples):
                # retrieve the index of nonzero elements
                index_list = retrieve_nonzero(samples[t])
                preds[t] = self.update_param(sample=samples[t], label=labels[t], nonzero_index=index_list)
                log_loss += cal_loss(probability=preds[t], true_label=labels[t]) / n_samples
            train_error = evaluate_model(preds=preds, labels=labels)
            if i % 10 == 0 & is_print:
                print("FM-after iteration %s, the total logloss is %s,"
                      " the training error is %s" % (i, log_loss, train_error))
            i += 1

    def load_model(self, file_path):
        with open(file_path, 'r') as f:
            s = f.read()
            model = pickle.loads(s)
            self.weights[:-1] = model['weights']
            self.weights[-1] = model['bias']
            self.V = model['V']

    def save_model(self, file_path):
        model = {"weights": self.weights[:-1], "bias": self.weights[-1], 'V': self.V}
        with open(file_path, 'w') as f:
            pickle.dump(model, f)


if __name__ == "__main__":

    # generate the datasets including the testing samples and training samples
    train_samples, train_labels = generate_samples(15, 100)
    test_samples, test_labels = generate_samples(15, 20)

    # load bread_cancer datasets from sklearn
    minint = load_breast_cancer()
    data_samples = minint.data
    target_samples = minint.target
    num_samples, dim_ = data_samples.shape

    # split all the samples into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(data_samples, target_samples, test_size=0.2, random_state=42)

    # define the hyper-parameter
    alpha_w_, alpha_v_, beta_w_, beta_v_ = 0.2, 0.2, 0.2, 0.2
    lambda_w1_, lambda_w2_, lambda_v1_, lambda_v2_ = 0.2, 0.2, 0.2, 0.2
    hiddens = 8
    sigma_ = 1.0
    iteration_ = 40

    # create the fm model
    fm = FM(dim=dim_, dim_map=hiddens, sigma=sigma_,
            alpha_w=alpha_w_, alpha_v=alpha_v_, beta_w=beta_w_, beta_v=beta_v_,
            lambda_w1=lambda_w1_, lambda_w2=lambda_w2_, lambda_v1=lambda_v1_, lambda_v2=lambda_v2_)

    fm.train_ftrl(X_train, y_train, iteration_, is_print=True)

    # test the unseen samples
    test_preds = fm.predict(X_test)
    test_error = evaluate_model(test_preds, y_test)
    test_auc = roc_auc_score(y_true=y_test, y_score=test_preds)
    my_auc = get_auc(scores=test_preds, labels=y_test)
    print("test-error: ", test_error)
    print("test-sklearn auc: ", test_auc)
    print("test-my auc: ", my_auc)

    # print the parameters of trained FM model
    print("weights: ", fm.weights)
    print("bias: ", fm.weights[-1])
    print("V: ", fm.V)




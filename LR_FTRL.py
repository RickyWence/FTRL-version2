#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-02 14:58:44
# @Author  : Ricky (liqiwang@corp.netease.com)
import os
import time
import pickle
import numpy as np
import sklearn.datasets as dt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
'''logistic regression model'''


def generator_nonzero(sample):
    """
    generate the nonzero index in the feature
    :param sample:  one sample          vector
    :return: generator
    """
    for j in range(len(sample)):
        if sample[j] != 0:
            yield j


def cal_loss(true_label, probability):
    """
    calculate the log_loss between ground true-label and prediction
    :param true_label: the ground truth label for the sample	{0, 1}
    :param probability: the prediction of the trained model		[0, 1]
    :return: logloss
    """
    probability = max(min(probability, 1. - 1e-15), 1e-15)
    return -np.log(probability) if true_label == 1 else -np.log(1 - probability)


def cal_loss2(true_label, probability):
    """
    calculate the softmax log_loss between ground true-label and prediction for one single sample
    note: the probability has been normalized (no need to max or min operation)
    :param true_label: the ground truth label vector for the sample         -array
    :param probability: the prediction vector of the trained model          -array
    :return: logloss
    """
    k = np.argmax(true_label)
    return -np.log(probability[k])


def evaluate_model(preds, labels):
    """
    evaluate the model errors on a set of data (not one single sample)
    :param preds: the prediction of unseen samples          (n_sample, n_label)
    :param labels: the ground truth labels                  (n_sample, n_label)
    :return:
    """
    shapes = len(labels.shape)
    if shapes == 2:
        # multi-class classification-find the max-index per row

        max_index = np.argmax(preds, axis=1)
        for i, p in enumerate(max_index):
            preds[i, p] = 1
        preds[preds < 1.] = 0
    else:
        # binary classification-default (n_sample, )
        preds[preds >= 0.5] = 1
        preds[preds < 0.5] = 0
    return np.abs(preds - labels).sum() / (len(labels) * shapes)* 100


def get_auc(scores, labels):
    """
    calculate the auc indicator on a set of data
    :param scores: the probability of each sample   [0, 1]-array
    :param labels: the ground truth labels          {0, 1}-array
    :return: auc indicator
    """
    data_shape = labels.shape
    pos_num = np.sum(labels, axis=0)
    neg_num = len(labels) - pos_num
    # rank scores
    rank_index = np.argsort(scores, axis=0, kind='quicksort')
    if len(data_shape) == 1:
        rank_sum = 0.0
        for i in range(data_shape[0]):
            if labels[rank_index[i]] == 1:
                rank_sum += (i + 1)
        # calculate the auc
        denominator = pos_num * neg_num
        if denominator == 0:
            res = 0
        else:
            res = (rank_sum - 0.5 * (pos_num + 1) * pos_num) / denominator

    else:
        rank_sum = np.zeros(data_shape[1])
        res = 0.0
        for i in range(data_shape[0]):
            for j in range(data_shape[1]):
                if labels[rank_index[i, j], j] == 1:
                    rank_sum[j] += (i + 1)
        # calculate the auc
        denominator = pos_num * neg_num
        for j in range(data_shape[1]):
            if denominator[j] == 0:
                res += 0.0
            else:
                numerator = rank_sum[j] - 0.5 * (pos_num[j] + 1) * pos_num[j]
                res += numerator / denominator[j]
        res = res / data_shape[1]
    return res


def logistic0(var):
    """
    calculate the logistic value of one variable
    :param var: the input variable
    :return: logistic value
    """
    var = max(min(var, 100), -100)
    return 1. / (1 + np.exp(-var))


def logistic(var):
    """
    extend to multi-dimension ndarray   (1,2,3,4)multi-dimensions
    :param var: float/int/ndarray
    :return:
    """
    if isinstance(var, np.ndarray):
        shapes = var.shape
        length = np.multiply.reduce(shapes)
        var = np.reshape(var, length)
        res = np.zeros(length)
        for i in range(length):
            res[i] = logistic0(var[i])
        res = np.reshape(res, shapes)
    else:
        res = logistic0(var)
    return res


def softmax(var):
    """
    calculate the softmax value of one vector variable
    :param var: the input vector
    :return: softmax vector
    """
    e_x = np.exp(var - np.max(var))
    output = e_x / e_x.sum()
    return output


def generate_samples(dimension, n_samples):
    """
    generate samples according to the user-defined requirements
    :param dimension:
    :param n_samples:
    :return:
    """
    samples = np.random.rand(n_samples, dimension)
    labels = np.random.randint(0, 2, (n_samples, ))
    return samples, labels


class LR:

    def __init__(self, dim, alpha, beta, lambda1, lambda2):
        """
        the constructor of LR class
        :param dim: the dimension of input features
        :param alpha: the alpha parameters for learning rate in the update of weights
        :param beta: the beta parameters for learning rate in the update of weights
        :param lambda1: L1 regularization
        :param lambda2: L2 regularization
        """
        self.dim = dim
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        # initialize the zis, nis, gradient, weights
        self._zs = np.zeros(self.dim + 1)
        self._ns = np.zeros(self.dim + 1)
        self.weights = np.zeros(self.dim + 1)

    def update_param(self, sample, label):
        """
        update the parameters: weights, zs, ns, gradients
        :param sample: the feature vector                -array vector
        :param label: the ground truth label             -value
        :param nonzero_index: the nonzero index list     -list
        """
        # update bias
        if np.abs(self._zs[-1]) > self.lambda1:
            fore = (self.beta + np.sqrt(self._ns[-1])) / self.alpha + self.lambda2
            self.weights[-1] = -1. / fore * (self._zs[-1] - np.sign(self._zs[-1]) * self.lambda1)
        else:
            self.weights[-1] = 0.0

        # update weights
        for index in generator_nonzero(sample):
            if np.abs(self._zs[index]) > self.lambda1:
                fore = (self.beta + np.sqrt(self._ns[index])) / self.alpha + self.lambda2
                self.weights[index] = -1. / fore * (self._zs[index] - np.sign(self._zs[index]) * self.lambda1)
            else:
                self.weights[index] = 0

        # predict the sample, compute the gradient of loss
        prediction = self.predict(sample)
        base_grad = prediction - label
        # update the zs, ns
        for j in generator_nonzero(sample):
            gradient = base_grad * sample[j]
            sigma = (np.sqrt(self._ns[j] + gradient ** 2) - np.sqrt(self._ns[j])) / self.alpha
            self._zs[j] += gradient - sigma * self.weights[j]
            self._ns[j] += gradient ** 2
        sigma = (np.sqrt(self._ns[-1] + base_grad ** 2) - np.sqrt(self._ns[-1])) / self.alpha
        self._zs[-1] += base_grad - sigma * self.weights[-1]
        self._ns[-1] += base_grad ** 2
        return prediction

    def predict(self, samples):
        """
        :param samples: the unseen sample  		        -array(n_samples, dimension)
        :return: prediction                             -array(n_samples, )
        """
        raw_output = np.dot(samples, self.weights[:-1]) + self.weights[-1]
        return logistic(raw_output)

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
                #index_list = retrieve_nonzero(samples[t])
                preds[t] = self.update_param(sample=samples[t], label=labels[t])
                log_loss += cal_loss(probability=preds[t], true_label=labels[t]) / n_samples
            train_error = evaluate_model(preds=preds, labels=labels)
            if i % 10 == 0 & is_print:
                print("LR-after iteration %s, the total logloss is %s,"
                      " the training error is %.2f%%" % (i, log_loss, train_error))
            i += 1


    def load_model(self, file_path):
        with open(file_path, 'rb+') as f:
            s = f.read()
            model = pickle.loads(s)
            self.weights[:-1] = model['weights']
            self.weights[-1] = model['bias']

    def save_model(self, file_path):
        model = {"weights": self.weights[:-1], "bias": self.weights[-1]}
        if not os.path.exists(file_path):
            # split the file and path
            path = file_path.split("/")[:-1]
            path = "/".join(path)
            os.makedirs(path)
        with open(file_path, 'wb+') as f:
            pickle.dump(model, f)


if __name__ == "__main__":

    # generate the datasets including the testing samples and training samples
    train_samples, train_labels = generate_samples(15, 100)
    test_samples, test_labels = generate_samples(15, 20)

    # load bread_cancer datasets from sklearn
    minint = dt.load_breast_cancer()
    data_samples = minint.data
    target_samples = minint.target
    num_samples, dim_ = data_samples.shape

    # split all the samples into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(data_samples, target_samples, test_size=0.2, random_state=42)

    # define the hyper-parameter
    alpha_, beta_, lambda_1, lambda_2 = 0.2, 0.2, 0.2, 0.2
    iteration_ = 100

    # create the lr model
    lr = LR(dim=dim_, alpha=alpha_, beta=beta_, lambda1=lambda_1, lambda2=lambda_2)

    # train the lr model
    start_time = time.time()
    lr.train_ftrl(X_train, y_train, iteration_, is_print=True)
    print("end time: ", time.time() - start_time)
    # test the unseen samples
    test_preds = lr.predict(X_test)
    test_error = evaluate_model(test_preds, y_test)

    test_auc = roc_auc_score(y_true=y_test, y_score=test_preds)
    my_auc = get_auc(scores=test_preds, labels=y_test)
    print("test-error: %.2f%%" % test_error)
    print("test-sklearn auc: ", test_auc)
    print("test-my auc: ", my_auc)

    # print the parameters of trained LR model
    print("weights: ", lr.weights[:-1])
    print("bias: ", lr.weights[-1])

    file_path = "../model/model.txt"
    lr.save_model(file_path)
    lr.load_model(file_path)


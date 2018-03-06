#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-02-05 13:44:54
# @Author  : Ricky (liqiwang@corp.netease.com)
# @Link    : https://mail.163.com/

from FM_FTRL import FM
from LR_FTRL import LR, evaluate_model, get_auc
import sklearn.datasets as dt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
'''using FTRL to update models (logistic regression or factorization machine)'''


class FTRL:

    def __init__(self, base, args_parse, iteration):
        """
        initialize the ftrl model
        :param base: the base model
        :param args_parse: the according parameters
        :param iteration: the stopping criterion
        """
        self.iteration = iteration
        self.base = base
        self.params = {}
        if self.base == "lr":
            dim = args_parse['dim']
            alpha = args_parse['alpha_w']
            beta = args_parse['beta_w']
            lambda1 = args_parse['lambda_w1']
            lambda2 = args_parse['lambda_w2']
            self.model = LR(dim=dim, alpha=alpha, beta=beta, lambda1=lambda1, lambda2=lambda2)
        else:
            dim = args_parse['dim']
            dim_map = args_parse['dim_map']
            sigma = args_parse['sigma']
            alpha_w = args_parse['alpha_w']
            alpha_v = args_parse['alpha_v']
            beta_w = args_parse['beta_w']
            beta_v = args_parse['beta_v']
            lambda_w1 = args_parse['lambda_w1']
            lambda_w2 = args_parse['lambda_w2']
            lambda_v1 = args_parse['lambda_v1']
            lambda_v2 = args_parse['lambda_v2']
            self.model = FM(dim=dim, dim_map=dim_map, sigma=sigma,
                            alpha_w=alpha_w, alpha_v=alpha_v, beta_w=beta_w, beta_v=beta_v,
                            lambda_w1=lambda_w1, lambda_w2=lambda_w2, lambda_v1=lambda_v1, lambda_v2=lambda_v2)

    def fit(self, train_samples, train_labels):
        """
        train model using ftrl optimization model
        :param train_samples:  the training samples         -shapes(n_samples, dimension)
        :param train_labels:   the training labels          -shapes(n_samples, )
        """
        self.model.train_ftrl(train_samples, train_labels, self.iteration, is_print=True)
        self.params['weights'] = self.model.weights
        self.params['bias'] = self.model.weights[-1]
        if self.base == "fm":
            self.params['V'] = self.model.V

    def predict(self, test_samples):
        """
        test the unseen samples using the trained model
        :param test_samples: the testing samples            -shapes(n_samples, dimension)
        :return: the predictions
        """
        test_preds = self.model.predict(test_samples)
        return test_preds

    def evaluate(self, test_samples, test_labels, metrics='error'):
        """
        evaluate the model using different metrics
        :param test_samples: the testing samples            -shapes(n_samples, dimension)
        :param test_labels: the testing labels              -shapes(n_samples,)
        :param metrics: auc or error
        :return: the evaluation
        """
        test_preds = self.predict(test_samples)
        if metrics == 'error':
            evaluation = evaluate_model(preds=test_preds, labels=test_labels)
        else:
            evaluation = roc_auc_score(y_true=test_labels, y_score=test_preds)
        return evaluation


if __name__ == "__main__":

    # load bread_cancer datasets from sklearn
    minint = dt.load_breast_cancer()
    data_samples = minint.data
    target_samples = minint.target
    num_samples, dim_ = data_samples.shape

    # split all the samples into training data and testing data
    X_train, X_test, y_train, y_test = train_test_split(data_samples, target_samples, test_size=0.2, random_state=42)

    # define the hyper-parameter
    hyper_params = {
        'dim': dim_,
        'dim_map': 8,
        'sigma': 1.0,
        'alpha_w': 0.2,
        'alpha_v': 0.2,
        'beta_w': 0.2,
        'beta_v': 0.2,
        'lambda_w1': 0.2,
        'lambda_w2': 0.2,
        'lambda_v1': 0.2,
        'lambda_v2': 0.2,
        }
    iteration_ = 20

    # create the fm model
    lr = FTRL(base="lr", args_parse=hyper_params, iteration=iteration_)
    fm = FTRL(base="fm", args_parse=hyper_params, iteration=iteration_)

    # train the fm model
    fm.fit(X_train, y_train)
    lr.fit(X_train, y_train)

    # test the unseen samples
    test_preds_lr = lr.predict(X_test)
    test_preds_fm = fm.predict(X_test)
    test_error_lr = evaluate_model(test_preds_lr, y_test)
    test_error_fm = evaluate_model(test_preds_fm, y_test)
    test_auc_lr = roc_auc_score(y_true=y_test, y_score=test_preds_lr)
    test_auc_fm = roc_auc_score(y_true=y_test, y_score=test_preds_fm)
    my_test_auc_lr = get_auc(scores=test_preds_lr, labels=y_test)
    my_test_auc_fm = get_auc(scores=test_preds_fm, labels=y_test)

    print("logistic regression-test error: %.2f%%" % test_error_lr)
    print("logistic regression-test auc: ", test_auc_lr)
    print("logistic regression-my test auc: ", my_test_auc_lr)

    print("factorization machine-test error: %.2f%%" % test_error_fm)
    print("factorization machine-test auc: ", test_auc_fm)
    print("factorization machine-my test auc: ", my_test_auc_fm)



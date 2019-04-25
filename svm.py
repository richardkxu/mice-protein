import os, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt


class SVMClassifier:
    def __init__(self, X, y, kerenl_type='linear', test_ratio=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
        self.clf = SVC(kernel=kerenl_type)

    def train_test(self):
        Cs = np.logspace(-6, -1, 10)
        best_clf = GridSearchCV(estimator=self.clf, param_grid=dict(C=Cs), n_jobs = -1)
        best_clf.fit(self.X_train, self.y_train)
        print("Best score: %0.2f".format(best_clf.best_score_))
        print("Best para: {}".format(best_clf.best_params_))
        best_clf.score(self.X_test, self.y_test)

    def visualize_feat_importance(self, classifier, feature_names, top_features=20):
        """
        https://medium.com/@aneesha/visualising-top-features-in-linear-svm-with-scikit-learn-and-matplotlib-3454ab18a14d
        :param classifier: 
        :param feature_names: 
        :param top_features: 
        :return: 
        """"""
        coef = classifier.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = [‘red’ if c < 0 else ‘blue’ for c in coef[top_coefficients]]
        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha=’right’)
        plt.show()

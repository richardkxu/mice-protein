import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


class SVMClassifier:
    def __init__(self, X, y, kerenl_type='linear', test_ratio=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
        self.kernel_type = kerenl_type

    def train_test(self):
        Cs = np.logspace(-6, -1, 10)
        self.best_clf = GridSearchCV(estimator=SVC(kernel=self.kernel_type), param_grid=dict(C=Cs), n_jobs = -1)
        self.best_clf.fit(self.X_train, self.y_train)
        print("Best training score: {0.3f}".format(self.best_clf.best_score_))
        print("Best para: {}".format(self.best_clf.best_params_))
        test_score = self.best_clf.score(self.X_test, self.y_test)
        print("Test performance: {0.3f}".format(test_score))

    def visualize_feat_importance(self, feature_names, top_features=20):
        coef = self.best_clf.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(15, 5))
        colors = []
        for c in coef[top_coefficients]:
            if c <0:
                colors.append('red')
            else:
                colors.append('blue')

        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha=’right’)
        plt.show()


if __name__ == '__main__':
    svm_clf = SVMClassifier(X, y)
    svm_clf.train_test()
    svm_clf.visualize_feat_importance(feature_names, top_features=20)

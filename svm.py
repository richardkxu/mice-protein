import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd


class SVMClassifier:
    def __init__(self, X, y, kerenl_type='linear', test_ratio=0.25):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=test_ratio, random_state=0)
        self.kernel_type = kerenl_type

    def train_test(self):
        Cs = np.logspace(-6, -1, 10)
        grid_search = GridSearchCV(estimator=SVC(kernel=self.kernel_type), param_grid=dict(C=Cs), n_jobs = -1)
        grid_search.fit(self.X_train, self.y_train)
        print("Best training score: {:0.3f}".format(grid_search.best_score_))
        print("Best para: {}".format(grid_search.best_params_))
        test_score = grid_search.score(self.X_test, self.y_test)
        print("Test performance: {:0.3f}".format(test_score))
        self.clf = grid_search.best_estimator_

    def visualize_feat_importance(self, feature_names, i, top_features=20):
        coef = self.clf.coef_.ravel()
        top_positive_coefficients = np.argsort(coef)[-top_features:]
        top_negative_coefficients = np.argsort(coef)[:top_features]
        top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
        # create plot
        plt.figure(figsize=(16, 9))
        colors = []
        for c in coef[top_coefficients]:
            if c <0:
                colors.append('red')
            else:
                colors.append('blue')

        plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
        feature_names = np.array(feature_names)
        plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
        fig_name = 'feat-importance' + str(i) + '.pdf'
        plt.savefig(fig_name)




if __name__ == '__main__':
    base_directory = "./data"
    kinds = ["behavior", "genotype", "treatment"]

    for i in range(len(kinds)):

        x = pd.read_csv(base_directory + "/" + kinds[i] + "_x.csv", sep=',',header=0,index_col=0 )
        x_value = x.values
        y = pd.read_csv(base_directory + "/" + kinds[i] + "_y.csv", sep=',', header=0, index_col=0)
        y_value = y.values
        header = pd.read_csv(base_directory + "/" + kinds[i] + "_x.csv", sep=',' ,index_col=0)
        filepath = base_directory + "/" + kinds[i] + "_x.csv"
        with open(filepath) as fp:
            line = fp.readline()
            feature_names = line.split(",")[1:]




        svm_clf = SVMClassifier(x_value, y_value)
        svm_clf.train_test()
        svm_clf.visualize_feat_importance(feature_names, i, top_features=20)

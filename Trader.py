import pandas as pd
import numpy as np
import sklearn

from sklearn.neural_network import MLPClassifier  # multi layer neural network
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.tree._tree import TREE_LEAF
import graphviz
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix



"""
trading classifier for stock prediction

uses 70/30 training/testing split
"""
class Trader(object):

    training_split = 0.70

    def split_training_testing_sets(self, x, y):
        training_items = int(len(y) * self.training_split)
        train_xs = list(x[:training_items])
        test_xs = list(x[training_items:])
        train_ys = list(y[:training_items])
        test_ys = list(y[training_items:])

        # print test_xs

        # print "**********training items ", len(y), training_items, len(train_xs), len(train_ys), len(test_xs)
        # for x in train_xs:
        #     for x2 in test_xs:
        #         if x == x2:
        #
        #             print "EQUAL", x, x2

        return train_xs, test_xs, train_ys, test_ys

    def decision_tree(self, debug=True, prune=5):
        """
        decision tree classifier with pruning
        """
        def visualize():
            """
            outputs "visualization.pdf" of D tree
            """
            dot_data = tree.export_graphviz(clf, out_file=None)
            graph = graphviz.Source(dot_data)
            graph.render("visualization")

        def prune_index(inner_tree, index, threshold):
            if inner_tree.value[index].min() < threshold:
                # turn node into a leaf by "unlinking" its children
                inner_tree.children_left[index] = TREE_LEAF
                inner_tree.children_right[index] = TREE_LEAF
            # if there are children, visit them as well
            if inner_tree.children_left[index] != TREE_LEAF:
                prune_index(inner_tree, inner_tree.children_left[index], threshold)
                prune_index(inner_tree, inner_tree.children_right[index], threshold)

        clf = tree.DecisionTreeClassifier()

        clf.fit(self.train_xs, self.train_ys)

        # print "pruning"
        # print sum(clf.tree_.children_left < 0)
        # start pruning from the root
        prune_index(clf.tree_, 0, prune) #5
        # print sum(clf.tree_.children_left < 0)

        # print "DTree prediction for tomorrow", clf.predict([self.flatten(self.df.tail(5).values.tolist())]) # for 5 days
        # print "DTree prediction for tomorrow", clf.predict(self.df.tail(1).values.tolist())
        if debug:
            print "DTree prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)

        visualize()
        return clf, accuracy


    def random_forest(self, debug=True):
        """
        see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        really weird: only saying GREEN no matter what!
        """
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(self.train_xs, self.train_ys)
        if debug:
            print(clf.feature_importances_)
            print "prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy

    def knn(self, debug=True, neighbors=2):
        """
        pretty good for day trading, terrbile for swing trading
        """
        clf = KNeighborsClassifier(n_neighbors=neighbors) # n=1 is even better but possibly overfitting
        clf.fit(self.train_xs, self.train_ys)
        if debug:
            print "KNN prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy

    def svm(self, debug=True, kernel='poly', degree=4):
         # sigmoid, rbf, poly is slow, linear is slow,
        clf = svm.SVC(kernel=kernel, degree=degree) # degree is ignored by anything that isnt poly
        print "here"
        clf.fit(self.train_xs, self.train_ys)
        if debug:
            print "\nSVM prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy

    def boost(self, debug=True, prune_val=5):
        """
        adaboost classifier
        """
        clf = AdaBoostClassifier(base_estimator=self.decision_tree(debug=False, prune=prune_val)[0], n_estimators=50, learning_rate=1.0)

        # clf = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)
        # parameter_space = {
        #     'base_estimator': [None], # none
        #     'n_estimators': [200], #200
        #     'learning_rate': [1] # 1
        # }
        # clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3) # 3 cross validation sets

        clf.fit(self.train_xs, self.train_ys)

        # print clf.best_params_

        if debug:
            print "\n\nBoosting for " + self.tradingType
            print "Boost prediction for tomorrow is", clf.predict([self.current_sample])
            print "\n\n"
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy

    def neural_network(self, debug=True, max_iter=200):
        """
        see https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        """
        # 10% validation fraction

        # clf = MLPClassifier(solver='lbfgs', alpha=1e-07,activation='tanh', max_iter=100, hidden_layer_sizes=[(50,50,50,50)])
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50,50,50), activation='relu', alpha=0.0000001, max_iter=max_iter)

        # ('Best parameters found:\n', {'alpha': 1e-07, 'activation': 'tanh', 'max_iter': 100, 'solver': 'lbfgs', 'hidden_layer_sizes': (50, 50, 50, 50)})
        # parameter_space = {
        #     'hidden_layer_sizes': [(100, 100, 100, 100), (50,50,50,50,50), (50,50,50)], #50,50,50
        #     'activation': ['tanh', 'relu'], # relu
        #     'solver': ['lbfgs', 'sgd', 'adam'], # lbgfs
        #     'alpha': [0.00000001, 0.0000001, 0.000001], # e-7
        #     'max_iter': [150, 200, 250], # 200
        # }
        # clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3) # 3 cross validation sets

        clf.fit(self.train_xs, self.train_ys)

        # print clf.best_params_
        if debug:
            # tomorrow = [[2657.439941,2672.379883,2657.330078,2664.760010,2664.760010,3814080000]]
            # print "NN prediction for tomorrow", clf.predict([self.flatten(self.df.tail(5).values.tolist())]) # for 5 days
            print "NN prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy



    def test_accuracy(self, clf, debug=True):
        """
        tests accuracy of fitted training data on test set
        """
        y_true, y_pred = self.test_ys, clf.predict(self.test_xs)
        return reports(y_true, y_pred, debug=debug)


    def flatten(self, past5Days):
        flattened = []
        for day in past5Days:
            for val in day:
                flattened.append(val)
        return flattened


def reports(y_true, y_pred, debug=True):
    accuracy = accuracy_score(y_true, y_pred)
    if debug:
        print('Results on the test set:')
        print(classification_report(y_true, y_pred))
        print '\nConfusion matrix:\n',confusion_matrix(y_true, y_pred)

        # print "TRAINING accuracy:"
        # print accuracy_score(self.train_ys, clf.predict(self.train_xs))

        print "\n Accuracy Score:"
        print accuracy

        print "diff", set(y_true) - set(y_pred)
    return accuracy

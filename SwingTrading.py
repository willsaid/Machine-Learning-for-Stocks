import pandas as pd
import numpy as np
import sklearn

from sklearn.neural_network import MLPClassifier  # multi layer neural network
from sklearn.model_selection import GridSearchCV

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

"""
swing trading classifier: predict if market (S&P) will open in GREEN (+1) or RED (-1) tomorrow

uses 70/30 training/testing split
"""

class SwingTrading(object):

    training_split = 0.70

    """Read in historical market data and create dataset"""
    def __init__(self):
        # s&p 500 (^gspc) 1-day candles from past 5 years
        self.df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)
        x, y = self.create_dataset()
        self.train_xs, self.test_xs, self.train_ys, self.test_ys = self.split_training_testing_sets(x, y)

        self.neural_network()
        # self.decision_tree()
        # self.random_forest()

    def create_dataset(self):
        """
        returns tuple (X, y) of the form:

        >>> X = [
                  [ 1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000
                    1796.199951,1799.939941,1791.829956,1799.839966,1799.839966,3312160000
                    1790.880005,1793.880005,1772.260010,1782.589966,1782.589966,4059690000 ]

                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000],
                    [1776.010010,1798.030029,1776.010010,1797.020020,1797.020020,3775990000],  # ...
                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000],  # day before yesterday
                    [1776.010010,1798.030029,1776.010010,1797.020020,1797.020020,3775990000],  # yesterday
                    [1791.030029,1795.979980,1772.880005,1781.560059,1781.560059,4045200000]   # today Open,High,Low,Close,Adj Close,Volume
                    .
                    .
                    .
                ]

        # 1 = next day opens green, -1 = red
        >>> y = [1.0, -1.0, . . .]

        """
        xs, ys = [], []

        for index, row in self.df[4:-1].iterrows():
            # past 5 days, including today
            # past4Days = self.df[:index].tail(5).values.tolist()
            # xs.append(self.flatten(past4Days))

            # just today
            xs.append(row.values)

            # determine if the following day was red or green
            nextRow = self.df[index:].head(2).tail(1).values # tomorrow
            nextOpen = nextRow[0][0] # OPEN IS FIRST COLUMN!
            isGreen = 1.0 if nextOpen > row['Open'] else -1.0
            ys.append(isGreen)

        # shuffle them around: this somehow results in a super high decision tree accuracy of over 70% , but it should be random right?
        import random
        zipped = list(zip(xs, ys))
        random.shuffle(zipped)
        return zip(*zipped)

    def flatten(self, past5Days):
        flattened = []
        for day in past5Days:
            for val in day:
                flattened.append(val)
        return flattened


    def split_training_testing_sets(self, x, y):

        training_items = int(len(y) * self.training_split)
        train_xs = list(x[:training_items])
        test_xs = list(x[training_items:])
        train_ys = list(y[:training_items])
        test_ys = list(y[training_items:])

        return train_xs, test_xs, train_ys, test_ys


    def test_accuracy(self, clf):
        """
        tests accuracy of fitted training data on test set
        """
        y_true, y_pred = self.test_ys, clf.predict(self.test_xs)

        from sklearn.metrics import classification_report, accuracy_score
        print('Results on the test set:')
        print(classification_report(y_true, y_pred))

        print "\n Accuracy Score:"
        print accuracy_score(y_true, y_pred)


        numCorrect = 0
        for X, expectedY in zip(self.test_xs, self.test_ys):
            actual = clf.predict([X])
            if actual == expectedY: numCorrect += 1
        accuracy = float(numCorrect) / len(self.test_ys)
        print "Accuracy is ", accuracy

        # print "y_true", y_pred
        print "diff", set(y_true) - set(y_pred)




    def decision_tree(self):
        """
        decision tree classifier with pruning
        """
        def visualize():
            """
            outputs "visualization.pdf" of D tree
            """
            import graphviz
            dot_data = tree.export_graphviz(clf, out_file=None)
            graph = graphviz.Source(dot_data)
            graph.render("visualization")

        def prune_index(inner_tree, index, threshold):
            from sklearn.tree._tree import TREE_LEAF
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

        print "pruning"
        print sum(clf.tree_.children_left < 0)
        # start pruning from the root
        prune_index(clf.tree_, 0, 5)
        print sum(clf.tree_.children_left < 0)

        print "DTree prediction for tomorrow", clf.predict([self.flatten(self.df.tail(5).values.tolist())]) # for 5 days
        # print "DTree prediction for tomorrow", clf.predict(self.df.tail(1).values.tolist())
        self.test_accuracy(clf)

        visualize()

    def random_forest(self):
        """
        see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

        really weird: only saying GREEN no matter what!
        """
        clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
        clf.fit(self.train_xs, self.train_ys)
        print(clf.feature_importances_)
        print "prediction for tomorrow", clf.predict(self.df.tail(1).values.tolist())
        self.test_accuracy(clf)

    def neural_network(self):
        """
        see https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        """
        # 10% validation fraction

        # clf = MLPClassifier(solver='lbfgs', alpha=1e-07,activation='tanh', max_iter=100, hidden_layer_sizes=[(50,50,50,50)])
        clf = MLPClassifier(solver='lbfgs', hidden_layer_sizes=[(50,50,50)])

        # ('Best parameters found:\n', {'alpha': 1e-07, 'activation': 'tanh', 'max_iter': 100, 'solver': 'lbfgs', 'hidden_layer_sizes': (50, 50, 50, 50)})
        parameter_space = {
            'hidden_layer_sizes': [(50,50,50,50), (50,50,50), (50,50), (50,)],
            'activation': ['tanh', 'relu'],
            'solver': ['lbfgs', 'sgd', 'adam'],
            'alpha': [0.0000001, 0.01, 0.1],
            'max_iter': [100, 200, 300],
        }

        clf = GridSearchCV(clf, parameter_space, n_jobs=-1, cv=3) # 3 cross validation sets

        clf.fit(self.train_xs, self.train_ys)

        # tomorrow = [[2657.439941,2672.379883,2657.330078,2664.760010,2664.760010,3814080000]]
        print "NN prediction for tomorrow", clf.predict([self.flatten(self.df.tail(5).values.tolist())]) # for 5 days
        # print "NN prediction for tomorrow", clf.predict(self.df.tail(1).values.tolist())

        self.test_accuracy(clf)












#

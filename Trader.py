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


from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import VarianceThreshold


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

    def neural_network(self, debug=True, max_iter=200, X=None, testX=None):
        """
        see https://scikit-learn.org/stable/modules/neural_networks_supervised.html
        """
        # 10% validation fraction

        if X is None:
            X = self.train_xs
        if testX is not None:
            self.test_xs = testX
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
        # print X is None
        # print np.array(self.xs).shape
        # print X
        print np.array(X).shape
        print np.array(self.train_ys).shape
        # print self.train_ys
        clf.fit(X, self.train_ys)

        # print clf.best_params_
        if debug:
            # tomorrow = [[2657.439941,2672.379883,2657.330078,2664.760010,2664.760010,3814080000]]
            # print "NN prediction for tomorrow", clf.predict([self.flatten(self.df.tail(5).values.tolist())]) # for 5 days
            print "NN prediction for tomorrow for " + self.tradingType, clf.predict([self.current_sample])
        accuracy = self.test_accuracy(clf, debug)
        return clf, accuracy


    def k_means_clustering(self, debug=True, n_clusters=2, X=None):
        """
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
        >>> X = np.array([[1, 2], [1, 4], [1, 0],
        ...               [10, 2], [10, 4], [10, 0]])
        >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        >>> kmeans.labels_
        array([1, 1, 1, 0, 0, 0], dtype=int32)
        >>> kmeans.predict([[0, 0], [12, 3]])
        array([1, 0], dtype=int32)
        >>> kmeans.cluster_centers_
        array([[10.,  2.],
               [ 1.,  2.]])
        """
        if X is None:
            X = self.xs
        kmeans = KMeans(n_clusters=n_clusters).fit(X)
        # if debug:
            # print "K Means labels", kmeans.labels_
            # print "K Means Centers", kmeans.cluster_centers_
            # print "K Means Prediction", kmeans.predict([self.current_sample])

            # swing trading:
            # print "K Means Test Predict [1.         1.00408393 1.00800404 1.01096457 1.01265234]:", kmeans.predict([[1.,         1.00408393, 1.00800404, 1.01096457, 1.01265234]])
            # print "K Means Test Predict [1.         0.99533576 0.99089404 0.98759187 0.9859582 ]:", kmeans.predict([[1.,         0.99533576, 0.99089404, 0.98759187, 0.9859582]])
        # accuracy = self.test_accuracy(kmeans, debug)
        return kmeans

    def expectation_max(self, debug=True, n_clusters=2):
        em = GaussianMixture(n_components = n_clusters).fit(self.train_xs)
        print "GaussianMixture Prediction:", em.predict([self.current_sample])
        print "Predicted Prob:", em.predict_proba([self.current_sample])
        accuracy = self.test_accuracy(em, debug)
        return em, accuracy


    def pca(self, debug=True, n_components=2, X=None):
        """
        >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        >>> pca = PCA(n_components=2)
        >>> pca.fit(X)
        PCA(copy=True, iterated_power='auto', n_components=2, random_state=None,
          svd_solver='auto', tol=0.0, whiten=False)
        >>> print(pca.explained_variance_ratio_)
        [0.9924... 0.0075...]
        >>> print(pca.singular_values_)
        [6.30061... 0.54980...]
        """
        if X is None: X = self.xs
        pca = PCA(n_components=n_components)
        pca.fit(X)
        # print "pca.explained_variance_ratio_", pca.explained_variance_ratio_, "\n\n"
        # print "pca.singular_values_", pca.singular_values_, "\n\n"
        # print "pca.score", pca.score(self.xs), "\n\n"
        # print "pca.score_samples", len(pca.score_samples(self.train_xs)), pca.score_samples(self.train_xs), "\n\n"
        # print "pca.transform", len(pca.transform(self.train_xs)), pca.transform(self.train_xs), "\n\n"
        eigenvectors = pca.components_
        eigenvalues = pca.explained_variance_
        # print "eigenvectors", eigenvectors
        # print "eigenvalues", eigenvalues, "\n\n"
        # accuracy = self.test_accuracy(pca, debug)
        # return pca, accuracy
        return pca.fit_transform(X)


    def ica(self, n_components=3):
        """
        >>> transformer = FastICA(n_components=7,
        ...         random_state=0)
        >>> X_transformed = transformer.fit_transform(X)
        >>> X_transformed.shape
        (1797, 7)
        """
        ica = FastICA(n_components=n_components)
        X_trans = ica.fit_transform(self.xs)
        print ica.components_
        return X_trans
        # print "X_transformed.shape", X_transformed.shape



    def rca(self,n_components=2):
        """
        Reduce dimensionality through Gaussian random projection
        The components of the random matrix are drawn from N(0, 1 / n_components).
        >>> X = np.random.rand(100, 10000)
        >>> transformer = random_projection.GaussianRandomProjection()
        >>> X_new = transformer.fit_transform(X)
        """
        rca = random_projection.GaussianRandomProjection(n_components=n_components)
        X_trans = rca.fit_transform(self.xs)
        return X_trans


    def lda(self,n_components=2):
        lda = LinearDiscriminantAnalysis(n_components=n_components)
        X_trans = lda.fit(self.xs, self.ys).transform(self.xs)
        return X_trans

    def var_threshold(self, percentage_reduction=0.8):
        feature_selection = VarianceThreshold(threshold=(percentage_reduction * (1 - percentage_reduction)))
        X_trans = feature_selection.fit_transform(self.xs)
        return X_trans



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

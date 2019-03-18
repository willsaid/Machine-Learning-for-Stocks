from SwingTrader import SwingTrader
from DayTrader import DayTrader
import Trader
from Candle import Candle
import analysis

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import scipy.stats as stats
import math
from matplotlib.colors import LogNorm
from sklearn import mixture
from numpy.testing import assert_array_almost_equal

"""
Unsupervised Learning Analysis
"""

def nerual_after_em():
    """
    use em clustering to reduce data,
    then rerun neural net
    """
    pass

def neural_after_kmeans():
    """
    use kmeans clustering to reduce data,
    then rerun neural net

    day trader
    """
    s = SwingTrader(excluding_last=252 * 11)
    kmeans = s.k_means_clustering(n_clusters=2, X=s.train_xs)
    kmeans_test = s.k_means_clustering(n_clusters=2, X=s.test_xs)

    trainx = kmeans.labels_
    testx = kmeans_test.labels_

    trainx = [[x] for x in trainx]
    testx = [[x] for x in testx]

    # print labels
    # print s.test_ys
    # print trainx
    # print testx
    print np.array(trainx).shape
    print np.array(s.train_xs).shape
    #
    # print np.array(trainx).shape
    # print np.array(s.train_ys).shape
    # # print s.train_xs
    # # print trainx
    # print 1/0
    # plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], marker='x', color='orange', s=40000)
    # plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='orange')
    # plt.show()

    # acc = analysis.day_trials(X=trainx, testX=testx)[0]
    acc = analysis.swing_several_sims(s=s, X=trainx, testX=testx)[0]
    print acc


def neural_swing_pca():
    s = SwingTrader(excluding_last=252 * 11)
    r = range(1, 6)
    accuracies = []
    for i in r:
        X = s.pca(n_components=i, X=s.train_xs)
        testX = s.pca(n_components=i, X=s.test_xs)
        acc = analysis.swing_several_sims(s=s, X=X, testX=testX)[0]
        accuracies.append(acc)

    title = 'Neural Net After PCA for Swing Trading'
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlabel('Num of Components')
    ax.set_ylabel('Accuracy')
    plt.plot(r, accuracies)
    plt.show()


def neural_with_pca():
    """
    old results

    in paper: an average accuracy of 46.28% and daily returns of 1.318%.
    takes about 15 sec to run
    >>>     analysis.day_trials()
    average accuracies:
    0.4574418604651163
    avg daily returns:
    1.3216907197464711 %

    in paper: 53.7% accuracy and 1.23% in daily returns on avg
    takes about 2 min to run
    >>>     analysis.swing_several_sims()
    # Avg accuracies and money:
    0.4830140946873871
    3.0788267587245786
    """
    # analysis.day_trials()
    # analysis.swing_several_sims()
    s = DayTrader()
    r = range(1, 21)
    accuracies = []
    for i in r:
        acc = 0
        for _ in range(2):
            X = s.pca(n_components=i, X=s.train_xs)
            testX = s.pca(n_components=i, X=s.test_xs)
            acc += analysis.day_trials(X=X, testX=testX)[0]
        accuracies.append(acc / 2.)

    title = 'Neural Net After PCA for Day Trading'
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlabel('Num of Components')
    ax.set_ylabel('Accuracy')
    plt.plot(r, accuracies)
    plt.show()

    # analysis.day_trials(X=s.train_xs, testX=s.test_xs)

def plot_kmeans(s, X_trans, day, n_clusters):
    title = 'K-Means after RP for Day Trading' if day else 'K-Means after RP for Swing Trading'
    fig, ax = plt.subplots()
    plt.title(title)
    ax.set_xlabel('First Price')
    ax.set_ylabel('Second Price')

    kmeans = s.k_means_clustering(n_clusters=n_clusters, X=X_trans)
    plt.scatter(X_trans[:,0],X_trans[:,1], c=kmeans.labels_, cmap='rainbow')
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], marker='x', color='orange', s=40000)
    plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='orange')
    # plt.legend(markerscale=3)
    plt.show()

def kmeans_nodimred():
    s = DayTrader()
    # X_trans = s.pca(n_components=3)
    fig, ax = plt.subplots()
    plot_kmeans(s, s.xs, True, 3)


def kmeans_pca():
    s = SwingTrader()
    X_trans = s.pca(n_components=3)
    fig, ax = plt.subplots()
    plot_kmeans(s, X_trans, True)

def kmeans_ica():
    s = DayTrader()
    X_trans = s.ica(n_components=4)
    plot_kmeans(s, X_trans, False, 4)

def kmeans_rca():
    s = DayTrader()
    X_trans = s.rca(n_components=2)
    plot_kmeans(s, X_trans,True, 2)





def var_threshold(s, percent):
    X_trans = s.var_threshold(percentage_reduction=percent)
    fig, ax = plt.subplots()
    plt.title('Swing Trading with Variance Threshold of ' + str(percent * 100) + "%")
    ax.set_xlabel('First Price')
    ax.set_ylabel('Second Price')
    print X_trans[0]
    # plt.plot(X_trans)
    plt.scatter(X_trans[:, 0], X_trans[:, 1])

def variance_threshold():
    s = SwingTrader()
    # var_threshold(s, 0.010000)
    var_threshold(s, 0.000250)
    var_threshold(s, 0.000100)
    var_threshold(s, 0.000025)
    var_threshold(s, 0.000001)

    plt.show()

    # var_threshold(s, 0.000050)
    # var_threshold(s, 0.000075)
    # var_threshold(s, 0.000010)
    # var_threshold(s, 0.0000001)

def lda():
    s = DayTrader()
    X_trans = s.lda()
    fig, ax = plt.subplots()
    plt.title('Random Projection Reduction to Two Days on Swing Trading')
    ax.set_xlabel('First Price')
    ax.set_ylabel('Second Price')
    print X_trans
    plt.plot(X_trans)
    # plt.scatter(X_trans[:, 0], X_trans[:, 1])
    plt.show()

def rca():
    s = SwingTrader()
    for i in range(10):
        X_trans = s.rca()
        fig, ax = plt.subplots()
        plt.title('Random Projection Reduction to Two Days on Swing Trading')
        ax.set_xlabel('First Price')
        ax.set_ylabel('Second Price')
        plt.scatter(X_trans[:, 0], X_trans[:, 1])
        plt.show()

def ica():
    X_trans = SwingTrader().ica()
    fig, ax = plt.subplots()
    plt.title('ICA: Dimensionality Reduction to only Past Two Days on Swing Trading')
    ax.set_xlabel('First Price')
    ax.set_ylabel('Second Price')
    plt.scatter(X_trans[:, 0], X_trans[:, 1])
    plt.show()

# s: trader
def pca_loss(s, n_components):
    pca = s.pca(n_components=n_components)
    X_train_pca = pca.fit_transform(s.xs)
    X_projected = pca.inverse_transform(X_train_pca)
    X_projected2 = X_train_pca.dot(pca.components_) + pca.mean_
    assert_array_almost_equal(X_projected, X_projected2)
    loss = ((s.xs - X_projected) ** 2).mean()
    print "loss", loss
    return loss

def pca_losses():
    s = SwingTrader()
    losses = []
    r = range(1,4)
    for i in r:
        losses.append(pca_loss(s,i))

    fig, ax = plt.subplots()
    ax.set_xlabel('Number of Features')
    ax.set_ylabel('Data Loss')
    plt.title('PCA: Reduction of Features and Data Loss for Swing Trader')
    plt.plot(r, losses)
    plt.show()

def pca():
    # d = DayTrader()
    # principalComponents = d.pca(n_components=2)

    s = SwingTrader()
    principalComponents = s.pca(n_components=2).transform(s.xs)

    fig, ax = plt.subplots()
    plt.title('PCA: Dimensionality Reduction to only Past Two Days on Swing Trading Data Set')
    ax.set_xlabel('First Price')
    ax.set_ylabel('Second Price')

    plt.scatter([x[0] for x in principalComponents], [x[1] for x in principalComponents])
    plt.show()


def test_k_means():
    # s = SwingTrader()
    # s.k_means_clustering(n_clusters=2)
    d = DayTrader()
    d.k_means_clustering(n_clusters=2)


def test_em():
    SwingTrader().expectation_max()
    DayTrader().expectation_max()



def scatter_sp_normal():
    diffs = SwingTrader().difference_from_yesterday

    # -10, -9.9, ... 9.9, 10
    # 200 total points.
    points = []
    for i in range(20):
        integer = i - 10 # like -9
        for dec in range(10):
            num = float(integer) + dec/10.0
            points.append(num)
    points.append(10.0)

    point_buckets = [0]*len(points)
    print len(points)
    for diff in diffs:
        percent_rounded = float(int(diff * 10)) / 10.
        index = int(percent_rounded * 10) + 100
        if index > 200:
            index = 200
        if index < 0:
            index = 0
        point_buckets[index] += 1


    stddev = np.std(point_buckets)
    mean = np.mean(diffs)

    print "mean:", mean
    print "variance", stddev

    point_buckets = [(x - mean) / stddev for x in point_buckets]
    # print point_buckets

    fig, ax = plt.subplots()
    plt.title('S&P 500 Distribution vs Normal')
    plt.scatter(points, point_buckets, label='S&P')
    ax.set_xlabel('Difference From Yesterday\'s Close, % to nearest tenth')
    ax.set_ylabel('Occurences Since 1950')

    x_axis = np.arange(-10, 10, 0.1)
    plt.plot(x_axis, stats.norm.pdf(x_axis,0,1), label='Normal Distribution')

    plt.gca().legend(('Normal','S&P'))

    # plt.legend(handles=[line1, line2])
    plt.show()





def gaussian():
    n_samples = 300

    # generate random sample, two components
    np.random.seed(0)

    # generate spherical data centered on (20, 20)
    shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-20., 30.)
    y = np.linspace(-20., 40.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                     levels=np.logspace(0, 3, 10))
    CB = plt.colorbar(CS, shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], .8)

    plt.title('Negative log-likelihood predicted by a GMM')
    plt.axis('tight')
    plt.show()










#

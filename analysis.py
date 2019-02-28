from SwingTrader import SwingTrader
from DayTrader import DayTrader
import Trader
from Candle import Candle

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

"""
analysis
"""

def export_data(swing=True):
    if swing:
        s = SwingTrader()
    else:
        s = DayTrader()
    lines = []
    for i in range(len(s.xs)):
        lines.append(s.xs[i] + [s.ys[i]])

    print "len is ", len(s.xs)

    with open('amd-out.txt', 'w') as f:
        for line in lines:
            for item in line:
                f.write("%s " % item)
            f.write("\n")

def export_day_trader():
    s = DayTrader()

    # limit to last 7, not last 20, data points
    trimmed_xs, trimmed_ys = [], []


    lines = []
    for i in range(len(s.xs)):
        trimmed_xs = s.xs[i][-5:]
        y = [s.ys[i]]
        lines.append(trimmed_xs + y)
        print trimmed_xs + y

    # print lines
    print "len is ", len(s.xs)

    with open('amd-out-trimmed.txt', 'w') as f:
        for line in lines:
            for item in line:
                f.write("%s " % item)
            f.write("\n")

def day_chart_datas():
    """
    chart how more data lowers error
    note its diff each time
    """
    df = pd.read_csv('amd.csv', index_col='timestamp', parse_dates=True)
    # 1949 / 16 = 120
    r = range(0, len(df), 120)[1:]
    train_errors = []
    test_errors = []
    for amount in r:
        d = DayTrader(df=df[:amount])
        clf, train_acc = d.decision_tree(debug=False)
        test_acc = day_simulations((clf, train_acc), d)
        train_errors.append(1.0 - train_acc)
        test_errors.append(1.0 - test_acc)

    print train_errors
    print test_errors

    plt.xticks(r)
    plt.title('Decision Trees: Amount of Data and Accuracy')
    plt.plot(r, train_errors, label="Training Error")
    plt.plot(r, test_errors, label="Testing Error")
    plt.legend()
    plt.show()


def day_svm_degrees():
    accs, moneys = [], []
    r = range(1,6)
    for degree in r:
        print '*******************************************\n' + str(degree) + '\n'
        acc, money = day_trials(outer_trials=3, inner_trials=3, degree=degree)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('SVM: Ideal Polynomial Degree')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def day_ideal_k():
    accs, moneys = [], []
    r = range(1,15)
    for neighbors in r:
        print '*******************************************\n' + str(neighbors) + '\n'
        acc, money = day_trials(outer_trials=3, inner_trials=5, neighbors=neighbors)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('Day Trading KNN: Ideal Neighbors')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def day_ideal_prune_boosted():
    accs, moneys = [], []
    r = range(0,100, 2)
    for val in r:
        print '*******************************************\n' + str(val) + '\n'
        acc, money = day_trials(outer_trials=2, inner_trials=5, prune_val=val)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('Ideal Prune Value for Boosted Decision Trees')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def day_svm_kernels():
    for kernel in ['sigmoid', 'rbf', 'poly', 'linear']:
        print '*******************************************\n' + kernel + '\n'
        day_trials(outer_trials=10, inner_trials=10, kernel=kernel)

def ideal_prune():
    """ around 3 to 5"""
    accs = []
    d = DayTrader()
    r = range(0, 35)
    for pruneval in r:
        clf, train_acc = d.decision_tree(debug=False, prune=pruneval)
        test_acc = day_simulations((clf, train_acc), d)
        accs.append(1.0 - test_acc)
    plt.xticks(r)
    plt.title('Decision Trees: Ideal Prune Value')
    plt.plot(r, accs, label="Error")
    plt.legend()
    plt.show()

def day_trials(outer_trials=50, inner_trials=3, kernel='poly', degree=5, neighbors=5, prune_val=5):
    allmoneys=[]
    allboosts=[]
    for _ in range(outer_trials):
        d = DayTrader()
        boosts, moneys = [], []
        for _ in range(0, inner_trials):
            print "*******BOOSTING"
            # acc, money = day_simulations(d.svm(debug=False, kernel=kernel, degree=degree), d, debug=False)
            # acc, money = day_simulations(d.knn(debug=False, neighbors=neighbors), d, debug=False)
            acc, money = day_simulations(d.boost(debug=False, prune_val=prune_val), d, debug=False)
            boosts.append(acc)
            moneys.append(money)
        # print "*************\n\naverage accuracies for :"
        # print np.mean(boosts)
        # print "avg money:", np.mean(moneys)
        # print "avg daily returns:"
        dr = (abs(np.mean(moneys)) ** (1.0 / len(d.testingX)) - 1) * 100
        if np.mean(moneys) < 0: dr *= -1
        allmoneys.append(dr)
        allboosts.append(np.mean(boosts))
        # print dr,"%"
    print "*************\n\naverage accuracies for :"
    print np.mean(allboosts)
    print "avg daily returns:"
    print np.mean(allmoneys),"%"
    return np.mean(allboosts), np.mean(allmoneys)

def baseline_daily_returns():
    """daily returns of stock on avg, like if i were to just hold it"""
    days = 5 # easier to just harcode it here
    df = pd.read_csv('amd.csv', index_col='timestamp', parse_dates=True)
    candles = Candle.dayCandlesFromDataframe(df)
    percent_gain = (candles[-1].close - candles[0].open) / candles[0].open - 1
    dr = percent_gain**(1.0/days) if percent_gain > 0 else -1*(percent_gain*-1)**(1.0/days)
    print "Daily Returns of this STOCK:\n", dr, "%"

def day_simulations(clf_acc, daytrader, debug=True):
    clf, trainingAcc = clf_acc
    correct=0
    total=0
    y_true, y_pred = daytrader.testingY, clf.predict(daytrader.testingX)
    accuracy = Trader.reports(y_true, y_pred, debug=debug)

    money = 1.0
    for i in range(len(daytrader.testingX)):
        gain = daytrader.testingGains[i]
        prediction = clf.predict([daytrader.testingX[i]])
        if prediction == 1:
            money += gain
        else:
            money -= gain

    # print "money is ", money
    # print "num trials", len(daytrader.testingGains)
    return accuracy, money




"""
********************************************************

SWING TRADER

********************************************************
"""
def baseline_swing_daily_returns():
    """daily returns of market on avg, like if i were to just hold it"""
    days = 252
    df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True).tail(days)
    candles = Candle.swingCandlesFromDataframe(df)
    percent_gain = (candles[-1].close - candles[0].open) / candles[0].open - 1
    print candles[-1].close
    print candles[-0].open
    print percent_gain
    dr = percent_gain**(1.0/days) if percent_gain > 0 else -1*(percent_gain*-1)**(1.0/days)
    print "Swing Baseline Daily Returns:\n", dr, "%"

def swing_nn_iterations():
    accs, moneys = [], []
    r = range(50,600,50)
    for iterations in r:
        print '*******************************************\n' + str(iterations) + '\n'
        acc, money = swing_several_sims(outer_trials=2, inner_trials=3, iterations=iterations)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('Swing Trading SVM: Ideal Polynomial Degree')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def swing_svm_degrees():
    accs, moneys = [], []
    r = range(1,6)
    for degree in r:
        print '*******************************************\n' + str(degree) + '\n'
        acc, money = swing_several_sims(outer_trials=2, inner_trials=2, degree=degree)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('Swing Trading SVM: Ideal Polynomial Degree')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def swing_ideal_k():
    accs, moneys = [], []
    r = range(1,8)
    for k in r:
        print '*******************************************\n' + str(k) + '\n'
        acc, money = swing_several_sims(outer_trials=2, inner_trials=2, neighbors=k)
        money -= 1.0 # both are like 0.44, 0.55 now
        accs.append(1.0 - acc)
        moneys.append(money)
    plt.xticks(r)
    plt.title('Swing Trading KNN: Ideal Neighbors')
    plt.plot(r, accs, label="Error")
    plt.plot(r, moneys, label="Average Daily Returns")
    plt.legend()
    plt.show()

def svm_kernels():
    for kernel in ['sigmoid', 'rbf', 'poly', 'linear']:
        print '*******************************************\n' + kernel + '\n'
        swing_several_sims(outer_trials=2, inner_trials=2, kernel=kernel)

def swing_chart_datas():
    """
    chart how more data lowers error
    """
    df = pd.read_csv('^GSPC.csv', index_col='Date', parse_dates=True)
    r = range(0, len(df), 600)[1:]
    train_errors = []
    test_errors = []
    for amount in r:
        s = SwingTrader(df=df[:amount])
        clf, train_acc = s.decision_tree(debug=False)
        candles = s.last_remaining()
        test_acc, money = swing_simulation_for_2018(s.decision_tree()[0], candles=s.last_remaining())
        train_errors.append(1.0 - train_acc)
        test_errors.append(1.0 - test_acc)

    print train_errors
    print test_errors

    plt.xticks(r)
    plt.title('Decision Trees: Amount of Data and Accuracy')
    plt.plot(r, train_errors, label="Training Error")
    plt.plot(r, test_errors, label="Testing Error")
    plt.legend()
    plt.show()


def swing_several_sims(past_days=5, outer_trials=2, inner_trials=20, kernel='poly', degree=5, neighbors=2, iterations=200):
    days = 252 * 11
    allaccs, allmoneys = [], []
    for _ in range(outer_trials):
        s = SwingTrader(past_days=past_days, excluding_last=days)
        accs, moneys = 0, 0
        for _ in range(inner_trials):
            boostAcc, boostMoney = swing_simulation_for_2018(s.knn(debug=False, neighbors=neighbors)[0], past_days=past_days, candles=s.last_remaining())
            # boostAcc, boostMoney = swing_simulation_for_2018(s.neural_network(debug=False, max_iter=iterations)[0], past_days=past_days, candles=s.last_remaining())
            accs += boostAcc
            moneys += boostMoney
        # print "\n\navg acc", accs / float(inner_trials)
        # print "avg money", moneys / float(outer_trials)
        allaccs.append(accs / float(inner_trials))
        allmoneys.append(moneys / float(outer_trials))
    print "*********\n\nAvg accuracies and money:"
    print np.mean(allaccs)
    print np.mean(allmoneys)
    print "daily returns:"
    mean = np.mean(allmoneys)
    if mean < 0: mean *= -1
    dr = mean**(1./days) - 1
    percent = dr * 100
    print percent, "%"
    return np.mean(allaccs), np.mean(allmoneys)

def optimal_days_to_swing():
    """
    Turns out 4 day trends have the highest accuracy for swinging
    4 day trends avg 30% returns
    """
    acc2s, acc3s = [], []
    moneys2, moneys3 = [],[]

    r = range(1, 10)
    trials = 3

    # avg of 3 instances
    for days in r:
        print "*******DAYS: ", days
        s = SwingTrader(past_days=days, excluding_last=252)
        avgBoost, avgNN = 0,0
        boostmons, nnmons = 0,0
        for trial in range(0, trials):
            boostAcc, boostMoney = simulation_for_2018(s.boost()[0], past_days=days, candles=s.last_remaining())
            nnAcc, nnMoney = simulation_for_2018(s.neural_network()[0], past_days=days, candles=s.last_remaining())
            avgBoost += boostAcc
            avgNN += nnAcc
            boostmons += boostMoney
            nnmons += nnMoney
        acc2s.append(avgBoost / float(trials))
        acc3s.append(avgNN / float(trials))
        moneys2.append(boostmons / float(trials))
        moneys3.append(nnmons / float(trials))

    print "Results"
    print acc2s, acc3s, np.mean(acc2s), np.mean(acc3s), moneys2, moneys3, np.mean(moneys2), np.mean(moneys3)

    plt.xticks(r)
    plt.title('Optimal Days to Swing')
    plt.plot(r, acc2s, label="boost")
    plt.plot(r, acc3s, label="nn")
    plt.plot(r, moneys2, label="boost-money")
    plt.plot(r, moneys3, label="nn-money")
    plt.legend()
    plt.show()


def swing_simulation_for_2018(clf=None, past_days=5, candles=None):
    """
    How much money would the swing classifier make in 2018, trained only on 1950-2017?

    I'll buy at the close and sell at open on what i think are GREEN days,
    and short at close on what I think will be RED days.

    returns (accuracy, money)
    """
    money = 1 # 1 dollar
    candles2018 = candles
    if clf == None:
        s = SwingTrader(past_days=past_days, excluding_last=252) # doesnt include last year at all
        candles2018 = s.last_remaining()
        clf = s.boost()[0]

    correct=0
    total=0

    y_true = []
    y_pred = []

    for candleIndex in range(past_days - 1, len(candles2018) - 1):

        past_X_candles = candles2018[candleIndex - past_days + 1 : candleIndex + 1]

        # THIS IS JUST CLOSE
        past_X_days = [[candle.close for candle in past_X_candles]]
        # THIS IS OPEN, CLOSE, HIGH, LOW only for GSPC!!!!!!!
        # past_X_days = [[candle.open, candle.high, candle.low, candle.close] for candle in past_X_candles]

        first_day = float(past_X_days[0][0]) # for normalization
        normalized_xs = [[x/first_day for x in y] for y in past_X_days]

        # print past_X_days
        # print first_day
        # print normalized_xs
        # print "candles2018[0:10]", [x.close for x in candles2018[:10]]
        #
        # print "candleIndex", candleIndex
        #
        #
        #
        # print "pastXDays", past_X_days

        prediction = int(clf.predict(normalized_xs)[0]) # 1 is predicting Green, -1 is Red
        tmmrw_open = candles2018[candleIndex+1].open
        today_close = candles2018[candleIndex].close
        actual = 1 if tmmrw_open > today_close else -1
        y_pred.append(prediction)
        y_true.append(actual)
        diff = abs(tmmrw_open - today_close)/first_day

        # ONLY LONG STRATEGY
        if prediction == 1 and actual == 1:
            money += diff
        elif prediction == 1 and actual == 0:
            money -= diff

        if actual == prediction: correct += 1
        # LONG AND SHORT STRATEGY
        # if actual == prediction:
        #     money += diff
        #     correct += 1
        # else:
        #     money -= diff

        total+=1


    _ = Trader.reports(y_true, y_pred)

    if total > 0:
        accuracy = correct / float(total)
        print "Accuracy was {}".format(accuracy)
        print "money went from 1.00 to ", money

        return accuracy, money
    else:
        print "less than zero"
        return 0,0

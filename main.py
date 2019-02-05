from SwingTrader import SwingTrader
from DayTrader import DayTrader
import Trader

import warnings
import numpy as np
"""
usage: python main.py

implements the following 5 machine learning algos on the SwingTrading and DayTrading datasets:
1. Decision Trees with pruning
2. Neural Networks
3. Boosting
4. Support Vector Machines
5. K-nearest Neighbors
"""
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

if __name__ == '__main__':
    print "SWING TRADER: WILL MARKET OPEN UP OR DOWN?"
    s = SwingTrader()
    s.decision_tree()
    s.boost()

    # SwingTrader().svm()
    # SwingTrader().decision_tree()
    # SwingTrader().random_forest()
    # SwingTrader().knn()
    # SwingTrader().boost()
    # SwingTrader().neural_network()
    # print "\n\n\n\n\n\nDAY TRADER: AMD NEXT 15 MIN"
    # DayTrader().svm()
    # DayTrader().decision_tree()
    # DayTrader().random_forest()
    # DayTrader().knn()
    # DayTrader().boost()
    # DayTrader().neural_network()

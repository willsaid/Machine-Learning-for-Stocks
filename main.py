import SwingTrading
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
    SwingTrading.SwingTrading()

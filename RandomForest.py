import numpy as np
import pandas as pd
import random as rand
import seaborn as sns
import time


class Random_Forest:
    
    def __init__(self, x_train, y_train, x_test, y_test, 
                 num_features = None, 
                 num_exemplars = None, 
                 n_estimators = 10,
                 min_leaf=1, 
                 max_depth=10):
        
        # init data
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test 
        
        # init hyperparameters
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.min_leaf = int(min_leaf)
        
        if num_features == None:
            num_features = int(x_train.shape[1])
        if num_exemplars == None:
            num_exemplars = x_train.shape[0]
        
        self.num_features = np.random.randint(0, num_features, num_features)
        self.num_exemplars = num_exemplars
        
    def MSE(self, y_pred, y_train):
        return np.sum((y_pred - y_train) ** 2) / (2 * y_train.shape[0])
    
    def MAE(self, y_pred, y_test):
        return (np.mean(np.abs(y_pred-y_test)))

    def fit(self, x, y):
        
        self.trees = [] # create tree for every estimator
        for n in range(self.n_estimators):
            rand_index = np.random.randint(y.shape[0], size = self.num_exemplars)
            self.trees.append(Decision_Tree(x.iloc[rand_index],
                                            y.iloc[rand_index],
                                            self.num_features, 
                                            self.min_leaf, 
                                            self.max_depth))
        
        
    def predict(self, x_test):
        
        y_trees = []
        
        # use every tree to make predictions
        for tree in self.trees:
            y_trees.append(tree.predict(x_test))
            
        # average every decision tree prediction for final y_pred
        y_pred = np.mean(y_trees, axis = 0)
        
        # calc mean square error
        MSE = self.MSE(y_pred, self.y_test)
        
        return y_pred, MSE

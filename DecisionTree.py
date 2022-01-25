
import numpy as np
import pandas as pd
import random as rand
import seaborn as sns
import time


class Decision_Tree:
    
    def __init__(self, x, y, 
                 num_features, 
                 min_leaf, 
                 max_depth):
        
        # init data
        self.x = x
        self.y = y
        self.num_rows = x.shape[0]
        
        # init hyperparameters
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        
        # data charactersitics
        self.num_features = num_features 
        self.val = np.mean(y.values[:])
        
        self.score = np.inf
        self.find_split()

        
    def std(self, count, ssum, squared_sum):
        return np.abs( (squared_sum / count) - (ssum / count)**2 )
    
    def find_split(self):
        
        for feature in self.num_features:
            self.check_split(feature)
        
        
        if self.score == np.inf or self.max_depth <= 0:
            return

        x = self.x.values[:, self.split_feature]

        left = np.nonzero(x <= self.split_val)[0]
        right = np.nonzero(x > self.split_val)[0]

        self.left_tree = Decision_Tree(self.x.iloc[left], self.y.iloc[left], self.num_features, self.min_leaf, self.max_depth - 1)
        self.right_tree = Decision_Tree(self.x.iloc[right], self.y.iloc[right], self.num_features, self.min_leaf, self.max_depth - 1)

    def check_split(self, feature):
        
        # sort data
        x = self.x.values[:, feature][np.argsort(self.x.values[:, feature])]
        y = self.y.values[:][np.argsort(self.x.values[:, feature])]
        
        # set counters
        left_count = 0
        left_sum = 0.0
        left_squared_sum = 0.0
        right_count = self.num_rows
        right_sum = np.sum(y)
        right_squared_sum = np.sum(y ** 2)
        
        # loop through data
        for i in range(0, self.num_rows - self.min_leaf):
            
            left_count += 1
            left_sum += y[i]
            left_squared_sum += y[i] ** 2
            
            right_count -= 1            
            right_sum -= y[i]
            right_squared_sum -= y[i] ** 2
            
            # use variance or std as impurity metric
            left_std = self.std(left_count, left_sum, left_squared_sum)
            right_std = self.std(right_count, right_sum, right_squared_sum)
            score = left_std * left_count + right_std * right_count

            if score < self.score:
                self.split_feature = feature
                self.score = score
                self.split_val = x[i]

    def predict_one(self, x):
        
        if self.score == np.inf or self.max_depth <= 0:
            return self.val

        if x[self.split_feature] <= self.split_val:
            return self.left_tree.predict_one(x)
        else:
            return self.right_tree.predict_one(x)
        
    def predict(self, x):
        
        vals = []
        for i in x.values:
            vals.append(self.predict_one(i))
        
        return np.array(vals)

    

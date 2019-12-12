#!/usr/bin/env python3
import gc
import warnings
import gc, sys, re, os, math
from time import strptime, mktime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
np.random.seed(1)
gc.collect()

def trainPCA(X_train, X_test):
    if X_train.shape[0] < X_train.shape[1]:
        if X_train.shape[0] < X_test.shape[0]:
            n_components=X_train.shape[0]
        else:
            n_components=X_test.shape[0]
    else:
        if X_train.shape[1] < X_test.shape[1]:
            n_components=X_train.shape[1]
        else:
            n_components=X_test.shape[1]
    
    X_PCA = PCA(n_components=n_components)
    
    X_train_PCA = X_PCA.fit_transform(X_train)
    X_train_variance = X_PCA.fit(X_train)
    print(f'X_train Variance: {sum(X_train_variance.explained_variance_ratio_)}')
    print(f'X_train Shape: {X_train.shape}')
    X_train_df = pd.DataFrame(X_train_PCA)
    
    X_test_PCA = X_PCA.fit_transform(X_test)
    X_test_variance = X_PCA.fit(X_test)
    print(f'X_train Variance: {sum(X_test_variance.explained_variance_ratio_)}')
    print(f'X_train Shape: {X_test_PCA.shape}')
    X_test_df = pd.DataFrame(X_test_PCA)
    
    return X_train_df, X_test_df
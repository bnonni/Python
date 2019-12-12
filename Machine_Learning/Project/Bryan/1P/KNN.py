#!/usr/bin/env python3
import gc
import warnings
import gc, sys, re, os, math
from time import strptime, mktime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from impyute.imputation.cs import fast_knn
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
warnings.filterwarnings('ignore')
np.random.seed(1)
gc.collect()
# %matplotlib inline
from AccuracyMetrics import *

def testK(X_train, y_train):
    # KNN to obtain optimal k for y_test1 values
    neighbors = np.arange(1,20)
    train_accuracy = np.empty(len(neighbors))
    for i,k in enumerate(neighbors):
        knn_ytest = KNeighborsClassifier(n_neighbors=k)
        knn_ytest.fit(X_train, y_train)
        train_accuracy[i] = knn_ytest.score(X_train, y_train)
    return neighbors, train_accuracy

def plotTestK(neighbors, train_accuracy):
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.xticks(np.arange(0, neighbors[-1], step=1))
    plt.ylabel('Accuracy')
    plt.title('KNN Varying Number of Neighbors')
    plt.show()
    
    
def optimizeK(X_train, y_train, X_test, y_test):
    neighbors = np.arange(1,20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i,k in enumerate(neighbors):
     knn = KNeighborsClassifier(n_neighbors=k)
     knn.fit(X_train, y_train)
     train_accuracy[i] = knn.score(X_train, y_train)
     test_accuracy[i] = knn.score(X_test, y_test)
    return neighbors, test_accuracy, train_accuracy

def plotK(neighbors, test_accuracy, train_accuracy):
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.xticks(np.arange(0, neighbors[-1], step=1))
    plt.ylabel('Accuracy')
    plt.title('KNN Varying Number of Neighbors')
    plt.show() 

def getTestLabelsKNN(n, X_train, y_train, X_test):
    knn_ytest = KNeighborsClassifier(n_neighbors=n)
    knn_ytest.fit(X_train, y_train)
    y_test = knn_ytest.predict(X_test)
    return y_test

def runKNN(n, X_train, y_train, X_test, y_test):
    n_neighbors=n
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    return score, y_pred

def getScoresKNN(n, y_test, y_pred):
    knn_cm, knn_cr, knn_acc = getAccuracyMetrics(y_test, y_pred)
    print(f'KNN Accuracy Score, k={n}: {knn_acc}\n')
    print(f'KNN Confusion Matrix, k={n}\n\n{knn_cm}')
    print(f'\n\t\t  KNN Classification Report, k={n}\n\n{knn_cr}')
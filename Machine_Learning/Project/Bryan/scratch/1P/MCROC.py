#!/usr/bin/env python3
import gc
import warnings
import gc, sys, re, os, math
from time import strptime, mktime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import *
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score

def calcMultiClassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
        labels = kwargs['labels']
        
    y_train_bin = label_binarize(y_train, classes=labels)
    n_classes = y_train_bin.shape[1]
    y_test_bin = label_binarize(y_test, classes=labels)
# SVC(kernel='linear', probability=True, random_state=0)
    clf = OneVsRestClassifier(AdaBoostClassifier())
    y_score = clf.fit(X_train, y_train_bin).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],  label=f'{model} Micro-Avg Area: {round(roc_auc["micro"], 4)}')
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'{model} Area: {round(roc_auc[i], 4)}')
        if tuner_val == None:
          plt.title(f'{model} ROC Curve, Label {labels[i]}')
        else:
          plt.title(f'{model} ROC Curve, Label {labels[i]}, {tuner}={tuner_val}')
        plt.plot([0, 1], [0, 1], label='tpr-fpr line')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.show()
    
        
        

"""
def calcMultiClassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
        label_len = kwargs['label_len']
        labels = kwargs['labels']
        
    y_bin = label_binarize(y_test, classes=labels)

    clf = OneVsRestClassifier(OneVsRestClassifier(SVC(kernel='linear', probability=True, random_state=0)))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(label_len):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_score)
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])



    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label=f'{model} micro-average ROC curve (area = {roc_auc["micro"]})')
    for i in range(label_len):
        if (tuner == '') or (tuner_val == None):
            plt.plot(fpr[i], tpr[i], label=f'{model} ROC curve, Label {i} (area = {roc_auc[i]})')
        else:
            plt.plot(fpr[i], tpr[i], label=f'{model} ROC curve, Label {i},  {tuner}={tuner_val}, Label {i} (area = {roc_auc[i]})')
        plt.plot([0, 1], [0, 1], label='tpr-fpr line')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        plt.legend()
        plt.show()
 """
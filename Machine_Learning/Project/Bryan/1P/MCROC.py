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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score

def calcMultiClassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
        label_len = kwargs['label_len']
        
    labels = np.arange(1, label_len+1)
    y_bin = label_binarize(y_test, classes=labels)

    clf = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=0))
    y_score = clf.fit(X_train, y_train).decision_function(X_test)
    y_pred = clf.predict(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(label_len):
        fpr[i+1], tpr[i+1], thresholds = roc_curve(y_bin[:, i], y_pred)
        roc_auc[i+1] = auc(fpr[i+1], tpr[i+1])
    
    for i in range(label_len):
        if math.isnan(roc_auc[i+1]):
            score = 0
        else: 
            score = round(roc_auc[i+1],2)
        if (tuner == '') or (tuner_val == None):
            plt.plot(fpr[i+1], tpr[i+1], label=f'{model}')
        else:
            plt.plot(fpr[i+1], tpr[i+1], label=f'{model}, {tuner}={tuner_val}')
        plt.plot([0, 1], [0, 1], label='tpr-fpr line')
        plt.xlabel('fpr')
        plt.ylabel('tpr')
        if dec == True:
            plt.title(f'{model} ROC Curve Label {i+1}, ROC_Score={score}')
            tuner_val -= 1
        else:
            plt.title(f'{model} ROC Curve Label {i+1}, ROC_Score={score}')
        plt.legend()
        plt.show()
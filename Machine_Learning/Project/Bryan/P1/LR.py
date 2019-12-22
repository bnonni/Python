#!/usr/bin/env python3
import gc
import warnings
import gc, sys, re, os, math
from time import strptime, mktime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
gc.collect()
warnings.filterwarnings('ignore')
np.random.seed(1)
# %matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
import operator

def runLogisticRegression(e, X_train, y_train, X_test, y_test):    
    c = np.arange(1, e+1)
    cma = {}
    cra = {}
    acc = {}
    preds = {}
    for i in c:    
        lr = LogisticRegression(C=i, multi_class='ovr', solver='lbfgs',random_state=0)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cma[i] = confusion_matrix(y_test, y_pred)
        cra[i] = classification_report(y_test, y_pred)
        acc[i] = (accuracy_score(y_test, y_pred))
        preds[i] = y_pred
    ky = max(acc.items(), key=operator.itemgetter(1))[0]
    val = float(max(acc.items(), key=operator.itemgetter(1))[1])
    return cma, cra, acc, preds, ky, val
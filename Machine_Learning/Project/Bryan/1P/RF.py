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
from sklearn.ensemble import RandomForestClassifier
from MCROC import *
from AccuracyMetrics import *

def runRandomForest(X_train, y_train, X_test, y_test, ll):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    rfc_cm, rfc_cr, rfc_acc = getAccuracyMetrics(y_test, rfc_pred)
    print(f'Random Forest Accuracy Score: {rfc_acc}\n')
    print(f'Random Forest Confusion Matrix\n\n{rfc_cm}\n')
    pd.crosstab(y_test, rfc_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(f'\n\t\tRandom Forest Classification Report\n\n{rfc_cr}')
    calcMultiClassROCAUC(X_train, y_train, X_test, rfc_pred, model='RF', tuner='', tuner_val=None, label_len=ll, dec=False)
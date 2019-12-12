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
from sklearn.svm import *
from MCROC import *
from AccuracyMetrics import *


def runSVM(kernel, ll, X_train, y_train, X_test, y_test):
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    svm_y_pred = svm.predict(X_test)
    svm_cm, svm_cr, svm_acc = getAccuracyMetrics(y_test, svm_y_pred)
    print(f'SVM {kernel} Kernel Accuracy Score: {svm_acc}\n')
    print(f'SVM {kernel} Kernel Confusion Matrix\n\n{svm_cm}\n')
    pd.crosstab(y_test, svm_y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(f'\n\t\t    SVM {kernel} Kernel Classification Report\n\n{svm_cr}')
    calcMultiClassROCAUC(X_train, y_train, X_test, svm_y_pred, model='SVM', tuner='kernel', tuner_val=kernel, label_len=ll, dec=False)
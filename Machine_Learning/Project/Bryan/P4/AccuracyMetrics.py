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

def getAccuracyMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cra = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return cm, cra, acc
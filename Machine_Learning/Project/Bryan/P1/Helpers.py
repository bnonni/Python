#!/usr/bin/env python3
from collections import Counter
import numpy as np
import pandas as pd
from impyute.imputation.cs import fast_knn

def countUniqueValues(data):
    keys = Counter(data).keys()
    vals = Counter(data).values()
    labels=[]
    for i in keys:
      labels.append(i)
    print(f'Keys: {keys}\nValues: {vals}\nLabels: {labels}')
    return keys, vals, labels

def printShapes(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Shape: {value}\n')
        
def printHeads(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Head: {value}\n')
        
def printTails(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Tail: {value}\n')
        
def replaceWithNan(df, r, c):
    n = df.iloc[r][c]
    for i in df.keys():
        df[i] = df[i].replace(n, np.nan)
    print(f'Bad Values {n} replaced with NaN.')
    return df

def runFastKNN(df, k):
    print(f'Running fast_knn, k={k}.')
    df = fast_knn(df.values, k=k)
    df = pd.DataFrame(df)
    return df
#!/usr/bin/env python3
from collections import Counter
import numpy as np
import pandas as pd
from impyute.imputation.cs import fast_knn

def countUniqueValues(data):
    print(f'Values: {Counter(data).keys()}\nKeys: {Counter(data).values()}')

def printShapes(**kwargs):
    for key, value in kwargs.items():
        print(f'\n{key} Shape: {value}')
        
def printHeads(**kwargs):
    for key, value in kwargs.items():
        print(f'\n{key} Head: {value}\n')
        
def printTails(**kwargs):
    for key, value in kwargs.items():
        print(f'\n{key} Tail: {value}')
        
def replaceWithNan(df, r, c):
    n = df.iloc[r][c]
    for i in df.keys():
        df[i] = df[i].replace(n, np.nan)
    print(f'Bad Values {n} replaced with NaN. Run')
    return df

def runFastKNN(df, k):
    print(f'Running fast_knn, k={k}.')
    df = fast_knn(df.values, k=k)
    df = pd.DataFrame(df)
    return df
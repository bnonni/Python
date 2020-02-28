#!/usr/bin/env python3
from collections import Counter
import numpy as np
import pandas as pd
from impyute.imputation.cs import fast_knn

def count_unique_values(data):
    keys = Counter(data).keys()
    vals = Counter(data).values()
    labels=[]
    for i in keys:
      labels.append(i)
    print(f'Keys: {keys}\nValues: {vals}\nLabels: {labels}')
    return keys, vals, labels

def print_shapes(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Shape: {value}\n')
        
def print_heads(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Head: {value}\n')
        
def print_tails(**kwargs):
    for key, value in kwargs.items():
        print(f'{key} Tail: {value}\n')
        
def replace_with_nan(df, row, column):
    n = df.iloc[row][column]
    for i in df.keys():
        df[i] = df[i].replace(n, np.nan)

    print(f'Bad Values {n} replaced with NaN.')

    return df

def run_fast_knn(df, k):
    print(f'Running fast_knn, k={k}.')

    df = fast_knn(df.values, k=k)
    df = pd.DataFrame(df)

    return df



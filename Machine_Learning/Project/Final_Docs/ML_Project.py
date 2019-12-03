#!/usr/bin/env python
# coding: utf-8

# In[361]:


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# Python magic
get_ipython().run_line_magic('matplotlib', 'inline')

# Base packages
import gc, sys, re, os
from time import strptime, mktime

# Data processing/preprocessing/modeling packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
from sklearn.preprocessing import *
# Modeling settings
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Testing & Validation packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score

# KNN
from sklearn.neighbors import KNeighborsClassifier
# Logistic Regression
from sklearn.linear_model import LogisticRegression
# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from pydotplus import *
# SVM
from sklearn.svm import SVC


# In[534]:


X_train1 = pd.read_csv('1/TrainData1.txt', delimiter='\s+', header=None)
X_train2 = pd.read_csv('1/TrainData2.txt', delimiter='\s+', header=None)
X_train3 = pd.read_csv('1/TrainData3.txt', delimiter='\s+', header=None)
X_train4 = pd.read_csv('1/TrainData4.txt', delimiter='\s+', header=None)
X_train5 = pd.read_csv('1/TrainData5.txt', delimiter='\s+', header=None)
X_train6 = pd.read_csv('1/TrainData6.txt', delimiter='\s+', header=None)


# In[535]:


y_train1 = pd.read_csv('1/TrainLabel1.txt', delimiter='\n', header=None)
y_train2 = pd.read_csv('1/TrainLabel2.txt', delimiter='\n', header=None)
y_train3 = pd.read_csv('1/TrainLabel3.txt', delimiter='\n', header=None)
y_train4 = pd.read_csv('1/TrainLabel4.txt', delimiter='\n', header=None)
y_train5 = pd.read_csv('1/TrainLabel5.txt', delimiter='\n', header=None)
y_train6 = pd.read_csv('1/TrainLabel6.txt', delimiter='\n', header=None)


# In[536]:


test1 = pd.read_csv('1/TestData1.txt', delimiter='\s+', header=None)
test2 = pd.read_csv('1/TestData2.txt', delimiter='\s+', header=None)
test3 = pd.read_csv('1/TestData3.txt', delimiter=',',   header=None)
test4 = pd.read_csv('1/TestData4.txt', delimiter='\s+', header=None)
test5 = pd.read_csv('1/TestData5.txt', delimiter='\s+', header=None)
test6 = pd.read_csv('1/TestData6.txt', delimiter='\s+', header=None)


# In[537]:


X_training = [X_train1, X_train2, X_train3, X_train4, X_train5, X_train6]
y_training = [y_train1, y_train2, y_train3, y_train4, y_train5, y_train6]
all_testing = [test1, test2, test3, test4, test5, test6]


# In[538]:


for i,x in enumerate(X_training):
    print(f'X_TrainData{i+1} Shape: {x.shape}')


# In[539]:


for i,y in enumerate(y_training):
    print(f'y_TrainData{i+1} Shape: {y.shape}')


# In[540]:


for j,y in enumerate(all_testing):
    print(f'TestData{j+1} Shape: {y.shape}')


# In[ ]:





# In[ ]:





# In[542]:


np.random.seed(42)
X = X_train1.values
y = y_train1[0].values


# In[545]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1, stratify=y)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[546]:


neighbors = np.arange(1,20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[547]:


for i,k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train, y_train)
    
    train_accuracy[i] = knn.score(X_train, y_train)
    
    test_accuracy[i] = knn.score(X_test, y_test)


# In[548]:


plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Varying Number of Neighbors')
plt.show()


# In[549]:


n_neighbors=6
knn = KNeighborsClassifier(n_neighbors)


# In[550]:


knn.fit(X_train, y_train)


# In[551]:


knn.score(X_test, y_test)


# In[552]:


y_pred = knn.predict(X_test)


# In[553]:


y_pred


# In[554]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[555]:


print(classification_report(y_test, y_pred))


# In[556]:


len(y_pred)


# In[557]:


y_test = pd.DataFrame(y_pred).sample(53)


# In[558]:


# # Logistic Regression
# log_reg = LogisticRegression(solver='lbfgs')
# log_reg.fit(X_train1, y_train1)


# In[ ]:





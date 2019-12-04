#!/usr/bin/env python3
# coding: utf-8

# ## 1. Multi-Class Classification:
# For the multiclass classification problem, there were six different datasets. Some of the datasets contain missing values. For example, TrainData1, TestData1 and TrainData3 contain some missing values (1.00000000000000e+99). Therefore, the first approach needs to handle the missing values for selecting the features. Then compare the accuracy on train dataset to find out which classifier gives best result for each dataset with cross validation to verify the accuracy based on test dataset.
# <center><div style='width:50%; height:50%'><img src='../images/Q1_table.jpeg'></div></center>
#
# Hint:
# * Missing Value Estimation
#     - (KNN method for imputation of the missing values)
# * Dimensionality Reduction
# * Use Several Classifiers/ Ensemble Method
#     - Logistic Regression (with different c values)
#     - Random Forest (with different estimator values)
#     - SVM (with different kernels)
#     - KNN (with k = 1,2,5,10,20)
#     - K (3,5,10) Fold Cross Validation
# * Performance Comparison
#     - Classification Accuracy, Precision, Recall, Sensitivity, Specificity
#     - AUC, ROC Curve

# In[517]:


import os
import re
import sys
import gc
from sklearn.svm import SVC
from pydotplus import *
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from impyute.imputation.cs import fast_knn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import *
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import *
import statistics as stat
import seaborn as sns
from matplotlib import offsetbox
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from time import strptime, mktime
import warnings
warnings.filterwarnings('ignore')
# Modeling settings
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
# In[353]:


def optimizeK(X_train, y_train, X_test, y_test):
    neighbors = np.arange(1, 20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i, k in enumerate(neighbors):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        train_accuracy[i] = knn.score(X_train, y_train)

        test_accuracy[i] = knn.score(X_test, y_test2)

    return neighbors, test_accuracy, train_accuracy


# In[361]:


def plotK(neighbors, test_accuracy, train_accuracy):
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.xticks(np.arange(0, neighbors[-1], step=1))
    plt.ylabel('Accuracy')
    plt.title('KNN Varying Number of Neighbors')
    plt.show()


# In[317]:


X_train2 = pd.read_csv('../data/1/TrainData2.txt',
                       delimiter='\s+', header=None)
X_train3 = pd.read_csv('../data/1/TrainData3.txt',
                       delimiter='\s+', header=None)
X_train4 = pd.read_csv('../data/1/TrainData4.txt',
                       delimiter='\s+', header=None)


# In[318]:


y_train2 = pd.read_csv('../data/1/TrainLabel2.txt',
                       delimiter='\n', header=None)
y_train3 = pd.read_csv('../data/1/TrainLabel3.txt',
                       delimiter='\n', header=None)
y_train4 = pd.read_csv('../data/1/TrainLabel4.txt',
                       delimiter='\n', header=None)


# In[319]:


X_test2 = pd.read_csv('../data/1/TestData2.txt', delimiter='\s+', header=None)
X_test3 = pd.read_csv('../data/1/TestData3.txt', delimiter=',',   header=None)
X_test4 = pd.read_csv('../data/1/TestData4.txt', delimiter='\s+', header=None)


# In[320]:


X_training = [X_train2, X_train3, X_train4]
y_training = [y_train2, y_train3, y_train4]
X_testing = [X_test2, X_test3, X_test4]


# In[321]:


for i, x in enumerate(X_training):
    print(f'X_TrainData{i+1} Shape: {x.shape}')


# In[322]:


for i, y in enumerate(y_training):
    print(f'y_TrainData{i+1} Shape: {y.shape}')


# In[323]:


for j, y in enumerate(X_testing):
    print(f'TestData{j+1} Shape: {y.shape}')


# # _Dataset 2_

# ### PCA for DS2

# In[324]:


X_train2.shape


# In[325]:


y_train2.shape


# In[326]:


X_test2.shape


# In[327]:


xTrain2PCA = PCA(n_components=74)
X_train2_pca = xTrain2PCA.fit_transform(X_train2)


# In[330]:


# 100 principle components can explain 99% of the data

X_train2_pca_var = xTrain2PCA.fit(X_train2)
print(sum(X_train2_pca_var.explained_variance_ratio_))
print(X_train2_pca.shape)


# In[332]:


# 74 principle components can explain 99% of the data
xTest2PCA = PCA(n_components=74)
X_test2_pca = xTest2PCA.fit_transform(X_test2)


# In[333]:


X_test2_pca_var = X_test_pca.fit(X_test2)
print(sum(X_test2_pca_var.explained_variance_ratio_))
print(X_test2_pca.shape)


# In[334]:


X_train2_components = pd.DataFrame(X_train2_pca)
X_train2_components.head(10)


# In[335]:


X_test2_components = pd.DataFrame(X_test2_pca)
X_test2_components.head(10)


# In[336]:


X_train2_components.shape


# In[337]:


y_train2.shape


# In[338]:


X_test2_components.shape


# ### KNN for DS2

# In[452]:


# KNN to obtain y_test1 values

knn = KNeighborsClassifier(n_neighbors=5)

knn = knn.fit(X_train2_components, y_train2)
y_test2_pred = knn.predict(X_test2_components)
print(y_test2_pred)


# In[453]:


y_test2 = y_test2_pred


# In[454]:


y_test2


# In[455]:


n, test_acc, train_acc = optimizeK(
    X_train2_components, y_train2, X_test2_components, y_test2)
plotK(n, test_acc, train_acc)


# In[456]:


n_neighbors = 3
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train2_components, y_train2)


# In[457]:


knn.score(X_test2_components, y_test2)


# In[458]:


knn_y_pred = knn.predict(X_test2_components)
knn_cm = confusion_matrix(y_test2, knn_y_pred)
print(knn_cm)


# In[459]:


pd.crosstab(y_test2, knn_y_pred, rownames=[
            'True'], colnames=['Predicted'], margins=True)


# In[460]:


knn_cr = classification_report(y_test2, knn_y_pred)
print(knn_cr)


# In[ ]:


# ytest2_bin = label_binarize(y_test2, classes=[0, 1, 2])
# roc_auc_score(y_test2, knn_y_pred_proba)


# In[509]:


# knn_y_pred_proba = knn.predict_proba(X_test2_components)[:,1]
# kypp = [knn_y_pred_proba[0]]


# In[498]:


# fpr, tpr, thresholds = roc_curve(y_test2, knn_y_pred_proba)


# In[499]:


# plt.plot([0,1],[0,1], 'k--')
# plt.plot(fpr, tpr, label='KNN')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title(f'KNN (n_neighbors={n_neighbors}) ROC Curve')
# plt.show()


# In[522]:


y_label = label_binarize(y_test2, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
n_classes = y_label.shape[1]


# In[524]:


classifier = OneVsRestClassifier(
    SVC(kernel='linear', probability=True, random_state=0))

# generate y_score
y_score = classifier.fit(
    X_train2_components, y_train2).decision_function(X_test2_components)
y_score.shape


# In[ ]:


fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test2[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

colors = cycle(['blue', 'red', 'green'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'''.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([-0.05, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic for multi-class data')
plt.legend(loc="lower right")
plt.show()
# roc_auc_score(y_test2, knn_y_pred_proba)


# In[ ]:


# In[ ]:


# In[ ]:


# ### Logistic Regression for DS2

# In[405]:


lr = LogisticRegression(random_state=0)
lr.fit(X_train2_components, y_train2)

lr_y_pred = lr.predict(X_test2_components)


# In[424]:


lr_cm = confusion_matrix(y_test2, lr_y_pred)
print(f'Logistic Regression Confusion Matrix\n\n{lr_cm}')


# In[435]:


print(' Logistic Regression Confusion Matrix\n ______________________________________',)
pd.crosstab(y_test2, knn_y_pred, rownames=[
            'True'], colnames=['Predicted'], margins=True)


# In[436]:


lr_cr = classification_report(y_test2, lr_y_pred)
print(f'             Logistic Regression Classification Report\n\n{lr_cr}')


# In[518]:


print('LR Accuracy:', accuracy_score(y_test2, lr_y_pred))


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# # Dataset 3

# In[68]:


# X_train3
X_train3.head()


# In[69]:


n = X_train3[12][0]

for i in X_train3.keys():
    X_train3[i] = X_train3[i].replace(n, np.nan)


# In[70]:


X_train3.head()


# In[ ]:


X_train3_clean = fast_knn(X_train1.values, k=100)


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[11]:


# In[12]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[ ]:


# In[17]:


# In[ ]:


# In[ ]:


# In[ ]:


# # Logistic Regression
# log_reg = LogisticRegression(solver='lbfgs')
# log_reg.fit(X_train1, y_train1)

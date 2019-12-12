#!/usr/bin/env python
# coding: utf-8

# ## 1. Multi-Class Classification:
# For the multiclass classification problem, there were six different datasets. Some of the datasets contain missing values. For example, TrainData1, TestData1 and TrainData3 contain some missing values (1.00000000000000e+99). Therefore, the first approach needs to handle the missing values for selecting the features. Then compare the accuracy on train dataset to find out which classifier gives best result for each dataset with cross validation to verify the accuracy based on test dataset.
# <center><div style='width:50%; height:50%'><img src='images/Q1_table.jpg'></div></center>
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

# In[346]:


import warnings
warnings.filterwarnings('ignore')

# Python magic
# %matplotlib inline

# Base packages
import gc, sys, re, os, math
from time import strptime, mktime

# Data processing/preprocessing/modeling packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.random.seed(1)

# Sklearn: KNN, SVM, LR, RF
from sklearn.neighbors import KNeighborsClassifier
from impyute.imputation.cs import fast_knn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import export_graphviz
from sklearn.svm import *
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import GridSearchCV
# DT packages
# from six import StringIO
# from IPython.display import Image
# from pydotplus import *


# In[347]:


def seperateClassifiers(X_test):
    classes = {}
    tmp = []
    for i in Y:
        tmp.append(i)
    


# In[348]:


X_train2 = pd.read_csv('data/1/TrainData2.txt', delimiter='\s+', header=None)
X_train3 = pd.read_csv('data/1/TrainData3.txt', delimiter='\s+', header=None)
X_train4 = pd.read_csv('data/1/TrainData4.txt', delimiter='\s+', header=None)


# In[349]:


y_train2 = pd.read_csv('data/1/TrainLabel2.txt', delimiter='\n', header=None)
y_train3 = pd.read_csv('data/1/TrainLabel3.txt', delimiter='\n', header=None)
y_train4 = pd.read_csv('data/1/TrainLabel4.txt', delimiter='\n', header=None)


# In[350]:


X_test2 = pd.read_csv('data/1/TestData2.txt', delimiter='\s+', header=None)
X_test3 = pd.read_csv('data/1/TestData3.txt', delimiter=',',   header=None)
X_test4 = pd.read_csv('data/1/TestData4.txt', delimiter='\s+', header=None)


# In[351]:


X_training = [X_train2, X_train3, X_train4]
y_training = [y_train2, y_train3, y_train4]
X_testing = [X_test2, X_test3, X_test4]


# In[352]:


for i,x in enumerate(X_training):
    print(f'X_TrainData{i+2} Shape: {x.shape}')


# In[353]:


for i,y in enumerate(y_training):
    print(f'y_TrainData{i+2} Shape: {y.shape}')


# In[354]:


for j,y in enumerate(X_testing):
    print(f'TestData{j+2} Shape: {y.shape}')


# # _Dataset 2_

# ### PCA for DS2

# In[355]:


X_train2.shape


# In[356]:


y_train2.shape


# In[357]:


y_train2.head()


# In[358]:


y_train2.tail()


# In[359]:


X_test2.shape


# In[360]:


xTrain2PCA = PCA(n_components=74)
X_train2_pca = xTrain2PCA.fit_transform(X_train2)


# In[361]:


# 100 principle components can explain 99% of the data

X_train2_pca_var = xTrain2PCA.fit(X_train2)
print(sum(X_train2_pca_var.explained_variance_ratio_))
print(X_train2_pca.shape)


# In[362]:


# 74 principle components can explain 99% of the data
xTest2PCA = PCA(n_components=74)
X_test2_pca = xTest2PCA.fit_transform(X_test2)


# In[363]:


X_test2_pca_var = xTest2PCA.fit(X_test2)
print(sum(X_test2_pca_var.explained_variance_ratio_))
print(X_test2_pca.shape)


# In[364]:


X_train2_cmp = pd.DataFrame(X_train2_pca)
X_train2_cmp.head(10)


# In[365]:


X_test2_cmp = pd.DataFrame(X_test2_pca)
X_test2_cmp.head(10)


# In[366]:


X_train2_cmp.shape


# In[367]:


y_train2.shape


# In[368]:


X_test2_cmp.shape


# In[369]:


def optimizeK(X_train, y_train, X_test, y_test):
    neighbors = np.arange(1,20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i,k in enumerate(neighbors):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        train_accuracy[i] = knn.score(X_train, y_train)

        test_accuracy[i] = knn.score(X_test, y_test2)
        
    return neighbors, test_accuracy, train_accuracy


# In[370]:


def plotK(neighbors, test_accuracy, train_accuracy):
        plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
        plt.plot(neighbors, train_accuracy, label='Training Accuracy')
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.xticks(np.arange(0, neighbors[-1], step=1))
        plt.ylabel('Accuracy')
        plt.title('KNN Varying Number of Neighbors')
        plt.show()


# In[371]:


def getAccuracyMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cra = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return cm, cra, acc


# ### KNN for DS2

# In[372]:


# KNN to obtain y_test1 values
neighbors = np.arange(1,20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):

    knn_ytest2 = KNeighborsClassifier(n_neighbors=k)

    knn_ytest2.fit(X_train2_cmp, y_train2)

    train_accuracy[i] = knn_ytest2.score(X_train2_cmp, y_train2)


# In[373]:


plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.xticks(np.arange(0, neighbors[-1], step=1))
plt.ylabel('Accuracy')
plt.title('KNN Varying Number of Neighbors')
plt.show()


# In[374]:


# KNN, k=2, to obtain y_test1 values

knn_ds2 = KNeighborsClassifier(n_neighbors=7)

knn_ds2.fit(X_train2_cmp, y_train2)
y_test2 = knn_ds2.predict(X_test2_cmp)


# In[375]:


# Run KNN to determine optimal K value

n, test_acc, train_acc = optimizeK(X_train2_cmp, y_train2, X_test2_cmp, y_test2)
plotK(n, test_acc, train_acc)


# In[376]:


n_neighbors=3
knn_ds2 = KNeighborsClassifier(n_neighbors)
knn_ds2.fit(X_train2_cmp, y_train2)


# In[377]:


knn_ds2.score(X_test2_cmp, y_test2)


# In[378]:


knn_ds2_y_pred = knn_ds2.predict(X_test2_cmp)


# In[379]:


knn_ds2_cm, knn_ds2_cra, knn_ds2_acc = getAccuracyMetrics(y_test2, knn_ds2_y_pred)


# In[380]:


print(f'KNN Accuracy Score, k={n_neighbors}: {knn_ds2_acc}\n')
print(f'KNN Confusion Matrix, k={n_neighbors}\n\n{knn_ds2_cm}')
pd.crosstab(y_test2, knn_ds2_y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[381]:


print(f'\n\t\t  KNN Classification Report, k={n_neighbors}\n\n{knn_ds2_cra}')


# In[520]:


# >>> def table_things(**kwargs):
# ...     for name, value in kwargs.items():
# ...         print( '{0} = {1}'.format(name, value))
# ...
# >>> table_things(apple = 'fruit', cabbage = 'vegetable')
# cabbage = vegetable
# apple = fruit


# In[523]:


def calculateMulticlassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
    label_len = 9
    labels = np.arange(1, label_len+1)
    y_bin = label_binarize(y_test, classes=labels)

    clf = OneVsRestClassifier(LinearSVC(random_state=0))
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


# In[524]:


calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, knn_ds2_y_pred, model='KNN', tuner='k', tuner_val=n_neighbors, dec=False)


# __________________________________________________________________________________________________________________________________________________

# In[525]:


def trainFitAccuracyLR(s, e, X_train, y_train, X_test, y_test):    
    c = np.arange(s, e+1)
    cma = {}
    cra = {}
    acc = {}
    preds = {}
    for i in c:    
        lr = LogisticRegression(C=i, solver='lbfgs',random_state=0)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cma[i] = confusion_matrix(y_test, y_pred)
        cra[i] = classification_report(y_test, y_pred)
        acc[i] = (accuracy_score(y_test, y_pred))
        preds[i] = y_pred
    return cma, cra, acc, preds


# ### Logistic Regression for DS2

# In[526]:


c = 10
lr_ds2_confusion_matricies, lr_ds2_classification_reports, lr_ds2_accuracies, lr_ds2_y_pred =    trainFitAccuracyLR(1, c, X_train2_cmp, y_train2, X_test2_cmp, y_test2)


# In[527]:


import operator
ky = max(lr_ds2_accuracies.items(), key=operator.itemgetter(1))[0]
val = float(max(lr_ds2_accuracies.items(), key=operator.itemgetter(1))[1])


# In[528]:


print(f'Logistic Regression Highest Accuracy Score, C={c}: {val}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_ds2_confusion_matricies[ky]}\n')
pd.crosstab(y_test2, lr_ds2_y_pred[ky], rownames=['True'], colnames=['Predicted'], margins=True)


# In[529]:


print(f'\n\t   Logistic Regression Classification Report C={c}\n\n{lr_classification_reports[ky]}')


# In[530]:


calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, lr_ds2_y_pred[ky], model='LR', tuner='c', tuner_val=c, dec=True)


# ## Random Forest for DS2

# In[531]:


rfc_ds2 = RandomForestClassifier()
rfc_ds2.fit(X_train2_cmp, y_train2)


# In[532]:


rfc_ds2_pred = rfc_ds2.predict(X_test2_cmp)


# In[533]:


rfc_ds2_cm, rfc_ds2_cr, rfc_ds2_acc = getAccuracyMetrics(y_test2, rfc_ds2_pred)


# In[534]:


print(f'Random Forest Accuracy Score: {rfc_ds2_acc}\n')
print(f'Random Forest Confusion Matrix\n\n{rfc_ds2_cm}\n')
pd.crosstab(y_test2, rfc_ds2_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[535]:


print(f'\n\t\tRandom Forest Classification Report\n\n{rfc_ds2_cr}')


# In[536]:


calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, rfc_ds2_pred, model='RF', tuner='', tuner_val=None, dec=False)


# ## SVM for DS2

# In[484]:


svm = SVC(kernel='linear')
svm.fit(X_train2_cmp, y_train2)


# In[538]:


svm_ds2_pred = svm.predict(X_test2_cmp)


# In[539]:


svm_ds2_cm, svm_ds2_cr, svm_ds2_acc = getAccuracyMetrics(y_test2, svm_ds2_y_pred)

print(f'SVM Accuracy Score: {svm_ds2_acc}\n')
print(f'SVM Confusion Matrix\n\n{svm_ds2_cm}\n')
pd.crosstab(y_test2, svm_ds2_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[487]:


print(f'\n\t\tRandom Forest Classification Report\n\n{svm_ds2_cr}')

calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, svm_ds2_y_pred, 9, 'linear', k)


# In[ ]:





# In[ ]:





# # Dataset 3

# In[ ]:


# # X_train3
# X_train3.head()


# In[ ]:


# n = X_train3[12][0]

# for i in X_train3.keys():
#     X_train3[i] = X_train3[i].replace(n, np.nan)


# In[ ]:


# X_train3.head()


# In[ ]:


# X_train3_clean = fast_knn(X_train1.values, k=100)


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


# # Logistic Regression
# log_reg = LogisticRegression(solver='lbfgs')
# log_reg.fit(X_train1, y_train1)


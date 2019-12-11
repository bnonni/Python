#!/usr/bin/env python
# coding: utf-8

# ## 1. Multi-Class Classification:
# For the multiclass classification problem, there were six different datasets. Some of the datasets contain missing values. For example, TrainData1, TestData1 and TrainData3 contain some missing values (1.00000000000000e+99). Therefore, the first approach needs to handle the missing values for selecting the features. Then compare the accuracy on train dataset to find out which classifier gives best result for each dataset with cross validation to verify the accuracy based on test dataset.
# <center><div style='width:50%; height:50%'><img src='/Users/hu5ky5n0w/Desktop/Github_Repos/Data/Python/Machine_Learning/Project/images/Q1_table.jpeg'></div></center>
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

# In[1]:


import warnings
warnings.filterwarnings('ignore')

# Base packages
import gc, sys, re, os
from time import strptime, mktime

# Data processing/preprocessing/modeling packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox

np.random.seed(1)

import seaborn as sns
import statistics as stat
from sklearn.preprocessing import *
# Modeling settings
plt.rc("font", size=14)
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Testing & Validation packages
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA

# SVM
from sklearn.svm import *

# KNN
from sklearn.neighbors import KNeighborsClassifier
from impyute.imputation.cs import fast_knn

# Logistic Regression
from sklearn.linear_model import LogisticRegression

# Random Forest
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from six import StringIO
from IPython.display import Image
from pydotplus import *

# SVM
from sklearn.svm import SVC, LinearSVC


# In[2]:


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


# In[10]:


def plotK(neighbors, test_accuracy, train_accuracy):
        plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
        plt.plot(neighbors, train_accuracy, label='Training Accuracy')
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.xticks(np.arange(0, neighbors[-1], step=1))
        plt.ylabel('Accuracy')
        plt.title('KNN Varying Number of Neighbors')
        plt.show()


# In[11]:


X_train2 = pd.read_csv('data/1/TrainData2.txt', delimiter='\s+', header=None)
X_train3 = pd.read_csv('data/1/TrainData3.txt', delimiter='\s+', header=None)
X_train4 = pd.read_csv('data/1/TrainData4.txt', delimiter='\s+', header=None)


# In[12]:


y_train2 = pd.read_csv('data/1/TrainLabel2.txt', delimiter='\n', header=None)
y_train3 = pd.read_csv('data/1/TrainLabel3.txt', delimiter='\n', header=None)
y_train4 = pd.read_csv('data/1/TrainLabel4.txt', delimiter='\n', header=None)


# In[13]:


X_test2 = pd.read_csv('data/1/TestData2.txt', delimiter='\s+', header=None)
X_test3 = pd.read_csv('data/1/TestData3.txt', delimiter=',',   header=None)
X_test4 = pd.read_csv('data/1/TestData4.txt', delimiter='\s+', header=None)


# In[14]:


X_training = [X_train2, X_train3, X_train4]
y_training = [y_train2, y_train3, y_train4]
X_testing = [X_test2, X_test3, X_test4]


# In[15]:


for i,x in enumerate(X_training):
    print(f'X_TrainData{i+1} Shape: {x.shape}')


# In[16]:


for i,y in enumerate(y_training):
    print(f'y_TrainData{i+1} Shape: {y.shape}')


# In[17]:


for j,y in enumerate(X_testing):
    print(f'TestData{j+1} Shape: {y.shape}')


# # _Dataset 2_

# ### PCA for DS2

# In[18]:


X_train2.shape


# In[19]:


y_train2.shape


# In[20]:


X_test2.shape


# In[21]:


xTrain2PCA = PCA(n_components=74)
X_train2_pca = xTrain2PCA.fit_transform(X_train2)


# In[22]:


# 100 principle components can explain 99% of the data

X_train2_pca_var = xTrain2PCA.fit(X_train2)
print(sum(X_train2_pca_var.explained_variance_ratio_))
print(X_train2_pca.shape)


# In[23]:


# 74 principle components can explain 99% of the data
xTest2PCA = PCA(n_components=74)
X_test2_pca = xTest2PCA.fit_transform(X_test2)


# In[24]:


X_test2_pca_var = xTest2PCA.fit(X_test2)
print(sum(X_test2_pca_var.explained_variance_ratio_))
print(X_test2_pca.shape)


# In[26]:


X_train2_cmp = pd.DataFrame(X_train2_pca)
X_train2_cmp.head(10)


# In[27]:


X_test2_cmp = pd.DataFrame(X_test2_pca)
X_test2_cmp.head(10)


# In[28]:


X_train2_cmp.shape


# In[29]:


y_train2.shape


# In[31]:


X_test2_cmp.shape


# ### KNN for DS2

# In[32]:


# KNN to obtain y_test1 values

knn = KNeighborsClassifier(n_neighbors=5)

knn = knn.fit(X_train2_cmp, y_train2)
y_test2_pred = knn.predict(X_test2_cmp)
print(y_test2_pred)


# In[34]:


y_test2 = y_test2_pred


# In[35]:


y_test2


# In[36]:


n, test_acc, train_acc = optimizeK(X_train2_cmp, y_train2, X_test2_cmp, y_test2)
plotK(n, test_acc, train_acc)


# In[37]:


n_neighbors=3
knn = KNeighborsClassifier(n_neighbors)
knn.fit(X_train2_cmp, y_train2)


# In[38]:


knn.score(X_test2_cmp, y_test2)


# In[39]:


knn_y_pred = knn.predict(X_test2_cmp)
knn_cm = confusion_matrix(y_test2, knn_y_pred)
print(knn_cm)


# In[40]:


pd.crosstab(y_test2, knn_y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[41]:


knn_cr = classification_report(y_test2, knn_y_pred)
print(knn_cr)


# In[32]:


# knn_y_pred_proba = knn.predict_proba(X_test2_components)[:,1]


# In[33]:


# ytest2_bin = label_binarize(y_test2, classes=[0, 1])
# ytest2_bin = ytest2_bin.reshape(-1, 1)
# knn_y_pred_proba = knn_y_pred_proba.reshape(-1, 1)


# In[42]:


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    return roc_auc_score(y_test, y_pred, average=average)


# In[43]:


multiclass_roc_auc_score(y_test2, knn_y_pred)


# In[44]:


y_bin = label_binarize(y_test2, classes=[0,1,2,3,4])
n_classes = 3


# In[74]:


clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train2_cmp, y_train2).decision_function(X_test2_cmp)


# In[85]:


a = [0, 0, 0, 0, 0]
b = [-0.65202941, 3.58522024, 1.70262234, 1.4361554, -1.67022448]

fpr, tpr, thresholds = roc_curve(a, b)
print(fpr, tpr)


# In[76]:


fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], thresholds = roc_curve(y_bin[:, i], y_score[:, i])
    print(f'ybin: {y_bin[i]}\nyscore: {y_score[i]}\n')
    print(f'fpr: {fpr[i]}\ntpr: {tpr[i]}\n')
    c = auc(fpr[i], tpr[i])
    if c == 'nan':
        pass
    else:
        roc_auc = c
        


# In[64]:


type(tpr.get(0)[0])


# In[56]:


for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


X_train2_components.head()


# ### Logistic Regression for DS2

# In[41]:


lr = LogisticRegression(random_state=0)
lr.fit(X_train2_components, y_train2)

lr_y_pred = lr.predict(X_test2_components)


# In[42]:


# rfe = RFE(lr, 20)
# rfe = rfe.fit(X_train2_components, y_train2.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)
# print(len(rfe.support_))


# In[43]:


# cols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 19, 22, 24, 34] 
# X=X_train2_components[cols]
# y=y_train2
# lr_X_train = X_train2_components.drop([11,16,18,20,21,23,25,26,27,28,29,30,31,32,33,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73


# In[44]:


lr.fit(X_train2_components, y_train2)

lr_y_pred = lr.predict(X_test2_components)


# In[45]:


lr_cm = confusion_matrix(y_test2, lr_y_pred)
print(f'Logistic Regression Confusion Matrix\n\n{lr_cm}')


# In[46]:


print(' Logistic Regression Confusion Matrix\n ______________________________________',)
pd.crosstab(y_test2, lr_y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[47]:


lr_cr = classification_report(y_test2, lr_y_pred)
print(f'             Logistic Regression Classification Report\n\n{lr_cr}')


# In[48]:


print('LR Accuracy:', accuracy_score(y_test2, lr_y_pred))


# ## Random Forest for DS2

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





# ## SVM for DS2

# # Dataset 3

# In[49]:


# X_train3
X_train3.head()


# In[50]:


n = X_train3[12][0]

for i in X_train3.keys():
    X_train3[i] = X_train3[i].replace(n, np.nan)


# In[51]:


X_train3.head()


# In[53]:


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


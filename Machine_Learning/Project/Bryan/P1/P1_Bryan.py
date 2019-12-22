#!/usr/bin/env python3
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
# In[00]:
os.chdir('/Users/hu5ky5n0w/Desktop/Github_Repos/Data/Python/Machine_Learning/Project/Bryan/1P')
# In[0]:
# take out the trash
import gc
import warnings
gc.collect()
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
from sklearn.svm import *
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import *
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, auc, accuracy_score
from sklearn.model_selection import GridSearchCV

# Custom Packages
from Helpers import *
from BPCA import *
from MCROC import *
from AccuracyMetrics import *
from KNN import *
from LR import *
from RF import *
from SVM import *


# In[3]:


X_train2 = pd.read_csv('../data/1/TrainData2.txt', delimiter='\s+', header=None)
y_train2 = pd.read_csv('../data/1/TrainLabel2.txt', delimiter='\n', header=None)
X_test2 = pd.read_csv('../data/1/TestData2.txt', delimiter='\s+', header=None)


# In[4]:


X_train3 = pd.read_csv('../data/1/TrainData3.txt', delimiter='\s+', header=None)
y_train3 = pd.read_csv('../data/1/TrainLabel3.txt', delimiter='\n', header=None)
X_test3 = pd.read_csv('../data/1/TestData3.txt', delimiter=',',   header=None)


# In[5]:


X_train4 = pd.read_csv('../data/1/TrainData4.txt', delimiter='\s+', header=None)
y_train4 = pd.read_csv('../data/1/TrainLabel4.txt', delimiter='\n', header=None)
X_test4 = pd.read_csv('../data/1/TestData4.txt', delimiter='\s+', header=None)


# In[6]:


X_train5 = pd.read_csv('../data/1/TrainData5.txt', delimiter='\s+', header=None)
y_train5 = pd.read_csv('../data/1/TrainLabel5.txt', delimiter='\n', header=None)
X_test5 = pd.read_csv('../data/1/TestData5.txt', delimiter='\s+', header=None)


# In[7]:


X_train6 = pd.read_csv('../data/1/TrainData6.txt', delimiter='\s+', header=None)
y_train6 = pd.read_csv('../data/1/TrainLabel6.txt', delimiter='\n', header=None)
X_test6 = pd.read_csv('../data/1/TestData6.txt', delimiter='\s+', header=None)


# In[8]:

AdaBoostClassifier()
X_training = [X_train2, X_train3, X_train4, X_train5, X_train6]
y_training = [y_train2, y_train3, y_train4, y_train5, y_train6]
X_testing = [X_test2, X_test3, X_test4, X_test5, X_test6]


# In[9]:


for i,x in enumerate(X_training):
    print(f'X_TrainData{i+2} Shape: {x.shape}')


# In[10]:


for i,y in enumerate(y_training):
    print(f'y_TrainData{i+2} Shape: {y.shape}')


# In[11]:


for j,y in enumerate(X_testing):
    print(f'TestData{j+2} Shape: {y.shape}')


# # _Dataset 2_

# ### PCA for DS2


X_train2.shape


# In[12]:


X_train2.head()


# In[13]:


y_train2.shape


# In[14]:


y_train2.head()


# In[15]:


y_train2.tail()


# In[16]:


X_test2.shape


# In[18]:


X_train2_cmp, X_test2_cmp = trainPCA(X_train2, X_test2)


# In[19]:


X_train2_cmp.shape


# In[20]:


y_train2.shape


# In[21]:


X_test2_cmp.shape


# ### KNN for DS2


# In[27]:


neighbors, train_accuracy = testK(X_train2_cmp, y_train2)
plotTestK(neighbors, train_accuracy)


# In[29]:


y_test2 = getTestLabelsKNN(7, X_train2_cmp, y_train2, X_test2_cmp)


# In[30]:


n, test_acc2, train_acc2 = optimizeK(X_train2_cmp, y_train2, X_test2_cmp, y_test2)
plotK(n, test_acc2, train_acc2)


# In[32]:


n_neighbors=3
knn_score2, knn_y_pred2 = runKNN(3, X_train2_cmp, y_train2, X_test2_cmp, y_test2)
print(f'Accuracy: {knn_score2}')


# In[34]:


getScoresKNN(n_neighbors, y_test2, knn_y_pred2)
pd.crosstab(y_test2, knn_y_pred2, rownames=['True'], colnames=['Predicted'], margins=True)


# In[36]:


calcMultiClassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, knn_y_pred2, model='KNN', tuner='k',\
 tuner_val=n_neighbors, label_len=9, dec=False)


# __________________________________________________________________________________________________________________________________________________

# ### Logistic Regression for DS2
# In[38]:


c2 = 10
lr_cma2, lr_cr2, lr_acc2, lr_y_pred2, ky2, val2  = runLogisticRegression(c2, X_train2_cmp, y_train2, X_test2_cmp, y_test2)

print(f'Logistic Regression Highest Accuracy Score, C={c2}: {val2}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma2[ky2]}\n')
pd.crosstab(y_test2, lr_y_pred2[ky2], rownames=['True'], colnames=['Predicted'], margins=True)


# In[41]:
print(f'\n\t   Logistic Regression Classification Report C={c2}\n\n{lr_cr2[ky2]}')


# In[42]:


calcMultiClassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, lr_y_pred2[ky2], model='LR', tuner='c', tuner_val=c2, label_len=9, dec=True)


# ## Random Forest for DS2
# In[44]:
runRandomForest(X_train2_cmp, y_train2, X_test2_cmp, y_test2, 9)

# ## SVM for DS2
# In[23]:

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train2_cmp, y_train2, X_test2_cmp, y_test2)


# # Dataset 3

# In[47]:


X_train3.shape


# In[48]:


X_train3.head()


# In[49]:


n = X_train3.iloc[0][12]

for i in X_train3.keys():
    X_train3[i] = X_train3[i].replace(n, np.nan)


# In[50]:


X_train3 = fast_knn(X_train3.values, k=50)


# In[51]:


X_train3 = pd.DataFrame(X_train3)


# In[52]:


y_train3.head()


# In[53]:


X_test3.shape


# In[54]:


X_test3.head()


# In[55]:


y_train3.head()


# In[56]:


y_train3.tail()


# ## KNN for DS3

# In[58]:


neighbors3, train_accuracy3 = testK(X_train3, y_train3)
plotTestK(neighbors3, train_accuracy3)


# In[59]:


# Run KNN to get y_test3 data
y_test3 = getTestLabelsKNN(1, X_train3, y_train3, X_test3)


# In[60]:


n3, test_acc3, train_acc3 = optimizeK(X_train3, y_train3, X_test3, y_test3)
plotK(n3, test_acc3, train_acc3)


# In[61]:


n_neighbors3=1
knn_score3, knn_y_pred3 = runKNN(n_neighbors3, X_train3, y_train3, X_test3, y_test3)
print(f'Accuracy: {knn_score3}')


# In[62]:


getScoresKNN(n_neighbors3, y_test3, knn_y_pred3)
pd.crosstab(y_test3, knn_y_pred3, rownames=['True'], colnames=['Predicted'], margins=True)
calcMultiClassROCAUC(X_train3, y_train3, X_test3, y_test3, model='KNN', tuner='k', tuner_val=n_neighbors3, label_len=9, dec=False)


# ## LR for DS3
# In[64]:


c3 = 20
lr_cma3, lr_cr3, lr_acc3, lr_y_pred3, ky3, val3 = runLogisticRegression(c3, X_train3, y_train3, X_test3, y_test3)

print(f'Logistic Regression Highest Accuracy Score, C={c3}: {val3}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma3[ky3]}\n')
pd.crosstab(y_test3, lr_y_pred3[ky3], rownames=['True'], colnames=['Predicted'], margins=True)
print(f'\n\t   Logistic Regression Classification Report C={c3}\n\n{lr_cr3[ky3]}')
calcMultiClassROCAUC(X_train3, y_train3, X_test3, lr_y_pred3[ky3], model='LR', tuner='c', tuner_val=c3, label_len=9, dec=True)


# ## Random Forest for DS3

# In[68]:


runRandomForest(X_train3, y_train3, X_test3, y_test3, 9)

# ## SVM for DS3

# In[70]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train3, y_train3, X_test3, y_test3)


# # Dataset 4

# In[48]:


X_train4.shape


# In[49]:


X_train4.head()


# In[28]:


X_test4.shape


# In[53]:


X_test4.head()


# In[54]:


y_train4.shape


# In[52]:


y_train4.head()


# In[55]:


X_test4.tail()


# In[56]:


X_train4_cmp, X_test4_cmp = trainPCA(X_train4, X_test4)


# In[57]:


X_train4_cmp.shape


# In[58]:


X_test4_cmp.shape


# ## KNN for DS4

# In[59]:


neighbors4, train_accuracy4 = testK(X_train4_cmp, y_train4)
plotTestK(neighbors4, train_accuracy4)


# In[60]:


y_test4 = getTestLabelsKNN(3, X_train4_cmp, y_train4, X_test4_cmp)
n4, test_acc4, train_acc4 = optimizeK(X_train4_cmp, y_train4, X_test4_cmp, y_test4)
plotK(n4, test_acc4, train_acc4)


# In[61]:


countUniqueValues(y_test4)


# In[39]:


knn_score4, knn_y_pred4 = runKNN(5, X_train4_cmp, y_train4, X_test4_cmp, y_test4)
print(f'Accuracy: {knn_score4}')


# In[40]:


n_neighbors4=5
getScoresKNN(n_neighbors4, y_test4, knn_y_pred4)
pd.crosstab(y_test4, knn_y_pred4, rownames=['True'], colnames=['Predicted'], margins=True)
calcMultiClassROCAUC(X_train4_cmp, y_train4, X_test4_cmp, knn_y_pred4, model='KNN', tuner='k',\
 tuner_val=n_neighbors4, label_len=9, dec=False)


# ## LR for DS4

# In[85]:


c4 = 20
lr_cma4, lr_cr4, lr_acc4, lr_y_pred4, ky4, val4 = runLogisticRegression(c4, X_train4, y_train4, X_test4, y_test4)
print(f'Logistic Regression Highest Accuracy Score, C={c4}: {val4}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma4[ky4]}\n')
pd.crosstab(y_test4, lr_y_pred4[ky4], rownames=['True'], colnames=['Predicted'], margins=True)


# In[89]:


print(f'\n\t   Logistic Regression Classification Report C={c4}\n\n{lr_cr4[ky4]}')
calcMultiClassROCAUC(X_train4, y_train4, X_test4, lr_y_pred4[ky4], model='LR', tuner='c', tuner_val=c4, label_len=9, dec=True)


# ## Random Forest for DS4

# In[90]:


runRandomForest(X_train4, y_train4, X_test4, y_test4, 9)


# ## SVM for DS4

# In[41]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train4_cmp, y_train4, X_test4_cmp, y_test4)
    
    
    
# # Dataset 5

# In[48]:


X_train5.shape


# In[49]:


X_train4.head()


# In[28]:


X_test4.shape


# In[53]:


X_test4.head()


# In[54]:


y_train4.shape


# In[52]:


y_train4.head()


# In[55]:


X_test4.tail()


# In[56]:


X_train4_cmp, X_test4_cmp = trainPCA(X_train4, X_test4)


# In[57]:


X_train4_cmp.shape


# In[58]:


X_test4_cmp.shape


# ## KNN for DS4

# In[59]:


neighbors4, train_accuracy4 = testK(X_train4_cmp, y_train4)
plotTestK(neighbors4, train_accuracy4)


# In[60]:


y_test4 = getTestLabelsKNN(3, X_train4_cmp, y_train4, X_test4_cmp)
n4, test_acc4, train_acc4 = optimizeK(X_train4_cmp, y_train4, X_test4_cmp, y_test4)
plotK(n4, test_acc4, train_acc4)


# In[61]:


countUniqueValues(y_test4)


# In[39]:


knn_score4, knn_y_pred4 = runKNN(5, X_train4_cmp, y_train4, X_test4_cmp, y_test4)
print(f'Accuracy: {knn_score4}')


# In[40]:


n_neighbors4=5
getScoresKNN(n_neighbors4, y_test4, knn_y_pred4)
pd.crosstab(y_test4, knn_y_pred4, rownames=['True'], colnames=['Predicted'], margins=True)
calcMultiClassROCAUC(X_train4_cmp, y_train4, X_test4_cmp, knn_y_pred4, model='KNN', tuner='k',\
 tuner_val=n_neighbors4, label_len=9, dec=False)


# ## LR for DS4

# In[85]:


c4 = 20
lr_cma4, lr_cr4, lr_acc4, lr_y_pred4, ky4, val4 = runLogisticRegression(c4, X_train4, y_train4, X_test4, y_test4)
print(f'Logistic Regression Highest Accuracy Score, C={c4}: {val4}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma4[ky4]}\n')
pd.crosstab(y_test4, lr_y_pred4[ky4], rownames=['True'], colnames=['Predicted'], margins=True)


# In[89]:


print(f'\n\t   Logistic Regression Classification Report C={c4}\n\n{lr_cr4[ky4]}')
calcMultiClassROCAUC(X_train4, y_train4, X_test4, lr_y_pred4[ky4], model='LR', tuner='c', tuner_val=c4, label_len=9, dec=True)


# ## Random Forest for DS4

# In[90]:


runRandomForest(X_train4, y_train4, X_test4, y_test4, 9)


# ## SVM for DS4

# In[41]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train4_cmp, y_train4, X_test4_cmp, y_test4)
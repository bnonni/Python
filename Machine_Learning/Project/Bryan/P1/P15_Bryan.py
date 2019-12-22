# In[00]:
# %reset -f
import os
os.chdir('/Users/hu5ky5n0w/Desktop/Github_Repos/Data/Python/Machine_Learning/Project/Bryan/1P')
# In[0]:
# take out the trash
import gc
import warnings
gc.collect()
warnings.filterwarnings('ignore')

# Python magic
# %matplotlib inline

# Data processing/preprocessing/modeling packages
import numpy as np
import pandas as pd
np.random.seed(1)

# In[1]:
# Custom Packages
from Helpers import *
from BPCA import *
from MCROC import *
from AccuracyMetrics import *
from KNN import *
from LR import *
from RF import *
from SVM import *

# In[6]:


X_train5 = pd.read_csv('../data/1/TrainData5.txt', delimiter='\s+', header=None)
y_train5 = pd.read_csv('../data/1/TrainLabel5.txt', delimiter='\n', header=None)
X_test5 = pd.read_csv('../data/1/TestData5.txt', delimiter='\s+', header=None)


# In[7]:


X_train6 = pd.read_csv('../data/1/TrainData6.txt', delimiter='\s+', header=None)
y_train6 = pd.read_csv('../data/1/TrainLabel6.txt', delimiter='\n', header=None)
X_test6 = pd.read_csv('../data/1/TestData6.txt', delimiter='\s+', header=None)


# %%

# # Dataset 5

# In[48]:


print(f'rows: {X_train5.shape[0]} by columns: {X_train5.shape[1]}')


# In[28]:


X_test5.shape


# In[53]:


X_test5.head()


# In[54]:


y_train5.shape


# In[52]:


y_train5.sample(20)


# In[55]:


X_test5.tail()


# In[56]:


# X_train5_cmp, X_test5 = trainPCA(X_train5, X_test5)


# In[57]:


# X_train5.shape


# In[58]:


# X_test5.shape


# ## KNN for DS4

# In[59]:


neighbors5, train_accuracy5 = testK(X_train5, y_train5)
plotTestK(neighbors5, train_accuracy5)

# In[61]:
for i in range(1,15):
 y_test5 = getTestLabelsKNN(i, X_train5, y_train5, X_test5)
 n5, test_acc5, train_acc5 = optimizeK(X_train5, y_train5, X_test5, y_test5)
 plotK(n5, test_acc5, train_acc5)
 
# In[61]:
y_test5 = getTestLabelsKNN(1, X_train5, y_train5, X_test5)

# In[60]:

keys, vals, no_labels = countUniqueValues(y_test5)
for i in keys:
 labels=[]
 labels.append(i)

# In[62]:

n5, test_acc5, train_acc5 = optimizeK(X_train5, y_train5, X_test5, y_test5)
plotK(n5, test_acc5, train_acc5)


# In[39]:


knn_score5, knn_y_pred5 = runKNN(15, X_train5, y_train5, X_test5, y_test5)
print(f'Accuracy: {knn_score5}')


# In[40]:


n_neighbors5=15
getScoresKNN(n_neighbors5, y_test5, knn_y_pred5)
pd.crosstab(y_test5, knn_y_pred5, rownames=['True'], colnames=['Predicted'], margins=True)
calcMultiClassROCAUC(X_train5, y_train5, X_test5, knn_y_pred5, model='KNN', tuner='k',\
 tuner_val=n_neighbors5, label_len=no_labels, labels=labels, dec=False)


# ## LR for DS5
# In[85]:

c5 = 20
lr_cma5, lr_cr5, lr_acc5, lr_y_pred5, ky5, val5 = runLogisticRegression(c5, X_train5, y_train5, X_test5, y_test5)
print(f'Logistic Regression Highest Accuracy Score, C={ky5}: {val5}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma5[ky5]}\n')
pd.crosstab(y_test5, lr_y_pred5[ky5], rownames=['True'], colnames=['Predicted'], margins=True)


# In[89]:


print(f'\n\t   Logistic Regression Classification Report C={ky5}\n\n{lr_cr5[ky5]}')
calcMultiClassROCAUC(X_train5, y_train5, X_test5, lr_y_pred5[ky5], model='LR', tuner='c',\
                     tuner_val=c5, label_len=no_labels, labels=labels, dec=True)

# ## Random Forest for DS4

# In[90]:


runRandomForest(X_train5, y_train5, X_test5, y_test5, 6)


# ## SVM for DS4

# In[41]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 6, X_train5, y_train5, X_test5, y_test5)
    
    
    
# Dataset 5

# ## KNN for DS4

# In[59]:


# neighbors5, train_accuracy5 = testK(X_train5, X_train5)
# plotTestK(neighbors5, train_accuracy5)
#
# # In[61]:
#
# for i in range(1,10):
#  y_test5 = getTestLabelsKNN(i, X_train5, y_train5, X_test5)
#  n5, test_acc5, train_acc5 = optimizeK(X_train5, y_train5, X_test5, y_test5)
#  plotK(n5, test_acc5, train_acc5)
#
# # In[40]:
# y_test5 = getTestLabelsKNN(9, X_train5, y_train5, X_test5)
#
# # In[60]:
#
# keys, vals, no_labels = countUniqueValues(y_test5)
# for i in keys.items():
#  labels = []
#  labels.append(i)
#
# # In[39]:
#
# knn_score5, knn_y_pred5 = runKNN(15, X_train5, y_train5, X_test5, y_test5)
# print(f'Accuracy: {knn_score4}')
#
#
# # In[40]:
#
# n_neighbors5=5
# getScoresKNN(n_neighbors5, y_test5, knn_y_pred5)
# pd.crosstab(y_test5, knn_y_pred5, rownames=['True'], colnames=['Predicted'], margins=True)
# calcMultiClassROCAUC(X_train5, y_train5, X_test5, knn_y_pred5, model='KNN', tuner='k',\
#  tuner_val=n_neighbors5, label_len=6, dec=False)
#
#
# # ## LR for DS4
#
# # In[85]:
#
#
# c5 = 20
# lr_cma5, lr_cr5, lr_acc5, lr_y_pred5, ky5, val5 = runLogisticRegression(c5, X_train5, y_train5, X_test5, y_test5)
# print(f'Logistic Regression Highest Accuracy Score, C={ky5}: {val5}\n')
# print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma5[ky5]}\n')
# pd.crosstab(y_test5, lr_y_pred5[ky5], rownames=['True'], colnames=['Predicted'], margins=True)
#
#
# # In[89]:
#
#
# print(f'\n\t   Logistic Regression Classification Report C={ky5}\n\n{lr_cr5[ky5]}')
# calcMultiClassROCAUC(X_train5, y_train5, X_test5, lr_y_pred5[ky5], model='LR', tuner='c', tuner_val=c5, label_len=6, labels=labels, dec=True)
#
#
# # ## Random Forest for DS4
#
# # In[90]:
#
#
# runRandomForest(X_train5, y_train5, X_test5, y_test5, 9, labels)
#
#
# # ## SVM for DS4
#
# # In[41]:
#
#
# kernels = ['linear', 'poly', 'rbf', 'sigmoid']
# for k in kernels:
#     print(f'SVM {k} Kernel results\n_________________________________________')
#     runSVM(k, 9, X_train5, y_train5, X_test5, y_test5)
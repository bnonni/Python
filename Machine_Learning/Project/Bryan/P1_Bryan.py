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

# In[1]:


# take out the trash
import gc
gc.collect()


# In[2]:


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


# In[3]:


X_train2 = pd.read_csv('data/1/TrainData2.txt', delimiter='\s+', header=None)
y_train2 = pd.read_csv('data/1/TrainLabel2.txt', delimiter='\n', header=None)
X_test2 = pd.read_csv('data/1/TestData2.txt', delimiter='\s+', header=None)


# In[4]:


X_train3 = pd.read_csv('data/1/TrainData3.txt', delimiter='\s+', header=None)
y_train3 = pd.read_csv('data/1/TrainLabel3.txt', delimiter='\n', header=None)
X_test3 = pd.read_csv('data/1/TestData3.txt', delimiter=',',   header=None)


# In[5]:


X_train4 = pd.read_csv('data/1/TrainData4.txt', delimiter='\s+', header=None)
y_train4 = pd.read_csv('data/1/TrainLabel4.txt', delimiter='\n', header=None)
X_test4 = pd.read_csv('data/1/TestData4.txt', delimiter='\s+', header=None)


# In[6]:


X_train5 = pd.read_csv('data/1/TrainData5.txt', delimiter='\s+', header=None)
y_train5 = pd.read_csv('data/1/TrainLabel5.txt', delimiter='\n', header=None)
X_test5 = pd.read_csv('data/1/TestData5.txt', delimiter='\s+', header=None)


# In[7]:


X_train6 = pd.read_csv('data/1/TrainData6.txt', delimiter='\s+', header=None)
y_train6 = pd.read_csv('data/1/TrainLabel6.txt', delimiter='\n', header=None)
X_test6 = pd.read_csv('data/1/TestData6.txt', delimiter='\s+', header=None)


# In[8]:


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

# In[11]:


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


# In[12]:


# n_components must be between 0 and min(n_samples, n_features)=112 with svd_solver='full'

def trainPCA(X_train, X_test):
    if X_train.shape[0] < X_train.shape[1]:
        if X_train.shape[0] < X_test.shape[0]:
            n_components=X_train.shape[0]
        else:
            n_components=X_test.shape[0]
    else:
        if X_train.shape[1] < X_test.shape[1]:
            n_components=X_train.shape[1]
        else:
            n_components=X_test.shape[1]
    
    X_PCA = PCA(n_components=n_components)
    
    X_train_PCA = X_PCA.fit_transform(X_train)
    X_train_variance = X_PCA.fit(X_train)
    print(f'X_train Variance: {sum(X_train_variance.explained_variance_ratio_)}')
    print(f'X_train Shape: {X_train.shape}')
    X_train_df = pd.DataFrame(X_train_PCA)
    
    X_test_PCA = X_PCA.fit_transform(X_test)
    X_test_variance = X_PCA.fit(X_test)
    print(f'X_train Variance: {sum(X_test_variance.explained_variance_ratio_)}')
    print(f'X_train Shape: {X_test_PCA.shape}')
    X_test_df = pd.DataFrame(X_test_PCA)
    
    return X_train_df, X_test_df


# In[13]:


def testK(X_train, y_train):
    # KNN to obtain optimal k for y_test1 values
    neighbors = np.arange(1,20)
    train_accuracy = np.empty(len(neighbors))

    for i,k in enumerate(neighbors):

        knn_ytest = KNeighborsClassifier(n_neighbors=k)

        knn_ytest.fit(X_train, y_train)

        train_accuracy[i] = knn_ytest.score(X_train, y_train)
    
    return neighbors, train_accuracy


# In[14]:


def plotTestK(neighbors, train_accuracy):
    plt.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.xlabel('Number of Neighbors')
    plt.xticks(np.arange(0, neighbors[-1], step=1))
    plt.ylabel('Accuracy')
    plt.title('KNN Varying Number of Neighbors')
    plt.show()


# In[15]:


def optimizeK(X_train, y_train, X_test, y_test):
    neighbors = np.arange(1,20)
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
    
    for i,k in enumerate(neighbors):

        knn = KNeighborsClassifier(n_neighbors=k)

        knn.fit(X_train, y_train)

        train_accuracy[i] = knn.score(X_train, y_train)

        test_accuracy[i] = knn.score(X_test, y_test)
        
    return neighbors, test_accuracy, train_accuracy


# In[16]:


def plotK(neighbors, test_accuracy, train_accuracy):
        plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
        plt.plot(neighbors, train_accuracy, label='Training Accuracy')
        plt.legend()
        plt.xlabel('Number of Neighbors')
        plt.xticks(np.arange(0, neighbors[-1], step=1))
        plt.ylabel('Accuracy')
        plt.title('KNN Varying Number of Neighbors')
        plt.show()


# In[17]:


def getAccuracyMetrics(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    cra = classification_report(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    return cm, cra, acc


# In[18]:


def calculateMulticlassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
        label_len = kwargs['label_len']
        
    labels = np.arange(1, label_len+1)
    y_bin = label_binarize(y_test, classes=labels)

    clf = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=0))
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


# ### KNN for DS2

# In[19]:


def getTestLabelsKNN(n, X_train, y_train, X_test):
    knn_ytest = KNeighborsClassifier(n_neighbors=n)
    knn_ytest.fit(X_train, y_train)
    y_test = knn_ytest.predict(X_test)
    return y_test


# In[20]:


def runKNN(n, X_train, y_train, X_test, y_test):
    n_neighbors=n
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    return score, y_pred


# In[21]:


def getScoresKNN(n, y_test, y_pred):
    knn_cm, knn_cr, knn_acc = getAccuracyMetrics(y_test, y_pred)
    print(f'KNN Accuracy Score, k={n}: {knn_acc}\n')
    print(f'KNN Confusion Matrix, k={n}\n\n{knn_cm}')
    print(f'\n\t\t  KNN Classification Report, k={n_neighbors}\n\n{knn_cr}')


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


calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, knn_y_pred2, model='KNN', tuner='k',                          tuner_val=n_neighbors, label_len=9, dec=False)


# __________________________________________________________________________________________________________________________________________________

# ### Logistic Regression for DS2

# In[37]:


def trainFitAccuracyLR(e, X_train, y_train, X_test, y_test):    
    c = np.arange(1, e+1)
    cma = {}
    cra = {}
    acc = {}
    preds = {}
    for i in c:    
        lr = LogisticRegression(C=i, multi_class='ovr', solver='lbfgs',random_state=0)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cma[i] = confusion_matrix(y_test, y_pred)
        cra[i] = classification_report(y_test, y_pred)
        acc[i] = (accuracy_score(y_test, y_pred))
        preds[i] = y_pred
    return cma, cra, acc, preds


# In[38]:


c2 = 10
lr_cma2, lr_cr2, lr_acc2, lr_y_pred2 =    trainFitAccuracyLR(c2, X_train2_cmp, y_train2, X_test2_cmp, y_test2)


# In[39]:


import operator
ky2 = max(lr_acc2.items(), key=operator.itemgetter(1))[0]
val2 = float(max(lr_acc2.items(), key=operator.itemgetter(1))[1])


# In[40]:


print(f'Logistic Regression Highest Accuracy Score, C={c2}: {val2}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma2[ky2]}\n')
pd.crosstab(y_test2, lr_y_pred2[ky2], rownames=['True'], colnames=['Predicted'], margins=True)


# In[41]:


print(f'\n\t   Logistic Regression Classification Report C={c2}\n\n{lr_cr2[ky2]}')


# In[42]:


calculateMulticlassROCAUC(X_train2_cmp, y_train2, X_test2_cmp, lr_y_pred2[ky2], model='LR', tuner='c', tuner_val=c2, label_len=9, dec=True)


# ## Random Forest for DS2

# In[43]:


def runRandomForest(X_train, y_train, X_test, y_test, ll):
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    rfc_pred = rfc.predict(X_test)
    rfc_cm, rfc_cr, rfc_acc = getAccuracyMetrics(y_test, rfc_pred)
    print(f'Random Forest Accuracy Score: {rfc_acc}\n')
    print(f'Random Forest Confusion Matrix\n\n{rfc_cm}\n')
    pd.crosstab(y_test, rfc_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(f'\n\t\tRandom Forest Classification Report\n\n{rfc_cr}')
    calculateMulticlassROCAUC(X_train, y_train, X_test, rfc_pred, model='RF', tuner='', tuner_val=None, label_len=ll, dec=False)


# In[44]:


runRandomForest(X_train2_cmp, y_train2, X_test2_cmp, y_test2, 9)


# ## SVM for DS2

# In[23]:


def runSVM(kernel, ll, X_train, y_train, X_test, y_test):
    svm = SVC(kernel=kernel)
    svm.fit(X_train, y_train)
    svm_y_pred = svm.predict(X_test)
    svm_cm, svm_cr, svm_acc = getAccuracyMetrics(y_test, svm_y_pred)
    print(f'SVM {kernel} Kernel Accuracy Score: {svm_acc}\n')
    print(f'SVM {kernel} Kernel Confusion Matrix\n\n{svm_cm}\n')
    pd.crosstab(y_test, svm_y_pred, rownames=['True'], colnames=['Predicted'], margins=True)
    print(f'\n\t\t    SVM {kernel} Kernel Classification Report\n\n{svm_cr}')
    calculateMulticlassROCAUC(X_train, y_train, X_test, svm_y_pred, model='SVM', tuner='kernel', tuner_val=kernel, label_len=ll, dec=False)


# In[46]:


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


# In[57]:


def calculateMulticlassROCAUC(X_train, y_train, X_test, y_test, **kwargs):
    for k,v in kwargs.items():
        model = kwargs['model']
        tuner = kwargs['tuner']
        tuner_val = kwargs['tuner_val']
        dec = kwargs['dec']
        label_len = kwargs['label_len']
        
    labels = np.arange(1, label_len+1)
    y_bin = label_binarize(y_test, classes=labels)

    clf = OneVsRestClassifier(LinearSVC(multi_class='ovr', random_state=0))
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


# ## KNN for DS3

# In[58]:


neighbors, train_accuracy = testK(X_train3, y_train3)
plotTestK(neighbors, train_accuracy)


# In[59]:


# Run KNN to get y_test3 data
y_test3 = getTestLabelsKNN(1, X_train3, y_train3, X_test3)


# In[60]:


n, test_acc3, train_acc3 = optimizeK(X_train3, y_train3, X_test3, y_test3)
plotK(n, test_acc3, train_acc3)


# In[61]:


n_neighbors=1
knn_score3, knn_y_pred3 = runKNN(n_neighbors, X_train3, y_train3, X_test3, y_test3)
print(f'Accuracy: {knn_score3}')


# In[62]:


getScoresKNN(n_neighbors, y_test3, knn_y_pred3)

pd.crosstab(y_test3, knn_y_pred3, rownames=['True'], colnames=['Predicted'], margins=True)

calculateMulticlassROCAUC(X_train3, y_train3, X_test3, y_test3, model='KNN', tuner='k', tuner_val=n_neighbors, label_len=9, dec=False)


# ## LR for DS3

# In[63]:


def trainFitAccuracyLR(e, X_train, y_train, X_test, y_test):    
    c = np.arange(1, e+1)
    cma = {}
    cra = {}
    acc = {}
    preds = {}
    for i in c:    
        lr = LogisticRegression(C=i, multi_class='ovr', solver='lbfgs',random_state=0)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        cma[i] = confusion_matrix(y_test, y_pred)
        cra[i] = classification_report(y_test, y_pred)
        acc[i] = (accuracy_score(y_test, y_pred))
        preds[i] = y_pred
    return cma, cra, acc, preds


# In[64]:


c3 = 20
lr_cma3, lr_cr3, lr_acc3, lr_y_pred3 = trainFitAccuracyLR(c3, X_train3, y_train3, X_test3, y_test3)


# In[65]:


import operator
ky3 = max(lr_acc3.items(), key=operator.itemgetter(1))[0]
val3 = float(max(lr_acc3.items(), key=operator.itemgetter(1))[1])


# In[66]:


print(f'Logistic Regression Highest Accuracy Score, C={c3}: {val3}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma3[ky3]}\n')
pd.crosstab(y_test3, lr_y_pred3[ky3], rownames=['True'], colnames=['Predicted'], margins=True)


# In[67]:


print(f'\n\t   Logistic Regression Classification Report C={c3}\n\n{lr_cr3[ky3]}')
calculateMulticlassROCAUC(X_train3, y_train3, X_test3, lr_y_pred3[ky3], model='LR', tuner='c', tuner_val=c3, label_len=9, dec=True)


# ## Random Forest for DS3

# In[68]:


runRandomForest(X_train3, y_train3, X_test3, y_test3, 9)


# In[69]:


def countUniqueValues(data):
    from collections import Counter
    print(f'Values: {Counter(data).keys()}\nKeys: {Counter(data).values()}')


# ## SVM for DS3

# In[70]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train3, y_train3, X_test3, y_test3)


# # Dataset 4

# In[25]:


X_train4.shape


# In[26]:


X_train4.head()


# In[27]:


y_train4.head()


# In[ ]:





# In[28]:


X_test4.shape


# In[29]:


X_test4


# In[30]:


X_test4.tail()


# In[31]:


X_train4_cmp, X_test4_cmp = trainPCA(X_train4, X_test4)


# In[32]:


X_train4_cmp.shape


# In[33]:


X_test4_cmp.shape


# ## KNN for DS4

# In[35]:


neighbors, train_accuracy = testK(X_train4_cmp, y_train4)
plotTestK(neighbors, train_accuracy)


# In[37]:


y_test4 = getTestLabelsKNN(3, X_train4_cmp, y_train4, X_test4_cmp)
n4, test_acc4, train_acc4 = optimizeK(X_train4_cmp, y_train4, X_test4_cmp, y_test4)
plotK(n4, test_acc4, train_acc4)


# In[38]:


countUniqueValues(y_test3)


# In[39]:


knn_score4, knn_y_pred4 = runKNN(5, X_train4_cmp, y_train4, X_test4_cmp, y_test4)
print(f'Accuracy: {knn_score4}')


# In[40]:


n_neighbors=5
getScoresKNN(n_neighbors, y_test4, knn_y_pred4)
pd.crosstab(y_test4, knn_y_pred4, rownames=['True'], colnames=['Predicted'], margins=True)
calculateMulticlassROCAUC(X_train4_cmp, y_train4, X_test4_cmp, knn_y_pred4, model='KNN', tuner='k',                          tuner_val=n_neighbors, label_len=9, dec=False)


# ## LR for DS4

# In[85]:


c4 = 20
lr_cma4, lr_cr4, lr_acc4, lr_y_pred4 = trainFitAccuracyLR(c4, X_train4, y_train4, X_test4, y_test4)


# In[86]:


import operator
ky4 = max(lr_acc4.items(), key=operator.itemgetter(1))[0]
val4 = float(max(lr_acc4.items(), key=operator.itemgetter(1))[1])


# In[87]:


print(f'Logistic Regression Highest Accuracy Score, C={c4}: {val4}\n')
print(f'Logistic Regression Most Accurate Confusion Matrix\n\n{lr_cma4[ky4]}\n')
pd.crosstab(y_test4, lr_y_pred4[ky4], rownames=['True'], colnames=['Predicted'], margins=True)


# In[89]:


print(f'\n\t   Logistic Regression Classification Report C={c4}\n\n{lr_cr4[ky4]}')
calculateMulticlassROCAUC(X_train4, y_train4, X_test4, lr_y_pred4[ky4], model='LR', tuner='c', tuner_val=c4, label_len=9, dec=True)


# ## Random Forest for DS4

# In[90]:


runRandomForest(X_train4, y_train4, X_test4, y_test4, 9)


# ## SVM for DS4

# In[41]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
for k in kernels:
    print(f'SVM {k} Kernel results\n_________________________________________')
    runSVM(k, 9, X_train4_cmp, y_train4, X_test4_cmp, y_test4)


# In[ ]:





# In[ ]:





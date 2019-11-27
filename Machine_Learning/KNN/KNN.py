#!/usr/bin/env python
# coding: utf-8

# In[233]:


get_ipython().run_line_magic('matplotlib', 'inline')
# Necessary Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Model and data processing packages
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


# In[242]:


np.random.seed(234)


# In[243]:


df = pd.read_csv('diabetes.csv')
df.head(5)


# In[244]:


df.shape


# In[245]:


X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[246]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)


# In[247]:


neighbors = np.arange(1,20)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))


# In[248]:


for i,k in enumerate(neighbors):
    
    knn = KNeighborsClassifier(n_neighbors=k)
    
    knn.fit(X_train, y_train)
    
    train_accuracy[i] = knn.score(X_train, y_train)
    
    test_accuracy[i] = knn.score(X_test, y_test)


# In[249]:


plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.title('KNN Varying Number of Neighbors')
plt.show()


# In[250]:


n_neighbors=18
knn = KNeighborsClassifier(n_neighbors)


# In[251]:


knn.fit(X_train, y_train)


# In[252]:


knn.score(X_test, y_test)


# In[253]:


y_pred = knn.predict(X_test)


# In[254]:


confusion_matrix(y_test, y_pred)


# In[255]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[256]:


print(classification_report(y_test, y_pred))


# In[257]:


y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[258]:


# FPR = False Positive Rate, TPR = True Positive Rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[259]:


plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='KNN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'KNN (n_neighbors={n_neighbors}) ROC Curve')
plt.show()


# In[260]:


roc_auc_score(y_test, y_pred_proba)


# In[261]:


param_grid = {'n_neighbors':np.arange(1,100)}


# In[262]:


knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5)
knn_cv.fit(X, y)


# In[263]:


knn_cv.best_score_


# In[264]:


knn_cv.best_params_


# In[265]:


n_neighbors=14
knn = KNeighborsClassifier(n_neighbors)


# In[266]:


knn.fit(X_train, y_train)


# In[267]:


knn.score(X_test, y_test)


# In[268]:


y_pred = knn.predict(X_test)


# In[269]:


confusion_matrix(y_test, y_pred)


# In[270]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[272]:


print(classification_report(y_test, y_pred))


# In[273]:


y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[274]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[275]:


plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='KNN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title(f'KNN (n_neighbors={n_neighbors}) ROC Curve')
plt.show()


# In[276]:


roc_auc_score(y_test, y_pred_proba)


# In[ ]:





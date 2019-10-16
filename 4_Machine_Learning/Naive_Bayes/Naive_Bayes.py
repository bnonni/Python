#!/usr/bin/env python
# coding: utf-8

# In[99]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
np.random.seed(44)


# In[100]:


def print_scores(y_test, y_pred, y_pred_prob):
    print('recall score:',recall_score(y_test, y_pred))
    print('precision score:',precision_score(y_test, y_pred))
    print('f1 score (weighted average of the precision and recall):',f1_score(y_test, y_pred))
    print('accuracy score:',accuracy_score(y_test,y_pred))


# In[101]:


df = pd.read_csv('diabetes.csv')
df.head()


# In[102]:


print("Outcome as pie chart:")
fig, ax = plt.subplots(1, 1)
ax.pie(df.Outcome.value_counts(),autopct='%1.1f%%', labels=['Diabetes','No Diabetes'], colors=['yellowgreen','r'])
plt.axis('equal')
plt.ylabel('')


# In[103]:


##### plot Time to see if there is any trend
print("Age")
print(df["Age"].tail(5))
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(df.Age[df.Outcome==0], bins=40, color='g',alpha=0.5)
ax1.set_title('Not Diabetes')
ax1.set_xlabel('Age')
ax1.set_ylabel('# of Cases')
ax2.hist(df.Age[df.Outcome==1], bins=40, color='r',alpha=0.5)
ax2.set_title('Diabetes')
ax2.set_xlabel('Age')
ax2.set_ylabel('# of Cases')


# In[104]:


fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)
ax3.hist(df.Pregnancies[df.Outcome==0],bins=50, color='g',alpha=0.5)
ax3.set_title('Not Diabetes') 
ax3.set_ylabel('# of Cases')
ax4.hist(df.Pregnancies[df.Outcome==1],bins=50, color='r',alpha=0.5)
ax4.set_title('Diabetes')
ax4.set_xlabel('Pregnancies')
ax4.set_ylabel('# of Cases')


# In[105]:


import seaborn as sns
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(28, 1)
plt.figure(figsize=(6,28*4))
for i, col in enumerate(df[df.iloc[:,1:8].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(df[col][df.Outcome == 1], kde=True, bins=50, color='r')
    sns.distplot(df[col][df.Outcome == 0], kde=True, bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_ylabel('# of cases')
    ax5.set_title('feature: ' + str(col))
plt.show()


# In[106]:


X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[107]:


print("Length of training labels:", len(y_train))
print("Length of testing labels:", len(y_test))
print("Length of training features:", len(X_train))
print("Length of testing features:", len(X_test))


# # Bernoulli Naive-Bayes

# In[108]:


bnb = BernoulliNB()


# In[109]:


bnb.fit(X_train, y_train)


# In[110]:


y_pred = bnb.predict(X_test)


# In[111]:


confusion_matrix(y_test, y_pred)


# In[112]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[113]:


print(classification_report(y_test, y_pred))


# In[114]:


y_pred_proba = bnb.predict_proba(X_test)[:,1]


# In[115]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='BNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('Bernoulli Naive-Bayes ROC Curve')
plt.show()


# In[116]:


roc_auc_score(y_test, y_pred_proba)


# # Gaussian Naive-Bayes

# In[117]:


gnb = GaussianNB()


# In[118]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[119]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[120]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[121]:


print(classification_report(y_test, y_pred))


# In[122]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[123]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr') 
plt.title('Gaussian Naive-Bayes ROC Curve')
plt.show()


# In[124]:


roc_auc_score(y_test, y_pred_proba)


# ### Drop BMI

# In[125]:


df = df.drop('BMI', axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
df


# In[126]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[127]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[128]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[129]:


print(classification_report(y_test, y_pred))


# In[130]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[131]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop BMI')
plt.show()


# In[132]:


roc_auc_score(y_test, y_pred_proba)


# ### Drop Pregnancies

# In[133]:


df = pd.read_csv('diabetes.csv')
df = df.drop('Pregnancies', axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[134]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[135]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[136]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[137]:


print(classification_report(y_test, y_pred))


# In[138]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[139]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop Pregnancies')
plt.show()


# In[140]:


roc_auc_score(y_test, y_pred_proba)


# ### Drop Insulin

# In[141]:


df = pd.read_csv('diabetes.csv')
df = df.drop('Insulin', axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[142]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[143]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[144]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[145]:


print(classification_report(y_test, y_pred))


# In[146]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[147]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop Insulin')
plt.show()


# In[148]:


roc_auc_score(y_test, y_pred_proba)


# ## Drop DiabetesPedigreeFunction

# In[149]:


df = pd.read_csv('diabetes.csv')
df = df.drop('DiabetesPedigreeFunction', axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[150]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[151]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[152]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[153]:


print(classification_report(y_test, y_pred))


# In[154]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[155]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop Pregnancies')
plt.show()


# In[156]:


roc_auc_score(y_test, y_pred_proba)


# ### Drop SkinThickness

# In[157]:


df = pd.read_csv('diabetes.csv')
df = df.drop('SkinThickness', axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[158]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[159]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[160]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[161]:


print(classification_report(y_test, y_pred))


# In[162]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[163]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop SkinThickness')
plt.show()


# In[164]:


roc_auc_score(y_test, y_pred_proba)


# ### Drop Insulin & SkinThickness

# In[165]:


df = pd.read_csv('diabetes.csv')
df = df.drop(['SkinThickness', 'Insulin'], axis=1)
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values


# In[166]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# In[167]:


gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[168]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# In[169]:


print(classification_report(y_test, y_pred))


# In[170]:


y_pred_proba = gnb.predict_proba(X_test)[:,1]


# In[171]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='GNB')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Gaussian Naive-Bayes ROC Curve: Drop Insulin & SkinThickness')
plt.show()


# In[172]:


roc_auc_score(y_test, y_pred_proba)


# In[ ]:





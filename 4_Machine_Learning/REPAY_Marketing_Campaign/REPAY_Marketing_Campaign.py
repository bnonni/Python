#!/usr/bin/env python
# coding: utf-8

# # Take Home Test Instructions
# 
# You are being provided with a single compressed csv file with the following fields :
# 
# - browser : the browser the campaign was provided on
# - visiting_time : the amount of time on the campaign site
# - total_amount_due : their account balance at time of visiting
# - previous_payment_amount : the amount of the last payment they made
# - previous_visitor : whether or not we have evidence that they have been shown the campaign before
# - day_of_week : the day of the week of viewing
# - campaign : which campaign they were shown
# - converted: whether or not they finished the sign up process after the campaign
# - traffic_source: where they were when funneled to the campaign. This value is set after the campaign has been shown as part of the signup process. It is not possible for us to detect this prior to sign up at this time.
# 
# 
# The experiment describes a series of marketing campaigns that have the goal of converting an interaction into a new user of a service we're currently starting to offer at the company.
# 
# Three different marketing campaigns were applied at random to a population of visitors to our company. Each visitor was selected at random and had a different entry route to an interaction (web, mobile, in store), was shown a marketing campaign selected at random, and then asked if they wished to sign up for the service.
# 
# 
# Generate a short report (< 2 pages including figures) that analyzes and explores the data set in order to answer the following question :
# 
# How would you utilize the results of this experiment in order maximize conversion ?
# 
# Please send back the report along with any code (or links to remote repositories) within 7 days.

# ## Assumptions
# 
# #### Random Variables:
# 1. Marketing Campaign (A, B, C) shown to visitor
# 2. Population of visitors to company
# 3. Selected visitor from population
# 
# $\implies$ Assumption: the visitor's entry route (i.e. traffic_source) is also random since this is controlled by the visitor being selected at random
# 
# $\implies$ Assumption: the visitor's browser choice is also random since this is controlled by the visitor being selected at random
# 
# $\implies$ Assumption: the day_of_week is random since this is controlled by the visitors

# ## Major Questions: 
# <ol><li>Which feature/features correlate the strongest with the classifier (convert/not convert)?</li>
#     <li>What is the best combination of features that accurately predicts the output (convert/not convert)?</li>
# </ol>
# 
# #### Process: 
# <ol><li>Exploratory Data Analysis. </li>
#     <ul><li>Plot histograms and distrbution plots to visualize trends. Specifically looking for trends between:</li>
#         <ul><li>feature-and-feature</li><li>feature-and-conversion</li></ul></ul>
#     <li>Feature Engineering.</li>
#     <li>Predictive Model Analysis Using Decision Tree.</li>
# </ol>

# In[214]:


# Import all packages
# get_ipython().run_line_magic('matplotlib', 'inline')

import os
import gc
import pydotplus
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

np.random.seed(42)


# In[215]:


# Read in data
df = pd.DataFrame(pd.read_csv('results.csv'))

# Review features
df.keys()


# <hr>

# # <u>Step 1: Exploratory Data Analysis</u>

# #### Initial view of dataset overall (head, tail, sample, shape, describe)

# In[216]:


# Review top 5 to get idea of data set
df.head()


# In[217]:


# Review last 5 for good measure
df.tail()


# In[218]:


# Show me a sample of 20 to better understand the data's randomness
df.sample(20)


# In[219]:


# How many samples and how many features? 
df.shape


# In[220]:


df.describe(include='all')


# ## What percentage of the samples converted vs. did not?
# #### Process: 
# <ul><li>Isolate converted (1) and not converted (0)</li>
#     <li>Create pie chart</li>
#     <li>QA using value_counts</li>
# </ul>

# In[221]:


fig, ax = plt.subplots(1, 1)
ax.pie(df.converted.value_counts(),autopct='%1.1f%%', labels=['Not Converted','Converted'], colors=['r', 'yellowgreen'])
plt.axis('equal')
plt.ylabel('')


# In[222]:


# Check pie chart accuracy
print(f'Count:\n{df.converted.value_counts()}\n\nPercent:\n{df.converted.value_counts(1)}')


# <hr>

# ## <u>Data Exploration - Discrete Features</u>

# #### 1. What is the count of each discrete feature?
# #### 2. What is the count of each discrete feature's value that produced a conversion (i.e. Browser=firefox)?
# #### 3. What proportion of the overall feature does that value represent?

# In[223]:


def cvrDoubleHist(v, x, t, l):
    fig, ax1 = plt.subplots(1, 1, sharex = True, figsize=(6,3))
    ax1.hist([x[df.converted==1],x[df.converted==0]], bins=v, color=['g','r'],alpha=0.5)
    ax1.set_title(t)
    ax1.set_xlabel(l)
    ax1.set_ylabel('# of Conversions')
    
def cvrSingleHist(v, x, t, l):
    fig, ax1 = plt.subplots(1, 1, sharex = True, figsize=(6,3))
    ax1.hist(x[df.converted==1], bins=v*2, color='g',alpha=0.5)
    ax1.set_title(t)
    ax1.set_xlabel(l)
    ax1.set_ylabel('# of Conversions')


# In[224]:


# Value counts of each features static entires.
discrete = [df.browser, df.day_of_week, df.campaign, df.traffic_source]

for x in discrete:
    name = x.name
    vc = x.value_counts()
    f_vc = x[df.converted==1].value_counts()
    fv_prop = round(f_vc/sum(f_vc)*100,2)
    print(f'{name} count: \n{vc}\n\n% of total {name}: \n{round(vc/sum(vc)*100, 2)}\n\ncnv cnt by {name} count:\n{f_vc}\n\n% cvr of each {name}\n{fv_prop}\n------------------------------')
    cvrDoubleHist(len(vc), x, f'{name} Conversions (green) / Non-Conversions (red)', name)
    cvrSingleHist(len(vc), x, f'{name} Conversions', name)


# In[ ]:





# In[ ]:





# ## <u>Data Analysis - Discrete Features</u>

# ## browser
# <p>Firefox and Chrome are the top 2 browsers with the most and 2nd most conversions. Since visitors are selected at random, we've made the assumption (above) that thee browser choice is also random within our dataset since we cannot control the browser choice and are taking the visitor at random. </p>
#     
# <p>Firefox and Chrome also happen to be the 2 most popular products on the market. Looking at the above browser counting data, we see that these browser are the two most-used browsers in our dataset making the relationship between browser-type and conversion rate possibly correlated to the fact that Safari and IE are less used therefore reducing the opportunity for these browsers to produce conversions.</p>
#     
# <p>If more people use these products, its likely the case that people prefer using them. Preference indicates that visitors like using these browsers, which implies some form of ease-of-use or positive interaction. Positive feelings while browsing directly influences consumer buying behavior positively. Moving forward, I would recommend disproportionaly focusing marketing budget and marketing campaigns on visitors using these browsers. This will maximize possible conversion per marketing dollar spent.</p>
# 
# <p>Speaking from a purely statistical perspective, the results are correlative and possible anomolous; We must temper action with widsom and avoid throwing ALL marketing at 2 browsers. We also must temper action with practicality; practically speaking, consumers use technology to access websites that they're comfortable with and enjoy using. If more of the market is on these two browsers, we have to meet our market half-way. It is a fact that the population was random and the visitor was random, so it would be safe to assume that this population is representation of a bigger population in the market.</p>

# ## day_of_week
# <p>Thursday is the clear winner for both traffic (i.e. number of visitors) and conversions. Actual day of the week is mostly arbitrary for this industry since its not a quick/impulse buying industry (unlike retail). From this data, I gather that Thursday was clearly a popular day for overall visitor traffic during the course of these campaigns; however, to let this heavily impact our future marketing strategy would be misguided. </p>
# 
# <p>While we cannot ignore this data point, we would need historical data and contextual data (i.e. industry standards, type of visitor, visitor's industry, etc.) to better understand why this day of the week was most popular.</p>
# 
# <p>Again, statistical speaking, the results are correlative; we would want to test our theory before launching full-scale campaigns on single days of the week. Practically speaking, it does make sense that consumers shop more on certain days of the week, and thus, would be in the right part of the buying cycle on certain days.</p>

# ## campaign
# <p>Campaign count is evenly distributed at 33.33% each: 20,000 for each A, B and C campaign. Obviously, this is good for the integrity of our test that no campaign was given preferential treatment.</p>
# 
# <p>Campaign C was had the most conversions representing 37% of all campaign related conversion. C is followed very closely, however, by Campaign B with 36% of all campaign related conversions. Campaign A had the least conversions representing 26% of all campaign related conversions.</p>
# 
# <p>Given how close C and B are in success rate, and taking into account the other, more influential factors thus far (i.e. browser and traffic_source), its hard to say that  either campaign was "more effective." Purely from the numbers, C did correspond to more conversions, but to what extend was the messaging in C attributable to the conversion? I would recommend a run-off test between these two campaigns while controlling for browser, traffic_source and other features that could lead to a conversion. This will help us find a more-clear winner for the type of marketing message.</p>

# ## traffic_source
# <p>Mobile is the clear winner for conversions making up 54%, but again, mobile also makes up 70% of all traffic. The same statistical/practical analysis applies. Statistically, its unwise to focus too much effort or money on mobile without solidifying that this is not an anomally. However, given the statistical randomness of the data set, we can take this to mean that more of our customers are using mobile devices, so we should practically take this into account.</p>

# <hr>

# ## <u>Data Analysis - Continuous Features</u>

# #### 1. What are the averages of each continuous feature?
# #### 2. What is the distribution of each continuous feature against the outcomes?
# #### 3. What do the distributions tell me about these features?

# In[225]:


df['%_payment'] = round((df['previous_payment_amount']/df['total_amount_due'])*100, 4)
df['visiting_time'] = round((df['visiting_time']/60),2)


# In[226]:


df.head()


# In[227]:


viz_df = df.drop(['browser', 'campaign', 'previous_visitor', 'day_of_week', 'traffic_source'], axis=1)
cols = ['visiting_time', 'total_amount_due', 'previous_payment_amount', '%_payment', 'converted']
viz_df = viz_df.reindex(columns=cols)
viz_df.head()


# In[228]:


def scatterContinous(x, y):
    fig, ax = plt.subplots(1, 1, sharex = True, figsize=(6,3))
    ax.scatter(x, y)
    ax.set_xlabel(x.name)
    ax.set_ylabel('Converted')


# In[229]:


continuous = [df.visiting_time, df.total_amount_due, df.previous_payment_amount]
for x in continuous:
    print(f'{x.name}: {x.mean()} \n')
    scatterContinous(x, df.converted)


# In[230]:


import seaborn as sns
import matplotlib.gridspec as gridspec

gs = gridspec.GridSpec(28, 1)
plt.figure(figsize=(6,28*4))
for i, col in enumerate(viz_df[viz_df.iloc[:,0:4].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(viz_df[col][viz_df.converted == 0], kde=True, bins=50, color='r')
    sns.distplot(viz_df[col][viz_df.converted == 1], kde=True, bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_ylabel('# of conversions / non-conversions')
    ax5.set_title('feature: ' + str(col))
plt.show()


# ## Decision Tree Classifier

# ### Use Decision Tree for Feature Predictive Accuracy

# In[231]:


# encode = df
# le = LabelEncoder()
# encode['browser'] = le.fit_transform(df.browser)
# encode['day_of_week'] = le.fit_transform(df.day_of_week)
# encode['campaign'] = le.fit_transform(df.campaign)
# encode['traffic_source'] = le.fit_transform(df.traffic_source)

encode = pd.get_dummies(df)
# encode = encode.drop('%_payment', axis=1)
encode.head(10)


# In[232]:


X = encode.drop(['converted', 'visiting_time', 'total_amount_due', 'previous_payment_amount'], axis=1)
y = encode['converted'].values


# In[233]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

DTC = DecisionTreeClassifier()


# In[234]:


DTC = DTC.fit(X_train, y_train)

print('Train Accuracy Score:', DTC.score(X_train, y_train))
print('Test Accuracy Score:', DTC.score(X_test, y_test))


# In[235]:


y_pred = DTC.predict(X_test)


# In[236]:


confusion_matrix(y_test, y_pred)


# In[237]:


pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)


# In[238]:


print(classification_report(y_test, y_pred))


# In[239]:


y_pred_proba = DTC.predict_proba(X_test)[:,1]

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='KNN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('DT ROC Curve')
plt.show()


# In[240]:


roc_auc_score(y_test, y_pred_proba)


# In[241]:


del df, encode
gc.collect()


# In[242]:


features = X.columns
dot_data = StringIO()
export_graphviz(DTC, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=features, class_names=['Not Converted','Converted'])


# In[ ]:


graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# graph.write_png('Conversion_Path.png')
Image(graph.create_png())


# In[244]:


dtc = DecisionTreeClassifier(criterion='entropy', max_depth=6)
dtc = dtc.fit(X_train, y_train)
print('Train Accuracy Score:', dtc.score(X_train, y_train))
print('Test Accuracy Score:', dtc.score(X_test, y_test))


# In[245]:


y_pred = dtc.predict(X_test)
confusion_matrix(y_test, y_pred)


# In[246]:


pd.crosstab(y_test, y_pred, rownames=['Real'], colnames=['Predicted'], margins=True)


# In[247]:


y_pred_proba = dtc.predict_proba(X_test)[:,1]


# In[248]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot([0,1],[0,1], 'k--')
plt.plot(fpr, tpr, label='KNN')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('DT ROC Curve')
plt.show()


# In[249]:


roc_auc_score(y_test, y_pred_proba)


# In[ ]:





# In[ ]:





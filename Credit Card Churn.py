#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,roc_curve,classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[2]:


df=pd.read_csv("C:/games/codsoft/Credit Card Churn/Churn_Modelling.csv")
df


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


sns.countplot(x = df["Exited"], data = df)


# In[6]:


df.drop(['RowNumber', 'CustomerId', 'Surname','Geography','Gender'],axis=1,inplace=True)


# In[7]:


x=df.drop(['Exited'],axis=1)
y=df['Exited']


# In[8]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[9]:


LR_model=LogisticRegression()
LR_model.fit(X_train,y_train)

RFC_model=RandomForestClassifier()
RFC_model.fit(X_train,y_train)


# In[10]:


LR_pred = LR_model.predict(X_test)

accuracy = accuracy_score(y_test, LR_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, LR_pred))

RFC_pred = RFC_model.predict(X_test)

accuracy = accuracy_score(y_test, RFC_pred)
print("Validation Accuracy:", accuracy)
print(classification_report(y_test, RFC_pred))


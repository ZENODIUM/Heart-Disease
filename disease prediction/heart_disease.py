#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


heart_data = pd.read_csv('heart.csv')


# In[4]:


heart_data.head()


# In[9]:


heart_data.info()


# In[6]:


heart_data.shape


# In[7]:


heart_data.info()


# In[10]:


heart_data['target'].value_counts()


# In[13]:


X = heart_data.drop(columns='target', axis = 1) # column, axis = 1
Y = heart_data['target']
X


# In[14]:


Y


# In[15]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[16]:


print(X.shape,X_train.shape,X_test.shape)


# In[29]:


model = LogisticRegression(solver='lbfgs', max_iter=1100)


# In[30]:


model.fit(X_train,Y_train)


# In[31]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)


# In[35]:


print("Train Accuracy" , training_data_accuracy)


# In[33]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)


# In[36]:


print("Test Accuracy", test_data_accuracy)

pickle.dump(model,open('saved_model','wb'))

# In[38]:


#Building a acutal system that predicts value by taking inputs
input_data = (52,1,0,125,212,0,1,168,0,1,2,2,3)


input_npar = np.asarray(input_data)

#reshaping as we are predicting for only one value 
input_reshaped = input_npar.reshape(1,-1) # for on instance

loaded_model = pickle.load(open('saved_model','rb'))


prediction = loaded_model.predict(input_reshaped)
if (prediction[0] == 0):
    print("Safe")
else:
    print("Heart disease")


# In[ ]:





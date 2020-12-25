#!/usr/bin/env python
# coding: utf-8

# In this notebook, let us build our model using LinearSVC which is best for text classification problems.

# **Objective:**

# Develop algorithms to classify genetic mutations based on clinical evidence (text)

# In[3]:


import numpy as np 
import pandas as pd
import os


# Training Data

# In[6]:
os.chdir('/Users/samruddhimhatre/Documents/Proxima/Data')
source= '/Users/samruddhimhatre/Documents/Proxima/Data'

training_variants = pd.read_csv("training_variants.txt")


# In[7]:


training_variants.head(5)


# In[9]:


training_text = pd.read_csv("training_text.txt",sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[10]:


training_text.head(5)


# In[12]:


training_text["Text"][0]


# In[13]:


training_merge = training_variants.merge(training_text,left_on="ID",right_on="ID")


# In[19]:


training_merge.head()


# In[20]:


training_merge.columns


# Testing Data

# In[21]:


testing_variants = pd.read_csv("test_variants.txt")


# In[22]:


testing_variants.head()


# In[24]:


testing_text = pd.read_csv("test_text.txt", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])


# In[25]:


testing_text.head()


# In[26]:


testing_merge = testing_variants.merge(testing_text,left_on="ID",right_on="ID")


# In[27]:


testing_merge.head()


# In[28]:


training_merge["Class"].unique()


# Describing both Training and Testing data

# In[29]:


training_merge.describe()


# In[30]:


testing_merge.describe()


# Split the training data to train and test for checking the model accuracy

# In[31]:


from sklearn.model_selection import train_test_split

train ,test = train_test_split(training_merge,test_size=0.2) 
np.random.seed(0)
train


# In[34]:


x_train = train['Text'].values
x_test = test['Text'].values
y_train = train['Class'].values
y_test = test['Class'].values


# In[62]:


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import svm
from xgboost import XGBClassifier


# Set pipeline to build a complete text processing model with Vectorizer, Transformer and LinearSVC

# In[83]:


text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', XGBClassifier())
])
text_clf = text_clf.fit(x_train,y_train)


# In[ ]:





# Getting 65% accuracy with only LinearSVC. Try different ensemble models to get more accurate model. 

# In[84]:


y_test_predicted = text_clf.predict(x_test)
np.mean(y_test_predicted == y_test)


# In[85]:


X_test_final = testing_merge['Text'].values


# In[86]:


predicted_class = text_clf.predict(X_test_final)


# In[87]:


testing_merge['predicted_class'] = predicted_class


# Appended the predicted values to the testing data

# In[89]:


testing_merge.head(5)


# Onehot encoding to get the predicted values as columns

# In[92]:


onehot = pd.get_dummies(testing_merge['predicted_class'])
testing_merge = testing_merge.join(onehot)


# In[91]:


testing_merge


# Preparing submission data

# In[54]:


# submission = testing_merge[["ID",1,2,3,4,5,6,7,8,9]]
# submission.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
# submission.head(5)


# In[93]:


import joblib 
joblib.dump(text_clf, 'filename.pkl') 


# If you really feel this will help you. Please upvote this and encourage me to write more. 

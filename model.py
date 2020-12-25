# Text analysis helper libraries
from gensim.summarization import summarize
from gensim.summarization import keywords

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer # caring to care
from nltk.corpus import stopwords
from string import punctuation

#from scipy.misc import imresize (did not run)
from PIL import Image
import numpy as np
from collections import Counter

# Word2Vec related libraries
from gensim.models import KeyedVectors
from sklearn import model_selection

#Importing dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from joblib import dump,load

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import os
os.chdir('/Users/samruddhimhatre/Desktop/Proxima/Data')
source= '/Users/samruddhimhatre/Desktop/Proxima/Data'
train_variant = pd.read_csv('training_variants.txt')
test_variant = pd.read_csv('test_variants.txt')
print(test_variant.head())
print(train_variant.head())
print(train_variant.shape)
print(test_variant.shape)
train_m = train_variant.isnull().sum()
print(train_m)
test_m = test_variant.isnull().sum()
print(test_m)
train_data = train_variant.dropna(axis = 0, how = "any")
test_data = test_variant.dropna(axis = 0, how = "any")
print(train_data)
print(test_data)
train_data["Class"].unique()
train_data["Gene"].unique()
train_data["Gene"].value_counts()
train_data["Variation"].unique()
train_data["Variation"].value_counts()
sns.countplot(x = train_data["Class"])
plt.show()
sns.countplot(x = train_data["Gene"], hue = train_data["Class"])
plt.show()
train_data.groupby(["Gene"])["Class"].value_counts()
train_data.groupby(["Variation"])["Class"].value_counts()
str_data = train_data.select_dtypes(include = ["object"])
str_dt = test_data.select_dtypes(include = ["object"])
int_data = train_data.select_dtypes(include = ["integer", "float"])
int_dt = test_data.select_dtypes(include = ["integer", "float"])

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
feature = str_data.apply(label.fit_transform)
feature = feature.join(int_data)
feature.head()

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()
Test = str_dt.apply(label.fit_transform)
Test = Test.join(int_dt)
print(Test.head())

from sklearn.preprocessing import OneHotEncoder 
onehotencoder = OneHotEncoder() 
data = onehotencoder.fit_transform(feature).toarray() 

y_train = feature["Class"]
y_train.head()
x_train = feature.drop(["Class", "ID"] ,axis = 1)
print(x_train.head())
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)
print(x_train.head())
print(y_train.head())

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=5000,max_depth=23,min_samples_split=10, random_state=10)

classifier.fit(x_train,y_train)
y_pred=classifier.predict_proba(x_test)

from sklearn import metrics
#print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
submit = pd.DataFrame(Test.ID)
submit = submit.join(pd.DataFrame(y_pred))
submit.columns = ['ID', 'class1','class2','class3','class4','class5','class6','class7','class8','class9']
print(submit.columns)
submit.to_csv('submission.csv', index=False)


import joblib 
joblib.dump(classifier, 'filename.pkl') 

# Save the trained model as a pickle string.
#saved_model = pickle.dumps(classifier)

# Load the pickled model


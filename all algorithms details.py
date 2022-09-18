#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", color_codes=True)
import warnings # current version of seaborn generates a bunch of warnings that we'll ignore
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVC


# In[4]:


iris = pd.read_csv("Iris (1).csv") # the iris dataset is now a Pandas DataFrame

# Let's see what's in the iris data - Jupyter notebooks print the result of the last thing you do
iris.head()


# In[5]:


iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")


# In[6]:


sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)


# In[7]:


sns.FacetGrid(iris, hue="Species", size=5)    .map(plt.scatter, "SepalLengthCm", "SepalWidthCm")    .add_legend()


# In[8]:


sns.boxplot(x="Species", y="PetalLengthCm", data=iris)


# In[9]:


ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray")


# In[10]:


sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)


# In[30]:


sns.FacetGrid(iris, hue="Species", size=6)    .map(sns.kdeplot, "PetalLengthCm")    .add_legend()


# In[31]:


sns.pairplot(iris.drop("Id", axis=1), hue="Species", size=3)


# In[32]:


from pandas.plotting import andrews_curves
andrews_curves(iris.drop("Id", axis=1), "Species")


# In[33]:


from pandas.plotting import parallel_coordinates
parallel_coordinates(iris.drop("Id", axis=1), "Species")


# In[34]:


from pandas.plotting import radviz
radviz(iris.drop("Id", axis=1), "Species")


# In[35]:


X = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[36]:


# LogisticRegression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('LogisticRegression accuracy is',accuracy_score(y_pred,y_test))


# In[51]:


# K-Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=8)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('K-Nearest Neighbours accuracy is',accuracy_score(y_pred,y_test))


# In[59]:


print("These are test values:", y_test,"...............................",
      "These are predicted values:" ,y_pred)


# In[39]:


# Support Vector Machine's 
from sklearn.svm import SVC

classifier = SVC()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Support Vector Machine accuracy is',accuracy_score(y_pred,y_test))


# In[40]:


# neural_network
from sklearn.neural_network import MLPClassifier

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('neural_network accuracy is',accuracy_score(y_pred,y_test))


# In[41]:


# Decision Tree's
from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Decision Tree accuracy is',accuracy_score(y_pred,y_test))


# In[42]:


# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Gaussian Naive Bayes accuracy is',accuracy_score(y_pred,y_test))


# In[43]:


# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Multinomial Naive Bayes accuracy is',accuracy_score(y_pred,y_test))


# In[65]:


# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train, y_train)

a_pred = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Bernoulli Naive Bayes accuracy is',accuracy_score(y_pred,y_test))


# In[66]:


print("These are test values:", y_test,"...............................",
      "These are predicted values:" ,a_pred)


# In[60]:


# Complement Naive Bayes
from sklearn.naive_bayes import ComplementNB
classifier = ComplementNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)g

# Summary of the predictions made by the classifier
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Complement Naive Bayes accuracy is',accuracy_score(y_pred,y_test))


# In[61]:


print(y_pred)


# In[ ]:


from sklearn.metrics import accuracy_score, log_loss
classifiers = [
    GaussianNB(),
    MultinomialNB(),
    BernoulliNB(),
    ComplementNB(),               
                  ]
 
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)
 
for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print('****Results****')
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print("Accuracy: {:.4%}".format(acc))
    
    log_entry = pd.DataFrame([[name, acc*100, 11]], columns=log_cols)
    log = log.append(log_entry)
    
    print("="*30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





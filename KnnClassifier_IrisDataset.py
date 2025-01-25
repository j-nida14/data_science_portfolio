#!/usr/bin/env python
# coding: utf-8

# In[40]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, accuracy_score



# In[41]:


#import the dataset
iris = load_iris(as_frame=True)

print(iris.data)
print(iris.target)


# In[42]:


#iris feature and target names
print(iris.feature_names)
print(iris.target_names)


# In[43]:


#to check teh shape of the dataset
print(iris.data.shape)


# In[44]:


#spliting data into training and testing sets
X= iris.data
y= iris.target
X_train, X_test,y_train, y_test= train_test_split(X,y,stratify=y, random_state=0, test_size=0.3)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# In[45]:


# test the accuracy of the kNN method on the test set for various values of k.
for k in range(1,20):
  classifier = KNeighborsClassifier(n_neighbors=k)    # build kNN
  classifier.fit(X_train, y_train)                    # train kNN
  y_pred = classifier.predict(X_test)                 # predict on test set

  accuracy = accuracy_score(y_test, y_pred)*100
  print(f":kNN accuracy (k={k}) is {accuracy:.2f}%")


# In[46]:





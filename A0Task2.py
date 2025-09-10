#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!jupyter nbconvert --to script A0Task2.ipynb 
# converts Jupyter notebook to python script 
# uncomment first line and run the cell to do so


# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[3]:


X_train = pd.read_csv('Data/train_in.csv', header=None).values
y_train = pd.read_csv('Data/train_out.csv', header=None).values.flatten()
X_test = pd.read_csv('Data/test_in.csv', header=None).values
y_test = pd.read_csv('Data/test_out.csv', header=None).values.flatten()
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)


# In[4]:


X_train_bias = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_test_bias = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
print("Train with bias shape:", X_train_bias.shape)
print("Test with bias shape:", X_test_bias.shape)


# In[13]:


num_features = X_train_bias.shape[1]
num_classes = 10


# In[6]:


def predict(X, W):
    scores = X @ W
    return np.argmax(scores, axis=1)

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


# In[14]:


np.random.seed(123)
W = np.random.randn(num_features, num_classes) * 0.01

epochs = 100
lr = 0.01
train_acc = []
test_acc = []

for epoch in range(epochs):
    for i in range(X_train_bias.shape[0]):
        xi = X_train_bias[i]
        yi = y_train[i]
        scores = xi @ W
        pred = np.argmax(scores)
        if pred != yi:
            W[:, yi] += lr * xi
            W[:, pred] -= lr * xi
    
    train_pred = predict(X_train_bias, W)
    test_pred = predict(X_test_bias, W)
    train_acc.append(accuracy(y_train, train_pred))
    test_acc.append(accuracy(y_test, test_pred))
    print(f"Epoch {epoch+1}: Train Acc={train_acc[-1]:.4f}, Test Acc={test_acc[-1]:.4f}")


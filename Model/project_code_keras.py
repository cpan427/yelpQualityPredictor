# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 00:39:00 2018

@author: John Martin
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import glorot_uniform
from read_data_oh import readData

def buildModel(n_x, unitsInLayer, optimizer ='adam', loss ='mse'):
    
    model = Sequential()
    
    #Input layer
    model.add( Dense(unitsInLayer[0], input_shape=(n_x,), activation ='relu', use_bias =True, kernel_initializer='glorot_uniform') )
    
    #Intermediate layer
    for i in range(len(unitsInLayer)-1):
        model.add( Dense(unitsInLayer[i+1], activation ='relu', use_bias =True, kernel_initializer='glorot_uniform') )
    
    #Last layer with no activation for regression
    model.add(Dense(1, kernel_initializer='glorot_uniform'))
    #Compile model
    model.compile(optimizer =optimizer, loss='mse')
    
    return model

#main

num_epochs = 10
batch_size = 32

X_train = None
Y_train = None
X_dev = None
Y_dev = None

X_test = None
Y_test = None

X_train,   Y_train   = readData('./data/100k_oneHot_train.csv')
X_dt, Y_dt = readData('./data/20k_oneHot_dev.csv')
X_dev,   Y_dev   = X_dt[:,0:5000], Y_dt[:,0:5000]
X_test,  Y_test  = X_dt[:,5001:], Y_dt[:,5001:]

n_x = X_train.shape[0]
hiddenUnits = (
        """
                [2 ,2 ,2 ,2 ] These are pretty good with the one hot
               ,[2, 2]
               ,[1, 1]
               """
               ,[5, 5, 5]
               ,[10, 10, 10]
               ,[2, 1, 2, 1, 2]
               ,[10, 20, 10]
               )
               

for i in range(len(hiddenUnits)):
    model = buildModel(n_x, hiddenUnits[i])
    
    print '\n\n\nTRAINING MODEL %s' %(i+1)
    #Train the model
    model.fit(X_train.T, Y_train.T,
              batch_size =batch_size, epochs =num_epochs,
              validation_data=(X_dev.T, Y_dev.T))


    prediction = model.predict(X_test.T)
    print(np.sum(Y_test>0), np.sum(Y_test==0), np.sum(Y_test<5))
    print(np.sum(Y_test))
    print(np.sum(prediction>0), np.sum(prediction==0), np.sum(prediction<5))
    print(np.sum(prediction))
    
    print(np.mean(prediction-Y_test.T))
   

#Score on test set
#test_result = model.evaluate(X_test.T, Y_test.T, batch_size =batch_size)
#print ("Loss = " + str(test_result[0]))
#print ("Test Accuracy = " + str(test_result[1]))
#print(test_result)

model.summary()



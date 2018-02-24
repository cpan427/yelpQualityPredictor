# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:45:53 2018

@author: John Martin
"""
import numpy as np
import time
import pandas

def parseLine(line):
    parts = line.split(',')
    
    x_l = np.concatenate((np.array(parts[2:4]) , np.array(parts[5:])))
    x = []
    for i in range(len(x_l)):
        x.append(int(x_l[i]))
    print(i)
    y = int(parts[4])
    return x, y

def readData(fileName = './data/100k_oneHot_train.csv'):
    DATA_FILE = fileName
    
    start = time.time()
    indices = []
    for i in range(13514):
        if i not in (0, 1, 4):
            indices.append(i)
    
    x = pandas.read_csv(DATA_FILE, usecols = indices, nrows =10000)
    print ('Time to read X')
    print (time.time()-start)
    start = time.time()
    print ('Time to read Y')
    y = pandas.read_csv(DATA_FILE, usecols = [4], nrows =10000)
    print(time.time()-start)
    
    # Step 1: read in data from the .txt file
    """
    inputFile = open(DATA_FILE,'r')
    
    singleLine = inputFile.readline()
    
    x = []
    y = []
    i = 0
    start = time.time()
    for singleLine in inputFile:
        
        
        if (i+1)%1000 == 0:
            print'Reading %s'%(i+1)
            print(time.time()-start)
        if (i+1)>1:
            break

        new_x,new_y = parseLine(singleLine)
        x.append(new_x)
        y.append(new_y)
        i+=1
        
    inputFile.close()
    """
    x_matrix = np.array(x)
    y_matrix = np.array(y)
    y_matrix = y_matrix.reshape((y_matrix.shape[0],1))
    return x_matrix.T, y_matrix.T

#readData(fileName = './data/20k_oneHot_dev.csv')
readData()

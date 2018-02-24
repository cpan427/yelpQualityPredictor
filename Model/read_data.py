# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 01:45:53 2018

@author: John Martin
"""
import numpy as np

def parseLine(line):
    parts = line.split(',')
    x = [int(parts[2]),
         int(parts[3]),
         int(parts[5]),
         int(parts[6]),
    ]
    y = int(parts[4])
    return x, y

def readData(fileName = './data/100k_numDate_train.csv'):
    DATA_FILE = fileName
    
    # Step 1: read in data from the .txt file
    inputFile = open(DATA_FILE,'r')
    
    singleLine = inputFile.readline()
    
    x = []
    y = []
    i = 0
    for singleLine in inputFile:
        
        
        if (i+1)%10000 == 0:
            print'Reading %s'%(i+1)

        new_x,new_y = parseLine(singleLine)
        x.append(new_x)
        y.append(new_y)
        i+=1
        
    inputFile.close()
    
    x_matrix = np.array(x)
    y_matrix = np.array(y)
    y_matrix = y_matrix.reshape((y_matrix.shape[0],1))
    return x_matrix.T, y_matrix.T

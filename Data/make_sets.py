import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DEV_SIZE = 250000
TEST_SIZE = 250000

# open entire data set
ttl = pd.read_csv('merged_raw_set.csv')
print("Opened data set")

TOTAL_NUM = len(ttl)

train, combo = train_test_split(ttl, test_size = float(DEV_SIZE + TEST_SIZE)/TOTAL_NUM)

dev, test = train_test_split(combo, test_size=TEST_SIZE/float(DEV_SIZE + TEST_SIZE))

# Writing to the files
print ("Writing to the files")
train.to_csv('train.csv')
print ("Train done; ", len(train), "examples")
dev.to_csv('dev.csv')
print ("Dev done; ", len(dev), "examples")
test.to_csv('test.csv')
print ("Test done; ", len(test), "examples")



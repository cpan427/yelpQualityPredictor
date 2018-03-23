import pandas as pd
import numpy as np
from collections import defaultdict
import csv



#vocabRead = open("words.txt", 'r')
#vocab = vocabRead.readlines()


#trainLab = open("answers_train.txt")

wordFreq = defaultdict(int)

def updateWords(sent):
	sent = sent.rstrip()
	words = sent.split()
	seenAlready = set()
	for w in words:
		if w not in seenAlready:
			wordFreq[w] += 1
			seenAlready.add(w)

trainSent = open("text_train.txt")
trainSent = trainSent.readlines()
for sent in trainSent:
	updateWords(sent)


devSent = open("text_dev.txt")
devSent = devSent.readlines()
for sent in devSent:
	updateWords(sent)

testSent = open("text_test.txt")
testSent = testSent.readlines()
for sent in testSent:
	updateWords(sent)

# Taken from the python documentation
file = open('freqs.csv', 'w', newline='\n')
fieldnames = ['word', 'appearanceCount']
writer = csv.DictWriter(file, fieldnames=fieldnames)
writer.writeheader()
for k in wordFreq.keys():
	temp = {}
	temp['word'] = k
	temp['appearanceCount'] = wordFreq[k]
	writer.writerow(temp)
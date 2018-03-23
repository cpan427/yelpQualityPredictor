import pandas as pd
import numpy as np
import re
import random as rand

pattern = re.compile('\\n')
noPunc = re.compile('[@#$%^&*(){\\\/?.!\'\",]:-')

print ("Starting...")

rev = pd.read_csv('yelp_review.csv')

text = rev['text'].as_matrix()
use = rev['useful'].as_matrix()

wordsTest = open("text_test.txt", "w")
revsTest = open("answers_test.txt", "w")
succTestCount = 0
succTestCountLab = 0

wordsDev = open("text_dev.txt", "w")
revsDev = open("answers_dev.txt", "w")
succDevCount = 0
succDevCountLab = 0

wordsTrain = open("text_train.txt", "w")
revsTrain = open("answers_train.txt", "w")
succCount = 0
succCountLab = 0

excluded = 0

print ("Looking at sentences")

sentLens = []
usefulDist = []

def chooseBatch(wordsTest, revsTest, wordsDev, revsDev, wordsTrain, revsTrain):
	num = rand.random()
	if (num <= 250000/5200000):
		return wordsTest, revsTest, 0
	elif (num <= 500000/5200000):
		return wordsDev, revsDev, 1
	return wordsTrain, revsTrain, 2


for i in range(len(text)):
	if (i % 100000 == 0):
		print ("Finished", i, "reviews")
	doneSent = False
	doneLab = False
	try:
		wrd = str(text[i].encode('ascii').decode('ascii'))
		wrd = wrd.rstrip()
		
		#wrd = wrd.replace('\n', '')
		#wrd = re.sub(pattern, ' ', wrd)
		wrd = wrd.lower()
		wrd = wrd.replace('\'', '')
		wrd = re.sub('[^0-9a-zA-Z]+', ' ', wrd)

		sent = wrd.split(" ")
		if (len(sent) <= 50):
			if (i % 10000 == 0):
				print (wrd)
			sentLens.append(len(sent))

			words, revs, whichSet = chooseBatch(wordsTest, revsTest, wordsDev, revsDev, wordsTrain, revsTrain)
			#print (wrd)
			words.write(wrd + "\n")
			if (whichSet == 0):
				succTestCount += 1
			elif (whichSet == 1):
				succDevCount += 1
			else:
				succCount += 1
			doneSent = True

			usefulDist.append(use[i])
			revs.write(str(use[i]) + "\n")
			
			if (whichSet == 0):
				succTestCountLab += 1
			elif (whichSet == 1):
				succDevCountLab += 1
			else:
				succCountLab += 1
			doneLab = True
		else:
			excluded += 1
	except UnicodeEncodeError:
		excluded += 1
	if (doneLab != doneSent):
		print (text[i], fun[i])

print ("Cool")
print ("Number of rows for train:", succCount, succCountLab)
print ("Number of rows for dev:", succDevCount, succDevCountLab)
print ("Number of rows for test:", succTestCount, succTestCountLab)
print ("Number of excluded (long and non-ASCII and non English) reviews:", excluded)
print ("Total Number of reviews included", (len(text) - excluded))
print (np.percentile(sentLens,[0, 25, 50, 75, 80, 90, 95, 100]))
print (np.percentile(usefulDist,[0, 25, 50, 75, 80, 90, 95, 100]))
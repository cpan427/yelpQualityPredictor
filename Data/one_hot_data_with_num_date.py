import csv
import time, datetime
import calendar
import collections
import re
from random import *

REVIEW_ID_COL = 0;
USER_ID_COL = 1
BUSINESS_ID_COL = 2
STARS_COL = 3
DATE_COL = 4
TEXT_COL = 5
USEFUL_COL = 6
FUNNY_COL = 7
COOL_COL = 8

DEV_SIZE = 25000
TEST_SIZE = 25000
TRAIN_SIZE = 50000
NUM_REVIEW = 5260000
DEV_RNG = DEV_SIZE/NUM_REVIEW
TEST_RNG = DEV_RNG + TEST_SIZE/NUM_REVIEW

trainSz = 0
devSz = 0
testSz = 0


pattern = re.compile('\W')

with open("yelp_review.csv", encoding="utf8") as csvfile:
	def beautifyDate(res): 
		# This function returns a floating point that gives the UTC
		#print (res)
		dt = time.strptime(res, '%Y-%m-%d')
		return calendar.timegm(dt)

	def cleanString(text):
		"""
		Things to note about the code: this code include punctuation and immediately adds non ASCII
		friendly into the <unk> pile
		"""
		res = collections.defaultdict(int)
		strings = text.lower()
		strings = strings.split(" ")
		for wrd in strings:
			try:
				wrd = re.sub(pattern, '', wrd)
				#print (wrd)
				wrd.encode('ascii')
				res[wrd] += 1
			except UnicodeEncodeError:
				#print (":( ", wrd)
				res["<unk>"] += 1
		return res

	def addOneHots(curr, vocab, text, isHeader):
		#print (text)
		if isHeader:
			for wRow in vocab:
				curr.append(wRow[0])
			return curr
		nUnks = 0
		words = cleanString(text)
		for wRow in vocab:
			wrd = wRow[0]
			if words[wrd] != 0:
				curr.append(words[wrd])
				del words[wrd]
			else:
				curr.append(0)
		for w in words:
			curr[-1] += words[w]
		return curr

	def addToFile(curr, trainSz, devSz, testSz):
		rNum = random()
		if (rNum < DEV_RNG && devSz <= DEV_SIZE):
			devWriter.writerow(curr)
			devSz += 1
			return trainSz, devSz, testSz
		if (rNum < TEST_RNG && testSz <= TEST_SIZE):
			testWriter.writerow(curr)
			testSz += 1
			return trainSz, devSz, testSz
		trainWriter.writerow(curr)
		trainSz += 1
		return trainSz, devSz, testSz


	# opening train file
	toyTrain = open("train.csv", 'w')
	trainWriter = csv.writer(toyTrain, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')


	# opening dev file
	toyDev = open("dev.csv", 'w')
	devWriter = csv.writer(toyDev, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
	

	# opening test file
	toyTest = open("test.csv", 'w')
	testWriter = csv.writer(toyTest, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
	

	# opening vocab file
	vocab = open("10k_vocab.csv")
	vocabList = list(csv.reader(vocab, delimiter=','))

	# Reading list of reviews
	print ("Creating List....")
	readCSV = list(csv.reader(csvfile, delimiter=','))
	print ("Number of examples:", len(readCSV))

	excludeSet = {REVIEW_ID_COL, TEXT_COL}


	fieldNames = readCSV[0]
	for i in excludeSet:
		fieldNames.remove(fieldNames[i])
	words = []
	for i in range(len(vocabList) -1):
		words.append(vocabList[i+1][0])
	fieldNames.extend(words)
	trainWriter.writerow(fieldNames)
	devWriter.writerow(fieldNames)
	testWriter.writerow(fieldNames)

	#print(fieldNames)
	addedCount = 0

	for row in readCSV[1:]:
		curr = []
		for i in range(len(row)):
			if i not in excludeSet:
				res = row[i]
				if (i == DATE_COL): 
					res = beautifyDate(res)
				curr.append(res)
				#print(row[i])
		curr = addOneHots(curr, vocabList, row[TEXT_COL], (i == 0))
		trainSz, devSz, testSz = addToFile(curr, trainSz, devSz, testSz)
		addedCount += 1
		if (addedCount % 1000 == 0):
			print ("Finished adding", addedCount, "rows")
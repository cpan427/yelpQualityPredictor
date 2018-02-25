"""
This file goes through the data to find the frequencies of words in the corpus
"""

import csv
import time, datetime
import calendar
from collections import defaultdict
import chardet
import re

REVIEW_ID_COL = 0;
USER_ID_COL = 1
BUSINESS_ID_COL = 2
STARS_COL = 3
DATE_COL = 4
TEXT_COL = 5
USEFUL_COL = 6
FUNNY_COL = 7
COOL_COL = 8

pattern = re.compile('\W')

with open("yelp_review.csv", encoding="utf8") as csvfile:
	wordFrequencies = defaultdict(int)
	def beautifyDate(res): 
		# This function returns a floating point that gives the UTC
		# print (res)
		dt = time.strptime(res, '%Y-%m-%d')
		return calendar.timegm(dt)

	def getAsciiFriendlyString(text, wordFrequencies):
		"""
		Things to note about the code: this code include punctuation and immediately adds non ASCII
		friendly into the <unk> pile
		"""
		strings = text.lower()
		strings = strings.split(" ")
		for wrd in strings:
			try:
				wrd = re.sub(pattern, '', wrd)
				#print (wrd)
				wrd.encode('ascii')
				wordFrequencies[wrd] += 1
			except UnicodeEncodeError:
				#print (":( ", wrd)
				wordFrequencies["<unk>"] += 1

	#getAsciiFriendlyString("mooing!@ cows are the best", wordFrequencies)
	#print (len(wordFrequencies))
	#for wrd in wordFrequencies:
		#print (wrd, wordFrequencies[wrd])
		#wrdFrqWriter.writerow([wrd])
	toyTrain = open("100k_numDate_train.csv", 'w')
	toyWriter = csv.writer(toyTrain, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
	print ("Creating List....")
	readCSV = list(csv.reader(csvfile, delimiter=','))
	print ("Finished creating list....")
	print ("Number of examples:", len(readCSV))
	excludeSet = {REVIEW_ID_COL};

	fieldNames = readCSV[0]
	print(fieldNames)

	readForOneHot = readCSV[1:]

	print ("Going through the words for the frequencies.")
	# Go through the set, finding the frequencies
	for row in readForOneHot:
		getAsciiFriendlyString(row[TEXT_COL], wordFrequencies)

	print (len(readForOneHot))
	# Write the frequencies to a file (so we don't have to do this again.....)
	print ("creating file with word frequencies")

	wrdFrq = open("yelp_word_frequencies.csv", 'w')
	wrdFrqWriter = csv.writer(wrdFrq, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL)
	wrdFrqWriter.writerow(["word", "frequency"])
	for wrd in wordFrequencies:
		wrdFrqWriter.writerow([wrd, wordFrequencies[wrd]])




		

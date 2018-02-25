import csv

MIN_APPEARANCES = 1000

with open("yelp_word_frequencies.csv") as csvfile:
	readCSV = list(csv.reader(csvfile, delimiter=','))
	toyTrain = open("10k_vocab.csv", 'w')
	toyWriter = csv.writer(toyTrain, delimiter=',',
		quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator = '\n')
	toyWriter.writerow(readCSV[0])
	count = 0
	readCSV = readCSV[1:]
	i = 1
	unkCount = 0
	# this code assumes that the CSV is skipping lines (change everything to for row in ...)
	while i < len(readCSV):
		row = readCSV[i]
		#print (row)
		if (int(row[1]) > MIN_APPEARANCES):
			count += 1
			toyWriter.writerow(row)
			#print (row)
		else:
			unkCount += int(row[1])
		i += 2

	toyWriter.writerow(["<unk>", unkCount])
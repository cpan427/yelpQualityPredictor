import pandas as pd
import numpy as np
from keras.layers import Dense, Embedding, Flatten, LSTM
from keras.models import Sequential
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras import optimizers

# Open the vocabulary set
vocabRead = open("words.txt", 'r')
vocab = vocabRead.readlines()

#Used this for help: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# def convertSentenceToIndices(sent, index, trainX):
# 	sent = sent.rstrip()
# 	words = sent.split()
# 	#print (words)
# 	i = 0
# 	for w in words:
# 		trainX[index, i] = wordToIndex[w]
# 		i += 1
# 	#paddingLen = maxSentLen - len(words)
# 	while i != maxSentLen:
# 		trainX[index, i] = pad_index
# 		i += 1
# 	#print (res)

# def sentencesToIndices(data):
# 	trainX = np.zeros((len(data), maxSentLen))
# 	index = 0
# 	for sent in data:
# 		convertSentenceToIndices(sent, index, trainX)
# 		index += 1
# 	return trainX

def getNumLabels(data):
	trainY = np.zeros((len(data), 1))
	index = 0
	for num in data:
		num = int(num)
		trainY[index] = (num)
		index += 1
	return trainY

input_dim = 500
output_dim = 50
maxSentLen = 50
vocab_size = len(vocab)
pad_index = 0

# Creating embedding matrix
# From: https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html

embeddings_index = {}
f = open('glove.6B.50d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size + 1, 50))
i = 0
for word in vocab:
    word = word.rstrip()
    #print(word)
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        i += 1
print (embedding_matrix.shape)

# importing the training set
trainSent = open("train/sentences.txt")
trainLab = open("train/labels.txt")
trainSentences = trainSent.readlines()
trainLabels = trainLab.readlines()

# importing the dev set
devSent = open("dev/sentences.txt")
devLab = open("dev/labels.txt")
devSentences = devSent.readlines()
devLabels = devLab.readlines()

# importing the test set
testSent = open("test/sentences.txt")
testLab = open("test/labels.txt")
testSentences = testSent.readlines()
testLabels = testLab.readlines()

# create embeddings for training set
docs = trainSentences
encoded_docs = [one_hot(d, vocab_size) for d in docs]
padded_docs = pad_sequences(encoded_docs, maxlen = maxSentLen, padding='post')
trainY = getNumLabels(trainLabels)

#print ("Example of training example:\n", padded_docs[15])
#print ("Corresponding output:\n", trainY[15])

# create embeddings for dev set
devDocs = devSentences
encoded_dev_docs = [one_hot(d, vocab_size) for d in devDocs]
padded_dev_docs = pad_sequences(encoded_dev_docs, maxlen = maxSentLen, padding='post')

# create embeddings for test set (NOTE: ONLY DO THIS AT THE END)
testDocs = testSentences
encoded_test_docs = [one_hot(d, vocab_size) for d in testDocs]
padded_test_docs = pad_sequences(encoded_test_docs, maxlen = maxSentLen, padding='post')

# Here, we construct the model
model = Sequential()
model.add(Embedding(vocab_size+1, output_dim, weights=[embedding_matrix], input_length = 50, embeddings_initializer='random_uniform'))
model.add(LSTM(50, activation='relu', dropout = 0.3, return_sequences=True))
model.add(Flatten())
model.add(Dense(1, activation='relu'))

model.compile(optimizer='adam',
              loss='mse')
print(model.summary())

# now, train the model
model.fit(x=padded_docs,y=trainY, batch_size = 2048, epochs=10)
model.save('lstm.h5')

# evalute using the dev set
devGuesses = model.predict(padded_dev_docs)
devY = getNumLabels(devLabels)
devLoss = model.evaluate(padded_dev_docs, devY)
print (devLoss)
np.savetxt("dev_preds_lstm.csv", devGuesses, delimiter=",")
np.savetxt("dev_acts_lstm.csv", devY, delimiter=",")
devResults = open("dev_results_lstm.txt", 'w')
for i in range(len(devY)):
	devResults.write(devSentences[i] + "\t" + str(devGuesses[i]) + "\t" + str(devY[i]) + "\n")

# evalute using the test set (NOTE: ONLY AT THE END!!!)
testGuesses = model.predict(padded_test_docs)
testY = getNumLabels(testLabels)
testLoss = model.evaluate(padded_test_docs, testY)
print (testLoss)
np.savetxt("test_preds_lstm.csv", testGuesses, delimiter=",")
np.savetxt("test_acts_lstm.csv", testY, delimiter=",")
testResults = open("test_results_lstm.txt", 'w')
for i in range(len(testY)):
	testResults.write(testSentences[i] + "\t" + str(testGuesses[i]) + "\t" + str(testY[i]) + "\n")

print("We finished!")

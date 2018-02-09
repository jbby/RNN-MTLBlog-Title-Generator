import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# load ascii text and covert to lowercase
filename = "lifestyle.txt"
print("File is: ", filename)
raw_text = open(filename, encoding='UTF8').read()
raw_text = raw_text.lower()

#Map char to ints in a dict
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))

#Grab data info
n_chars = len(raw_text)
n_vocab = len(chars)
print("Total Characters: ", n_chars)
print("Total Vocab: ", n_vocab)

#Sliding window for the input 
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print("Total Patterns: ", n_patterns)

#Reshape and normalize
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
X = X / float(n_vocab)
# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#RNN model with LSTM. 2 layers, make sure return_sequences = TRUE
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))

#For resuming any training
#filename = "weights\\Newsweights-improvement-16-1.5702-bigger.hdf5"
#model.load_weights(filename)

model.compile(loss='categorical_crossentropy', optimizer='adam')
#Checkpoint
filepath="weights\\lifestyle-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

#Fit the model
model.fit(X, y, epochs=1000, batch_size=64, callbacks=callbacks_list)
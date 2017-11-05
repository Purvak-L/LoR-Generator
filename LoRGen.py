# Text generataion using Keras

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import LSTM

raw_txt = open("lor.txt",encoding='utf8').read()
# Error 'Charmap codec cannot decode ox9D format' To resolve the error add type of encoding

raw_txt = raw_txt.lower()
chars = sorted(list(set(raw_txt)))
#print(chars)

# Assigning integers to characters to construct a neural network
char_to_int = dict((character, integer) for integer, character in enumerate(chars))
#print(char_to_int)

number_of_char = len(raw_txt)
number_of_vocab = len(chars)

print('number_of_char:', number_of_char)
print('number_of_vocab',number_of_vocab)

seq_length = 100
data_X = []
data_Y = []

for i in range (0, number_of_char - seq_length,1):
	seq_in = raw_txt[i:i+seq_length]
	seq_out = raw_txt[i+seq_length]
	data_X.append([char_to_int[char] for char in seq_in])
	data_Y.append(char_to_int[seq_out])


number_of_patterns = len(data_X)
print('Number of patterns',number_of_patterns)

import numpy as np

# for keras [samples, time steps, features]

X = np.reshape(data_X,(number_of_patterns,seq_length,1))
X = X/float(number_of_vocab)
y = np_utils.to_categorical(data_Y)
'''
#print(X)
print(char_to_int)
print(raw_txt[100])
print(data_Y)
print(y[0])
'''

#LSTM MODEL

model = Sequential()
model.add(LSTM(256, input_shape = (X.shape[1],X.shape[2]),return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation = 'softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')

from keras.callbacks import ModelCheckpoint


filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)

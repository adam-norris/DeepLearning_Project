import numpy as np
import keras
from keras.preprocessing.sequence import pad_sequences

class DataFeeder(keras.utils.Sequence):

	def __init__(self, data, labels, tokenizer, batch_size, max_length, num_classes, shuffle=True):

		self.max_length = max_length
		self.batch_size = batch_size
		self.labels = labels
		self.data = data
		self.tokenizer = tokenizer
		self.num_classes = num_classes
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):

		return int(np.floor(len(self.data) / self.batch_size))

	def __getitem__(self, index):

		# Generate indexes of the batch
		indexes_temp = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Generate data
		X, y = self.__data_generation(indexes_temp)

		return X, y

	def on_epoch_end(self):
		self.indexes = np.arange(len(self.data))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, indexes_temp):

		# Initialization
		X = [self.data[x] for x in indexes_temp]	
		X_seq = self.tokenizer.texts_to_sequences(X)
		X_seq = pad_sequences(X_seq, self.max_length)	
		X_mat = self.tokenizer.texts_to_matrix(X, mode='count')
		y = [self.labels[x] for x in indexes_temp]

		return [X_seq, X_mat], keras.utils.to_categorical(y, self.num_classes)

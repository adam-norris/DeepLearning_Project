import numpy as np
import keras
from trainer.preprocess import to_ascii

from keras.preprocessing.sequence import pad_sequences

class DataFeeder(keras.utils.Sequence):

	def __init__(self, data, labels, tokenizer, mapping, batch_size, max_word_count, max_char_count, num_classes, NGRAM_SIZE, shuffle=True):

		self.max_word_count = max_word_count
		self.max_char_count = max_char_count
		self.tokenizer = tokenizer
		self.batch_size = batch_size
		self.labels = labels
		self.data = data
		self.mapping = mapping
		self.num_classes = num_classes
		self.shuffle = shuffle
		self.on_epoch_end()
		self.NGRAM_SIZE = NGRAM_SIZE

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

		# get data
		X = [self.data[x] for x in indexes_temp]
		y = keras.utils.to_categorical([self.labels[x] for x in indexes_temp], self.num_classes)

		word_sequence = self.tokenizer.texts_to_sequences(to_ascii(X))
		word_sequence = pad_sequences(word_sequence, self.max_word_count, padding='post')


		word_count = self.tokenizer.texts_to_matrix(X, mode='count')
		word_existence = self.tokenizer.texts_to_matrix(X, mode='binary')

		## generate n-grams	
		## take a look at the report to understand what's goin on here	
		char_seq = [[self.mapping[char] if char in self.mapping else 0 for char in line] for line in X]
		char_seq = pad_sequences(char_seq, self.max_char_count)

		ngram_output_dim = self.max_char_count - self.NGRAM_SIZE + 1

		ngrams = np.zeros((self.batch_size, ngram_output_dim))
		ngrams += char_seq[:, :ngram_output_dim]

		for i in range(1, self.NGRAM_SIZE):
			ngrams *= len(self.mapping)
			ngrams += char_seq[:, i:ngram_output_dim+i]


		#change this list here to feed the model with different data
		return [word_count, ngrams], y

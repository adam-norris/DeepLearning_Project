## This is the main class. Take a look at this file to unterstand what's going on.
## Have a look at model.py to see the model implementation

import keras as keras
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')


import numpy as np
import argparse
import os
import gcsfs
import sklearn
import collections

from trainer.model import create_model_ngrams
from trainer.model import create_model_word2vec
from trainer.model import create_model_wordcount
from trainer.model import create_model_ngrams_lstm
from trainer.model import concat_models
from trainer.preprocess import load_data
from trainer.preprocess import preprocess
from trainer.preprocess import load_embedding_matrix
from trainer.preprocess import to_ascii
from trainer.dataFeeder import DataFeeder
from trainer.words2vec import train_words2vec
from trainer.history import visualize_history
from trainer.history import save_history
from tensorflow.python.lib.io import file_io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD, Adam

def main(args):

	NGRAM_SIZE = 2
	EMBEDDING_DIM = 300
	NUMBER_OF_CHARS = 200

	job_dir = args.job_dir
	train_file = args.train_files
	eval_file = args.eval_files
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	gcloud_project = args.gcloud_project
	words2vec = args.words2vec
	dictionary_file = args.dictionary_file
	use_cpu = args.use_cpu

	##Setting up the path for saving logs
	logs_path = job_dir

	##Using the GPU
	device = None
	if use_cpu == "true" :
		device = '/device:CPU:0'
	else :
		device = '/device:GPU:0'
	with tf.device(device):

		## Load data
		train_data, train_labels, eval_data, eval_labels = load_data(train_file, eval_file, gcloud_project)

		## Do preprocessing
		train_data, train_labels, eval_data, eval_labels, num_classes, labelEncoder = preprocess(train_data, train_labels, eval_data, eval_labels)


		if words2vec == "false":

			## word2vec
			tokenizer = Tokenizer()
			tokenizer.fit_on_texts(to_ascii(train_data))

			word_index = tokenizer.word_index
			num_words = len(word_index) + 1
			max_word_count = max([len(x) for x in tokenizer.texts_to_sequences(to_ascii(train_data))])

			## ngrams
			char_counts = collections.Counter(' '.join(train_data))
			chars = [None]
			for char, count in char_counts.most_common(NUMBER_OF_CHARS):
				chars.append(char)
			mapping = dict((c, i) for i, c in enumerate(chars))
			
			print("chars: ", mapping)

			num_chars = len(mapping)
			max_char_count = max([len(x) for x in train_data])

			## Load embedding matrix
			embedding_matrix, EMBEDDING_DIM = load_embedding_matrix(dictionary_file, word_index, num_words, gcloud_project)


			## Make data generators
			training_generator = DataFeeder(train_data, train_labels, tokenizer, mapping, batch_size, max_word_count, max_char_count, num_classes, NGRAM_SIZE)
			validation_generator = DataFeeder(eval_data, eval_labels, tokenizer, mapping, batch_size, max_word_count, max_char_count, num_classes, NGRAM_SIZE)


			## Initializing the model
			num_outputs = 100
			model_ngrams = create_model_ngrams(max_char_count, 20, num_chars, NGRAM_SIZE)
			model_ngrams_lstm = create_model_ngrams_lstm(max_char_count, 20, num_chars, NGRAM_SIZE)
			model_word2vec = create_model_word2vec(max_word_count, 50, num_words, EMBEDDING_DIM, embedding_matrix)
			model_wordcount = create_model_wordcount(num_words, 30)
			model_wordexistence = create_model_wordcount(num_words, 30)


			## Here the magic happens. Just insert the models you want to test into the list and also modify 
			## the list in the return statement at the end of the data Feeder class
			Model = concat_models([model_wordcount, model_ngrams], num_classes)

			## Compling the model
			Model.compile(optimizer = 'adam', loss = "categorical_crossentropy", metrics = ["accuracy"]);


			## Printing the modle summary
			Model.summary()

			## Adding the callback for TensorBoard and History
			tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)

			verbose = 2
			if gcloud_project == "":
				verbose = 1

			#fitting the model
			history = Model.fit_generator(generator=training_generator, epochs=num_epochs, 
				verbose=verbose, callbacks=[tensorboard], use_multiprocessing=True,
				validation_data=validation_generator)
		
			# This code can not be run in the cloud as it reqires graphical output
			if gcloud_project == "":
				visualize_history(history)
				save_history(history)


			## create report to see accuracy for each class
			validation_generator.on_epoch_end()
			
			batch = 0
			x_test, y_test = validation_generator.__getitem__(batch)
			batch += 1
			if type(x_test) == list :
				while x_test[0].shape[0] < 500:
					x_test_add, y_test_add = validation_generator.__getitem__(batch)
					batch += 1
					for i in range(len(x_test)):
						x_test[i] = np.concatenate((x_test[i], x_test_add[i]), axis=0)
					y_test = np.concatenate((y_test, y_test_add), axis=0)
			else :
				while x_test.shape[0] < 500:
					x_test_add, y_test_add = validation_generator.__getitem__(batch)
					batch += 1
					x_test = np.concatenate((x_test, x_test_add), axis=0)
					y_test = np.concatenate((y_test, y_test_add), axis=0)

			Y_test = np.argmax(y_test, axis=1)
			y_pred = Model.predict(x_test)
			y_pred = np.argmax(y_pred,axis=1)


			print("classes:")
			for i in range(num_classes) :
				print(str(i), ":", labelEncoder.inverse_transform([i])[0])
			print(sklearn.metrics.classification_report(Y_test, y_pred))


			# Save model.h5 on to google storage
			Model.save('model.h5')
			with file_io.FileIO('model.h5', mode='rb') as input_f:
				with file_io.FileIO(job_dir + 'model/model.h5', mode='wb') as output_f:
					output_f.write(input_f.read())

		else:
			## words2vec training
			train_words2vec(to_ascii(train_data), dictionary_file, EMBEDDING_DIM)


##Running the app
if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# Input Arguments
	parser.add_argument(
	  '--job-dir',
	  help='GCS location to write checkpoints and export models',
	  required=True
	)
	parser.add_argument(
	  '--train-files',
	  help='GCS location to get training data',
	  required=True
	)
	parser.add_argument(
	  '--eval-files',
	  help='GCS location to get eval data',
	  required=True
	)
	parser.add_argument(
	  '--num-epochs',
	  help='Number of epochs',
	  type=int,
	  default=1
	)
	parser.add_argument(
	  '--batch-size',
	  help='Number of epochs',
	  type=int,
	  default=100
	)
	parser.add_argument(
	  '--gcloud-project',
	  help='Number of epochs',
	  default=""
	)
	parser.add_argument(
	  '--words2vec',
	  help='Train the words2vec model',
	  default="false"
	)
	parser.add_argument(
	  '--dictionary_file',
	  help='Path to the word2vec dictionary',
	  required=True
	)
	parser.add_argument(
	  '--use-cpu',
	  help='Use the CPU',
	  default="false"
	)

	args, _ = parser.parse_known_args()

	main(args)

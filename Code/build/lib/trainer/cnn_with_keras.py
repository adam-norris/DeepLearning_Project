import keras as keras
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')


import numpy as np
import argparse
import os
import gcsfs

from trainer.model import create_model
from trainer.preprocess import load_data
from trainer.preprocess import preprocess
from trainer.preprocess import load_embedding_matrix
from trainer.dataFeeder import DataFeeder
from trainer.words2vec import train_words2vec
from tensorflow.python.lib.io import file_io
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


def main(args):

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
		train_data, train_labels, eval_data, eval_labels, num_classes = preprocess(train_data, train_labels, eval_data, eval_labels)


		if words2vec == "false":

			## Setup tokenizer
			tokenizer = Tokenizer()
			tokenizer.fit_on_texts(train_data)

			word_index = tokenizer.word_index
			num_words = len(word_index) + 1
			max_length = max([len(x) for x in tokenizer.texts_to_sequences(train_data)])		

			## Make data generators
			training_generator = DataFeeder(train_data, train_labels, tokenizer, batch_size, max_length, num_classes)
			validation_generator = DataFeeder(eval_data, eval_labels, tokenizer, batch_size, max_length, num_classes)

			## Load embedding matrix
			embedding_matrix, EMBEDDING_DIM = load_embedding_matrix(dictionary_file, word_index, num_words, gcloud_project)


			## Initializing the model
			Model = create_model(max_length, num_classes, num_words, EMBEDDING_DIM, embedding_matrix);

			## Compling the model
			Model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics = ["accuracy"]);


			## Printing the modle summary
			Model.summary()

			## Adding the callback for TensorBoard and History
			tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)


			#fitting the model
			Model.fit_generator(generator=training_generator, epochs=num_epochs, 
				verbose=1, callbacks=[tensorboard], use_multiprocessing=True,
				validation_data=validation_generator)

			# Save model.h5 on to google storage
			Model.save('model.h5')
			with file_io.FileIO('model.h5', mode='rb') as input_f:
				with file_io.FileIO(job_dir + 'model/model.h5', mode='wb') as output_f:
					output_f.write(input_f.read())

		else:
			## words2vec training

			train_words2vec(train_data, dictionary_file)


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

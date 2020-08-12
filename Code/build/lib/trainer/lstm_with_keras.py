import keras as keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv2D
from keras.layers import Dropout, MaxPooling2D
import keras.backend as K
from keras import callbacks
import tensorflow as tf
K.set_image_data_format('channels_last')
import numpy as np
import argparse
import shutil
import pandas as pd
import os
import re
import gcsfs
from tensorflow.python.lib.io import file_io
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer


def model(input_shape, num_classes):
##First input

	model = keras.Sequential()
	
	## LSTM Hyper Parameters
	hidden_size = 100
	
	## LSTM Architecture
	model.add(Embedding(512, hidden_size, input_shape=input_shape))
	model.add(LSTM(hidden_size, return_sequences=True))
	model.add(LSTM(hidden_size, return_sequences=True))
	model.add(Dropout(0.5,name = 'dropout'))
	#model.add(TimeDistributed(Dense(vocabulary)))
	model.add(Activation('softmax'))

	return model

def main(args):

	job_dir = args.job_dir
	train_file = args.train_files
	eval_file = args.eval_files
	num_epochs = args.num_epochs
	batch_size = args.batch_size
	gcloud_project = args.gcloud_project

	##Setting up the path for saving logs
	logs_path = job_dir

	##Using the GPU
	with tf.device('/device:GPU:0'):

		trainData = None
		evalData = None
	
		##Loading the data
		if gcloud_project != "":
			fs = gcsfs.GCSFileSystem(project=gcloud_project)
			with fs.open(train_file) as f:
    				trainData = pd.read_csv(f, encoding='utf-8')
			with fs.open(eval_file) as f:
				evalData = pd.read_csv(f, encoding='utf-8')
		else :	
			trainData = pd.read_csv(train_file, encoding='utf-8')
			evalData = pd.read_csv(eval_file, encoding='utf-8')

		train_data = trainData["text"]
		train_labels = trainData["screen_name"]
		eval_data = evalData["text"]
		eval_labels = evalData["screen_name"]

		labels = list(set(train_labels))
		num_classes = len(labels)

		##Pre processing the data
		labelEncoder = LabelEncoder()
		labelEncoder.fit(labels)
		train_labels = labelEncoder.transform(train_labels)
		eval_labels = labelEncoder.transform(eval_labels)

		## Remove none ASCII characters
		train_data = [re.sub(r'[^\x00-\x7F]+',' ', x) for x in train_data]
		eval_data = [re.sub(r'[^\x00-\x7F]+',' ', x) for x in eval_data]
        
        ##TODO
        ## Make all letters lower-case
        
        ## Replace slang and abbreviations
        
        ## Remove Stop-words
        
        ## Word Stemming
        
        ## Spelling Correction
        
        ## Remove Punctuation

		## Remove links
		train_data = [re.sub(r'(http(s)?[^\s]*)|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', 'URL', x) for x in train_data]
		eval_data = [re.sub(r'(http(s)?[^\s]*)|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', 'URL', x) for x in eval_data]

		## Remove Hashtags
		train_data = [re.sub(r'\#[^\s]+', 'HASHTAG', x) for x in train_data]
		eval_data = [re.sub(r'\#[^\s]+', 'HASHTAG', x) for x in eval_data]

		## Remove user references
		train_data = [re.sub(r'@[^\s]+', 'USER', x) for x in train_data]
		eval_data = [re.sub(r'@[^\s]+', 'USER', x) for x in eval_data]

		## Setup tokenizer
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(train_data)

		## Apply Tokenizer
		train_data = tokenizer.texts_to_matrix(train_data, mode='count')
		eval_data = tokenizer.texts_to_matrix(eval_data, mode='count')

		train_labels = keras.utils.np_utils.to_categorical(train_labels, num_classes)
		eval_labels = keras.utils.np_utils.to_categorical(eval_labels, num_classes)



		## Initializing the model
		Model = model(train_data.shape[1:], num_classes);

		## Compling the model
		Model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics = ["accuracy"]);




		## Printing the modle summary
		Model.summary()

		## Adding the callback for TensorBoard and History
		tensorboard = callbacks.TensorBoard(log_dir=logs_path, histogram_freq=0, write_graph=True, write_images=True)




		##fitting the model
		Model.fit(x=train_data, y=train_labels, epochs=num_epochs, verbose=1, batch_size=batch_size, callbacks=[tensorboard], validation_data=(eval_data,eval_labels) )

		# Save model.h5 on to google storage
		Model.save('model.h5')
		with file_io.FileIO('model.h5', mode='rb') as input_f:
            		with file_io.FileIO(job_dir + 'model/model.h5', mode='wb') as output_f:
                		output_f.write(input_f.read())


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
	  default=500
	)
	parser.add_argument(
	  '--gcloud-project',
	  help='Number of epochs',
	  default=""
	)

	args, _ = parser.parse_known_args()

	main(args)

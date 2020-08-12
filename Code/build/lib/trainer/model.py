from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv1D, Conv2D, Embedding, MaxPooling1D, Lambda, concatenate
from keras.layers import Dropout, MaxPooling2D
from keras.initializers import Constant
import keras
import tensorflow as tf
import numpy as np


def create_model(max_length, num_classes, num_words, EMBEDDING_DIM, embedding_matrix) :

	## Input Layer
	input_x = Input(shape=(max_length,))

	## Embedding Layer
	x = Embedding(num_words, EMBEDDING_DIM, embeddings_initializer=Constant(embedding_matrix), trainable=False)(input_x)

	## Convolutional Layers
	x = Conv1D(256, 4, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = MaxPooling1D(2)(x)

	x = Conv1D(256, 4, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = MaxPooling1D(3)(x)

	x = Flatten()(x)
	
	input_y = Input(shape=(num_words,))
	y = Dense(512, activation='relu')(input_y)
	y = Dropout(0.5)(y)
	y = Dense(1024, activation='relu')(y)
	y = Dropout(0.5)(y)	

	## Combine result of x and y
	combined = concatenate([x, y])

	## Dense Layers
	output = Dense(512, activation='relu')(combined)
	output = Dropout(0.5)(output)
	output = Dense(num_classes, activation='softmax')(output)


	return Model(inputs=[input_x, input_y], outputs=output)


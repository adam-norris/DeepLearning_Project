from keras import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, Conv1D, Conv2D, Conv3D, Embedding, MaxPooling1D, Lambda, concatenate, Reshape, Permute, LSTM
from keras.layers import Dropout, MaxPooling2D
from keras.initializers import Constant
from keras import regularizers
import keras
import tensorflow as tf
import numpy as np


def create_model_ngrams(max_length, num_outputs, num_chars, NGRAM_SIZE) :

	EMBEDDING_DIM = 300
	FILTERS = 50
	input_dim = max_length-NGRAM_SIZE+1

	reg = regularizers.l2(0.0)

	## Input Layer
	input_layer = Input(shape=(input_dim,))

	embedding = Embedding(input_dim=num_chars**NGRAM_SIZE, output_dim=EMBEDDING_DIM, input_length=input_dim)(input_layer)
	embedding = Dropout(0.5)(embedding)
	
	x = Conv1D(FILTERS, kernel_size= 3, activation='sigmoid', kernel_regularizer=reg)(embedding)
	x = MaxPooling1D(input_dim - 2)(x)
	y = Conv1D(FILTERS, kernel_size= 4, activation='sigmoid', kernel_regularizer=reg)(embedding)
	y = MaxPooling1D(input_dim - 3)(y)
	z = Conv1D(FILTERS, kernel_size= 5, activation='sigmoid', kernel_regularizer=reg)(embedding)
	z = MaxPooling1D(input_dim - 4)(z)

	output = concatenate([x, y, z])

	output = Flatten()(output)

	return [input_layer, output]

def create_model_word2vec(max_length, num_outputs, num_words, EMBEDDING_DIM, embedding_matrix) :

	input_layer = Input(shape=(max_length,))

	x = Embedding(num_words,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=max_length,
			trainable=False)(input_layer)

	y = Embedding(num_words,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=max_length,
			trainable=True)(input_layer)

	z = concatenate([x, y], axis=1)

	z = Conv1D(filters=250, kernel_size=3, activation='relu')(z)
	z = MaxPooling1D(2)(z)

	z = Conv1D(filters=250, kernel_size=4, activation='relu')(z)
	z = MaxPooling1D(2)(z)

	z = Conv1D(filters=500, kernel_size=5, activation='relu')(z)
	z = MaxPooling1D(2)(z)

	z = Dropout(0.5)(z)

	z = Flatten()(z)

	z = Dense(512, activation='relu')(z)
	z = Dropout(0.5)(z)
	
	return [input_layer, z]

	
def create_model_wordcount(num_words, num_outputs) :
	
	input_layer = Input(shape=(num_words,))

	x = Dense(2048, activation='relu')(input_layer)

	x = Dense(64, activation='relu')(x)
	x = Dropout(0.8)(x)

	return [input_layer, x]

def create_model_ngrams_lstm(max_length, num_outputs, num_chars, NGRAM_SIZE):
	## Input Layer
	input_layer = Input(shape=(max_length-NGRAM_SIZE+1,))

	embedding = Embedding(input_dim=num_chars**NGRAM_SIZE, output_dim=300, input_length=max_length-NGRAM_SIZE+1)(input_layer)
	embedding = Dropout(0.25)(embedding)
	
	#Architecture but with single LSTM layer
	x = Conv1D(500, kernel_size=5, activation="relu")
	x = LSTM(50)(embedding)
	#x = MaxPooling1D(max_length-NGRAM_SIZE+1 - 2)(x)
	#x = Reshape((500,))(x)

	#output = Flatten()(x)
	output = Dense(num_outputs, activation='relu')(x)
	
	return [input_layer, output]

def concat_models(layers, num_classes) :

	inputs = list()
	outputs = list()
	for x in layers :
		inputs.append(x[0])
		outputs.append(x[1])

	if len(inputs) > 1 :
		x = concatenate(outputs)
	else :
		x = outputs[0]

	x = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.0))(x)

	return Model(inputs=inputs, outputs=x)


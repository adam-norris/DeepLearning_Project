#!/usr/bin/env python

def visualize_history(history):
	
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	import csv


	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper left')
	plt.show()

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.show()

def save_history(history):

	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	import csv


	with open('tmp/history.csv', 'w') as f:  # Just use 'w' mode in 3.x
		w = csv.writer(f)
		w.writerow(history.history.keys())
		w.writerows(zip(*history.history.values()))


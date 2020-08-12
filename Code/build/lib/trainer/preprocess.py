from sklearn.preprocessing import LabelEncoder
import re
import pandas as pd
import numpy as np

def load_data(train_file, eval_file, gcloud_project) :

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

	## Get needed data from csv
	train_data = trainData["text"]
	train_labels = trainData["screen_name"]
	eval_data = evalData["text"]
	eval_labels = evalData["screen_name"]

	return train_data, train_labels, eval_data, eval_labels

def preprocess(train_data, train_labels, eval_data, eval_labels) :

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
	
	## Remove links
	train_data = [re.sub(r'(http(s)?[^\s]*)|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', 'URL', x) for x in train_data]
	eval_data = [re.sub(r'(http(s)?[^\s]*)|(\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*)', 'URL', x) for x in eval_data]

	## Remove Hashtags
	train_data = [re.sub(r'\#[^\s]+', 'HASHTAG', x) for x in train_data]
	eval_data = [re.sub(r'\#[^\s]+', 'HASHTAG', x) for x in eval_data]

	## Remove user references
	train_data = [re.sub(r'@[^\s]+', 'USER', x) for x in train_data]
	eval_data = [re.sub(r'@[^\s]+', 'USER', x) for x in eval_data]

	return train_data, train_labels, eval_data, eval_labels, num_classes

def load_embedding_matrix(dictionary_file, word_index, num_words, gcloud_project) :

	## Load dictionary file
	dictrionary_data = None
	if gcloud_project != "":
		with fs.open(dictionary_file) as f:
			dictionaryData = f.readlines()
				
	else :	
		with open(dictionary_file, 'r') as f:
			dictionaryData = f.readlines()

	## Convert to numpy array
	embeddings_index = {}	
	for line in dictionaryData[1:]:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:])
		embeddings_index[word] = coefs
	EMBEDDING_DIM = int(dictionaryData[0].split()[1])
	dictionary_data = None

	embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
	for word, i in word_index.items():
		if i > num_words:
			continue
		embedding_vector = embeddings_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	
	return embedding_matrix, EMBEDDING_DIM

import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim

EMBEDDING_DIM = 100

def train_words2vec(train_data, dictionary_file):

	nltk.download('stopwords')
	nltk.download('punkt')
	
	lines = list()
	stop_words = set(stopwords.words('english'))

	for line in train_data:
		tokens = word_tokenize(line)
		tokens = [w.lower() for w in tokens]
		table = str.maketrans('', '', string.punctuation)
		stripped = [w.translate(table) for w in tokens]
		words = [word for word in stripped if word.isalpha()]
		words = [w for w in words if not w in stop_words]
		lines.append(words)
		
	print("length: ", len(lines))

	model = gensim.models.Word2Vec(sentences=lines, size=EMBEDDING_DIM, window=5, workers=4, min_count=1)
	words = list(model.wv.vocab)
	print('Vocabulary size: %d' % len(words))
	print('Saving dictionary')
	model.wv.save_word2vec_format(dictionary_file, binary=False)
		
	

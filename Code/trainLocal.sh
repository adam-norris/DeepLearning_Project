#!/bin/bash

gcloud ai-platform local train \
	--module-name=trainer.cnn_with_keras \
	--package-path=./trainer \
  	--package-path ./trainer \
  	-- \
  	--train-files=dataset/train-data.csv \
  	--eval-files=dataset/eval-data.csv \
	--dictionary_file=dataset/word2vec_dictionary.txt \
  	--job-dir ./tmp/ "$@"

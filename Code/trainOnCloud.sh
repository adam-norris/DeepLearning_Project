#!/bin/bash

datetime=$(date +"%Y%m%d_%H%M%S")

job_name=$datetime
args=()

while (( "$#" )); do
key="$1"

case $key in
    -j|--job_name)
    job_name="$2"
    shift # past argument
    shift # past value
    ;;
    *)    # unknown option
    args+=("$1")
    shift # past argument
    ;;
esac
done

gcloud ai-platform jobs submit training JOB$job_name \
	--module-name=trainer.cnn_with_keras \
	--package-path=./trainer \
	--region=us-east1 \
	--config=trainer/cloudml-gpu.yaml \
	--job-dir=gs://keras-uni/output/Job-$job_name/ \
	-- \
	--train-files=keras-uni/dataset/train-data.csv \
	--dictionary_file=keras-uni/dataset/word2vec_dictionary.txt \
	--gcloud-project=deep-learning-uni-245010 \
	--eval-files=keras-uni/dataset/eval-data.csv "${args[@]}"

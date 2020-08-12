#!/bin/bash

cd CIFAR
if [ -f "CIFAR.tar.gz" ]; then
    echo "CIFARE data already downloaded"
else
    wget https://files.pushshift.io/twitter/US_PoliticalTweets.tar.gz -O CIFAR.tar.gz
    tar -xzvf CIFAR.tar.gz
    python3 JsonToCSV.py
fi

cd ..

cd Downloaded
python3 tweet_dumper.py
python3 concat.py

cd ../CIFAR
python3 filterAccounts.py

cd ..
python3 joinResults.py

echo ""
echo "Output saved to tweets.csv"

python3 splitEvalData.py

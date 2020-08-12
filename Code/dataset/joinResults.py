#!/usr/bin/env python
# encoding: utf-8

from os import listdir
from os.path import isfile, join
import glob
import pandas as pd
import csv
import os

NUM_ACCOUNTS = 10

files = ["Downloaded/data.csv", "CIFAR/tweets_filtered.csv"]
if os.path.isfile("tweets.csv"):
	print("Including old tweets")
	os.rename("tweets.csv", "tweets-old.csv")
	files.append("tweets-old.csv")

print("Combining files:")
for f in files:
	print(f)

csv_data = pd.concat([pd.read_csv(f, encoding='utf-8') for f in files], sort=False)

print("Removing duplicates...")
csv_data = csv_data.drop_duplicates(subset=["id"], keep='first')


file = open("accounts", 'r')
accounts = file.read().splitlines()
file.close()

tweet_count = list()

for acc in accounts :
	count = len(csv_data.loc[csv_data['screen_name'] == acc])
	print("Collected ", count,  "tweets for user ", acc)
	tweet_count.append((count, acc))

tweet_count.sort(key=lambda tup: tup[0], reverse=True)
donald_trump = [x for x in tweet_count if x[1] == "realDonaldTrump"][0]
tweet_count = tweet_count[:NUM_ACCOUNTS-1]
tweet_count.append(donald_trump)

print("Using data of:")
for x in tweet_count:
	print(x[1], " [", x[0], "]")

biggest_accounts = [x[1] for x in tweet_count]

data = csv_data.loc[csv_data['screen_name'].isin(biggest_accounts)]
data['id'] = pd.Series(data['id'], dtype=int)

print("Writing file...")
data.to_csv("tweets.csv", encoding='utf8', index=False, float_format="%g")

file.close()

if os.path.isfile("tweets-old.csv"):
	os.remove("tweets-old.csv")

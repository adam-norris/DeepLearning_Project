#!/usr/bin/env python
# encoding: utf-8

import pandas as pd

EVAL_TWEETS_PER_USER = 300

f = open("accounts", 'r')
accounts = f.read().splitlines()
f.close()

data = pd.read_csv("tweets.csv", encoding='utf-8')

train_data = list()
eval_data = list()

for acc in accounts :
	subset = data.loc[data["screen_name"] == acc]
	eval_data.append(subset[:EVAL_TWEETS_PER_USER])
	train_data.append(subset[EVAL_TWEETS_PER_USER:])

train_data = pd.concat(train_data)
eval_data = pd.concat(eval_data)

train_data = train_data.sample(frac=1)

train_data.to_csv("train-data.csv", index=False, encoding='utf-8')
eval_data.to_csv("eval-data.csv", index=False, encoding='utf-8')

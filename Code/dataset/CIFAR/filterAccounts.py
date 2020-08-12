#!/usr/bin/env python
# encoding: utf-8

import csv

print("Filtering file for accounts...")

accounts = open("../accounts", "r",)
acc_list = accounts.read().splitlines()
accounts.close()

csv_file = csv.reader(open("tweets.csv", "r", encoding='utf-8'))

data = [x for x in csv_file if x[1] in acc_list]

output = csv.writer(open("tweets_filtered.csv", "w", encoding='utf-8'))
output.writerow(["user_id", "screen_name", "id", "text"])

for row in data:
	output.writerow(row)

print("done")

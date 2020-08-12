#!/usr/bin/env python
# encoding: utf-8

import csv
import json

print("Converting Json to CSV...")

json_file = open("tweets.json", 'r')
csv_file = csv.writer(open("tweets.csv", "w"))

csv_file.writerow(["user_id", "screen_name", "id", "text"])
  
for l in json_file:
    x = json.loads(l)
    if not x["retweeted"]:
    	output = [x["user_id"], x["screen_name"], x["id"], x["text"]]
    	csv_file.writerow(output)

json_file.close()
print("done")

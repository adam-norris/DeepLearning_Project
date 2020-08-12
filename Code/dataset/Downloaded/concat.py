#!/usr/bin/env python
# encoding: utf-8

from os import listdir
from os.path import isfile, join
import glob
import pandas as pd

files = ["output/" + f for f in listdir("output") if isfile(join("output", f))]
print("Combining files:")
for f in files:
	print(f)

csv = pd.concat([pd.read_csv(f, encoding='utf-8') for f in files])
csv.to_csv( "data.csv", index=False, encoding='utf-8')

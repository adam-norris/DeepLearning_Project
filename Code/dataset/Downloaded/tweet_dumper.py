#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv
import sys
import os
import time

#Twitter API credentials
consumer_key = "TCrq54UHk2JBSqcFfe0gSJUtE"
consumer_secret = "4fL8It7NGXQPU5fGWqlTMjrBMgCQ09jYnumg9TsauRqXW6Pq4B"
access_key = "714569440029966337-127mWZP2B9z3hwGM2yqRRjBhmP0pg8w"
access_secret = "3PAeS24llYlu39TIxVwNOhdnLxh8t5EhM2MR3kU2ioeO9"


def get_all_tweets(screen_name):
	#Twitter only allows access to a users most recent 3240 tweets with this method
	
	#authorize twitter, initialize tweepy
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	api = tweepy.API(auth)

	try:
		u=api.get_user(screen_name)
	except tweepy.RateLimitError as e:
		return False		
	except tweepy.TweepError:
		return True
	
	#initialize a list to hold all the tweepy Tweets
	alltweets = []	
	
	#make initial request for most recent tweets (200 is the maximum allowed count)
	success = False		
	while (not success) :
		success = True
		try:
			new_tweets = api.user_timeline(screen_name = screen_name,count=200)
		except tweepy.RateLimitError as e:
			print("Limit exeeded. Waiting 5 minutes...")
			time.sleep(60*5)
			success = False		
		except tweepy.TweepError:
			print(e)
			success = False
		
	if len(new_tweets) == 0:
		return True


	#save most recent tweets
	alltweets.extend(new_tweets)
	
	#save the id of the oldest tweet less one
	oldest = alltweets[-1].id - 1
	
	#keep grabbing tweets until there are no tweets left to grab
	while len(new_tweets) > 0:
		print("getting tweets before ", (oldest))
		
		#all subsiquent requests use the max_id param to prevent duplicates
		success = False	
		while (not success) :
			success = True
			try:
				new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
			except tweepy.RateLimitError as e:
				print("Limit exeeded. Waiting 5 minutes...")
				time.sleep(60*5)
				success = False		
			except tweepy.TweepError:
				print(e)
				success = False
		
		#save most recent tweets
		alltweets.extend(new_tweets)
		
		#update the id of the oldest tweet less one
		oldest = alltweets[-1].id - 1
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
	outtweets = [[tweet.user.id, tweet.user.screen_name, tweet.id, tweet.text] for tweet in alltweets if not tweet.retweeted]

	#write the csv	
	with open('output/%s_tweets.csv' % screen_name, 'w', encoding='utf-8') as f:
		writer = csv.writer(f)
		writer.writerow(["user_id", "screen_name", "id", "text"])
		writer.writerows(outtweets)
	
	return True


if __name__ == '__main__':
	#pass in the username of the account you want to download
	f = open("../accounts", 'r')
	accounts = f.read().splitlines()
	f.close() 

	if os.path.exists("output") :
		for root, dirs, files in os.walk("output", topdown=False):
			for name in files:
				os.remove(os.path.join(root, name))
			for name in dirs:
				os.rmdir(os.path.join(root, name))		
	else :
		os.mkdir("output")

	for acc in accounts:
		print(acc)
		while(not get_all_tweets(acc)): 
			print("Retry...")

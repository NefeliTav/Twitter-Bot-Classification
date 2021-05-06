import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyspark
from pyspark.sql import *
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
import csv
import tweepy
from tweepy import OAuthHandler

consumer_key = "phCKVDVUS7nBmCvN5aJZWwrxo"
consumer_secret = "3k7gMiVmxPPDI0C6kTc8uMTL0nSdNfeeU82OGcNVftkaMmujlR"
access_token = "1389540894022545408-lovri9oSZKdLryO5JXqvfwuLeruEGq"
access_token_secret = "N0XU2cNl14QsO13IRDHCSFmuZFpELKqNG2pA9mkwaQfrg"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

"""
redirect_url = auth.get_authorization_url()
print(redirect_url)
user_pint_input = input("What's the pin value?")
auth.get_access_token(user_pint_input)
"""

api = tweepy.API(auth)


with open('twitter_human_bots_dataset_clean.csv', 'r', encoding='latin1') as inp, open('clean.csv', 'w', newline='') as out:
    next(inp)
    writer = csv.writer(out)
    writer.writerow(['id', 'name', 'screen name', 'location', 'description', 'followers',
                    'friends', 'image', 'favourites', 'verified', 'tweets', 'account_type'])
    for row in csv.reader(inp):
        try:
            user = api.get_user(row[0])

            tweets = api.user_timeline(screen_name=user.screen_name,
                                       count=200,
                                       include_rts=False,
                                       tweet_mode='extended'
                                       )
            twenty_tweets = []
            for info in tweets[:20]:
                twenty_tweets.append(info.full_text)
            writer.writerow([row[0], user.name, user.screen_name, user.location, user.description, user.followers_count,
                            user.friends_count, user.default_profile_image, user.favourites_count, user.verified, twenty_tweets, row[1]])

        except tweepy.TweepError:
            pass

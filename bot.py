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

# create the session
conf = SparkConf().set("spark.ui.port", "4050").set('spark.executor.memory',
                                                    '4G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G')

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()


consumer_key = "phCKVDVUS7nBmCvN5aJZWwrxo"
consumer_secret = "3k7gMiVmxPPDI0C6kTc8uMTL0nSdNfeeU82OGcNVftkaMmujlR"
access_token = "1389540894022545408-lovri9oSZKdLryO5JXqvfwuLeruEGq"
access_token_secret = "N0XU2cNl14QsO13IRDHCSFmuZFpELKqNG2pA9mkwaQfrg"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# redirect_url = auth.get_authorization_url()
# print(redirect_url)
# user_pint_input = input("What's the pin value?")
# auth.get_access_token(user_pint_input)

api = tweepy.API(auth)

# user_timeline = user.timeline()
# for tweet in user_timeline:
#    print(tweet)
df = spark.read.load("twitter_human_bots_dataset.csv",
                     format="csv",
                     sep=",",
                     inferSchema="true",
                     header="true"
                     )

filename = open("./twitter_human_bots_dataset.csv", "r")
reader = csv.reader(filename, delimiter=",")
next(reader)

filename_clean = open("./twitter_human_bots_dataset_clean.csv", "w")
writer = csv.writer(filename_clean, delimiter=",")

for row in df.collect():
    try:
        user = api.get_user(row[0])
        print(user.screen_name)
        # writer.writerow([row])
    except:
        pass

    # print(user.followers_count)
    # for friend in user.friends():
    #    print(friend.screen_name)

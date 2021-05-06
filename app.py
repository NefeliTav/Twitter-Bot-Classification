from flask import Flask, redirect, url_for, render_template, request
#from pyspark import SparkContext
#sc = SparkContext('local')
"""
import tweepy
from tweepy import OAuthHandler

consumer_key = "phCKVDVUS7nBmCvN5aJZWwrxo"
consumer_secret = "3k7gMiVmxPPDI0C6kTc8uMTL0nSdNfeeU82OGcNVftkaMmujlR"
access_token = "1389540894022545408-lovri9oSZKdLryO5JXqvfwuLeruEGq"
access_token_secret = "N0XU2cNl14QsO13IRDHCSFmuZFpELKqNG2pA9mkwaQfrg"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
"""
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def check():
    if request.method == "POST":
        user = request.form["name"]
        #usr = api.get_user(user)
        # print(usr)

    return render_template("index.html")


if __name__ == "__main__":
    app.run()

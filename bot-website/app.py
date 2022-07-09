from flask import Flask, render_template, request
import mlflow.spark
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
import tweepy
import langdetect
import datetime
import nltk
import re
from pyspark.sql.functions import col
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *

spark = SparkSession.builder.master("local").getOrCreate()
now = datetime.datetime.now()
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
lancaster = LancasterStemmer()
RANDOM_SEED = 42  # For reproducibility

consumer_key = "phCKVDVUS7nBmCvN5aJZWwrxo"
consumer_secret = "3k7gMiVmxPPDI0C6kTc8uMTL0nSdNfeeU82OGcNVftkaMmujlR"
access_token = "1389540894022545408-lovri9oSZKdLryO5JXqvfwuLeruEGq"
access_token_secret = "N0XU2cNl14QsO13IRDHCSFmuZFpELKqNG2pA9mkwaQfrg"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)


NUMERICAL_FEATURES = ["follower_count",
                      "friends_count",
                      "listed_count",
                      "statuses_count",
                      "retweets",
                      "with_url",
                      "with_mention",
                      "created_at",
                      "avg_cosine"
                      ]
CATEGORICAL_FEATURES = ["geo_enabled",
                        "verified",
                        "has_extended_profile",
                        "default_profile",
                        "default_profile_image",
                        ]

TEXTUAL_FEATURES = ["screen_name",
                    "description",
                    "tweet_text"
                    ]

TARGET_VARIABLE = "account_type"


def clean(text):
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", " ", text)
    return text


def cosine_sim(X, Y):
    X_list = word_tokenize(X)
    Y_list = word_tokenize(Y)

    sw = stopwords.words('english')
    l1 = []
    l2 = []

    # Remove stop words from the string
    X_set = {w for w in X_list if not w in sw}
    Y_set = {w for w in Y_list if not w in sw}

    stemmed_X = set()
    for word in X_set:
        stemmed_X.add(lancaster.stem(word))
    stemmed_Y = set()
    for word in Y_set:
        stemmed_Y.add(lancaster.stem(word))

    rvector = stemmed_X.union(stemmed_Y)
    for w in rvector:
        if w in stemmed_X:
            l1.append(1)  # Create a vector
        else:
            l1.append(0)
        if w in stemmed_Y:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # Cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]

    try:
        sum1 = 0
        sum2 = 0
        for item in l1:
            sum1 += item
        for item in l2:
            sum2 += item
        cosine = c / float((sum1 * sum2) ** 0.5)
        return cosine
    except ZeroDivisionError:
        return 0


def remove_emoji(string):
    return string.encode('ascii', 'ignore').decode('ascii')


# Subtract profiles' creation dates from current date to convert this field into pure integers
def to_days(then):
    now = datetime.datetime.now()
    date_time_obj = datetime.datetime.strptime(
        then, '%Y-%m-%d %H:%M:%S').date()
    diff = (now.date() - date_time_obj)
    diff = str(diff).split(' ')
    return int(diff[0])


def find_user(usr):
    text_list = []
    try:
        # Count
        retweets = 0
        with_mention = 0
        with_url = 0
        text = ""

        # Find user
        user = api.get_user(usr)
        tweets = api.user_timeline(
            screen_name=user.screen_name, count=200, include_rts=True, tweet_mode='extended')

        # Read 130 of user's tweets
        for tweet in tweets:
            try:
                if tweet.retweeted_status:
                    retweets += 1
            except AttributeError:
                # Combine all tweets into one big text
                to_add = remove_emoji(tweet.full_text).replace("\n", " ")
                to_add = clean(to_add)
                try:
                    # Keep only English texts
                    if langdetect.detect(to_add) != 'en':
                        continue
                except langdetect.lang_detect_exception.LangDetectException:
                    continue
                text_list.append(to_add)
                text = text + " " + to_add
            if tweet.entities['urls']:
                with_url += 1
            if tweet.entities['user_mentions']:
                with_mention += 1

        text = " ".join(text.split())

        # Find retweets,mentions and urls per tweet
        if len(tweets) >= 1:
            retweets = retweets / len(tweets)
            with_mention = with_mention / len(tweets)
            with_url = with_url/len(tweets)
        else:
            retweets = 0
            with_url = 0
            with_mention = 0

        # Get Average Cosine similarity between every Tweet
        cosine_count = 0
        cosine_sum = 0
        avg_cosine = 0
        for first_tweet in text_list:
            for second_tweet in text_list:
                if first_tweet != second_tweet:
                    cosine_sum += cosine_sim(first_tweet, second_tweet)
                    cosine_count += 1
        if (cosine_count != 0):
            avg_cosine = cosine_sum / cosine_count

            # Clean description
        description = " ".join((re.sub(
            r"(?:\@|http?\://|https?\://|www)\S+", "", remove_emoji(user.description).replace("\n", " "))).split())

        # Create a dataframe to store the give account's data
        bot_df = spark.createDataFrame(
            [
                ("human", user.followers_count, user.friends_count, user.listed_count, user.statuses_count, str(user.geo_enabled), str(user.verified),
                 str(user.created_at), str(user.has_extended_profile), str(user.default_profile), str(user.default_profile_image), retweets, with_url, with_mention, avg_cosine, description, text),
            ],
            ['account_type', 'follower_count', 'friends_count', 'listed_count', 'statuses_count', 'geo_enabled', 'verified',
             'created_at', 'has_extended_profile', 'default_profile', 'default_profile_image', 'retweets', 'with_url', 'with_mention', 'avg_cosine', 'description', 'tweet_text']  # add your column names here
        )

        # Get the number of days since the account was created
        bot_df = bot_df.withColumn(
            "created_at", to_days_UDF(col("created_at")))

        # Cast numerical features from string to int/float
        bot_df = bot_df.selectExpr("account_type", "cast(follower_count as int) follower_count", "cast(friends_count as int) friends_count", "cast(listed_count as int) listed_count", "cast(statuses_count as int) statuses_count", "cast(retweets as float) retweets",
                                   "cast(with_url as float) with_url", "cast(with_mention as float) with_mention", "geo_enabled", "verified", "has_extended_profile", "default_profile", "default_profile_image", "cast(created_at as int) created_at", "cast(avg_cosine as float) avg_cosine")

        # Get the result and convert to pandas for display
        test_predictions = cv_model.transform(bot_df)
        bot_pdf = test_predictions.toPandas()
        # Return the result
        if bot_pdf["prediction"][0] == 0:
            return "bot"
        else:
            return "human"

    # If the account is not found
    except tweepy.TweepError:
        return "unknown"


to_days_UDF = spark.udf.register("to_days", to_days)
cv_model = mlflow.spark.load_model('./model')

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def check():
    if request.method == "POST":

        user = request.form["name"]

        # Check if user provided in frontend is human or bot
        usr = find_user(user)
        print(usr)

    return render_template("index.html", usr=usr)


if __name__ == "__main__":
    app.run()

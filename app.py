from flask import Flask, redirect, url_for, render_template, request
import string
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.corpus import stopwords
from tweepy import OAuthHandler
import tweepy
import langdetect
import csv
import datetime
import nltk
import re
from pyspark.sql.functions import udf, col, lower, trim, regexp_replace
from pyspark.ml.feature import StopWordsRemover, Word2Vec, RegexTokenizer, Tokenizer, MinMaxScaler
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler, OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import FMClassifier, GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.session import SparkSession
from pyspark import SparkContext, SparkConf, SparkFiles
from pyspark.sql.functions import udf
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql import *
import pyspark
import pandas as pd


spark = SparkSession.builder.master("local").getOrCreate()
now = datetime.datetime.now()
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lancaster = LancasterStemmer()
RANDOM_SEED = 42  # for reproducibility

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
                      "created_at"
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

    # remove stop words from the string
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
            l1.append(1)  # create a vector
        else:
            l1.append(0)
        if w in stemmed_Y:
            l2.append(1)
        else:
            l2.append(0)
    c = 0

    # cosine formula
    for i in range(len(rvector)):
        c += l1[i] * l2[i]

    try:
        cosine = c / float((sum(l1) * sum(l2)) ** 0.5)
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


def random_forest_pipeline(train,
                           numerical_features,
                           categorical_features,
                           target_variable,
                           with_std=True,
                           with_mean=True,
                           k_fold=5):

    indexers = [StringIndexer(inputCol=c, outputCol="{0}_indexed".format(
        c), handleInvalid="keep") for c in categorical_features]

    # Indexing the target column (i.e., transform human/bot into 0/1) and rename it as "label"
    label_indexer = StringIndexer(inputCol=target_variable, outputCol="label")

    # Assemble all the features (both one-hot-encoded categorical and numerical) into a single vector
    assembler = VectorAssembler(inputCols=[indexer.getOutputCol(
    ) for indexer in indexers] + numerical_features, outputCol="features")

    # Populate the stages of the pipeline with all the preprocessing steps
    stages = indexers + [label_indexer] + [assembler]  # + ...

    # Create the random forest transformer
    # change `featuresCol=std_features` if scaler is used
    rf = RandomForestClassifier(featuresCol="features", labelCol="label")

    # Add the random forest transformer to the pipeline stages (i.e., the last one)
    stages += [rf]

    # Set up the pipeline
    pipeline = Pipeline(stages=stages)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # With 3 values for rf.maxDepth and 3 values for rf.numTrees
    # this grid will have 3 x 3 = 9 parameter settings for CrossValidator to choose from.
    param_grid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [3, 5, 8]) \
        .addGrid(rf.numTrees, [10, 50, 100]) \
        .build()
    cross_val = CrossValidator(estimator=pipeline,
                               estimatorParamMaps=param_grid,
                               # default = "areaUnderROC", alternatively "areaUnderPR"
                               evaluator=BinaryClassificationEvaluator(
                                   metricName="areaUnderROC"),
                               numFolds=k_fold,
                               # this flag allows us to store ALL the models trained during k-fold cross validation
                               collectSubModels=True
                               )

    # Run cross-validation, and choose the best set of parameters.
    cv_model = cross_val.fit(train)

    return cv_model


def find_user(usr):
    text_list = []
    try:
        # count
        retweets = 0
        with_mention = 0
        with_url = 0
        text = ""

        user = api.get_user(usr)
        tweets = api.user_timeline(
            screen_name=user.screen_name, count=130, include_rts=True, tweet_mode='extended')

        # read 130 of user's tweets
        for tweet in tweets:
            try:
                if tweet.retweeted_status:
                    retweets += 1
            except AttributeError:
                # combine all tweets into one big text
                to_add = remove_emoji(tweet.full_text).replace("\n", " ")
                to_add = clean(to_add)
                try:
                    # keep only english texts
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
        # find retweets,mentions and urls per tweet
        if len(tweets) >= 1:
            retweets = retweets / len(tweets)
            with_mention = with_mention / len(tweets)
            with_url = with_url/len(tweets)
        else:
            retweets = 0
            with_url = 0
            with_mention = 0

        # clean description
        description = " ".join((re.sub(r"(?:\@|http?\://|https?\://|www)\S+",
                                       "", remove_emoji(user.description).replace("\n", " "))).split())
        # create a new clean and complete csv file with the dataset

        bot_df = spark.createDataFrame(
            [
                ("human", user.followers_count, user.friends_count, user.listed_count, user.statuses_count, str(user.geo_enabled), str(user.verified),
                 str(user.created_at), str(user.has_extended_profile), str(user.default_profile), str(user.default_profile_image), retweets, with_url, with_mention, description, text),
            ],
            ['account_type', 'follower_count', 'friends_count', 'listed_count', 'statuses_count', 'geo_enabled', 'verified',
             'created_at', 'has_extended_profile', 'default_profile', 'default_profile_image', 'retweets', 'with_url', 'with_mention', 'description', 'tweet_text']  # add your column names here
        )

        bot_df_text = bot_df

        to_days_UDF = spark.udf.register("to_days", to_days)
        bot_df = bot_df.withColumn(
            "created_at", to_days_UDF(col("created_at")))

        # Cast numerical features from string to int/float
        bot_df = bot_df.selectExpr("account_type", "cast(follower_count as int) follower_count", "cast(friends_count as int) friends_count", "cast(listed_count as int) listed_count", "cast(statuses_count as int) statuses_count",
                                   "cast(retweets as float) retweets", "cast(with_url as float) with_url", "cast(with_mention as float) with_mention", "geo_enabled", "verified", "has_extended_profile", "default_profile", "default_profile_image", "cast(created_at as int) created_at")

        # Textual fields
        bot_df_text = bot_df_text.selectExpr(
            "account_type", "description", "tweet_text")

        test_predictions = cv_model.transform(bot_df)

        # test_predictions.select("features", "prediction").show(1)

        bot_pdf = test_predictions.toPandas()

        if bot_pdf["prediction"][0] == 0:
            return "bot"
        else:
            return "human"

    except tweepy.TweepError:
        pass


# create dataframe using dataset
df = spark.read.csv("final.csv", header='true')

df_text = df

# Drop duplicates
df.dropDuplicates(["id"])

to_days_UDF = spark.udf.register("to_days", to_days)
df = df.withColumn("created_at", to_days_UDF(col("created_at")))

# Cast numerical features from string to int/float
df = df.selectExpr("account_type", "cast(follower_count as int) follower_count", "cast(friends_count as int) friends_count", "cast(listed_count as int) listed_count", "cast(statuses_count as int) statuses_count", "cast(retweets as float) retweets",
                   "cast(with_url as float) with_url", "cast(with_mention as float) with_mention", "geo_enabled", "verified", "has_extended_profile", "default_profile", "default_profile_image", "cast(created_at as int) created_at")

# Textual fields
df_text = df_text.selectExpr("account_type", "description", "tweet_text")

train_df, test_df = df.randomSplit([0.8, 0.2], seed=RANDOM_SEED)

# train
cv_model = random_forest_pipeline(
    train_df, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET_VARIABLE)

# test
test_predictions = cv_model.transform(test_df)


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def check():
    if request.method == "POST":

        user = request.form["name"]

        # check if user provided in frontend is human or bot
        usr = find_user(user)
        print(usr)

    return render_template("index.html", usr=usr)


if __name__ == "__main__":
    app.run()

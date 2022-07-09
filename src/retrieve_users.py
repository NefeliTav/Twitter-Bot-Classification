import csv
import langdetect
import tweepy
from tweepy import OAuthHandler
import re
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')
lancaster = LancasterStemmer()


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


# tweepy authentication
consumer_key = "phCKVDVUS7nBmCvN5aJZWwrxo"
consumer_secret = "3k7gMiVmxPPDI0C6kTc8uMTL0nSdNfeeU82OGcNVftkaMmujlR"
access_token = "1389540894022545408-lovri9oSZKdLryO5JXqvfwuLeruEGq"
access_token_secret = "N0XU2cNl14QsO13IRDHCSFmuZFpELKqNG2pA9mkwaQfrg"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

# clean data and retrieve the interesting info for each user
with open('../dataset/twitter_human_bots_dataset.csv', 'r', encoding='latin1') as inp, open('../dataset/twitter_human_bots_dataset_clean.csv', 'w', newline='') as out:
    # next(inp)
    writer = csv.writer(out)
    writer.writerow(['id', 'account_type', 'screen_name', 'follower_count', 'friends_count', 'listed_count', 'statuses_count', 'geo_enabled', 'verified',
                    'created_at', 'has_extended_profile', 'default_profile', 'default_profile_image', 'retweets', 'with_url', 'with_mention', 'avg_cosine', 'description', 'tweet_text'])

    # check every user id
    for row in csv.reader(inp):
        text_list = []
        try:
           # count
            retweets = 0
            with_mention = 0
            with_url = 0
            text = ""

            # find user
            user = api.get_user(row[0])
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

            cosine_count = 0
            cosine_sum = 0
            avg_cos = 0
            for first_tweet in text_list:
                for second_tweet in text_list:
                    if first_tweet != second_tweet:
                        cosine_sum += cosine_sim(first_tweet, second_tweet)
                        cosine_count += 1
            if (cosine_count != 0):
                avg_cos = cosine_sum / cosine_count

            # clean description
            description = " ".join((re.sub(
                r"(?:\@|http?\://|https?\://|www)\S+", "", remove_emoji(user.description).replace("\n", " "))).split())
            # create a new clean and complete csv file with the dataset
            writer.writerow([row[0], row[1], user.screen_name, user.followers_count, user.friends_count, user.listed_count, user.statuses_count, user.geo_enabled, user.verified,
                            user.created_at, user.has_extended_profile, user.default_profile, user.default_profile_image, retweets, with_url, with_mention, avg_cos, description, text])

        except tweepy.TweepError:
            pass

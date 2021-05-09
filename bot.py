import csv
import langdetect
import tweepy
from tweepy import OAuthHandler
import re


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
with open('twitter_human_bots_dataset.csv', 'r', encoding='latin1') as inp, open('twitter_human_bots_dataset_clean.csv', 'w', newline='') as out:
    next(inp)
    writer = csv.writer(out)
    writer.writerow(['id', 'account_type', 'screen_name', 'follower_count', 'friends_count', 'listed_count', 'statuses_count', 'geo_enabled', 'verified',
                    'created_at', 'has_extended_profile', 'default_profile', 'default_profile_image', 'retweets', 'with_url', 'with_mention', 'description', 'tweet_text'])

    # check every user id
    for row in csv.reader(inp):
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
                    text = text + " " + tweet.full_text

                if tweet.entities['urls']:
                    with_url += 1
                if tweet.entities['user_mentions']:
                    with_mention += 1

            text = remove_emoji(text).replace("\n", " ")
            text = re.sub(r"http\S+", "", text)
            text = re.sub(
                r"(?:\@|http?\://|https?\://|www)\S+", "", text)
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

            try:
                # keep only english texts
                if langdetect.detect(text) != 'en':
                    continue
            except langdetect.lang_detect_exception.LangDetectException:
                continue

            # clean description
            description = " ".join((re.sub(
                r"(?:\@|http?\://|https?\://|www)\S+", "", remove_emoji(user.description).replace("\n", " "))).split()),
            # create a new clean and complete csv file with the dataset
            writer.writerow([row[0], row[1], user.screen_name, user.followers_count, user.friends_count, user.listed_count, user.statuses_count, user.geo_enabled, user.verified,
                            user.created_at, user.has_extended_profile, user.default_profile, user.default_profile_image, retweets, with_url, with_mention, description, text])

        except tweepy.TweepError:
            pass

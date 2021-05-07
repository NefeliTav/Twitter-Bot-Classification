import csv
import tweepy
import langdetect
from tweepy import OAuthHandler

# tweepy authentication
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

# clean data and retrieve the interesting info for each user
with open('twitter_human_bots_dataset.csv', 'r', encoding='latin1') as inp, open('clean_dataset.csv', 'w', newline='') as out:
    next(inp)
    writer = csv.writer(out)
    writer.writerow(['id', 'account_type', 'screen_name', 'description', 'follower_count', 'geo_enabled', 'friends_count',  # fields to be taken into consideration
                     'listed_count', 'created_at', 'statuses_count', 'has_extended_profile', 'default_profile',
                     'default_profile_image', 'URL', 'retweets', 'with_url', 'with_mention', 'tweet_text'])

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
                    pass
                if tweet.entities['urls']:
                    with_url += 1
                if tweet.entities['user_mentions']:
                    with_mention += 1

                # combine all tweets into one big text
                text = text + " " + tweet.full_text
            text = text.replace("\n", " ")
            if "expanded_url" not in user.entities.keys():
                URL = "NO_URL"
            else:
                URL = user.entities["expanded_url"]

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

            # create a new clean and complete csv file with the dataset
            writer.writerow([row[0], row[1], user.screen_name, user.description.replace("\n", " "), user.followers_count,
                             user.geo_enabled, user.friends_count, user.listed_count, user.created_at, user.statuses_count,
                             user.has_extended_profile, user.default_profile, user.default_profile_image, URL, retweets, with_url,
                             with_mention, text])

        except tweepy.TweepError:
            pass

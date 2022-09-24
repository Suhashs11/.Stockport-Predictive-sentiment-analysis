from tweepy import API
from tweepy import Cursor
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import numpy as np
import pandas as pd
import re
import time
import api_credentials 


class TwitterAuthenticator():
    

    def authenticate_twitter_app(self):
        TWITTER_ACCESS_TOKEN = api_credentials.TWITTER_ACCESS_TOKEN
        TWITTER_ACCESS_TOKEN_SECRET = api_credentials.TWITTER_ACCESS_TOKEN_SECRET
        TWITTER_CONSUMER_KEY = api_credentials.TWITTER_CONSUMER_KEY
        TWITTER_CONSUMER_SECRET = api_credentials.TWITTER_CONSUMER_SECRET
        auth = OAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
        auth.set_access_token(TWITTER_ACCESS_TOKEN,
                              TWITTER_ACCESS_TOKEN_SECRET)
        return auth


class TwitterClient():
   

    def __init__(self, twitter_user=None):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()
        self.twitter_client = API(self.auth)
        self.twitter_user = twitter_user

    def get_twitter_client_api(self):
        return self.twitter_client

    def get_user_timeline_tweets(self, num_tweets):
        tweets = []
        for tweet in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
            tweets.append(tweet)
        return tweets


class TwitterStreamer():
    

    def __init__(self):
        self.auth = TwitterAuthenticator().authenticate_twitter_app()

    def stream_tweets(self, hash_tag_list):
        listener = TwitterListener()
        stream = Stream(self.auth, listener)

        # Filter Twitter streams by keywords
        stream.filter(track=hash_tag_list)


class TwitterListener(StreamListener):
   

    def on_error(self, status):
        if status == 420:
            # Return False on_data method in case rate limit occurs.
            return False
        print(status)


class TweetPreprocessor():

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def tweets_to_data_frame(self, tweets):
        
        df = pd.DataFrame(data=[self.clean_tweet(tweet.text)
                          for tweet in tweets], columns=['tweet'])

        df['id'] = np.array([tweet.id for tweet in tweets])

        # The lastest tweet id is conveiniently the largest number
        latest_id = df.id.max()

        # # If you want to drop retweets from data frame
        # df = df[~df.tweet.str.contains("RT")]

        return df, latest_id




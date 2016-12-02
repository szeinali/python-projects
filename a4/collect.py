from collections import Counter
import matplotlib.pyplot as plot
import networkx as nx
import sys
import time
from TwitterAPI import TwitterAPI
import json


consumer_key = 'qxQIYB9jB2rs1qEjdEMkuxaQ3'
consumer_secret = '4bHc37rbou7qVu3qsX0PChee8usqAS2Sh90pGfjRVrsopEjiPB'
access_token = '161404345-zjuqHVFHbEU85qTKxzqhPq4GRhMnsNqVkINwFCVD'
access_token_secret = 'jndXVGzVCyG3O8nbZiJZ2bbbLHYDLPy3wwCqiBveuhSul'

tweet_list=[]
def get_twitter():
    return TwitterAPI(consumer_key, consumer_secret, access_token, access_token_secret)

def tweets(twitter, tweet_count):
    tweet = []
    while True:
        try:
            for response in twitter.request('statuses/filter',{'track':'narendramodi,@narendramodi'}):
                if 'text' in response:
                        tweet.append(response)
                        if len(tweet) % 100 == 0:
                            print('found %d tweet' % len(tweet))
                            with open('result2','w') as fout:
                                json.dump(tweet,fout)
                        if len(tweet) >= tweet_count:
                            return tweet
        except:
            print("Unexpected error:", sys.exc_info()[0])
    return tweet


def robust_request(twitter, resource, params, max_tries=5):
    for x in range(max_tries):
        request = twitter.request(resource, params)
        if request.status_code == 200:
            return request
        else:
            with open('result1', 'w') as fout:
                json.dump(tweet_list, fout)

            print('Got error %s \nsleeping for 15 minutes.' % request.text)
            sys.stderr.flush()
            time.sleep(60 * 15)
            
def tweet_collection(tweet,twitter):
    tweet_dict={}
    for x in range(len(tweet)):
        request1 = robust_request(twitter,'friends/list', {'screen_name':tweet[x]['user']['screen_name'] })
        friends = [r for r in request1]
        tweet[x]['user'].update({'friends':friends})
        tweet_dict={'user':tweet[x]['user'],'text':tweet[x]['text']}
        tweet_list.append(tweet_dict)
        print(tweet_list[x]['user']['screen_name'])
    
def main():
    twitter = get_twitter()
    print('Connection established')
    tweet = tweets(twitter, 1000)
    tweet_collection(tweet,twitter)
    print('Collection of tweets completed.Please browse for the file in home folder')

if __name__ == '__main__':
    main()

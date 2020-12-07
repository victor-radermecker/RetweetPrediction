# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:46:55 2020

@author: Antoine
"""
import pandas as pd
import re
import numpy as np
import texthero as hero
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

def clean_hashtag(row):
    return " ".join(filter(lambda x:x[0]!='#', row.split()))

def delete_duplicates(string):
    words = string.split(',')
    words = list(map(str.strip, words))
    return " ".join(sorted(set(words), key=words.index))

#Remove hashtags from text
def find_deplace_hashtags(df):
    '''
    This function moves the "#" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_hashtags = df.apply(lambda x: re.findall("[#]\w+", x.text), axis=1)
    
    clean_hashtags = []
    for ls in extra_hashtags:
        res = ', '.join(ls)
        res = res.replace('#','')
        clean_hashtags.append(res)
    
    #print(clean_hashtags)
    df.text = df.apply(lambda x: clean_hashtag(x.text), axis=1)
    
    #add them to hashtags columns
    df['hashtags'] = df['hashtags'] + ', ' + pd.Series(clean_hashtags)
    df.hashtags = df.apply(lambda x : delete_duplicates(str(x.hashtags)), axis=1)
    df['hashtags'] = df['hashtags'].str.replace('nan, ', '')
    
def clean_urls(row):
    return " ".join(filter(lambda x:x[0:5]!='https', row.split()))

def find_deplace_urls(df):
    '''
    This function moves the "#" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_urls = df.apply(lambda x: re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x.text), axis=1)
#df.apply(lambda x: re.findall(r"https.*", x.text), axis=1)
    
    clean_urls_res = []
    for ls in extra_urls:
        res = ', '.join(ls)
        clean_urls_res.append(res)
   
    df.text = df.apply(lambda x: clean_urls(x.text), axis=1)
        
    #add them to urls columns  
    df.urls = df.apply(lambda x : delete_duplicates(str(x.urls)), axis=1)
    df["urls"] = df["urls"].str.cat(clean_urls_res, sep=', ')
    df.urls = df.urls.replace('nan, ',np.NaN)
    df['urls'] = df['urls'].str.replace('nan, ', '')
    df['urls'].loc[df['urls'] == ''] = np.NaN

    
def clean_at(row):
    return " ".join(filter(lambda x:x[0]!='@', row.split()))

#Remove hashtags from text
def find_deplace_ats(df):
    '''
    This function moves the "@" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_hashtags = df.apply(lambda x: re.findall("[@]\w+", x.text), axis=1)
    
    clean_ats = []
    for ls in extra_hashtags:
        res = ', '.join(ls)
        res = res.replace('@','')
        clean_ats.append(res)
    
    df.text = df.apply(lambda x: clean_at(x.text), axis=1)
    
    #add them to hashtags columns
    df['user_mentions'] = df['user_mentions'] + ', ' + pd.Series(clean_ats)
    df.user_mentions = df.apply(lambda x : delete_duplicates(str(x.user_mentions)), axis=1)
    df['user_mentions'] = df['user_mentions'].str.replace('nan, ', '')


'''Preprocessing beginning'''
data = pd.read_csv("train.csv")

#Tweet length
data['tweet_length'] = data['text'].str.len()
print("Tweet_length")

#Timestamp
data['timestamp'] = data['timestamp'].apply(lambda x: int(str(x)[:-3]))
print('timestamp')

#Urls
find_deplace_urls(data)
print('urls')

#Mentions
find_deplace_ats(data)
print('Mentions')

#lowercase
data['text'] = data['text'].apply(lambda x :' '.join([word.lower() for word in x.split()]))
print('lowercase')

find_deplace_hashtags(data)
print('hashtags')

#Eliminate non-ascii chars
data['text'] = data['text'].str.replace(r'[^\x00-\x7F]+','', regex=True)
print('ascii')

special_char_list = [':',',', ';', '.','?', '!', '}', ')', '{', '(']
for special_char in special_char_list:
    data['text'] = data['text'].str.replace(special_char, ' ')
print('specialchars')

#Common meaningless words
stop = stopwords.words("english")
data['text'] = data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print('stopwords')

#Stemming (transformed into their roots)
ps = PorterStemmer()
data['text'] = data['text'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split()]))
print('stemmer')
 
#Lemmatization (words with common roots)
lemmatizer = WordNetLemmatizer()
data['text'] = data['text'].apply(lambda x :' '.join([lemmatizer.lemmatize(word, 'v') for word in x.split()]))
print('lemmatization')
 
#Count Hashtags, urls, mentions
data['hashtags'].loc[data['hashtags'].isna() == 1] = ""
data['hashtags_number'] = data['hashtags'].str.count(' ') +1
data['hashtags_number'].loc[data['hashtags']==""] = 0

data['urls'].loc[data['urls'].isna() == 1] = ""
data['urls_number'] = data['urls'].str.count(', ') +1
data['urls_number'].loc[data['urls']==""] = 0

data['user_mentions'].loc[data['user_mentions'].isna() == 1] = ""
data['user_mentions_number'] = data['user_mentions'].str.count(' ') +1
data['user_mentions_number'].loc[data['user_mentions']==""] = 0
print("Count")

#Normalization
data['user_verified'] = data['user_verified'] * 1
data['user_statuses_count'] = data['user_statuses_count'] / data['user_statuses_count'].max()
data['user_followers_count'] = data['user_followers_count'] /data['user_followers_count'].max()
data['user_friends_count'] = data['user_friends_count'] / data['user_friends_count'].max()
data['hashtags_number'] /= data['hashtags_number'].max()
data['urls_number'] /= data['urls_number'].max()
data['user_mentions_number'] /= data['user_mentions_number'].max()
data['tweet_length']/=data['tweet_length'].max()
data['timestamp'] = (data['timestamp'] - data['timestamp'].min())/(data['timestamp'].max()-data['timestamp'].min())
print('Normalization')

#Text fillNa
data['text'].loc[data['text'].isna()==1] = ""

del data['urls'], data['text'], data['hashtags']
del data['user_mentions'], data['id']

#Export
data.to_csv(r'train_processed.csv', index=False)
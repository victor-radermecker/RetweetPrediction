# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:46:55 2020

@author: Antoine
"""
import pandas as pd
import re
import numpy as np

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


train_data = pd.read_csv("train.csv")



del train_data['timestamp']
del train_data['id']

print('start')
find_deplace_urls(train_data)
print('urls')
find_deplace_ats(train_data)
print('@')

#lowercase
train_data['text'] = train_data['text'].apply(lambda x :' '.join([word.lower() for word in x.split()]))
print('lowercase')

find_deplace_hashtags(train_data)
print('hashtags')

#Eliminate non-ascii chars
train_data['text'] = train_data['text'].str.replace(r'[^\x00-\x7F]+','', regex=True)
print('ascii')


special_char_list = [':',',', ';', '.','?', '!', '}', ')', '{', '(']
for special_char in special_char_list:
    train_data['text'] = train_data['text'].str.replace(special_char, ' ')
print('specialchars')

#Common meaningless words
stop = stopwords.words("english")
train_data['text'] = train_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
print('stopwords')


#Stemming (transformed into their roots)
ps = PorterStemmer()
train_data['text'] = train_data['text'].apply(lambda x : ' '.join([ps.stem(word) for word in x.split()]))
print('stemmer')
 
#Lemmatization (words with common roots)
lemmatizer = WordNetLemmatizer()
train_data['text'] = train_data['text'].apply(lambda x :' '.join([lemmatizer.lemmatize(word, 'v') for word in x.split()]))
print('lemmatization')
 

#Normalization
train_data['user_verified'] = train_data['user_verified'] * 1
train_data['user_statuses_count'] = train_data['user_statuses_count'] / train_data['user_statuses_count'].max()
train_data['user_followers_count'] = train_data['user_followers_count'] /train_data['user_followers_count'].max()
train_data['user_friends_count'] = train_data['user_friends_count'] / train_data['user_friends_count'].max()

#Export
train_data.to_csv(r'C:\Users\Antoine\Desktop\3A\INF554\Kaggle\covid19-retweet-prediction-challenge-2020\export_dataframe.csv', index=False)

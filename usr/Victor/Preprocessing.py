# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:46:55 2020

@author: Antoine
"""
import pandas as pd
import re
import numpy as np
from datetime import datetime
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
def find_deplace_hashtags(train_data):
    '''
    This function moves the "#" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_hashtags = train_data.apply(lambda x: re.findall("[#]\w+", x.text), axis=1)
    
    clean_hashtags = []
    for ls in extra_hashtags:
        res = ', '.join(ls)
        res = res.replace('#','')
        clean_hashtags.append(res)
    
    #print(clean_hashtags)
    train_data.text = train_data.apply(lambda x: clean_hashtag(x.text), axis=1)
    
    #add them to hashtags columns
    train_data['hashtags'] = train_data['hashtags'] + ', ' + pd.Series(clean_hashtags)
    train_data.hashtags = train_data.apply(lambda x : delete_duplicates(str(x.hashtags)), axis=1)
    train_data['hashtags'] = train_data['hashtags'].str.replace('nan', '')
    train_data['hashtags'].loc[train_data['hashtags'] == ''] = np.NaN

    
def clean_urls(row):
    return " ".join(filter(lambda x:x[0:5]!='https', row.split()))

def find_deplace_urls(train_data):
    '''
    This function moves the "#" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_urls = train_data.apply(lambda x: re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', x.text), axis=1)
    #train_data.apply(lambda x: re.findall(r"https.*", x.text), axis=1)
    
    clean_urls_res = []
    for ls in extra_urls:
        res = ', '.join(ls)
        clean_urls_res.append(res)
   
    train_data.text = train_data.apply(lambda x: clean_urls(x.text), axis=1)
        
    #add them to urls columns  
    train_data['urls'] = train_data.apply(lambda x : delete_duplicates(str(x.urls)), axis=1)
    train_data['urls'] = train_data['urls'].str.cat(clean_urls_res, sep=', ')
    train_data['urls'] = train_data.urls.replace('nan, ',np.NaN)
    train_data['urls'] = train_data['urls'].str.replace('nan, ', '')
    train_data['urls'].loc[train_data['urls'] == ''] = np.NaN

    
def clean_at(row):
    return " ".join(filter(lambda x:x[0]!='@', row.split()))

#Remove hashtags from text
def find_deplace_ats(train_data):
    '''
    This function moves the "@" from the text directly to the 'hashtags' columns.
    '''
    #Find extra hashtags
    extra_hashtags = train_data.apply(lambda x: re.findall("[@]\w+", x.text), axis=1)
    
    clean_ats = []
    for ls in extra_hashtags:
        res = ', '.join(ls)
        res = res.replace('@','')
        clean_ats.append(res)
    
    train_data.text = train_data.apply(lambda x: clean_at(x.text), axis=1)
    
    #add them to hashtags columns
    train_data['user_mentions'] = train_data['user_mentions'] + ', ' + pd.Series(clean_ats)
    train_data.user_mentions = train_data.apply(lambda x : delete_duplicates(str(x.user_mentions)), axis=1)
    train_data['user_mentions'] = train_data['user_mentions'].str.replace('nan', '')
    train_data['user_mentions'].loc[train_data['user_mentions'] == ''] = np.NaN

def number_elements(x):
    if x != '-':
        return len(x.split(' '))
    else:
        return 0

train_data = pd.read_csv("../../data/evaluation.csv")

print('start')

#Add length of tweets
print('Adding length')
train_data['text_len'] = train_data.text.apply(lambda x: len(str(x)))

#Get date
print('Add date')
time = train_data['timestamp'].apply(lambda x: datetime.utcfromtimestamp(int(str(x))/1000).strftime("%Y-%m-%d %H:%M:%S"))
time = pd.DataFrame(time)
time.timestamp = pd.to_datetime(time.timestamp)
train_data['hour'] = time.timestamp.dt.hour 

#Urls cleaning
print('urls')
find_deplace_urls(train_data)
print('@')
find_deplace_ats(train_data)

#lowercase
train_data['text'] = train_data['text'].apply(lambda x :' '.join([word.lower() for word in x.split()]))
print('lowercase')

#Hashtags cleaning
find_deplace_hashtags(train_data)
print('hashtags')


#Removing NaN values 
print('Removing NaN values')
train_data['user_mentions'] = train_data['user_mentions'].fillna(value='-')
train_data['urls'] = train_data['urls'].fillna(value='-')
train_data['hashtags'] = train_data['hashtags'].fillna(value='-')

train_data['nbr_user_mentions'] = train_data['user_mentions'].apply(lambda x: number_elements(x))
train_data['nbr_hashtags'] = train_data['hashtags'].apply(lambda x: number_elements(x))
train_data['nbr_urls'] = train_data['urls'].apply(lambda x: number_elements(x))


#Eliminate non-ascii chars
train_data['text'] = train_data['text'].str.replace(r'[^\x00-\x7F]+','', regex=True)
print('ascii')


special_char_list = [':',',', ';', '.','?', '!', '}', ')', '{', '(']
for special_char in special_char_list:
    train_data['text'] = train_data['text'].str.replace(special_char, ' ')
print('specialchars')

#Common meaningless words
import nltk as nltk
nltk.download("stopwords")

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
#train_data['user_statuses_count'] = train_data['user_statuses_count'] / train_data['user_statuses_count'].max()
#train_data['user_followers_count'] = train_data['user_followers_count'] /train_data['user_followers_count'].max()
#train_data['user_friends_count'] = train_data['user_friends_count'] / train_data['user_friends_count'].max()

#Counting number of hashtags, urls and user_mentions
       
#Export
train_data.reset_index()

print('Ready for export')
train_data.to_csv(r'../../data/eval_clean_final.csv', index=False)



from nltk.tokenize import word_tokenize, TweetTokenizer

import re
import string
import pandas as pd
import numpy as np


def target_encode_test(dataframe_train, dataframe_test, column, target, crossing=10, rate=20):
    """
    Performs target encoding on 'column' using Kfold cross-vaidation

    Parameters
    ----------
    dataframe : Pandas DataFrame
        Dataframe that contains columns
    
    column : string
        Column to apply target encoding

    target : string
        Target columns use to encode 'column'

    suffix (optional) : string, default = 'target'
        String used to rename the new target-encoded column

    Returns
    ----------
    dataframe : Pandas Dataframe
        Dataframe containing the transformed columns
    """
    temp_train = dataframe_train[[column, target]].copy()
    temp_test = dataframe_test[[column]].copy()

    prior = np.mean(temp_train['target'])
    
    lamb = lambda x : (1/(1+np.exp(-(x-crossing)/rate))) if x > 0 else 0 
    
    train_df = temp_train.groupby(column)[target].agg(['count', 'mean']).reset_index(drop=False)
    test_df = temp_test.merge(train_df, on=column, how='left')[['count', 'mean']]
    test_df.fillna({'count':0, 'mean':0}, inplace=True)
    test_df['weight'] =  test_df['count'].apply(lamb)
    test_df['Z'] = test_df['weight'] * test_df['mean'] + (1 - test_df['weight']) * prior
    #print(test_df[['Z']])
    return  test_df[['Z']].values

pm = re.compile(r'@\w+')

def mention_count(tweet):
    """
    Returns the number of mentions in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        count : int
    """
    count = len(pm.findall(tweet))
    return count


ph = re.compile(r'#\w+')

def hashtag_count(tweet):
    """
    Returns the number of hashtags in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        count : int
    """
    count = len(ph.findall(tweet))
    return count


purls = re.compile(r'http[s]?:\/\/\S+')

def url_count(tweet):
    """
    Returns the number of urls in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        count : int
    """
    count = len(purls.findall(tweet))
    return count


def token_count(tweet):
    """
    Returns the number of unprocessed tokens in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        len : int
    """
    tokens = tweet.split()
    return len(tokens)


def char_count(tweet):
    """
    Returns the number of characters in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        len : int
    """
    return len(tweet)


def avg_word_length(tweet):
    """
    Returns the average length of words of each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        avg_len : float
    """
    tokens = tweet.split()
    lens = [len(word) for word in tokens]
    avg_len = sum(lens)/len(lens)
    return avg_len

def median_word_length(tweet):
    """
    Returns the median of length of words in each tweet

    Parameters
    ----------
    tweet : str
        Text of tweet
        
    Returns
    ----------
        med_len : float
    """
    tokens = tweet.split()
    lens = [len(word) for word in tokens]
    med_len = np.median(lens)
    return med_len


def lowercase(tweet):
    """
    Convert capitalized words to lowercase.

    Parameters
    ----------
    tweet : str
        Text of tweet

    Returns
    ----------
    tweet : str
        Lowercased version of tweet
        
    """
    tweet = tweet.lower()
    return tweet
    

def tokenize(tweet, tokenizer = 'word_tokenize'):
    """
    Returns a list of tokens from 'tweet' using 'tokenizer'.

    Parameters
    ----------
    tweet : str
        Text of tweet

    tokenizer : str, default = 'word_tokenizer'
        Controls which tokenizer is used. Options are: 'word_tokenize', 'TweetTokenizer'

    Returns
    ----------
    tokens : list
        list of tokens
        
    """
    if tokenizer == 'word_tokenize':
        tokens = word_tokenize(tweet)
        
    elif tokenizer == 'TweetTokenizer':
        tknzr =  TweetTokenizer(reduce_len=True)
        tokens =  tknzr.tokenize(tweet)

    return tokens


def remove_punctuation(tokens):
    """
    Returns a list of tokens with punctuation removed.

    Parameters
    ----------
    tokens : list
        List of tokens

    stopwords : list
        List of stopwords to remove

    Returns
    ----------
    tokens : list
        
    """
    tokens = [word for word in tokens if word not in string.punctuation]
    return tokens
    

def remove_stopwords(tokens, stopwords):
    """
    Returns a list of tokens with stopwords removed.

    Parameters
    ----------
    tokens : list
        List of tokens

    stopwords : list
        List of stopwords to remove

    Returns
    ----------
    tokens : list
        
    """
    tokens = [token for token in tokens if token not in stopwords]
    return tokens


def remove_regex(tokens, regex):
    """
    Removes token that match regex pattern.

    Parameters
    ----------
    tokens : list
        List of tokens

    regex : string
        Regular expresion pattern

    Returns
    ----------
    tokens : list
        
    """

    tokens = [token for token in tokens if not re.match(regex, token)]
    return tokens


def remove_cols(X, cols_to_remove):
    return X[[col for col in X.columns if col not in cols_to_remove]]


def keep_col(X, col):
    return X[col]


def generate_dictionary(preds):
    results = [{'id': key, 'prediction':value[1]} for key, value in enumerate(preds)]
    return results
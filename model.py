import joblib
import numpy as np
import pandas as pd
from utils import *

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords


train = pd.read_csv('processed_train.csv')


def load_model():
    model = joblib.load('model.joblib')
    return model


def load_pipeline():
    pipe = joblib.load('pipeline.joblib')
    return pipe


def read_input(X):
    """
    Reads data into a pandas dataframe.
    """
    test_df = pd.DataFrame(X)
    #test_df = pd.read_json(X)
    return test_df

def read_input_2(X):
    """
    Reads data into a pandas dataframe.
    """
    #test_df = pd.DataFrame(X)
    test_df = pd.read_json(X)
    return test_df


def process_input(input_data):
    """
    Applies cleaning and processing steps.
    """
    test = input_data.copy()

    test['keyword'] = test['keyword'].fillna('Unknown')

    test['keyword_target_enc'] = target_encode_test(train, test, 'keyword', 'target')
    test['number_of_mentions'] = test['text'].apply(mention_count)
    test['number_of_hashtags'] = test['text'].apply(hashtag_count)
    test['number_of_urls'] = test['text'].apply(url_count)

    test['token_count'] = test['text'].apply(token_count)
    test['char_count'] = test['text'].apply(char_count)
    test['avg_word_length'] = test['text'].apply(avg_word_length)
    test['median_word_length'] = test['text'].apply(median_word_length)

    stpws = stopwords.words('english')

    test['tokens'] = test['text'].apply(lowercase)
    test['tokens'] = test['tokens'].apply(tokenize, tokenizer='TweetTokenizer')
    test['tokens'] = test['tokens'].apply(remove_punctuation)
    test['tokens'] = test['tokens'].apply(remove_regex, regex=r'\d+')

    test['tokens'] = test['tokens'].apply(remove_stopwords, stopwords=stpws)

    test['new_text'] = test['tokens'].apply(lambda token_list: ' '.join(token_list))

    return test


def format_input(test):
    """
    Selects columns and converts Pandas DataFrame into numpy array.
    """
    feature_cols = list(train.columns)
    remove = ['id', 'keyword', 'location', 'text', 'target', 'tokens']
    feature_cols = [col for col in feature_cols if col not in remove]

    X = test[feature_cols]
    return X


def predict_label(input_data):
    """
    Predict class labels for input data.
    """
    clf = load_model()
    pipe = load_pipeline()
    
    #test = read_input(input_data)
    test = process_input(input_data)
    X = format_input(test)

    X = pipe.transform(X)
    y_pred = clf.predict_proba(X)

    return y_pred
    


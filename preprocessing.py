import glob
import os
import re
import string
import torch
import nltk
import contractions
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt


# preprocessing function for a BoW model
def preprocess_lgbm(reviews, lemmatizer, stop_words):
    punct = string.punctuation
    data = []

    for review in reviews:
        text = review
        # remove html tags
        text = re.sub('<.*?>', '', text)
        # expand contractions (haven't -> have not)
        text = contractions.fix(text)
        # get rid of non-alphabetic characters
        text = re.sub(r'[^a-zA-Z ]+', ' ', text)

        # tokenize text and turn it into lemmas
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t not in stop_words and t not in punct]
        lemmas = [lemmatizer.lemmatize(t) for t in tokens]

        data.append(lemmas)

    return data


def preprocess_bert(reviews, tokenizer):
    data = {"input_ids": [], "attention_mask": []}

    for review in reviews:
        text = review
        tokens = tokenizer(text, truncation=True, padding="max_length", max_length=512)
        data["input_ids"].append(tokens['input_ids'])
        data["attention_mask"].append(tokens['attention_mask'])

    data["input_ids"] = torch.Tensor(data["input_ids"]).type(torch.LongTensor)
    data["attention_mask"] = torch.Tensor(data["attention_mask"]).type(torch.LongTensor)
    return data


def process_output(ratings, thresh):
    new_ratings = []
    labels = []
    for rating in ratings:
        if rating > 1:
            rating = 1
        elif rating < 0.1:
            rating = 0.1

        label = 1 if rating > thresh else 0
        labels.append(label)
        new_ratings.append(round(rating * 10))

    return new_ratings, labels


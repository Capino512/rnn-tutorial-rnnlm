#! /usr/bin/env python

import csv
import itertools
import operator
import numpy as np
import nltk
import sys
import os
import time
from datetime import datetime
from utils import *
from rnn_theano import RNNTheano
from rnn_numpy import RNNNumpy

class Config:
    _DATASET_FILE = os.environ.get('DATASET_FILE', './data/small-dataset.csv')
    _MODEL_FILE = os.environ.get('MODEL_FILE', './data/small_rnn_model.npz')

    _VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '20'))
    _UNKNOWN_TOKEN = "UNKNOWN_TOKEN"
    _SENTENCE_START_TOKEN = "SENTENCE_START"
    _SENTENCE_END_TOKEN = "SENTENCE_END"

    _HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '20'))
    _LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
    _NEPOCH = int(os.environ.get('NEPOCH', '10'))

# Read the data and append SENTENCE_START and SENTENCE_END tokens
sentences = read_sentences_from_csv(Config._DATASET_FILE, Config._SENTENCE_START_TOKEN, Config._SENTENCE_END_TOKEN)

# Tokenize the sentences into words
tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

index_to_word, word_to_index = build_vocabulary(tokenized_sentences, Config._VOCABULARY_SIZE, Config._UNKNOWN_TOKEN)

# Replace all words not in our vocabulary with the unknown token
for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i] = [w if w in word_to_index else Config._UNKNOWN_TOKEN for w in sent]

# Create the training data
X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

def train_numpy():
    model = RNNNumpy(Config._VOCABULARY_SIZE, hidden_dim=Config._HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], Config._LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    model.train_with_sgd(X_train, y_train, nepoch=Config._NEPOCH, learning_rate=Config._LEARNING_RATE)
    # train_with_sgd(model, X_train, y_train, nepoch=_NEPOCH, learning_rate=_LEARNING_RATE)

    if Config._MODEL_FILE != None:
        print "start saving model..."
        save_model_parameters_numpy(Config._MODEL_FILE, model)
        print "model saved!"

def train_theano():
    model = RNNTheano(Config._VOCABULARY_SIZE, hidden_dim=Config._HIDDEN_DIM)
    t1 = time.time()
    model.sgd_step(X_train[10], y_train[10], Config._LEARNING_RATE)
    t2 = time.time()
    print "SGD Step time: %f milliseconds" % ((t2 - t1) * 1000.)

    model.train_with_sgd(X_train, y_train, nepoch=Config._NEPOCH, learning_rate=Config._LEARNING_RATE)

    if Config._MODEL_FILE != None:
        print "start saving model..."
        save_model_parameters_theano(Config._MODEL_FILE, model)
        print "model saved!"

train_theano()
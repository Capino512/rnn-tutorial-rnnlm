from utils import *
from rnn_theano import RNNTheano
import os

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

def load_trained_model():
    model = RNNTheano(Config._VOCABULARY_SIZE, hidden_dim = Config._HIDDEN_DIM)
    print 'start loading...'
    load_model_parameters_theano(Config._MODEL_FILE, model)
    print 'load over!'
    return model

def generate_sentences(model):
    # Read the data and append SENTENCE_START and SENTENCE_END tokens
    sentences = read_sentences_from_csv(Config._DATASET_FILE, Config._SENTENCE_START_TOKEN, Config._SENTENCE_END_TOKEN)

    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

    index_to_word, word_to_index = build_vocabulary(tokenized_sentences, Config._VOCABULARY_SIZE, Config._UNKNOWN_TOKEN)

    num_sentences = 10
    senten_min_length = 6

    for i in range(num_sentences):
        sent = []
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = generate_sentence_by_theano_rnn(model, 50, word_to_index, index_to_word, Config._SENTENCE_START_TOKEN,
            	Config._SENTENCE_END_TOKEN, Config._UNKNOWN_TOKEN)
        print " ".join(sent)

def generate_sentence_by_theano_rnn(model, max_length, word_to_index, index_to_word, sentence_start_token, sentence_end_token, unknown_token):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    length = 0
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
        length += 1
        if length >= max_length:
            new_sentence.append(word_to_index[sentence_end_token])
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

model = load_trained_model()
generate_sentences(model)
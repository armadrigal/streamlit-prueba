import unicodedata
import string
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import pickle

with open("Word2Vect.pickle", "rb") as f:
    model_word2vect = pickle.load(f)

model_cnn = keras.models.load_model('model_cnn.h5')

def process_review(review):
    review = review.strip()
    review = review.lstrip()
    review = review.rstrip()
    review = review.lower() 
    review = ''.join((c for c in unicodedata.normalize('NFD',review) if unicodedata.category(c) != 'Mn'))
    for i in range(100):
        review = review.replace('  ',' ')
    replacements = ''
    for i in list(range(0,32))+list(range(33,97))+list(range(123,1000)):
        review = review.replace(chr(i),'')
    abc = string.ascii_lowercase
    for word in review.split():
        remove_word = False
        for char in abc:
            if word.find(3*char) != -1:
                remove_word = True
                break
        if (word.find('a') == -1 and 
            word.find('e') == -1 and 
            word.find('i') == -1 and 
            word.find('o') == -1 and 
            word.find('u') == -1):
            remove_word = True
        if(remove_word):
            index = review.find(word)
            review = review[0:index] + review[index+len(word)+1:len(review)]
    review = review.strip()
    review = review.lstrip()
    review = review.rstrip()
    return review

def tokenize_sentences(sentences):
    tokenized_sentences = []
    for sentence in sentences:
        tokenized_sentences.append(sentence.split())
    return tokenized_sentences

def vectorice_sentences(sentences):
    vectoriced_sentences = []
    for sentence in sentences:
        vectoriced_sentence = []
        for word in sentence:
            if word in model_word2vect.wv.key_to_index:
                vectoriced_sentence.append(model_word2vect.wv.key_to_index[word])
            else:
                vectoriced_sentence.append(0)
        vectoriced_sentences.append(vectoriced_sentence)
    return vectoriced_sentences

def predictions(sentences):
    sentences_ = []
    for sentence in sentences:
        print(sentence)
        sentences_.append(process_review(sentence))
    sentences = tokenize_sentences(sentences_)
    sentences = vectorice_sentences(sentences)
    sentences = pad_sequences(sentences, maxlen=50)
    pred = model_cnn.predict(sentences)
    return pred

pred = predictions(['A entrega foi super rápida e o pendente é lindo! Igual a foto mesmo!'])[0,0]
print('prediccion', pred)
# coding=utf-8
HELP = """
   Gensim model
"""
import os, sys, itertools, time, pandas as pd, numpy as np, pickle, gc, re
from typing import Callable, Tuple, Union
from box import Box
import random
import nltk
from pathlib import Path

#################################################################################################
from utilmy import log, log2


def help():
    from utilmy import help_create
    ss = HELP + help_create("utilmy.nlp.util_model")
    print(ss)


#################################################################################################
def test_all():
    test_gensim1()


def generate_random_bigrams(n_words=100, word_length=4, bigrams_length=5000):
    import string
    words = []
    while len(words) != n_words:
        word = ''.join(random.SystemRandom().choice(string.ascii_lowercase) for _ in range(word_length))
        if word not in words:
            words.append(word)

    paragraph = [random.choice(words) for i in range(bigrams_length + 1)]
    bigrams = list(nltk.bigrams(paragraph))
    return bigrams


def write_random_sentences_from_bigrams_to_file(dirout, n_sentences=14000):
    if not os.path.exists(dirout):
        from utilmy import os_makedirs
        os_makedirs(dirout)
    bigrams = generate_random_bigrams()
    with open(dirout, mode='w+') as fp:
        for i in range(n_sentences):
            rand_item = random.choice(bigrams)
            third_word = random.choice([i[1] for i in bigrams if i[0] == rand_item[1]])
            sent = ' '.join(rand_item)
            sent += ' ' + third_word
            fp.write(sent + "\n")


def test_gensim1():
    log("test_gensim")
    dir0 = os.getcwd()
    pars = Box({})
    pars.min_n = 6;
    pars.max_n = 6;
    pars.window = 3;
    pars.vector_size = 3

    write_random_sentences_from_bigrams_to_file(dirout='./testdata/mytext1.txt')
    gensim_model_train_save(None, dirout='./modelout1/model.bin', dirinput='./testdata/mytext1.txt',
                            epochs=1,
                            pars=pars)
    # gensim_model_check(dir0 + '/modelout1/model.bin')

    # model = gensim_model_load( dir0 + '/modelout1/model.bin')
    write_random_sentences_from_bigrams_to_file(dirout='./testdata/mytext2.txt')
    gensim_model_train_save(model_or_path='./modelout1/model.bin', dirout=dir0 + './modelout2/model.bin',
                            dirinput=dir0 + './testdata/mytext2.txt', epochs=1)
    # gensim_model_check(dir0 +  '/modelout2/model.bin')

    model = gensim_model_load(dir0 + './modelout2/model.bin')
    write_random_sentences_from_bigrams_to_file(dirout=dir0 + './testdata/mytext2.txt')
    gensim_model_train_save(model_or_path=model, dirout=dir0 + './modelout2/model.bin',
                            dirinput=dir0 + './testdata/mytext2.txt', epochs=1)
    # gensim_model_check(dir0 +  '/modelout2/model.bin')


#################################################################################################
def gensim_model_load(dirin, modeltype='fastext', **kw):
    """
    Loads the FastText model from the given path

    :param dirin: the path of the saved model
    :param modeltye:
    :param kw:
    :return: loaded model
    """
    if modeltype == 'fastext':
        from gensim.models import FastText
        loaded_model = FastText.load(f'{dirin}')  ## Full path

    return loaded_model


def gensim_model_train_save(model_or_path=None, dirinput='lee_background.cor', dirout="./modelout/model",
                            epochs=1, pars: dict = None, **kw):
    """ Trains the Fast text model and saves the model
      classgensim.models.fasttext.FastText(sentences=None, corpus_file=None, sg=0, hs=0, vector_size=100,
      alpha=0.025, window=5, min_count=5, max_vocab_size=None, word_ngrams=1, sample=0.001,
      seed=1, workers=3, min_alpha=0.0001, negative=5, ns_exponent=0.75, cbow_mean=1,
      hashfxn=<built-in function hash>, epochs=5, null_word=0, min_n=3, max_n=6,
      sorted_vocab=1, bucket=2000000, trim_rule=None,
      batch_words=10000, callbacks=(), max_final_vocab=None, shrink_windows=True)

    https://radimrehurek.com/gensim/models/fasttext.html


    train(corpus_iterable=None, corpus_file=None, total_examples=None, total_words=None, epochs=None, start_alpha=None,
          end_alpha=None, word_count=0, queue_factor=2,
          report_delay=1.0, compute_loss=False, callbacks=(), **kwargs


    :param model: The model to train
    :param dirinput: the filepath of the input data
    :param dirout: directory to save the model
    :epochs: number of epochs to train the model
    :pars: parameters of the creating FastText
    :return:
    """
    from gensim.test.utils import datapath
    from gensim.models import FastText
    if model_or_path is None:
        pars = {} if pars is None else pars
        # model = FastText(vector_size=vector_size, window=window, min_count=min_count)
        model = FastText(**pars)

    elif isinstance(model_or_path, str):
        model_path = model_or_path  ### path  is provided !!!
        model = gensim_model_load(model_path)
    else:
        model = model_or_path  ### actual model

    log("#### Input data building  ", model)
    corpus_file = dirinput
    if not os.path.exists(corpus_file):
        corpus_file = datapath(dirinput)

    to_update = True if model.wv else False
    model.build_vocab(corpus_file=corpus_file, update=to_update)
    nwords = model.corpus_total_words

    log("#### Model training   ", nwords)
    log('model ram', model.estimate_memory(vocab_size=nwords, report=None))
    log(nwords, model.get_latest_training_loss())

    model.train(corpus_file=corpus_file, total_words=nwords, epochs=epochs)
    log(model.get_latest_training_loss())
    log(model)

    from utilmy import os_makedirs
    os_makedirs(dirout)
    model.save(f'{dirout}')


def gensim_model_check(model_or_path):
    ''' various model check
          score(sentences, total_sentences=1000000, chunksize=100, queue_factor=2, report_delay=1)
          Score the log probability for a sequence of sentences. This does not change the fitted model in any way (see train() for that).

          Gensim has currently only implemented score for the hierarchical softmax scheme, so you need to have run word2vec with hs=1 and negative=0 for this to work.
          Note that you should specify total_sentences; you’ll run into problems if you ask to score more than this number of sentences but it is inefficient to set the value too high.

       Parameters
       sentences (iterable of list of str) – The sentences iterable can be simply a list of lists of tokens, but for larger corpora, consider an iterable that streams the sentences directly from disk/network. See BrownCorpus, Text8Corpus or LineSentence in word2vec module for such examples.
       total_sentences (int, optional) – Count of sentences.
       chunksize (int, optional) – Chunksize of jobs
       queue_factor (int, optional) – Multiplier for size of queue (number of workers * queue_factor).
       report_delay (float, optional) – Seconds to wait before reporting progress.

    '''
    if isinstance(model_or_path, str):
        model_path = model_or_path  ### path  is provided !!!
        model = gensim_model_load(model_path)
    else:
        model = model_or_path  ### actual model
    from gensim.test.utils import datapath

    print('Log Accuracy:    ', model.wv.evaluate_word_analogies(datapath('questions-words.txt'))[0])

    print('distance of the word {w1} and {w2} is {d}'.format(w1=model.wv.index_to_key[0],
                                                             w2=model.wv.index_to_key[1],
                                                             d=model.wv.distance(model.wv.index_to_key[0],
                                                                                 model.wv.index_to_key[1])))

    print('Most similar words to    ', model.wv.index_to_key[0])
    print(model.wv.most_similar(model.wv.index_to_key[0]))


def text_preprocess(sentence, lemmatizer, stop_words):
    """ Preprocessing Function
    :param sentence: sentence to preprocess
    :param lemmatizer: the class which lemmatizes the words
    :param stop_words: stop_words in english http://xpo6.com/list-of-english-stop-words/
    :return: preprocessed sentence
    """
    import nltk
    sentence = sentence.lower()
    sentence = re.sub(r'[^a-z]', ' ', sentence)  ### ascii only
    # sentence = re.sub(r'\s+', ' ', sentence)  ### Removes all multiple whitespaces with a whitespace in a sentence
    # sentence = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence) if word not in stop_words]
    return ' '.join(sentence)


def text_generate_random_sentences(dirout=None, n_sentences=5, ):
    """
    Generates Random sentences and Preprocesses them

    :param n_sentences: number of sentences to generate
    :param dirout: filepath do write the generated sentences
    :return: generated sentences
    """
    from essential_generators import DocumentGenerator
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    gen = DocumentGenerator()
    lemmatizer = WordNetLemmatizer()
    # stop_words = set(stopwords.words('english'))
    stop_words = []
    sentences = [text_preprocess(gen.sentence(), lemmatizer, stop_words) for i in range(n_sentences)]
    # sentences = [ gen.sentence()  for i in range(n_sentences)]

    from utilmy import os_makedirs

    if dirout is None:
        return sentences
    else:
        os_makedirs(dirout)
        with open(dirout, mode='w') as fp:
            for x in sentences:
                fp.write(x + "\n")
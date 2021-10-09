# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""
Methods for feature extraction and preprocessing
util_feature: input/output is pandas


#####################################################################################################
######### Term Frequency   ##########################################################################
If you need the term frequency (term count) vectors for different tasks, use Tfidftransformer.
If you need to compute tf-idf scores on documents within your “training” dataset, use Tfidfvectorizer
If you need to compute tf-idf scores on documents outside your “training” dataset, use either one, both will work.



#####################################################################################################
### The sklearn.feature_extraction.text submodule gathers utilities to build feature vectors from text documents.

feature_extraction.text.CountVectorizer([ÿ])  Convert a collection of text documents to a matrix of token counts
feature_extraction.text.HashingVectorizer([ÿ])  Convert a collection of text documents to a matrix of token occurrences
feature_extraction.text.TfidfVectorizer([ÿ])  Convert a collection of raw documents to a matrix of TF-IDF features.



"""
import os, sys, math, json, copy,re ,string
from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
import scipy as sci
import nltk
import sklearn as sk

########### Local Import #####################################################################
from nltk.corpus import stopwords
# Stemming and Lemmatizing
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn import preprocessing
from sklearn.feature_extraction.text import (CountVectorizer, TfidfTransformer,
                                             TfidfVectorizer)
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# import spacy
# import gensim



#### Add path for python import
path1 = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).replace("\\", "/") + "/"
print(path1)
sys.path.append( path1)
from source.bin.column_encoder import MinHashEncoder


print("os.getcwd", os.getcwd())


##############################################################################################
punctuations = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"


#############################################################################################
#############################################################################################
def get_stopwords(lang):
    if lang == "en":
        return json.load(open("stopwords_en.json"))["word"]


porter = PorterStemmer()


def coltext_stemporter(text):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split(" ")
    stemmed_tokens = [porter.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


wordnet = WordNetLemmatizer()


def coltext_lemmatizer(text):
    # data_stem['TWEET_SENT_1'] = data_stem['TWEET_SENT_1'].apply(stem_texts)
    tokens = text.split()
    stemmed_tokens = [wordnet.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


snowball = SnowballStemmer("english")


def coltext_stemmer(text, sep=" "):
    tokens = text.split(sep)
    stemmed_tokens = [snowball.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)


def coltext_stopwords(text, stopwords=None, sep=" "):
    tokens = text.split(sep)
    tokens = [t.strip() for t in tokens if t.strip() not in stopwords]
    return " ".join(tokens)


def pd_coltext_fillna(df, colname, val=""):
    return df[colname].fillna(val)


def pd_coltext_clean(dfref, colname, stopwords):
    if isinstance(colname, str):
        raise Exception("colname should be list of colname")

    df = dfref[colname]
    # fromword = [ r"\b({w})\b".format(w=w)  for w in fromword    ]
    # print(fromword)
    for col in colname:
        df[col] = df[col].fillna("")
        df[col] = df[col].str.lower()
        df[col] = df[col].apply(lambda x: x.translate(string.punctuation))
        df[col] = df[col].apply(lambda x: x.translate(string.digits))
        df[col] = df[col].apply(lambda x: re.sub("[!@,#$+%*:()'-]", " ", x))

        df[col] = df[col].apply(lambda x: coltext_stopwords(x, stopwords=stopwords))
    return df


def pd_coltext_clean_advanced(dfref, colname, fromword, toword):
    df = dfref[colname]
    # fromword = [r"\b({w})\b".format(w=w) for w in fromword]
    fromword = set(fromword)
    # print(fromword)
    for col in colname:
        df[col] = df[col].fillna("")
        df[col] = df[col].str.lower()
        df[col] = df[col].replace(fromword, toword, regex=True)
    return df


def pd_coltext_wordfreq(df, coltext, sep=" "):
    """
    :param df:
    :param coltext:  text where word frequency should be extracted
    :param nb_to_show:
    :return:
    """
    dfres = df[coltext].apply(lambda x: pd.value_counts(x.split(sep))).sum(axis=0).reset_index()
    dfres.columns = ["word", "freq"]
    dfres = dfres.sort_values("freq", ascending=0)
    return dfres


def pd_fromdict(ddict, colname):
    """
    :param ddict:
    :param colname:
    :return:
    """
    colname = ("c0", "c1") if colname is None else colname
    klist, xlist = [], []
    for k, x in ddict.items():
        klist.append(k)
        xlist.append(x)
    df = pd.DataFrame({colname[0]: klist, colname[1]: xlist})
    df = df.sort_values(by=colname[1], ascending=False)
    return df


def pd_coltext_encoder(df):
    """
    https://dirty-cat.github.io/stable/auto_examples/02_fit_predict_plot_employee_salaries.html#sphx-glr-auto-examples-02-fit-predict-plot-employee-salaries-py

    :param df:
    :return:
    """
    pass


def pd_coltext_countvect(df, coltext, word_tokeep=None, word_minfreq=1, return_val="dataframe,param"):
    """
    Function that adds count of a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        coltext:   column of df to apply tf-idf to
    Returns:
        concat_df:      dataframe with a new column for each word
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    if not isinstance(coltext, str):
        raise Exception("coltext should be column string")

    # Calculate count word
    vect = CountVectorizer(
        min_df=word_minfreq,
        ngram_range=(1, 3),
        strip_accents="unicode",
        lowercase=True,
        analyzer="word",
        token_pattern=r"\w+",
        stop_words=None,
    )

    if word_tokeep is None:
        v = vect.fit_transform(df[coltext])
    else:
        vect.fit(word_tokeep)
        v = vect.transform(df[coltext])

    v = v.toarray()
    voca = vect.get_feature_names()
    # print(v.shape)
    count_list = np.asarray(v.sum(axis=0))
    word_dict = dict(zip(word_tokeep, count_list))
    # print(len(word_tokeep))
    # voca = vect.vocabulary_

    df_vector = pd.DataFrame(v)
    df_vector.columns = vect.vocabulary_
    if return_val == "dataframe,param":
        return df_vector, word_dict
    else:
        return df_vector


def pd_coltext_tdidf(df, coltext, word_tokeep=None, word_minfreq=1, return_val="dataframe,param"):
    """
    Function that adds tf-idf of a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        col_tofilter:   column of df to apply tf-idf to
    Returns:
        concat_df:      dataframe with a new column for each word
        https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    """
    from sklearn.feature_extraction.text import CountVectorizer

    for coltext_ in coltext:
        if not isinstance(coltext_, str):
            raise Exception("coltext should be string")

    if word_tokeep is None:
        cv = CountVectorizer(
            min_df=1,
            ngram_range=(1, 3),
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"\w+",
            stop_words=None,
        )
        X = cv.fit_transform(df[coltext])
        word_tokeep = cv.get_feature_names()
        count_list = np.asarray(X.sum(axis=0))
        word_dict = dict(zip(word_tokeep, count_list))
        # print(len(word_tokeep))

    # Calculate td-idf vector
    vectorizer = TfidfVectorizer(stop_words='english' ,vocabulary=word_tokeep)
    vectorizer.fit(word_tokeep)
    v = vectorizer.transform(df[coltext])
    v = v.toarray()
    # print(v.shape)

    voca = vectorizer.vocabulary_

    df_vector = pd.DataFrame(v)
    # df_new = pd.concat([df, df_vector],axis=1)
    if return_val == "dataframe,param":
        return df_vector, voca
    else:
        return df_vector


def pd_coltext_minhash(dfref, colname, n_component=2, model_pretrain_dict=None,
                       return_val="dataframe,param"):
    """
    dfhash, colcat_hash_param = pd_colcat_minhash(df, colcat, n_component=[2] * len(colcat),
                                              return_val="dataframe,param")
    :param dfref:
    :param colname:
    :param n_component:
    :param return_val:
    :return:
    """
    from source.bin.column_encoder import MinHashEncoder

    df = dfref[colname]
    model_pretrain_dict = {} if model_pretrain_dict is None else model_pretrain_dict
    enc_dict = {}
    dfall = None
    for i, col in enumerate(colname):

        if model_pretrain_dict.get(col) is None:
            clf = MinHashEncoder(n_component[i])
            clf = clf.fit(df[col])
        else:
            clf = copy.deepcopy(model_pretrain_dict[col])

        #dfcat = clf.transform(df[col])
        #dfcat.columns  = ["{col}_hash_{t}".format(col=col, t=t) for t in range(0,  len(dfcat.columns))]

        v = clf.transform(df[col].values)
        dfcat = pd.DataFrame( v, columns=["{col}_hash_{t}".format(col=col, t=t) for t in range(0, v.shape[1])],
                              index= df.index)

        enc_dict[col] = copy.deepcopy(clf)
        dfall = pd.concat((dfall, dfcat), axis=1) if dfall is not None else dfcat

    if return_val == "dataframe,param":
        return dfall, enc_dict

    else:
        return dfall


def pd_coltext_hashing(df, coltext, n_features=20):
    """
    Function that adds Hash a given column for words in a text corpus.
    Arguments:
        df:             original dataframe
        word_tokeep: corpus of words to look into
        col_tofilter:   column of df to apply tf-idf to

    Returns:
        concat_df:      dataframe with a new column for each word
    """
    from sklearn.feature_extraction.text import HashingVectorizer

    vectorizer = HashingVectorizer(n_features=n_features)
    vector = vectorizer.transform(df[coltext])
    print(vector.shape)
    colname = ["c" + str(i) for i in range(0, n_features)]

    df_vector = pd.DataFrame(vector.toarray(), columns=colname)
    return df_vector


def pd_coltext_tdidf_multi(df, coltext, coltext_freq, ntoken=100, word_tokeep_dict=None, stopwords=None,
                           return_val="dataframe,param",):
    dftext_tdidf = {}
    word_tokeep_dict_new = {}
    for col in coltext:
        if word_tokeep_dict is None:
            word_tokeep = coltext_freq[col]["word"].values[:ntoken]
            word_tokeep = [t for t in word_tokeep if t not in stopwords]
        else:
            word_tokeep = word_tokeep_dict[col]

        dftext_tdidf[col], word_tokeep_dict_new[col] = pd_coltext_tdidf(
            df, col, word_tokeep=word_tokeep, word_minfreq=1, return_val="dataframe,param"
        )

    if return_val == "dataframe,param":
        return dftext_tdidf, word_tokeep_dict_new
    else:
        return dftext_tdidf

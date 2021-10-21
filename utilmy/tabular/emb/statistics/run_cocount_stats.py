# Create corpus to train model 
# Train the model and save 
# Create co-count matrix 
# Prepair cocount similarity file and model and save fasttext_1 and cocount_1 results.
# Generate corpus based on unigram and cocount frequency 
# Train the model and save 
# Prepair cocount similarity file and model and save fasttext_2 and cocount_2 results.
# Add 'ss' as prefix and suffix to last generated corpus 
# Train the model and save 
# Prepair cocount similarity file and model and save fasttext_ss and cocount_ss results.


import re
import numpy as np
import pandas as pd 
from util_rank import *

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from essential_generators import DocumentGenerator
from sklearn.feature_extraction.text import CountVectorizer

def generate_corpus(outfile="data.cor", unique_words_needed=1000):
    """
    function to generate trainable data for model
    outfile: file where corpus will be saved
    unique_words_needed: total unique words in all corpus.
    no returns, just saves the corpus file to outfile location.
    """
    gen = DocumentGenerator()
    lemmatizer = WordNetLemmatizer()
    unique_words = set([])

    page = ""
    while len(unique_words) < unique_words_needed:
        sentence = gen.sentence()
        sentence = sentence.lower()
        sentence = re.sub(r'[^a-z]', ' ', sentence)   ### ascii only
        sentence = re.sub(r'\s+', ' ', sentence)  ### Removes all multiple whitespaces with a whitespace in a sentence
        sentence = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence) if word not in stopwords.words('english')]

        sentence = " ".join(sentence)
        sentence = re.sub(r'\b\w{1,2}\b', '', sentence)

        if len(sentence.split(' ')) < 10:
            continue

        page += sentence + "\n"
        
        for word in sentence.split(' '):
            unique_words.add(word)

    with open(outfile, mode='w') as fp:        
        fp.write(page)

    fp.close()


def train_model(dirinput="./data.cor", dirout="./modelout/model.bin", **params):
    """
    dirinput: full path of data.cor(corpus) file
    no return, just saves the model to dirout location.
    """
    from util_model import gensim_model_train_save
    pars = {'min_count':2}
    for key, value in params:
        pars[key] = value
    gensim_model_train_save(dirinput=dirinput, dirout="./modelout/model.bin", pars=pars)


def load_model(dirin="./modelout/model.bin"):
    """
    dirinput: location of saved gensim model
    """
    from gensim.models import FastText
    model = FastText.load(f'{dirin}')
    return model


def cocount_matrix(dirin="gen_text_dist3.txt"):
    """
    dirin: corpus file to generate cocount matrix.
    returns: cocount matrix, normalize cocount matrix, dictionary with all top similar words(wb) from a given word(wa), dictionary with all word_to_index in matrix
    """
    f = open(dirin, 'r').read().split('\n')
    doc = []
    for sen in f:
        temp = sen.split(' ')
        try:
            i = temp.index('')
            temp.remove(i)
        except:
            pass
        doc.append(" ".join(temp))

    count_model = CountVectorizer(ngram_range=(1,1)) 
    X = count_model.fit_transform(doc)

    Xc = (X.T * X) 

    Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
    # print(Xc.todense()) # print out matrix in dense format

    matrix = Xc.todense()

    ## normalization
    norm_matrix = np.log(1+matrix)

    def id_to_w(idx, w_to_id):
        for key in w_to_id:
            if w_to_id[key] == idx:
                return key

    from numpy import dot
    from numpy.linalg import norm

    top_similar_dic = {}
    w_to_id = count_model.vocabulary_
    for w in w_to_id:
        temp_dic = {}
        
        a = matrix[:, w_to_id[w]]
        a = np.squeeze(np.asarray(a))
        temp_l = []
        for i,v in enumerate(a):
            temp_l.append([i,v])
        temp_l = sorted(temp_l, key=lambda x: x[1], reverse=True)

        for i, v in temp_l:
            temp_dic[id_to_w(i, w_to_id)] = v
                
        top_similar_dic[w] = temp_dic

    return matrix, norm_matrix, top_similar_dic, w_to_id


def get_top_k(w, top_similar_dic, top=5):
    """
    w: word
    top_similar_dic: dictionary with all top similar words(wb) from a given word(wa)
    top: top-k values similar in output.
    """
    dic = top_similar_dic[w]
    temp = sorted(dic, key=lambda x: abs(dic[x]), reverse=True)
    
    resp = {}
    if len(temp)>top:
        temp = temp[:top]
        
    for w in temp:
        resp[w] = dic[w]
        
    return temp, resp

def save_all_stats(model, top_similar_dic, corpus_file="data.cor", top=20, output_dir="./no_ss_test"):
    """
    model: gensim model, 
    top_similar_dic: dictionary with all top similar words(wb) from a given word(wa)
    corpus_file: corpus location
    top: top-k values in output
    output_dir: name of folder where all test csv files will be saved.
    """
    from collections import Counter

    f = open(corpus_file, 'r').read()
    f = f.replace('\n', ' ')
    dic =  Counter(f.split(' '))

    frequent_words = []
    for w in model.wv.index2entity:
        if w == "": continue
        frequent_words.append([w, dic[w]])

    frequent_words = sorted(frequent_words, key=lambda x: x[1], reverse=True)

    if len(frequent_words)>100:
        frequent_words = frequent_words[:100]

    frequent_words = [w for w,_ in frequent_words]

    rank_biased_overlap_list = []
    rank_topk_kendall_list = []
    w_id= []
    w_id_list = []
    w_id_ref = []

    for i in range(1, len(frequent_words)):
        if frequent_words[i] == "": continue
        sw, _ = get_top_k(frequent_words[i], top_similar_dic, top=top)
        nsw = [x[0] for x in model.wv.most_similar(frequent_words[i], topn=top)]

        w_id.append(frequent_words[i])
        w_id_ref.append(",".join(nsw))
        w_id_list.append(",".join(sw))

        sw = np.array(sw)
        nsw = np.array(nsw)
        
        if sw.shape[0] == 0 or nsw.shape[0]==0: continue

        rank_biased_overlap_list.append(rank_biased_overlap(list1=sw, list2=nsw, p=0.9))
        rank_topk_kendall_list.append(rank_topk_kendall(a=sw, b=nsw, topk=top,p=0))

    
    newdf = pd.DataFrame()
    newdf['id'] = w_id
    newdf['id_list'] = w_id_list
    newdf['ref_list'] = w_id_ref
    newdf.to_csv('tempdf.csv', index=False)

    rank_topk_check(dirin='newdf2.csv', dirout=output_dir)

    newdf = pd.DataFrame()
    newdf['rank_biased_overlap'] = rank_biased_overlap_list 
    newdf['rank_topk_kendall'] = rank_topk_kendall_list 
    newdf.to_csv(output_dir+'/topk_tests.csv', index=False)


def generate_corpus_from_cocount_and_unigram(dirin="./data.cor", dirout="gen_text_dist3.txt", unique_words=100, sentences_count=1000):
    """
    dirin: location of corpus file to generate new text based on A0 A1 A2 rule:
        A0 : random word from unigram 
        A1 : random word from cocount probality 
        A2: random word from cocount probality 
        sentence: A1 A0 A2,
    dirout: saves the corpus to that location
    unique_words: total unique words in corpus 
    sentences_count: total number of sentences in corpus.
    """
    from collections import Counter

    f = open(dirin, 'r').read()
    f = f.replace('\n', ' ')
    dic =  Counter(f.split(' '))

    del dic['']

    unigram_dis = {}
    for w in dic:
        unigram_dis[w] = dic[w]/len(dic)

    frequent_words = []
    for w in dic:
        if w == "": continue
        frequent_words.append([w, dic[w]])

    frequent_words = sorted(frequent_words, key=lambda x: x[1], reverse=True)

    if len(frequent_words)>unique_words:
        frequent_words = frequent_words[:unique_words]
    else:
        raise Exception('total unique_words in corpus already lower then required')

    frequent_words = [w for w,_ in frequent_words]
    unigram_p = [unigram_dis[w] for w in frequent_words]

    nrom_prod = 1/sum(unigram_p)
    unigram_p = np.array(unigram_p)
    unigram_p = unigram_p*nrom_prod

    assert len(frequent_words) == len(unigram_p)

    matrix, _, _, w_to_id = cocount_matrix(dirin="gen_text_dist3.txt")
    unigram_word_ids = [w_to_id[w] for w in frequent_words]
    unigram_word_ids = sorted(unigram_word_ids)
    matrix = matrix[unigram_word_ids][:,unigram_word_ids]

    cocount_words = []
    for idx in unigram_word_ids:
        for w in w_to_id:
            if w_to_id[w] == idx:
                cocount_words.append(w)

    # normalization
    pnorm_matrix = np.log(1+matrix)
    for i, row in enumerate(pnorm_matrix):
        pnorm_matrix[i,:] = row/np.sum(row)
    
    pnorm_matrix[np.isnan(pnorm_matrix)] = 0

    A0 = np.random.choice(frequent_words, size=sentences_count, p=unigram_p)
    gen_text_list = []
    for w in A0:
        sample = []
        p = np.squeeze(np.asarray(pnorm_matrix[cocount_words.index(A0[0]),:]))
        A1 = np.random.choice(cocount_words, size=1, p=p)
        A2 = np.random.choice(cocount_words, size=1, p=p)
        
        sample.append(A1[0])
        sample.append(w)    
        sample.append(A2[0])    
        gen_text_list.append(sample)

    gen_text = ""
    for sen in gen_text_list:
        sample = " ".join(sen)
        gen_text += sample + '\n'

    with open(dirout, "w") as text_file:
        text_file.write(gen_text)

    text_file.close()


def add_prefix_and_suffix_corpus(dirin="gen_text_dist3.txt", dirout="gen_text_dist4.txt"):
    """
    to add ss in front and end of each sentence, 
    dirin: original corpus location 
    dirout: updated corpus save location.
    """
    gen_text = open(dirin, 'r').read().split('\n')
    gen_text = gen_text[:1000]
    new_gen = ""
    for sen in gen_text:
        sen = sen.split(' ')
        temp = ['ss']
        temp.extend(sen)
        temp.append('ss')
        newsen = " ".join(temp)
        new_gen += newsen + '\n'

    with open(dirout, "w") as text_file:
        text_file.write(new_gen)
    text_file.close()


def run_all():
    # Create corpus to train model 
    generate_corpus()


    # Train the model and save 
    train_model()


    # Create co-count matrix 
    matrix, norm_matrix, top_similar_dic, w_to_id = cocount_matrix(dirin="data.cor")


    # Prepair cocount similarity file and model and save fasttext_1 and cocount_1 results.\
    model = load_model(dirin="./modelout/model.bin")
    save_all_stats(model, top_similar_dic, corpus_file="data.cor", top=20, output_dir="./data_test")


    # Generate corpus based on unigram and cocount frequency 
    generate_corpus_from_cocount_and_unigram(dirin="./data.cor", dirout="gen_text_dist3.txt", unique_words=100, sentences_count=1000)


    # Train the model and save 
    train_model(dirinput="./gen_text_dist3.txt", dirout="./modelout/model2.bin", min_n=9, max_n=9)


    # Prepair cocount similarity file and model and save fasttext_2 and cocount_2 results.
    model = load_model(dirin="./modelout/model2.bin")
    matrix, norm_matrix, top_similar_dic, w_to_id = cocount_matrix(dirin="./gen_text_dist3.txt")
    save_all_stats(model, top_similar_dic, corpus_file="data.cor", top=20, output_dir="./data_test_without_ss")


    # Add 'ss' as prefix and suffix to last generated corpus 
    add_prefix_and_suffix_corpus(dirin="gen_text_dist3.txt", dirout="gen_text_dist4.txt")


    # Train the model and save 
    train_model(dirinput="./gen_text_dist4.txt", dirout="./modelout/model3.bin", min_n=9, max_n=9)


    # Prepair cocount similarity file and model and save fasttext_ss and cocount_ss results.
    model = load_model(dirin="./modelout/model2.bin")
    matrix, norm_matrix, top_similar_dic, w_to_id = cocount_matrix(dirin="./gen_text_dist4.txt")
    save_all_stats(model, top_similar_dic, corpus_file="data.cor", top=20, output_dir="./data_test_with_ss")


if __name__ == "__main__":
    run_all()
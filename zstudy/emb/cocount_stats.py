import numpy as np 
import pandas as pd 
from numpy import dot
from numpy.linalg import norm
from util_rank import *
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import pickle
from util_rank import *

# loading corpus and getting cocount matrix 
def get_cocout(corpus_filename='data.cor'):
    """
    input: corpus filename
    return: cocount matrix, word to id 
    """
    f = open(corpus_filename, 'r').read().split('\n')
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
    w_to_id = count_model.vocabulary_
    return matrix, w_to_id




def get_similar_dict(matrix, w_to_id, dic):
    """
    matrix: cocount matrix
    w_to_id: word to id
    dic: word frequency count dictionary
    """
    top_similar_dic = {}
    for w in w_to_id:
        temp_dic = {}
        for nw in w_to_id:
            if (w != nw) and (dic[w]>2) and (dic[nw]>2):
                a = matrix[:, w_to_id[w]]
                a = np.squeeze(np.asarray(a))
                b = matrix[:, w_to_id[nw]]
                b = np.squeeze(np.asarray(b))
                cos_sim = dot(a, b)/(norm(a)*norm(b))
                cos_sim = np.arccos(cos_sim) / np.pi
                temp_dic[nw] = cos_sim
                
        top_similar_dic[w] = temp_dic


    with open('similar_cocount.pickle', 'wb') as handle:
        pickle.dump(top_similar_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('similar_cocount.pickle', 'rb') as handle:
        top_similar_dic = pickle.load(handle)

    return top_similar_dic

def get_top_k(w, top_similar_dic, top=5):
    """
    w: word 
    top_similar_dic: dictionary of similar cosine scores
    top: number of words in results
    """
    dic = top_similar_dic[w]
    temp = sorted(dic, key=lambda x: abs(dic[x]), reverse=True)
    
    resp = {}
    if len(temp)>top:
        temp = temp[:top]
        
    for w in temp:
        resp[w] = dic[w]
        
    return temp, resp

def load_model(dirin='./modelout/model.bin'):
    from gensim.models import FastText
    model = FastText.load(f'{dirin}')
    return model


if __name__ == '__main__':
    f = open('data.cor', 'r').read()
    f = f.replace('\n', ' ')
    dic =  Counter(f.split(' '))

    frequent_words = []
    for w in dic:
        if dic[w] > 2:
            frequent_words.append(w)

    rank_biased_overlap_list = []
    rank_topk_kendall_list = []
    total = []

    # cocount 
    matrix, w_to_id = get_cocout()
    # norm
    matrix = np.log(1+matrix)
    
    top_similar_dic = get_similar_dict(matrix, w_to_id, dic=dic)

    # model 
    model = load_model()

    for i in range(1, len(frequent_words)):
        if frequent_words[i] == "": continue
        sw, _ = get_top_k(frequent_words[i], top_similar_dic, top=10)
        nsw = [x[0] for x in model.wv.most_similar(frequent_words[i], topn=10)]

        sw = np.array(sw)
        nsw = np.array(nsw)

        rank_biased_overlap_list.append(rank_biased_overlap(list1=sw, list2=nsw, p=0.9))
        rank_topk_kendall_list.append(rank_topk_kendall(a=sw, b=nsw, topk=10,p=0))

        c = 0
        for w in sw:
            if w in nsw:
                c+=1
        total.append(c/10)


    print('rank_biased_overlap_mean', np.mean(rank_biased_overlap_list))
    print('rank_topk_kendall_mean', np.mean(rank_topk_kendall_list))

    with open("stats_cocount.txt", "w") as text_file:
        text_file.write(f"rank_biased_overlap_mean: {np.mean(rank_biased_overlap_list)} \n")
        text_file.write(f"rank_topk_kendall_mean: {np.mean(rank_topk_kendall_list)}")
            
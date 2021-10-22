import pickle
from collections import Counter
from gensim.models import FastText

def get_top_k_grams(w, top_similar_dic, top=5):
    dic = top_similar_dic[w]
    temp = sorted(dic, key=lambda x: abs(dic[x]), reverse=True)
    
    resp = {}
    if len(temp)>top:
        temp = temp[:top]
        
    for w in temp:
        resp[w] = dic[w]
        
    return temp, resp


f = open('data.cor', 'r').read()
f = f.replace('\n', ' ')
dic =  Counter(f.split(' '))

count = 0
sw = 0
word_keys = []
for w in dic:
    if dic[w] > 0:
        count+=1 
        word_keys.append(w)
    if len(w) <2:
        sw+=1

frequent_words = []
for w in dic:
    if dic[w] > 2:
        frequent_words.append(w)

with open('similar.pickle', 'rb') as handle:
    top_similar_dic = pickle.load(handle)

## Model
dirin = './modelout/model.bin'
model = FastText.load(f'{dirin}')

from util_rank import *
rank_biased_overlap_list = []
rank_topk_kendall_list = []

for i in range(1, len(frequent_words)):
    try:
        sw, _ = get_top_k_grams(frequent_words[i], top_similar_dic, top=10)
        nsw = [x[0] for x in model.wv.most_similar(frequent_words[i], topn=10)]
        
        sw = np.array(sw)
        nsw = np.array(nsw)
        
        rank_biased_overlap_list.append(rank_biased_overlap(list1=sw, list2=nsw, p=0.9))
        rank_topk_kendall_list.append(rank_topk_kendall(a=sw, b=nsw, topk=10,p=0))
        
    except:
        break

print('rank_biased_overlap_mean', np.mean(rank_biased_overlap_list))
print('rank_topk_kendall_mean', np.mean(rank_topk_kendall_list))

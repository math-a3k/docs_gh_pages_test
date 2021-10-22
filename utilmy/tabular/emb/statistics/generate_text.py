import numpy as np 
import pandas as pd 

from collections import Counter

f = open('data.cor', 'r').read()
f = f.replace('\n', ' ')
dic =  Counter(f.split(' '))

del dic['']

unigram_dis = {}
for w in dic:
    unigram_dis[w] = dic[w]/len(dic)


frequent_100 = []
temp = []
c = 0
for w in dic:
    if dic[w] > 2:
        c+=1
        frequent_100.append(w)
        
    if dic[w] == 2:
        temp.append(w)
frequent_100 = np.array(frequent_100)
frequent_100 = np.hstack((frequent_100, np.random.choice(temp, size=100-c)))

unigram_words = [key for key in frequent_100]
unigram_p     = [unigram_dis[key] for key in unigram_words] 

nrom_prod = 1/sum(unigram_p)
unigram_p = np.array(unigram_p)
unigram_p = unigram_p*nrom_prod

assert len(unigram_words) == len(unigram_p)

f = open('data.cor', 'r').read().split('\n')
doc = []
for sen in f:
    temp = sen.split(' ')
    try:
        i = temp.index('')
        temp.remove(i)
    except:
        pass
    doc.append(" ".join(temp))
    
from sklearn.feature_extraction.text import CountVectorizer
count_model = CountVectorizer(ngram_range=(1,1)) 
X = count_model.fit_transform(doc)

Xc = (X.T * X) 

Xc.setdiag(0) # sometimes you want to fill same word cooccurence to 0
# print(Xc.todense()) # print out matrix in dense format

matrix = Xc.todense()

w_to_id = count_model.vocabulary_
unigram_word_ids = [w_to_id[w] for w in unigram_words]
unigram_word_ids = sorted(unigram_word_ids)
matrix = matrix[unigram_word_ids][:,unigram_word_ids]
pnorm_matrix = np.log(1+matrix)

for i, row in enumerate(pnorm_matrix):
    pnorm_matrix[i,:] = row/np.sum(row)
    
pnorm_matrix[np.isnan(pnorm_matrix)] = 0
cocount_words = []
for idx in unigram_word_ids:
    for w in w_to_id:
        if w_to_id[w] == idx:
            cocount_words.append(w)

# conditions 
A0 = np.random.choice(unigram_words, size=1000, p=unigram_p)
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

# with open("gen_text_dist.txt", "w") as text_file:
#     text_file.write(gen_text)
# text_file.close()
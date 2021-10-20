import pickle
import numpy as np 
from numpy import dot
from numpy.linalg import norm
from collections import Counter
from tqdm import tqdm 

def get_contex_words(l):
    x = []; y = []
    window = 2
    for i, word in enumerate(l):
#         if dic[word] > 1:
        for w in range(window):
            if i + 1 + w < len(l): 
                x.append(word)
                y.append(l[(i + 1 + w)])

            if i - w - 1 >= 0:
                x.append(word)
                y.append(l[(i - w - 1)])
    return (x,y)

# get_contex_words(l=['hamilton', 'advocated', 'to', 'program', 'robot', 'navigation', 'and', 'sensation', 'includes', 'theories', 'of', 'modern'])

class Encoding():
    def __init__(self, dic):
        self.data = dic 
        self.w_to_id = {}
        self.id_to_w = {}
        self._word_id()
        
    def __len__(self):
        return len(self.data)
    
    def _word_id(self):
        count = 0
        for key in self.data:
            if self.data[key] > 2:
                self.w_to_id[key] = count 
                self.id_to_w[count] = key 
                count+=1 
            
    def __call__(self, w):
        temp = [0]*len(self.w_to_id)
        temp[self.w_to_id[w]] = 1 
        return np.array(temp) 

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
E = Encoding(dic=dic)

f = open('data.cor', 'r').read().split('\n')
X=[]; Y=[];
for sentence in f:
    l = sentence.split(' ')
    x, y = get_contex_words(l)
    X.extend(x)
    Y.extend(y)

X_in = []
y_out = []
for i, w in enumerate(X):
    yw = Y[i]
    
    if (dic[w]>2) and (dic[yw]>2):
        v = E(w)
        X_in.append(v)

        v = E(w)
        y_out.append(v)
    
X_in = np.array(X_in)
y_out = np.array(y_out)

## embedding model 

from keras.models import Input, Model
from keras.layers import Dense
# Defining the size of the embedding
embed_size = 10

# Defining the neural network
inp = Input(shape=(X_in.shape[1],))
x = Dense(units=embed_size, activation='linear')(inp)
x = Dense(units=y_out.shape[1], activation='softmax')(x)
m = Model(inputs=inp, outputs=x)
m.compile(loss = 'categorical_crossentropy', optimizer = 'adam')

# Optimizing the network weights
m.fit(
    x=X_in, 
    y=y_out, 
    batch_size=32,
    epochs=1000
    )

# The input layer 
weights = m.get_weights()[0]

# Creating a dictionary to store the embeddings in. The key is a unique word and 
# the value is the numeric vector
embedding_dict = {}
for word in dic: 
    if dic[word] > 1:
        embedding_dict.update({
            word: weights[E.w_to_id.get(word)]})

top_similar_dic = {}
for w in tqdm(embedding_dict):
    temp_dic = {}
    for nw in embedding_dict:
        if (w != nw) and (dic[w]>2) and (dic[nw]>2):
            a = embedding_dict[w]
            b = embedding_dict[nw]
            cos_sim = dot(a, b)/(norm(a)*norm(b))
            temp_dic[nw] = cos_sim
            
    top_similar_dic[w] = temp_dic



with open('similar.pickle', 'wb') as handle:
    pickle.dump(top_similar_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
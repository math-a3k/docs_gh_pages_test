#!/usr/bin/python
# '*' coding: utf8 '*'
import math
import numpy
import sys
import csv
import os
from collections import defaultdict
from pprint import pprint
import random
import argparse
from functools import cmp_to_key

import inspect
import rank_fusion_functions

functions = {}
    
for a,b, in inspect.getmembers(rank_fusion_functions, inspect.isfunction):
    functions[a] = b

#class to display floats with 4 decimal spaces
class prettyfloat(float):
    def __repr__(self):        return "%0.7f" % self
    def __str__(self):
         return "%0.7f" % self
         
class prettyint(int):
    def __repr__(self):
        return "%03d" % self
    def __str__(self):
         return "%03d" % self

def get_fusion_alg(text):
    return functions[text.lower()]

def parse_svmlight_rank(filepath):
    ranks = {}
    with open(filepath, 'r') as f:
        for row in f:
            topic = row.split(' ')[1][4:]
            doc = row.split('#')[1].split(' ')[2]

            features = row.split('#')[0]

            features = features.strip().split(' ')[2:]
            
            for feature in features:
                feature = feature.split(":")
                feat_id = feature[0]
                feat_val = feature[1]
                if feat_val != 'NULL':
                    if topic not in ranks:
                        ranks[topic] = {}
                    if doc not in ranks[topic]:
                        ranks[topic][doc] = []
                    ranks[topic][doc].append((0, int(feat_val), feat_id))
    return ranks
 
def sort_by_score_and_id(elem1,elem2):
    if elem1[1] == elem2[1]:
        return elem1[0] < elem2[0]
    else:
        return elem2[1] - elem1[1]

def parse_svmlight_score(filepath):
    scores = {}
    rel_judge = {}
    with open(filepath, 'r') as f:
        for row in f:
            rel = row.split(' ')[0]
            topic = row.split(' ')[1][4:]
            doc = row.split('#')[1].split(' ')[2]
            features = row.split('#')[0]
            features = features.strip().split(' ')[2:]
            if not topic in rel_judge:
                rel_judge[topic] = {}
            rel_judge[topic][doc] = rel
            for feature in features:
                feature = feature.split(":")
                feat_id = feature[0]
                feat_val = feature[1]
                if float(feat_val) != 0:
                    if topic not in scores:
                        scores[topic] = {}
                    if feat_id not in scores[topic]:
                        scores[topic][feat_id] = []
                    scores[topic][feat_id].append( (doc,float(feat_val)))
    ranks = {}
    for topic in sorted(scores, key=lambda e: int(e)):
        for feat_id in scores[topic]:
            sorted_topic = sorted(scores[topic][feat_id], key=cmp_to_key(sort_by_score_and_id))
            i = 1
            for doc,feat_val in sorted_topic:
                if topic not in ranks:
                    ranks[topic] = {}
                if doc not in ranks[topic]:
                    ranks[topic][doc] = []
                ranks[topic][doc].append((float(feat_val), i, feat_id))
                i+=1
    return ranks, rel_judge
 

# parse TREC style file
def parse_trec(filepath,idIsFilename=False): 
    ranks = {}
    lowest = float('inf')
    highest = float('-inf')
    total = 0
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            topic = row[0]
            doc = row[2]
            rank = int(row[3])
            score = float(row[4])
            engine = row[5]
            if idIsFilename:
                engine=filepath.split('/')[-1].split('.')[0]

            if rank <= 1000:
                if score > highest:
                    highest = score
                if score < lowest:
                    lowest = score

                if topic not in ranks:
                    ranks[topic] = {}

                if doc not in ranks[topic]:
                    ranks[topic][doc] = (score, rank, engine)

    return ranks, lowest, highest

def norm_minmax(ranks, lowest, highest): #normalizar as scores por MIN_MAX
    lowest -= 0.001
    n_ranks = {}
    for topic in sorted(ranks):
        if topic not in n_ranks:
            n_ranks[topic] = {}
        for doc in ranks[topic]:
            score, rank, engine = ranks[topic][doc]
            if highest == lowest:
                n_score = 0
            else:
                n_score = (score - lowest) / (highest-lowest)
            n_ranks[topic][doc] = (n_score, rank, engine)
    return n_ranks

def norm_zscore(ranks, lowest, highest): #normalizar as scores por ZSCORE
    n_ranks = {}
    for topic in sorted(ranks):
        if topic not in n_ranks:
            n_ranks[topic] = {}
        scores = []
        for doc in ranks[topic]:
            score, rank, engine = ranks[topic][doc]
            scores.append(score)
        mean = numpy.mean(scores)
        stdDev = numpy.std(scores)
        n_scores = []
        for doc in ranks[topic]:
            score, rank, engine = ranks[topic][doc]
            n_score = (score - mean) / stdDev
            n_scores.append(n_score)
        min = numpy.min(n_scores)
        max = numpy.max(n_scores)
        for doc in ranks[topic]:
            score, rank, engine = ranks[topic][doc]
            if stdDev == 0:
                n_score = 0
            else:    
                n_score = ((score - mean) / stdDev )
            n_ranks[topic][doc] = (n_score, rank, engine)
    return n_ranks



def comb(rank_list, fusion_function, params):
    c_ranks = {}
    
    topic_list = []
    
    for rank in rank_list:
        for topic in sorted(rank): 
            if topic not in topic_list: 
                topic_list.append(topic)
    
    for topic in sorted(topic_list): #para cada query
        all_docs = []
        
        for ranks in rank_list: #juntar as listas iniciais numa só lista
            if topic in ranks:
                all_docs.append(ranks[topic])

        doc_id_scores = defaultdict(list)

        for doc in all_docs: #agrupar os resultados de todas as listas num dicionario de listas de pares de scores e ranks: doc_id_scores[doc_id] = [(0,1;2),(0,4;1),....]
            for doc_id, score_and_rank in doc.items():
                doc_id_scores[doc_id].append(score_and_rank)

        ranks = []
            
        #if (fusion_function.__name__ == 'condor'):
        #    ranks = condor(doc_id_scores)
        #else:
        for doc_id in doc_id_scores: #fundir as listas [(0,1;2),(0,4;1),....], resultando numa score
            score = fusion_function(doc_id_scores[doc_id],params)
            ranks.append((doc_id, score))
        # a lista de (doc, score) é ordenada à posteriori
        #random.shuffle(ranks)
        
        c_ranks[int(topic)] = ranks

    return c_ranks

def print_comb(ranks,max_k,outstream,rank_name):
    if max_k > 0:
        for topic in sorted(ranks):
            total = 1
            for doc in sorted(ranks[topic], key=lambda elem: elem[1], reverse=True):#imprime os resultados ordenados por score
                outstream.write('{0}\tQ0\t{1}\t{2}\t{3}\t{4}\n'.format(topic, doc[0], total, prettyfloat(doc[1]),rank_name))

                total = total + 1
                if total > max_k:
                    break

def folder_merge(base_path,norm,merge_function,params,max_k,rank_name,output):
    combList = []
    
    for dirname, dirnames, filenames in os.walk(base_path):
        for filename in sorted(filenames):
            ranks1, low1, high1 = parse_trec(os.path.join(dirname, filename)) #ler uma lista no formato do TREC
            if norm != None:
                ranks1 = norm(ranks1, low1, high1)
            combList.append(ranks1)

        break
    c_ranks = comb(combList,get_fusion_alg(merge_function.lower()),params) #combinar listas
    
    if output == sys.stdout:
        f = output
        print_comb(c_ranks,max_k,f,args.name)
    else:
        with open(output, 'w') as f:
            print_comb(c_ranks,max_k,f,args.name)

def file_merge(base_path,norm,merge_function,params,max_k,rank_name,output):
    combList = []
    

    rank_list,_ = parse_svmlight_score(base_path) #ler uma lista no formato do TREC
    fusion_function = get_fusion_alg(merge_function.lower())
    
    c_ranks = {}
        
    for topic in sorted(rank_list.keys()): #para cada query
        ranks = []
        for doc_id in rank_list[topic]: #agrupar os resultados de todas as listas num dicionario de listas de pares de scores e ranks: doc_id_scores[doc_id] = [(0,1;2),(0,4;1),....]
            score = fusion_function(rank_list[topic][doc_id],params)
            ranks.append((doc_id, score))
        # a lista de (doc, score) é ordenada à posteriori
        #random.shuffle(ranks)
        
        c_ranks[int(topic)] = ranks

    if output == sys.stdout:
        f = output
        print_comb(c_ranks,max_k,f,args.name)
    else:
        with open(output, 'w') as f:
            print_comb(c_ranks,max_k,f,args.name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('path', help='path for the runs to parse')
    parser.add_argument('-f','--fusion', help='function for fusion', default='rrf')
    parser.add_argument('-n','--norm', help='score normalization', default=None, choices=['minmax','zscore'])
    parser.add_argument('-k','--max_k', help='print rank until which position',type=int, default=1000)
    parser.add_argument('-i','--max_k_input', help='print rank until which position',type=int, default=1000)
    parser.add_argument('-e','--name', help='name of the rank for fusion', default='')
    parser.add_argument('-m','--mode', help='type of fusion', choices=['trec','svm'])
    parser.add_argument('-o','--output', help='output file. Leave empty to output to std. out', default=sys.stdout)
    parser.add_argument('-a','--args', help='arguments for the fusion function', default=[], nargs='*')
    args = parser.parse_args()
    
    if args.name == '':
        args.name = args.fusion 
        if len(args.args) > 0:
            args.name += '_' + '_'.join(args.args)

    if args.norm != None:
        if args.norm == 'minmax':
            args.norm = globals()['norm_minmax']
        elif args.norm == 'zscore':
            args.norm = globals()['norm_zscore']

    if args.mode == 'trec':
        folder_merge(args.path,args.norm,args.fusion,args.args,args.max_k,args.name,args.output)
    else:
        file_merge(args.path,args.norm,args.fusion,args.args,args.max_k,args.name,args.output)
    #imageCLEFMerge(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4:])
    #mergeFolder(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4:])
    #trecMergeBestToWorse(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4:])

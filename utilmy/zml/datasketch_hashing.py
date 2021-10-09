from datasketch import MinHash, MinHashLSH
import pandas as pd
import time
import warnings
#Temperory to load some data
import pickle as pkl

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

'''
For each of the entry create a hash function
'''
def create_hash(df, column_name, threshold, num_perm):
    #Create LSH 
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    
    #Intialize list
    hash_lines = []
    for index, sentence in df.itertuples():
        
        #Get tokens of individual elements
        tokens = sentence.split()
        
        #Create local hash funtion
        v = MinHash(num_perm=num_perm)
        
        for j in set(tokens):
            v.update(j.encode('utf8'))
            
        #Append
        hash_lines.append(v)
        lsh.insert(str(index), v)
    return hash_lines, lsh

'''
For each of the hash function find a cluster and assign unique id to the dataframe cluster_id
'''

def find_clusters(df, column_name, threshold, num_perm):
 
    #MAster cluster ids
    all_cluster_ids = []
    
    #REturn from hash list
    hash_lines, lsh = create_hash(df, column_name=column_name, threshold = threshold, num_perm = num_perm)

    #For each local hash find the cluster ids and assign to the dataframe and return dataframe
    cluster_count = 1
    for ind, i in enumerate(hash_lines):

        if ind in all_cluster_ids:
            continue
            
        duplicate_elements = lsh.query(i)
        duplicate_elements_int = list(map(int, duplicate_elements))
        #print(duplicate_elements_int)
        df.at[duplicate_elements_int, 'cluster_id'] = cluster_count
        cluster_count+=1
        
        all_cluster_ids += duplicate_elements_int

    return df


if __name__ == "__main__":
    
    with open("Sentences.pkl",'rb') as f:
        sentences = pkl.load(f)
    
    column_name = "sentence"
    threshold = 0.7
    num_perm = 10
    num_items = 100000
    
    df = pd.DataFrame(sentences, columns = [column_name])
    s = time.time()
    df_1 = find_clusters(df.head(num_items), column_name, threshold, num_perm )
    e = time.time()
    print(e-s)

    
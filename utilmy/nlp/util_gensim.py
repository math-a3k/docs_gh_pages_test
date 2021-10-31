# coding=utf-8
HELP = """
   Gensim model
"""
import os, sys, itertools, time, pandas as pd, numpy as np, pickle, gc, re
from typing import Callable, Tuple, Union
from box import Box


#################################################################################################
from utilmy import log, log2


def help():
    from utilmy import help_create
    ss = HELP + help_create("utilmy.nlp.util_model")
    print(ss)

#################################################################################################
def test_all():
   test_gensim1()
   
def test_gensim1():
    log("test_gensim")
    dir0 = os.getcwd()  
    pars = Box({})
    pars.min_n=6 ;  pars.max_n=6; pars.window=3;  pars.vector_size=3
   
    text_generate_random_sentences(dirout=  dir0 + '/testdata/mytext1.txt')
    gensim_model_train_save(None, dirout= dir0 + '/modelout1/model.bin', dirinput=  dir0 +  '/testdata/mytext1.txt', epochs=1,
                            pars = pars)
    #gensim_model_check(dir0 + '/modelout1/model.bin')
   
   
    # model = gensim_model_load( dir0 + '/modelout1/model.bin')
    text_generate_random_sentences( dirout=  dir0 +  '/testdata/mytext2.txt')      
    gensim_model_train_save(model_or_path = dir0 + '/modelout1/model.bin', dirout= dir0 + '/modelout2/model.bin', dirinput= dir0 +  '/testdata/mytext2.txt', epochs=1)
    # gensim_model_check(dir0 +  '/modelout2/model.bin')


    model = gensim_model_load( dir0 + '/modelout2/model.bin')
    text_generate_random_sentences( dirout=  dir0 +  '/testdata/mytext2.txt')      
    gensim_model_train_save(model_or_path = model, dirout= dir0 + '/modelout2/model.bin', dirinput= dir0 +  '/testdata/mytext2.txt', epochs=1)
    # gensim_model_check(dir0 +  '/modelout2/model.bin')

      
      
#################################################################################################
####  Need global storage.

db_cocount_name  = {}  ######   worda  ---> List of cocount wordb
db_cocount_proba = {}  #######  wordb  --> Freuqnecy of cocoun wordb


def bigram_load_convert(path):
  pass
  db_cocount_name  = {}  ######   worda  ---> List of cocount wordb
  db_cocount_proba = {}  #######  wordb  --> Freuqnecy of cocoun wordb



def bigram_write_seq(rr=0, dirin=None, dirout=None, tag="") :
   ####  python prepro.py  ccnt_write_seq --rr 0  

   istart = int(0      + rr     * 1.4 * 10**6)
   iend   = int(istart + (rr+1) * 1.4 * 10**6)
   # iend   = 3  #  istart + (rr+1) * 2*10**6

   qq=2

   ### load Bi-gram tables from disk file
   flist = sorted( glob.glob( dirin ) )
   os_makedirs(dirout)

   df         = pd_read_file( flist ,  npool=5)  ### cols= ['a', 'cnt'],
   df.columns = [ 'a', 'b' , 'count' ]   ###Bi-grams tables
   df         = df[ (df.cnt < 10**7 ) & ( df.cnt >= 3 ) ]

   log(dirin, df, df.columns)
   if df is None or len(df) < 1 :
       return 1

   log('Start writing')
   v_ranid = df['ranid'].values
   v_cnt   = df['cnt'].values
   del df ; gc.collect()
   jj      = 0
   out     = ""

   with open(dirout, mode='a') as fp :
       for ii in range(len(v_ranid)) :
          if ii <  istart   : continue
          if ii >= iend     : break
          # ranid =  297840888214120000
          ranid   = int(  v_ranid[ii] )
          siid0   = ""  # map_ranid_to_siid(ranid, -1)
          itemtag = ""  # str(db_itemid_itemtag.get( siid0, ""))
          # if len(itemtag) < 5 : continue
          if ii % 5000 == 0   : log(ii, jj, ranid, siid0,  itemtag  )

          lname, lproba = bigram_get_list(ranid, mode='name,proba')
          if len(lname) < 1 : return ""
          lproba = np.log( lproba )   #### Log proba to reduce the gap
          pnorm  = lproba / np.sum(lproba)

          kmax = 15 * int( max(1, np.log( v_cnt[ii]  ))  )
          jj   = jj + kmax
          for kk in range(0, kmax):
              ss = bigram_get_seq3( ranid, itemtag,  lname,  pnorm= pnorm )
              if len(ss) > 5 :
                   out= out + ss + "\n"

          if len(out)  > 5000 :
              fp.write(out)
              out = ""

       if len(out)> 0 :  fp.write(out)
       log( 'all finished', jj )


def bigram_get_seq3(ranid, itemtag, lname, pnorm):
 ## Generate sequence of ssid, based on co-count proba.

 list_a1 = ccount_get_sample(lname, pnorm=pnorm, k=2 )

 try :
    ss = str(list_a1[0]) + " "  + str(ranid)  + " " +  str( list_a1[1])
    return ss
 except :
    return ""



def bigram_get_list(ranid, mode='name, proba'):
   lname =  db_cocount_name.get(ranid, "")
   if len(lname) < 1 :
       if 'proba' in mode : return [],[]
       else  : return []

   # if verbose : log(ranid, 'list',  str(lname)[:100] )
   lname =  [int(t) for t in  lname.split(",") ]


   if 'proba' in mode :
      lproba = db_cocount_proba.get(ranid,"").split(",")
      lproba = [ float(t) for t in lproba  ]
      # if verbose : log( 'name', str(lname)[:60],  str(lproba)[:60] )
      return lname, lproba

   return lname


def ccount_get_sample(lname, lproba=None, pnorm=None, k=5):
 if pnorm is None :
    pnorm = lproba / np.sum(lproba)

 ll = np.random.choice(lname, size=k,  p= pnorm )
 # ll = [ lname[0] for i in range(k) ]
 # log(ll)
 return ll


def np_intersec(va, vb):
  return [  x  for x in va if x in set(vb) ]





            

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

            
            
#################################################################################################
def gensim_model_load(dirin,  modeltype='fastext', **kw):
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
    else :
         model = model_or_path   ### actual model     

    log("#### Input data building  ", model )
    corpus_file = datapath(dirinput)
    to_update = True  if model.wv else False
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

   

def gensim_model_check(model_path):
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
    from gensim.test.utils import datapath  
    model = gensim_model_load(model_path)
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
    sentence = re.sub(r'[^a-z]', ' ', sentence)   ### ascii only
    # sentence = re.sub(r'\s+', ' ', sentence)  ### Removes all multiple whitespaces with a whitespace in a sentence
    # sentence = [lemmatizer.lemmatize(word) for word in nltk.word_tokenize(sentence) if word not in stop_words]
    return ' '.join(sentence)


def text_generate_random_sentences( dirout=None, n_sentences=5,):
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

   
   
   

####################################################################################################
####################################################################################################
def embedding_model_to_parquet(model_vector_path="model.vec", nmax=500):
    from gensim.models import KeyedVectors
    from collections import OrderedDict
    def isvalid(t):
        return True

    log("loading model.vec")  ## [Warning] Takes a lot of time
    en_model = KeyedVectors.load_word2vec_format(model_vector_path)

    # Limit number of tokens to be visualized
    limit = nmax if nmax > 0 else len(en_model.vocab)  # 5000
    vector_dim = en_model[list(en_model.vocab.keys())[0]].shape[0]

    jj = 0
    words = OrderedDict()
    embs = np.zeros((limit, vector_dim))
    for i, word in enumerate(en_model.vocab):
        if jj >= limit: break
        if isvalid(word):
            words[word] = jj  # .append(word)
            embs[jj, :] = en_model[word]
            jj = jj + 1

    embs = embs[:len(words), :]

    df_label = pd.DataFrame(words.keys(), columns=['id'])
    return embs, words, df_label


def embedding_to_parquet(dirin=None, dirout=None, skip=0, nmax=10 ** 8,
                         is_linevalid_fun=None):  ##   python emb.py   embedding_to_parquet  &
    #### FastText/ Word2Vec to parquet files    9808032 for purhase

    log(dirout);
    os_makedirs(dirout);
    time.sleep(4)

    if is_linevalid_fun is None:  #### Validate line
        def is_linevalid_fun(w):
            return len(w) > 5  ### not too small tag

    i = 0;
    kk = -1;
    words = [];
    embs = [];
    ntot = 0
    with open(dirin, mode='r') as fp:
        while i < nmax + 1:
            i = i + 1
            ss = fp.readline()
            if not ss: break
            if i < skip: continue

            ss = ss.strip().split(" ")
            if not is_linevalid_fun(ss[0]): continue

            words.append(ss[0])
            embs.append(",".join(ss[1:]))

            if i % 200000 == 0:
                kk = kk + 1
                df = pd.DataFrame({'id': words, 'emb': embs})
                log(df.shape, ntot)
                if i < 2: log(df)
                pd_to_file(df, dirout + f"/df_emb_{kk}.parquet", show=0)
                ntot += len(df)
                words, embs = [], []

    kk = kk + 1
    df = pd.DataFrame({'id': words, 'emb': embs})
    ntot += len(df)
    dirout2 = dirout + f"/df_emb_{kk}.parquet"
    pd_to_file(df, dirout2, show=1)
    log('ntotal', ntot, dirout2)
    return os.path.dirname(dirout2)


def embedding_load_parquet(dirin="df.parquet", nmax=500):
    """  id, emb (string , separated)
    
    """
    log('loading', dirin)
    col_embed = 'pred_emb'
    colid = 'id'
    # nmax    = nmax if nmax > 0 else  len(df)   ### 5000

    flist = list(glob.glob(dirin))

    df = pd_read_file(flist, npool=max(1, int(len(flist) / 4)))
    df = df.iloc[:nmax, :]
    df = df.rename(columns={col_embed: 'emb'})

    df = df[df['emb'].apply(lambda x: len(x) > 10)]  ### Filter small vector
    log(df.head(5).T, df.columns, df.shape)
    log(df, df.dtypes)

    ###########################################################################
    ###### Split embed numpy array, id_map list,  #############################
    embs = np_str_to_array(df['emb'].values, l2_norm=True, mdim=200)
    id_map = {name: i for i, name in enumerate(df[colid].values)}
    log(",", str(embs)[:50], ",", str(id_map)[:50])

    #####  Keep only label infos  ####
    del df['emb']
    return embs, id_map, df


def np_str_to_array(vv, l2_norm=True, mdim=200):
    ### Extract list of string into numpy
    # mdim = len(vv[0].split(","))
    # mdim = 200
    from sklearn import preprocessing
    import faiss
    X = np.zeros((len(vv), mdim), dtype='float32')
    for i, r in enumerate(vv):
        try:
            vi = [float(v) for v in r.split(',')]
            X[i, :] = vi
        except Exception as e:
            log(i, e)

    if l2_norm:
        # preprocessing.normalize(X, norm='l2', copy=False)
        faiss.normalize_L2(X)  ### Inplace L2 normalization
    log("Normalized X")
    return X

    def faiss_create_index(df_or_path=None, col='emb', dir_out="", db_type="IVF4096,Flat", nfile=1000, emb_dim=200):
        """
          1 billion size vector creation 
        """
        import faiss
        # nfile      = 1000
        emb_dim = 200

        if df_or_path is None:  df_or_path = dir_cpa2 + "/emb/erquet"
        dirout = "/".join(os.path.dirname(df_or_path).split("/")[:-1]) + "/faiss/"  # dir
        os.makedirs(dirout, exist_ok=True);
        log('dirout', dirout)
        log('dirin', df_or_path);
        time.sleep(10)

        if isinstance(df_or_path, str):
            flist = sorted(glob.glob(df_or_path))[:nfile]
            log('Loading', df_or_path)
            df = pd_read_file(flist, n_pool=20, verbose=False)
        else:
            df = df_or_path
        # df  = df.iloc[:9000, :]        
        log(df)

        tag = f"_" + str(len(df))
        df = df.sort_values('id')
        df['idx'] = np.arange(0, len(df))
        pd_to_file(df[['idx', 'id']].rename(columns={"id": 'item_tag_vran'}),
                   dirout + f"/map_idx{tag}.parquet", show=1)  #### Keeping maping faiss idx, item_tag

        log("### Convert parquet to numpy   ", dirout)
        X = np.zeros((len(df), emb_dim), dtype=np.float32)
        vv = df[col].values
        del df;
        gc.collect()
        for i, r in enumerate(vv):
            try:
                vi = [float(v) for v in r.split(',')]
                X[i, :] = vi
            except Exception as e:
                log(i, e)

        log("Preprocess X")
        faiss.normalize_L2(X)  ### Inplace L2 normalization
        log(X)

        nt = min(len(X), int(max(400000, len(X) * 0.075)))
        Xt = X[np.random.randint(len(X), size=nt), :]
        log('Nsample training', nt)

        ####################################################    
        D = emb_dim  ### actual  embedding size
        N = len(X)  # 1000000

        # Param of PQ for 1 billion
        M = 40  # 16  ###  200 / 5 = 40  The number of sub-vector. Typically this is 8, 16, 32, etc.
        nbits = 8  ### bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
        nlist = 6000  ###  # Param of IVF,  Number of cells (space partition). Typical value is sqrt(N)
        hnsw_m = 32  ###  # Param of HNSW Number of neighbors for HNSW. This is typically 32

        # Setup  distance -> similarity in uncompressed space is  dis = 2 - 2 * sim, https://github.com/facebookresearch/faiss/issues/632
        quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
        index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)

        log('###### Train indexer')
        index.train(Xt)  # Train

        log('###### Add vectors')
        index.add(X)  # Add

        log('###### Test values ')
        index.nprobe = 8  # Runtime param. The number of cells that are visited for search.
        dists, ids = index.search(x=X[:3], k=4)  ## top4
        log(dists, ids)

        log("##### Save Index    ")
        dirout2 = dirout + f"/faiss_trained{tag}.index"
        log(dirout2)
        faiss.write_index(index, dirout2)
        return dirout2

    def faiss_topk(df=None, root=None, colid='id', colemb='emb', faiss_index=None, topk=200, npool=1, nrows=10 ** 7,
                   nfile=1000):  ##
        """ id, dist_list, id_list ex

           https://github.com/facebookresearch/faiss/issues/632

           This represents the quantization error for vectors inside the dataset.
            For vectors in denser areas of the space, the quantization error is lower because the quantization centroids are bigger and vice versa.
            Therefore, there is no limit to this error that is valid over the whole space. However, it is possible to recompute the exact distances once you have the nearest neighbors, by accessing the uncompressed vectors.

            distance -> similarity in uncompressed space is

            dis = 2 - 2 * sim

       """
        # nfile  = 1000      ; nrows= 10**7
        # topk   = 500

        if faiss_index is None:
            faiss_index = ""
            # index       = root + "/faiss/faiss_trained_1100.index"
            # faiss_index = root + "/faiss/faiss_trained_13311813.index"
            # faiss_index = root + "/faiss/faiss_trained_9808032.index"
        log('Faiss Index: ', faiss_index)
        if isinstance(faiss_index, str):
            faiss_path = faiss_index
            faiss_index = faiss_load_index(db_path=faiss_index)
        faiss_index.nprobe = 12  # Runtime param. The number of cells that are visited for search.

        ########################################################################
        if isinstance(df, list):  ### Multi processing part
            if len(df) < 1: return 1
            flist = df[0]
            root = os.path.abspath(os.path.dirname(flist[0] + "/../../"))  ### bug in multipro
            dirin = root + "/df/"
            dir_out = root + "/topk/"

        elif df is None:  ## Default
            # root =  dir_rec + "/emb/emb/ichiba_clk_202006_202012d_itemtagb_202009_12/seq_merge_2020_2021_pur/"
            root = dir_cpa2 + "/emb/emb/ichiba_order_20210901b_itemtagb2/seq_1000000000/"
            dirin = root + "/df/*.parquet"
            dir_out = root + "/topk/"
            flist = sorted(glob.glob(dirin))

        else:  ### df == string path
            root = os.path.abspath(os.path.dirname(df) + "/../")
            log(root)
            dirin = root + "/df/*.parquet"
            dir_out = root + "/topk/"
            flist = sorted(glob.glob(dirin))

        log('dir_in', dirin);
        log('dir_out', dir_out);
        time.sleep(2)
        flist = flist[:nfile]
        if len(flist) < 1: return 1
        log('Nfile', len(flist), flist)
        # return 1

        ####### Parallel Mode ################################################
        if npool > 1 and len(flist) > npool:
            log('Parallel mode')
            from utilmy.parallel import multiproc_run
            ll_list = multiproc_tochunk(flist, npool=npool)
            multiproc_run(faiss_topk, ll_list, npool, verbose=True, start_delay=5,
                          input_fixed={'faiss_index': faiss_path}, )
            return 1

        ####### Single Mode #################################################
        dirmap = faiss_path.replace("faiss_trained", "map_idx").replace(".index", '.parquet')
        map_idx_dict = db_load_dict(dirmap, colkey='idx', colval='item_tag_vran')

        chunk = 200000
        kk = 0
        os.makedirs(dir_out, exist_ok=True)
        dirout2 = dir_out
        flist = [t for t in flist if len(t) > 8]
        log('\n\nN Files', len(flist), str(flist)[-100:])
        for fi in flist:
            if os.path.isfile(dir_out + "/" + fi.split("/")[-1]): continue
            # nrows= 5000
            df = pd_read_file(fi, n_pool=1)
            df = df.iloc[:nrows, :]
            log(fi, df.shape)
            df = df.sort_values('id')

            dfall = pd.DataFrame();
            nchunk = int(len(df) // chunk)
            for i in range(0, nchunk + 1):
                if i * chunk >= len(df): break
                i2 = i + 1 if i < nchunk else 3 * (i + 1)

                x0 = np_str_to_array(df[colemb].iloc[i * chunk:(i2 * chunk)].values, l2_norm=True)
                log('X topk')
                topk_dist, topk_idx = faiss_index.search(x0, topk)
                log('X', topk_idx.shape)

                dfi = df.iloc[i * chunk:(i2 * chunk), :][[colid]]
                dfi[f'{colid}_list'] = np_matrix_to_str2(topk_idx, map_idx_dict)  ### to item_tag_vran
                # dfi[ f'dist_list']  = np_matrix_to_str( topk_dist )
                dfi[f'sim_list'] = np_matrix_to_str_sim(topk_dist)

                dfall = pd.concat((dfall, dfi))

            dirout2 = dir_out + "/" + fi.split("/")[-1]
            # log(dfall['id_list'])
            pd_to_file(dfall, dirout2, show=1)
            kk = kk + 1
            if kk == 1: dfall.iloc[:100, :].to_csv(dirout2.replace(".parquet", ".csv"), sep="\t")

        log('All finished')
        return os.path.dirname(dirout2)

    def np_matrix_to_str2(m, map_dict):
        res = []
        for v in m:
            ss = ""
            for xi in v:
                ss += str(map_dict.get(xi, "")) + ","
            res.append(ss[:-1])
        return res

    def np_matrix_to_str(m):
        res = []
        for v in m:
            ss = ""
            for xi in v:
                ss += str(xi) + ","
            res.append(ss[:-1])
        return res

    def np_vector_to_str(m, sep=","):
        ss = ""
        for xi in m:
            ss += f"{xi}{sep}"
        return ss[:-1]

    def np_matrix_to_str_sim(m):  ### Simcore = 1 - 0.5 * dist**2
        res = []
        for v in m:
            ss = ""
            for di in v:
                ss += str(1 - 0.5 * di) + ","
            res.append(ss[:-1])
        return res

    def np_str_to_array(vv, l2_norm=True, mdim=200):
        ### Extract list into numpy
        # log(vv)
        # mdim = len(vv[0].split(","))
        # mdim = 200
        from sklearn import preprocessing
        import faiss
        X = np.zeros((len(vv), mdim), dtype='float32')
        for i, r in enumerate(vv):
            try:
                vi = [float(v) for v in r.split(',')]
                X[i, :] = vi
            except Exception as e:
                log(i, e)

        if l2_norm:
            # preprocessing.normalize(X, norm='l2', copy=False)
            faiss.normalize_L2(X)  ### Inplace L2 normalization
        log("Normalized X")
        return X



###################################################################################################
if __name__ == "__main__":
    import fire ;
    fire.Fire()

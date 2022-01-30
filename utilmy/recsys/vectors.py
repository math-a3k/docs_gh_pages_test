"""
Interface to Insert/Query Vectors


"""
def log(*s):
   print(*s, flush=True)



def test():
  
  #### Raw Storage
  vectorq = VectorStorage(engine='disk_numpy', engine_pars={'dir': 'mypath/', })  
  
  vec_all_df = pd.DataFrame([
    [121213, [ 3242,4324324,34242,42], 'cat1'],
    [121213, [ 3242,4324324,34242,42], 'cat1'],
    [121213, [ 3242,4324324,34242,42], 'cat1'],
     
       ], colums = ['id', 'vect', 'col1'] )  
  vectorq.set(vec_list_dict,  )  
  #### Store as numpy npz on disk
  
  
  #### Load All from table
  vec_list_all = vectorq.get_table_all(table_name)
  
  
  #### Faiss Query  
  vectorq = VectorQuery(engine='faiss', engine_pars={'dir': 'mypath/', })  
  vectorq.insert(vec_list_all, )  
  vectorq.save('mypath')
  res = vectorq.query(vec_list, topk=100,)  
  res['index'] , res['dist']
    
    
    
                                                     
    

class VectorStore(object):
    def __init__(self, engine='mongodb', engine_pars:dict=None, table='test'):
        """ Interface to Store vectors on disk or on database,
            Only retrieve by ids or tags from DB like Redis, or MongoDB
        """
        self.engine = 'mongodb' #### disk/mongodb/cassandra

    def connect(self, table='test'):


    def table_create(self, table='test', vector_size=128, distance='Euclid'):


    def table_update(self, table='test', optimizers_config=None):


    def table_info(self,):


    def table_shape(self,):



    def get_multi(self, key_list:list, **kw):
        """"  Retrieve vectors by their id. (from MongoDB, Cassandra, NoSQL DB)
        """


    def set_multi(self, keyval_dict:dict, **kw):
        """"  Insert vectors by their id. (from MongoDB, Cassandra, NoSQL DB)
        """


        
        
        
        

class VectorQuery(object):
    def __init__(self, engine='faiss', engine_pars:dict=None, table='test'):
        """  Generic Interface for Approx NN like faiss, HSSW
        """
        from box import Box
        self.engine = 'faiss'
        self.engine_pars = Box(engine_pars) 
        self.table  = table


    def connect(self, table='test'):

        if self.engine == 'faiss'
                self.client = self.connect(self.table)



    def table_create(self, table='test', vector_size=128, distance='Euclid'):
        ## Dot, Cosine

        if self.table == 'faiss' :




    def table_update(self, table='test', optimizers_config=None):



    def table_info(self, table=None):


    def table_shape(self, table=None):


    def query(self, vector_list, filter_dict=None, topk=5, append_payload=True, mode='dict', filter_cond='must', **kw):

        if filter_cond  == 'should' :   filter1 = self.create_filter_should(filter_dict)
        else :                          filter1 = self.create_filter(filter_dict)
        # log(filter1)
        res = self.client.search(  collection_name=self.table,
                            query_vector=vector,
                            query_filter=filter1,
                            append_payload=append_payload,
                            top=topk)

        if 'full' in mode : return res

        ### only result, point: sim scores
        topks =  [payload for point, payload in res]

        if 'pandas' in mode :
           if len(topks) < 1 : return pd.DataFrame()
           df = {k: [] for k in topks[0].keys() }
           for v in topks:
             for k,x in v.items():

                df[k].append(x[0])
           df = pd.DataFrame(df)
           return df

        return topks


    def insert(self, df,  colemb='emb', colsfeat=None, colid=None, batch_size=256, debug=False, verbose=True, npool=1, kbatch=24000):
        """
        """
        log('QueryVector Insert', len(df))
        if colid is not None :
           df  = df.drop_duplicates(colid)
           ids = df[colid].values

        n       = len(df)
        vectors = df[colemb].values
        if isinstance(df[colemb].values[0], str) :
           vectors = np.array([  np.array([ float(x) for x in  t.split(",")]) for t in vectors ],  dtype='float32')

        dim_size = self.table_shape()[1]
        assert vectors.shape[1] == dim_size, 'Wrong shape ' + str(dim_size )

        if colsfeat is None :
           colsfeat    = [c for c in df.columns if c not in  [colemb] ]

        payload = df[colsfeat].to_dict(orient='records')
        # payload = None
        if verbose :
           log(df[colsfeat]) ;  log(vectors.shape, vectors.dtype, len (payload))

        log('Size, Vector, NFeats:',  n,  vectors.shape, len(payload),   len(colsfeat))
        del df ; gc.collect()
        kmax = int(n  // kbatch)
        for k in range(0, kmax+1) :
            if k*kbatch >= n : break
            j2 = (k+1)*kbatch if k < kmax else n
            log(k, kbatch*k, j2)

            try :
              self.client.upload_collection(
                collection_name= self.table,
                vectors = vectors[k*kbatch:j2,:],
                payload = payload[k*kbatch:j2],
                ids     = ids[k*kbatch:j2] if ids is not None else None ,  # Vector ids will be assigned automatically
                batch_size = batch_size  # How many vectors will be uploaded in a single request?
               ,parallel   = npool
              )
            except Exception as e:
              log(e, k )
        if verbose : log( self.table_info() )




class VectorStoreCache(object):
    """  Cache layer to retrieve vectore more easily.
    """
    def __init__(self, prefix="m001", cass_query=None, nmax=10**6, use_dict=False  ) :
      self.ddict      = fixedDict(limit=nmax)

      #### Cass
      self.cass_query = cass_getconn() if cass_query is None else cass_query
      self.cass_table = "ndata.item_model"
      self.prefix     = model_prefix
      self.npref      = len(self.prefix)+1


    def get_multi(self, siids, use_dict=True, update_cache=True):
        """ Return a dict  {siid : {val} }
        """
        global db_ivector
        log('get item vect', len(siids))
        res = {}; miss = [] ;
        if use_dict :
            for sid in siids :
              try :       res[sid] = self.ddict[sid]
              except :
                 try :    res[sid] = db_ivector[sid]
                 except : miss.append(sid)
        else :
            for sid in siids :
                 try :     res[sid] = db_ivector[sid]
                 except :  miss.append(sid)

        log('In cache:', len(res) )
        if len(miss) < 1 : return res

        res2 = cass_get_itemvect_dict(miss, cass_query= self.cass_query, prefix=self.prefix, )
        log('Nemb from Cass', len(res2) )
        for key,val in res2.items():
          try :
             res[key]        = val
             if update_cache: db_ivector[key] = val
             if use_dict:     self.ddict[key] = val
          except Exception as e:
             log('error', key, val, e)
             db_ivector.set(key, False, expire=1e-9)  ### Delete items

        log('Got N emb:', len(res))
        return res




if 'import':
    import numpy as np, pandas as pd, json, os, random, uuid, time, gc


    
    
    

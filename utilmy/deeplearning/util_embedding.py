HELP="""

https://try2explore.com/questions/10109123

https://mpld3.github.io/examples/index.html


"""
import warnings ;warnings.filterwarnings("ignore")
from warnings import simplefilter  ; simplefilter(action='ignore', category=FutureWarning)
with warnings.catch_warnings():
    
    import random, os, sys, numpy as np, pandas as pd, time, gc, copy, glob
    from datetime import datetime ; from typing import List

    import matplotlib.pyplot as plt

    from scipy.cluster.hierarchy import ward, dendrogram
    from sklearn.cluster import KMeans
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_similarity
    from tqdm import tqdm

    from utilmy import pd_read_file, os_makedirs, pd_to_file

    from box import Box 


##################################################################################################################
from utilmy import log, log2


################################################################################################################
import mpld3

CSS = """
    text.mpld3-text, div.mpld3-tooltip {
      font-family:Arial, Helvetica, sans-serif;
    }
    g.mpld3-xaxis, g.mpld3-yaxis {
    display: none; }
    """

class TopToolbar(mpld3.plugins.PluginBase):
    """Plugin for moving toolbar to top of figure"""

    JAVASCRIPT = """
    mpld3.register_plugin("toptoolbar", TopToolbar);
    TopToolbar.prototype = Object.create(mpld3.Plugin.prototype);
    TopToolbar.prototype.constructor = TopToolbar;
    function TopToolbar(fig, props){
        mpld3.Plugin.call(this, fig, props);
    };
    TopToolbar.prototype.draw = function(){
      // the toolbar svg doesn't exist
      // yet, so first draw it
      this.fig.toolbar.draw();
      // then change the y position to be
      // at the top of the figure
      this.fig.toolbar.toolbar.attr("x", 150);
      this.fig.toolbar.toolbar.attr("y", 400);
      // then remove the draw function,
      // so that it is not called again
      this.fig.toolbar.draw = function() {}
    }
    """

    def __init__(self):
        self.dict_ = {"type": "toptoolbar"}


        
################################################################################################################
class vizEmbedding:
    def __init__(self, path="myembed.parquet", num_clusters=5, sep=";", config:dict=None):
        """ 
           Many issues with numba, numpy, pyarrow !!!!
           pip install  pynndescent==0.5.4  numba==0.53.1  umap-learn==0.5.1  llvmlite==0.36.0   numpy==1.19.1   --no-deps
        
           self = Box({})
           self.path = "C:/D/gitdev/cpa/data/model.vec"

           from utilmy.viz.embedding import vizEmbedding
           myviz = vizEmbedding(path = "C:/D/gitdev/cpa/data/model.vec")
           myviz.run_all(nmax=5000)

           myviz.dim_reduction(mode='mds')
           myviz.create_visualization(dir_out="ztmp/vis/")        
        
        """
        self.path         = path
        self.sep          = sep
        self.num_clusters = num_clusters
        self.dist         = None

    def run_all(self, mode="mds", col_embed='embed', ndim=2, nmax= 5000, dir_out="ztmp/", ntest=10000):
       self.dim_reduction( mode, col_embed, ndim=ndim, nmax= nmax, dir_out=dir_out, ntest=ntest)
       self.create_clusters(after_dim_reduction=True)
       self.create_visualization(dir_out, mode='d3', cols_label=None, show_server=False)


    def dim_reduction(self, mode="mds", col_embed='embed', ndim=2, nmax= 5000, dir_out=None, ntest=10000, npool=2 ): 
        
        if ".vec"     in self.path :        
          embs, id_map, df_labels  = embedding_load_word2vec(self.path, nmax= nmax)
        
        if ".parquet" in self.path :        
          embs, id_map, df_labels  = embedding_load_parquet(self.path, nmax= nmax)

            
        if mode == 'mds' :
            ### Co-variance matrix
            dist = 1 - cosine_similarity(embs)
            mds = MDS(n_components=ndim, dissimilarity="precomputed", random_state=1)
            mds.fit(dist)  # shape (n_components, n_samples)
            pos = mds.transform(dist)  # shape (n_components, n_samples)
            
            
        if mode == 'umap' :
            y_label = None
            from umap import UMAP, AlignedUMAP, ParametricUMAP
            clf = UMAP( set_op_mix_ratio=0.25, ## Preserve outlier
                        densmap=False, dens_lambda=5.0,          ## Preserve density
                        n_components= ndim,
                        n_neighbors=7,  metric='euclidean',
                        metric_kwds=None, output_metric='euclidean',
                        output_metric_kwds=None, n_epochs=None,
                        learning_rate=1.0, init='spectral',
                        min_dist=0.0, spread=1.0, low_memory=True, n_jobs= npool,
                        local_connectivity=1.0,
                        repulsion_strength=1.0, negative_sample_rate=5,
                        transform_queue_size=4.0, a=None, b=None, random_state=None,
                        angular_rp_forest=False, target_n_neighbors=-1,
                        target_metric='categorical', target_metric_kwds=None,
                        target_weight=0.5, transform_seed=42, transform_mode='embedding',
                        force_approximation_algorithm= True, verbose=False,
                        unique=False,  dens_frac=0.3,
                        dens_var_shift=0.1, output_dens=False, disconnection_distance=None)

            clf.fit( embs[ np.random.choice( len(embs), size= ntest )  , : ] , y=y_label )                      
            pos  = clf.transform( embs )          

        self.embs      = embs
        self.id_map    = id_map
        self.df_labels = df_labels        
        self.pos       = pos

        if dir_out is not None :
            os.makedirs(dir_out, exist_ok=True)
            df = pd.DataFrame(pos, columns=['x', 'y'] )
            for ci in [ 'x', 'y' ] :
               df[ ci ] = df[ ci ].astype('float32')
   
            # log(df, df.dtypes)
            pd_to_file(df.iloc[:100, :],  f"{dir_out}/embs_xy_{mode}.csv" )
            pd_to_file(df,                f"{dir_out}/embs_xy_{mode}.parquet" , show=1)
                

    def create_clusters(self, after_dim_reduction=True):
        
        import hdbscan
        #km = hdbscan.HDBSCAN(min_samples=3, min_cluster_size=10)  #.fit_predict(self.pos)
        km = KMeans(n_clusters=self.num_clusters)

        if after_dim_reduction :
           km.fit( self.pos)
        else :
           km.fit( self.embs)


        self.clusters      = km.labels_.tolist()        
        self.cluster_color = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(self.num_clusters)]        
        self.cluster_names = {i: f'Cluster {i}' for i in range(self.num_clusters)}
        
        
    def create_visualization(self, dir_out="ztmp/", mode='d3', cols_label=None, show_server=False,  **kw ):
        """

        """
        os.makedirs(dir_out, exist_ok=True)
        cols_label          = [] if cols_label is None else cols_label 
        text_label_and_text = []
        for i,x in self.df_labels.iterrows():
          ss = x["id"]  
          for ci in cols_label:  
             ss = ss + ":" + x[ci]
          text_label_and_text.append(ss) 

        #######################################################################################
        # create data frame that has the result of the MDS plus the cluster numbers and titles
        df = pd.DataFrame(dict(x= self.pos[:, 0], 
                               y= self.pos[:, 1], 
                               clusters= self.clusters, title=text_label_and_text))
        df.to_parquet(f"{dir_out}/embs_xy_cluster.parquet")


        # group by cluster
        groups_clusters = df.groupby('clusters')

        # set up plot
        fig, ax = plt.subplots(figsize=(25, 15))  # set size
        ax.margins(0.05)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        # note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return
        # the appropriate color/label
        for name, group in groups_clusters:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=12, label= self.cluster_names[name],
                    color=self.cluster_color[name],
                    mec='none')
            ax.set_aspect('auto')
            ax.tick_params(axis='x',  # changes apply to the x-axis
                           which='both',  # both major and minor ticks are affected
                           bottom='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelbottom='off')
            ax.tick_params(axis='y',  # changes apply to the y-axis
                           which='both',  # both major and minor ticks are affected
                           left='off',  # ticks along the bottom edge are off
                           top='off',  # ticks along the top edge are off
                           labelleft='off')

        ax.legend(numpoints=1)  # show legend with only 1 point

        # add label in x,y position with the label as the
        for i in range(len(df)):
            ax.text(df.loc[i]['x'], df.loc[i]['y'], df.loc[i]['title'], size=8)

        # uncomment the below to save the plot if need be
        plt.savefig(f'{dir_out}/clusters_static-{datetime.now().strftime("%Y-%m-%d %H-%M-%S")}.png', dpi=200)

        # Plot
        fig, ax = plt.subplots(figsize=(20, 15))  # set plot size
        ax.margins(0.03)  # Optional, just adds 5% padding to the autoscaling

        # iterate through groups to layer the plot
        for name, group in groups_clusters:
            points = ax.plot(group.x, group.y, marker='o', linestyle='', ms=7, label= self.cluster_names[name], mec='none',
                             color=self.cluster_color[name])
            ax.set_aspect('auto')
            labels = [i for i in group.title]

            # set tooltip using points, labels and the already defined 'css'
            tooltip = mpld3.plugins.PointHTMLTooltip(points[0], labels, voffset=10, hoffset=10, css=CSS)
            # connect tooltip to fig
            mpld3.plugins.connect(fig, tooltip, TopToolbar())

            # set tick marks as blank
            ax.axes.get_xaxis().set_ticks([])
            ax.axes.get_yaxis().set_ticks([])

            # set axis as blank
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)

        ax.legend(numpoints=1)  # show legend with only one dot


        ##### Export ############################################################
        mpld3.save_html(fig,  f"{dir_out}/embeds.html")
        log('Visualization',    f"{dir_out}/embeds.html" )

        ### Windows specifc
        if os.name == 'nt': os.system(f'start chrome "{dir_out}/embeds.html" ')


        if show_server :
           # mpld3.show(fig=None, ip='127.0.0.1', port=8888, n_retries=50, local=True, open_browser=True, http_server=None, **kwargs)[source] 
           mpld3.show()  # show the plot



    def draw_hiearchy(self):
        linkage_matrix = ward(self.dist)  # define the linkage_matrix using ward clustering pre-computed distances
        fig, ax = plt.subplots(figsize=(15, 20))  # set size
        ax = dendrogram(linkage_matrix, orientation="right", labels=self.text_labels)
        plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        plt.tight_layout()
        plt.savefig('dendogram_clusters.png', dpi=200)


           
#########################################################################################################
def embedding_to_parquet(dirin=None, dirout=None, skip=0, nmax=10**8, 
                         is_linevalid_fun=None):   ##   python emb.py   embedding_to_parquet  &
    #### FastText/ Word2Vec to parquet files    9808032 for purhase
    log(dirout) ; os_makedirs(dirout)  ; time.sleep(4)

    if is_linevalid_fun is None : #### Validate line
        def is_linevalid_fun(w):
            return len(w)> 5  ### not too small tag

    i = 0; kk=-1; words =[]; embs= []; ntot=0
    with open(dirin, mode='r') as fp:
        while i < nmax+1  :
            i  = i + 1
            ss = fp.readline()
            if not ss  : break
            if i < skip: continue

            ss = ss.strip().split(" ")            
            if not is_linevalid_fun(ss[0]): continue

            words.append(ss[0])
            embs.append( ",".join(ss[1:]) )

            if i % 200000 == 0 :
              kk = kk + 1                
              df = pd.DataFrame({ 'id' : words, 'emb' : embs }  )  
              log(df.shape, ntot)  
              if i < 2: log(df)  
              pd_to_file(df, dirout + f"/df_emb_{kk}.parquet", show=0)
              ntot += len(df)
              words, embs = [], []  

    kk      = kk + 1                
    df      = pd.DataFrame({ 'id' : words, 'emb' : embs }  )  
    ntot   += len(df)
    dirout2 = dirout + f"/df_emb_{kk}.parquet"
    pd_to_file(df, dirout2, show=1 )
    log('ntotal', ntot, dirout2 )
    return os.path.dirname(dirout2)


def embedding_load_parquet(dirin="df.parquet", nmax = 500):
    """  id, emb (string , separated)
    
    """
    log('loading', dirin)     
    col_embed = 'pred_emb'
    colid     = 'id'
    nmax    = nmax if nmax > 0 else  len(df)   ### 5000
    
    flist = list( glob.glob(dirin) )
    
    df  = pd_read_file( flist, npool= max(1, int( len(flist) / 4) ) )
    df  = df.iloc[:nmax, :]
    df  = df.rename(columns={ col_embed: 'emb'})
    
    df  = df[ df['emb'].apply( lambda x: len(x)> 10  ) ]  ### Filter small vector
    log(df.head(5).T, df.columns, df.shape)
    log(df, df.dtypes)    


    ###########################################################################
    ###### Split embed numpy array, id_map list,  #############################
    embs    = np_str_to_array(df['emb'].values,  l2_norm=True,     mdim = 200)
    id_map  = { name: i for i,name in enumerate(df[colid].values) }     
    log(",", str(embs)[:50], ",", str(id_map)[:50] )
    
    #####  Keep only label infos  ####
    del df['emb']                  
    return embs, id_map, df 


def np_str_to_array(vv,  l2_norm=True,     mdim = 200):
    ### Extract list of string into numpy
    #mdim = len(vv[0].split(","))
    # mdim = 200
    from sklearn import preprocessing
    import faiss
    X = np.zeros(( len(vv) , mdim  ), dtype='float32')
    for i, r in enumerate(vv) :
        try :
          vi      = [ float(v) for v in r.split(',')]        
          X[i, :] = vi
        except Exception as e:
          log(i, e)

    if l2_norm:
       # preprocessing.normalize(X, norm='l2', copy=False)
       faiss.normalize_L2(X)  ### Inplace L2 normalization
    log("Normalized X")        
    return X
    
    
def viz_run(dirin="in/model.vec", dirout="ztmp/", nmax=100):
   ###   python emb.py run    &  
   nmax    =  500000
   mode    =  'umap'
   tag     = f"{nmax}_{mode}"

   #### Generate HTML  ############################################ 
   log(dirin)
   myviz = vizEmbedding(path = dirin )
   myviz.run_all(nmax=nmax, dir_out= dirout, mode=mode, ntest=50000)

         
        

    
def topk(topk=100, dname=None, pattern="df_*1000*.parquet", filter1=None):
    """  python emb.py  topk    |& tee -a  /zzlog.py
    
    """
    from utilmy import pd_read_file
    if dname is None :
       dname = "seq_100000000"
    
    ksample = 500
         
    ###################################################################
    dname    = dname.replace("/", "_").replace(".", "-")    
    in_dir   = r0 + dname
    out_dir  = in_dir + "/topk/"
    os.makedirs(out_dir, exist_ok=True)
    log(in_dir)
    
    #### Load emb data  ###############################################
    df        = pd_read_file(  in_dir + f"/{pattern}", n_pool=10 )
    df.index = np.arange(0, len(df))
    log(df)
    # df['emb'] = df['emb'].apply(lambda x :  list( np.array(x) /np.sqrt(np.dot(x,x)) ) )   ###Norm Vector

        
    #### Element X0 ####################################################
    llids   = list(df.sample(frac=1.0)['id'].values)
    vectors =  np_str_to_array(df['emb'].values,  mdim=200)   
    
    # faiss_create_index(df_or_path=None, col='emb', dir_out="",  db_type = "IVF4096,Flat", nfile=1000, emb_dim=200)
    
    for ii,idr in enumerate(llids) :        
        if ii >= ksample : break
        dfi     = df[ df['id'] == idr ] 
        if len(dfi) < 1: continue
        x0      = np.array(dfi['emb'].values[0]).astype(np.float32)
        xname   = dfi['id'].values[0]
        log(xname)

        ##### Setup Faiss queey ########################################
        x0      = x0.reshape(1, -1).astype('float32')  
        # log(x0.shape, vectors.shape)
        dist, rank = topk_nearest_vector(x0, vectors, topk= topk) 
        df1              = df.iloc[rank[0], :]
        df1['topk_dist'] = dist[0]
        df1['topk_rank'] = np.arange(0, len(df1))
        log( df1 )
        del df1['emb']
        df1.to_csv( out_dir + f"/topk_{xname}_{filter1}.csv"  , sep=",")


        
        
def topk_nearest_vector(x0, vector_list, topk=3) :
   """ Retrieve top k nearest vectors using FAISS
   """
   import faiss  
   index = faiss.index_factory(x0.shape[1], 'Flat')
   index.add(vector_list)
   dist, indice = index.search(x0, topk)
   return dist, indice
   
        

 
def sim_score2(path=""):
    """
       Sim Score using FAISS
    
    """
    import faiss
    x0 = [ 0.1, .2, 0.3]
    x  = np.array([x0]).astype(np.float32)

    index = faiss.index_factory(3, "Flat", faiss.METRIC_INNER_PRODUCT)
    log(index.ntotal)
    faiss.normalize_L2(x)
    index.add(x)
    distance, index = index.search(x, 5)
    log(f'Distance by FAISS:{distance}')
    return distance, index
    
    #To Tally the results check the cosine similarity of the following example
    #from scipy import spatial
    #result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
    #print('Distance by FAISS:{}'.format(result))

    


###################################################################################################
if __name__ == "__main__":
    import fire
    fire.Fire()



    

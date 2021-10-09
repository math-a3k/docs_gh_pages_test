import numpy as np, sys,os, copy, random, pandas as pd, json

from scipy.stats import bernoulli
from random import sample
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import utils.active_learning as active_learning
# tfd = tf.contrib.distributions
import seaborn as sns; sns.set(style="ticks", color_codes=True)


### Need TF 1.4    pip install tensorflow=1.14
import tensorflow as tf
print(tf.__version__)

#sys.path.append( os.path.dirname(os.path.abspath(__file__)) + "/repo/VAEM/" )

import utils.process as process
import utils.params as params
import utils.active_learning as active_learning
import models.decoders as decoders
import models.encoders as encoders
import models.model as vae_model
import utils.reward as reward

#######################################################################################
verbosity =2

def log(*s):
    print(*s, flush=True)

def log2(*s):
    if verbosity >= 2 :
      print(*s, flush=True)


####################################################################################################
global model, session

def init(*kw, **kwargs):
    global model, session
    model = Model(*kw, **kwargs)
    session = None

# Making Args Global Variable 
args = params.Params('hyperparameters/bank_plot.json')

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)  
rs = 42 # random seed
fast_plot = 0

####################################################################################################
##### Custom code  #################################################################################
cols_ref_formodel = ['none']  ### No column group

########## Loading Of Data #########################################################################
def load_data(filePath,categories,cat_col,num_cols,discrete_cols,targetCol,nsample,delimiter):
    global Data_decompressed, Mask_decompressed
    '''
        Data will be loaded from repo/VAEM/data/bank and then preprocessing starts
    '''

    
    seed = 3000
    dataframe = pd.read_csv(filePath,delimiter=delimiter)
    if nsample != -1:
        dataframe = dataframe.iloc[:nsample,:]
    print(f'dataframe Shape: {dataframe.shape}')
    print(dataframe.info())
    label_column=targetCol
    matrix1 = dataframe.copy()
    #process.encode_catrtogrial_column(dataframe, [targetCol])
    #matrix1_cat = dataframe.iloc[:,cat_col]
    #matrix1_num = dataframe.iloc[:,num_cols]
    #cols = dataframe.columns
    for column in categories:
        process.encode_catrtogrial_column(matrix1, [column])
    
    Data = ((matrix1.values).astype(float))[0:,:]

    # the data will be mapped to interval [min_Data,max_Data]. Usually this will be [0,1] but you can also specify other values.
    max_Data = 0.7 
    min_Data = 0.3 
    # list of categorical variables
    list_cat = np.array(cat_col)
    # list of numerical variables
    list_flt = np.array(num_cols)
    # among numerical variables, which ones are discrete. This is referred as continuous-discrete variables in Appendix C.1.3 in our paper.
    # Examples include variables that take integer values, for example month, day of week, number of custumors etc. Other examples include numerical variables that are recorded on a discrete grid (for example salary). 
    list_discrete = np.array(discrete_cols)

    # sort the variables in the data matrix, so that categorical variables appears first. The resulting data matrix is Data_sub
    list_discrete_in_flt = (np.in1d(list_flt, list_discrete).nonzero()[0])
    list_discrete_compressed = list_discrete_in_flt + len(list_cat)

    if len(list_flt)>0 and len(list_cat)>0:
        list_var = np.concatenate((list_cat,list_flt))
    elif len(list_flt)>0:
        list_var = list_flt
    else:
        list_var = list_cat
    Data_sub = Data[:,list_var]
    dic_var_type = np.zeros(Data_sub.shape[1])
    dic_var_type[0:len(list_cat)] = 1

    # In this notebook we assume the raw data matrix is fully observed
    Mask = np.ones(Data_sub.shape)
    # Normalize/squash the data matrix
    Data_std = (Data_sub - Data_sub.min(axis=0)) / (Data_sub.max(axis=0) - Data_sub.min(axis=0))
    scaling_factor = (Data_sub.max(axis=0) - Data_sub.min(axis=0))/(max_Data - min_Data)
    Data_sub = Data_std * (max_Data - min_Data) + min_Data

    # decompress categorical data into one hot representation
    
    Data_cat = Data[:,list_cat].copy()
    Data_flt = Data[:,list_flt].copy()
    Data_compressed = np.concatenate((Data_cat,Data_flt),axis = 1)
    Data_decompressed, Mask_decompressed, cat_dims, DIM_FLT = process.data_preprocess(Data_sub,Mask,dic_var_type)
    Data_train_decompressed, Data_test_decompressed, mask_train_decompressed, mask_test_decompressed,mask_train_compressed, mask_test_compressed,Data_train_compressed, Data_test_compressed = train_test_split(
            Data_decompressed, Mask_decompressed,Mask,Data_compressed,test_size=0.1, random_state=rs)

    list_discrete = list_discrete_in_flt + (cat_dims.sum()).astype(int)

    Data_decompressed = np.concatenate((Data_train_decompressed, Data_test_decompressed), axis=0)
    Data_train_orig = Data_train_decompressed.copy()
    Data_test_orig = Data_test_decompressed.copy()

    # Note that here we have added some noise to continuous-discrete variables to help training. Alternatively, you can also disable this by changing the noise ratio to 0.
    Data_noisy_decompressed,records_d, intervals_d = process.noisy_transform(Data_decompressed, list_discrete, noise_ratio = 0.99)
    noise_record = Data_noisy_decompressed - Data_decompressed
    Data_train_noisy_decompressed = Data_noisy_decompressed[0:Data_train_decompressed.shape[0],:]
    Data_test_noisy_decompressed = Data_noisy_decompressed[Data_train_decompressed.shape[0]:,:]

    data_decode = (Data_train_decompressed,Data_train_compressed, Data_train_noisy_decompressed,mask_train_decompressed,Data_test_decompressed,mask_test_compressed,mask_test_decompressed,cat_dims,DIM_FLT,dic_var_type)

    return data_decode, records_d


def encode2(data_decode,list_discrete,records_d,fast_plot):
    # Extracting Masked Decompressed Data from data_decode function obtained from load_data function
    Data_train_decomp,Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode
    
    vae = p_vae_active_learning(Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type,args,list_discrete,records_d,estimation_method=1)

    tf.reset_default_graph()
    x_recon,z_posterior,x_recon_cat_p = vae.get_imputation( Data_train_noisy_decomp, mask_train_decomp*0,cat_dims,dic_var_type) ## one hot already converted to integer
    x_real = process.compress_data(Data_train_decomp,cat_dims, dic_var_type) ## x_real still needs conversion
    x_real_cat_p = Data_train_decomp[:,0:(cat_dims.sum()).astype(int)]
    
    max_Data = 0.7
    min_Data = 0.3  
    Data_std = (x_real - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    scaling_factor = (x_real.max(axis=0) - x_real.min(axis=0))/(max_Data - min_Data)
    Data_real = Data_std * (max_Data - min_Data) + min_Data

    fast_plot = 1

    sub_id = [1,2,10]

    if fast_plot:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        Data_real = pd.DataFrame(Data_real[:,sub_id])
        g = sns.pairplot(Data_real.sample(min(10000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    
    
    Data_fake_noisy= x_recon
    Data_fake = process.invert_noise(Data_fake_noisy,list_discrete,records_d) 

    Data_std = (Data_fake - x_real.min(axis=0)) / (x_real.max(axis=0) - x_real.min(axis=0))
    Data_fake = Data_std * (max_Data - min_Data) + min_Data


    sub_id = [1,2,10]

    if fast_plot:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))
    else:
        g = sns.pairplot(pd.DataFrame(Data_fake[:,sub_id]).sample(min(1000,x_real.shape[0])),diag_kind = 'kde')
        g = g.map_diag(sns.distplot, bins = 50,norm_hist = True)
        g = g.map_upper(plt.scatter,marker='+')
        g = g.map_lower(sns.kdeplot, cmap="hot",shade=True,bw=.1)
        g.set(xlim=(min_Data,max_Data), ylim = (min_Data,max_Data))

    return vae,scaling_factor


def decode2(data_decode,scaling_factor,list_discrete,records_d,plot=False):
    args = params.Params('hyperparameters/bank_SAIA.json')

    Data_train_decomp,Data_train_comp, Data_train_noisy_decomp,mask_train_decomp,Data_test_decomp,mask_test_comp,mask_test_decomp,cat_dims,DIM_FLT,dic_var_type = data_decode
    print('Decode Training Start')
    vae = p_vae_active_learning(Data_train_comp,
                                                Data_train_noisy_decomp,
                                                mask_train_decomp,
                                                Data_test_decomp,
                                                mask_test_comp,
                                                mask_test_decomp,
                                                cat_dims,DIM_FLT,dic_var_type,args,list_discrete,records_d)


    npzfile = np.load(args.output_dir+'/UCI_rmse_curve_SING.npz')
    IC_SING=npzfile['information_curve']*scaling_factor[-1]

    if plot:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
        plt.figure(0)
        L = IC_SING.shape[1]
        fig, ax1 = plt.subplots()
        # These are in unitless percentages of the figure size. (0,0 is bottom left)
        left, bottom, width, height = [0.45, 0.4, 0.45, 0.45]
        ax1.plot(np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0),'ys',linestyle = '-.', label = 'dataset')
        ax1.errorbar(np.arange(IC_SING.shape[2]),np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).mean(axis=0), yerr=np.sqrt((IC_SING[:,:,0:]**2).mean(axis=1)).std(axis = 0)/np.sqrt(IC_SING.shape[0]),ecolor='y',fmt = 'ys')
        plt.xlabel('Steps',fontsize=18)
        plt.ylabel('avg. test. RMSE',fontsize=18)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax1.legend(bbox_to_anchor=(0.0, 1.02, 1., .102), mode = "expand", loc=3,
                ncol=1, borderaxespad=0.,prop={'size': 20}, frameon=False)
        ax1.ticklabel_format(useOffset=False)
        plt.show()

    return vae

def save_model2(model,output_dir):
    model.save(output_dir)

def p_vae_active_learning(Data_train_compressed, Data_train,mask_train,Data_test,mask_test_compressed,mask_test,cat_dims,dim_flt,dic_var_type,args,list_discrete,records_d, estimation_method=1):
    global mask
    list_stage = args.list_stage
    list_strategy = args.list_strategy
    epochs = args.epochs
    #epochs = 1
    latent_dim = args.latent_dim
    batch_size = args.batch_size
    p = args.p
    K = args.K
    M = args.M
    Repeat = args.repeat
    iteration = args. iteration
    
    '''
    This function train or loads a VAEM model, and performs SAIA using SING or full EDDI strategy.
    Note that we assume that the last column of x is the target variable of interest
    :param Data_train_compressed: preprocessed traning data matrix without one-hot encodings. Note that we assume that the columns of the data matrix is re-ordered, so that the categorical variables appears first, and then the continuous variables afterwards.
    :param Data_train: preprocessed traning data matrix with one-hot encodings. Is is re-ordered as Data_train_compressed.
    :param mask_train: mask matrix that indicates the missingness of training data,with one-hot encodings. 1=observed, 0 = missing
    :param Data_test: test data matrix, with one-hot encodings.
    :param mask_test: mask matrix that indicates the missingness of test data, with one-hot encodings.. 1=observed, 0 = missing
    :param mask_test_compressed: mask matrix that indicates the missingness of test data, without one-hot encodings. 1=observed, 0 = missing.
    :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
    :param dim_flt: number of continuous variables.
    :param dic_var_type: a list that contains the statistical types for each variables
    :param args.epochs: number of epochs for training.
    :param args.latent_dim: latent dimension of partial VAE.
    :param args.K: dimension of feature map of PNP encoder
    :param args.M: number of samples used for MC sampling
    :param args.Repeat: number of repeats.
    :param estimation_method: what method to use for single ordering information reward estimation.
            In order to calculate the single best ordering, we need to somehow marginalize (average) the
            information reward over the data set (in this case, the test set).
            we provide two methods of marginalization.
            - estimation_method = 0: information reward marginalized using the model distribution p_{vae_model}(x_o).
            - estimation_method = 1: information reward marginalized using the data distribution p_{data}(x_o)
    :param args.list_stage: a list of stages that you wish to perform training. 1 = train marginal VAEs, 2 = train dependency VAEs, 3 = fine-tune, 4 = load a model without training.
    :param args.list_strategy: a list of strategies that is used for SAIA. 0 = Random, 1 = SING
    
    :return: None (active learning results are saved to args.output_dir)
    '''
    n_test = Data_test.shape[0]
    n_train = Data_train.shape[0]
    dim_cat = len(np.argwhere(cat_dims != -1))
    OBS_DIM = dim_cat + dim_flt
    al_steps = OBS_DIM
    # information curves
    information_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    information_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_RAND = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    rmse_curve_SING = np.zeros(
        (Repeat, n_test, al_steps - 1 + 1))
    # history of optimal actions
    action_SING = np.zeros((Repeat, n_test,
                            al_steps - 1))
    # history of information reward values
    R_hist_SING = np.zeros(
        (Repeat, al_steps - 1, n_test,
         OBS_DIM - 1))

    for r in range(Repeat):
        ## train partial VAE
        reward_estimation = reward.lindley(M, cat_dims, list_discrete, dic_var_type, records_d,)
        tf.reset_default_graph()
        for stag in range(len(list_stage)):
            stage = list_stage[stag]
            vae = train_p_vae(stage, Data_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration,list_discrete,records_d,args)  

        ## Perform active variable selection
        if len(list_strategy)>0:
            for strat in range(len(list_strategy)):
                strategy = list_strategy[strat]
                if strategy == 0:### random strategy
                    ## create arrays to store data and missingness
                    x = Data_test[:, :]  #
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM))
                    mask[:, -1] = 0  # we will never observe target value

                    ## initialize array that stores optimal actions (i_optimal)
                    i_optimal = [ nums for nums in range(OBS_DIM - 1 ) ]
                    i_optimal = np.tile(i_optimal, [n_test, 1])
                    random.shuffle([random.shuffle(c) for c in i_optimal])

                    ## evaluate likelihood and rmse at initial stage (no observation)
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_RAND[r, :, 0] = negative_predictive_llh
                    rmse_curve_RAND[r, :, 0] = predictive_rmse
                    for t in range(al_steps - 1 ):
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        io = np.eye(OBS_DIM)[i_optimal[:, t]]
                        mask = mask + io
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask, cat_dims, dic_var_type,M)
                        information_curve_RAND[r, :, t +1] = negative_predictive_llh
                        rmse_curve_RAND[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_RAND.npz'),
                             information_curve=information_curve_RAND)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_RAND.npz'),
                             information_curve=rmse_curve_RAND)

                if strategy == 1:### single ordering strategy
                    im_SING = np.zeros((Repeat, al_steps - 1 , M,
                                    n_test, Data_train.shape[1] ))
                    #SING is obtrained by maximize mean information reward for each step for the test set to be consistant with the description in the paper.
                    #We can also get this order by using a subset of training set to obtain the optimal ordering and apply this to the testset.
                    x = Data_test[:, :]  #
#                     x,_, _ = noisy_transform(x, list_discrete, noise_ratio)
    #                 x = np.reshape(x, [n_test, OBS_DIM])
                    mask = np.zeros((n_test, OBS_DIM)) # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
                    mask2 = np.zeros((n_test, OBS_DIM)) # this stores the mask indicating that which features has been selected of each data
                    mask[:, -1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
                    negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                        x, mask,cat_dims, dic_var_type, M)
                    information_curve_SING[r, :, 0] = negative_predictive_llh
                    rmse_curve_SING[r, :, 0] = predictive_rmse

                    for t in range(al_steps - 1 ): # t is a indicator of step
                        print("Repeat = {:.1f}".format(r))
                        print("Strategy = {:.1f}".format(strategy))
                        print("Step = {:.1f}".format(t))
                        ## note that for single ordering, there are two rewards.
                        # The first one (R) is calculated based on no observations.
                        # This is used for active learning phase, since single ordering should not depend on observations.
                        # The second one (R_eval) is calculated in the same way as chain rule approximation. This is only used for visualization.
                        if t ==-1:
                            im_0 = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            im = Data_train_compressed.reshape((1,Data_train_compressed.shape[0],-1))
                            R = -1e40 * np.ones((Data_train_compressed.shape[0], OBS_DIM - 1))
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                        else:
                            R = -1e40 * np.ones((n_test, OBS_DIM - 1))
                            im_0 = reward_estimation.completion(x, mask*0,vae) # sample from model prior
                            im = reward_estimation.completion(x, mask, vae) # sample conditional on observations
                            im_SING[r, t, :, :, :] = im
                            for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
                                loc = np.where(mask2[:, u] == 0)[0]
                                if estimation_method == 0:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask,  vae, im_0,loc)
                                else:
                                    R[loc, u] = reward_estimation.R_lindley_chain(u, x, mask, vae, im, loc)
                            R_hist_SING[r, t, :, :] = R
                        
                        i_optimal = (R.mean(axis=0)).argmax() # optimal decision based on reward averaged on all data
                        i_optimal = np.tile(i_optimal, [n_test])
                        io = np.eye(OBS_DIM)[i_optimal]
                        action_SING[r, :, t] = i_optimal
                        mask = mask + io*mask_test_compressed # this mask takes into account both data missingness and missingness of unselected features
                        negative_predictive_llh, predictive_rmse = vae.predictive_loss(
                            x, mask,cat_dims, dic_var_type, M)
                        mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
                        information_curve_SING[r, :, t +
                                               1] = negative_predictive_llh
                        rmse_curve_SING[r, :, t+1] = predictive_rmse
                    np.savez(os.path.join(args.output_dir, 'UCI_information_curve_SING.npz'),
                             information_curve=information_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_rmse_curve_SING.npz'),
                             information_curve=rmse_curve_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_action_SING.npz'), action=action_SING)
                    np.savez(os.path.join(args.output_dir, 'UCI_R_hist_SING.npz'), R_hist=R_hist_SING)

#             Save results
    return vae



def train_p_vae(stage, x_train, Data_train,mask_train, epochs, latent_dim,cat_dims,dim_flt,batch_size, p, K,iteration,list_discrete,records_d,args):
    global obs_dim
    '''
        This function trains the partial VAE.
        :param stage: stage of training 
        :param x_train: initial inducing points
        :param Data_train: training Data matrix, N by D
        :param mask_train: mask matrix that indicates the missingness. 1=observed, 0 = missing
        :param epochs: number of epochs of training
        :param LATENT_DIM: latent dimension for partial VAE model
        :param cat_dims: a list that indicates the number of potential outcomes for non-continuous variables.
        :param dim_flt: number of continuous variables.
        :param batch_size: batch_size.
        :param p: dropout rate for creating additional missingness during training
        :param K: dimension of feature map of PNP encoder
        :param iteration: how many mini-batches are used each epoch. set to -1 to run the full epoch.
        :return: trained VAE, together with the test data used for testing.
        '''
    # we have three stages of training.
    # stage 1 = training marginal VAEs, stage 2 = training dependency network, (see Section 2 in our paper)
    # stage 3 = add predictor and improve predictive performance (See Appendix C in our paper)
    if stage ==1:
        load_model = 0
        disc_mode = 'non_disc'
    elif stage == 2:
        load_model = 1
        disc_mode = 'non_local'
    elif stage == 3:
        load_model = 1
        disc_mode = 'joint'
    obs_dim = Data_train.shape[1]
    n_train = Data_train.shape[0]
    list_train = np.arange(n_train)
    batch_size = np.minimum(batch_size,n_train)
    ####### construct
    kwargs = {
        'stage':stage,
        'K': K,
        'latent_dim': latent_dim,
        'batch_size': batch_size,
        'encoder': encoders.vaem_encoders(obs_dim,cat_dims,dim_flt,K, latent_dim ),
        'decoder': decoders.vaem_decoders(obs_dim,cat_dims,list_discrete,records_d),
        'obs_dim': obs_dim,
        'cat_dims': cat_dims,
        'dim_flt': dim_flt,
        'load_model':load_model,
        'decoder_path': os.path.join(args.output_dir, 'generator.tensorflow'),
        'encoder_path': os.path.join(args.output_dir, 'encoder.tensorflow'),
        'x_train':x_train,
        'list_discrete':list_discrete,
        'records_d':records_d,
    }
    vae = vae_model.partial_vaem(**kwargs)
    if iteration == -1:
        n_it = int(np.ceil(n_train / kwargs['batch_size']))
    else:
        n_it = iteration
    hist_loss_full = np.zeros(epochs)
    hist_loss_cat = np.zeros(epochs)
    hist_loss_flt = np.zeros(epochs)
    hist_loss_z_local = np.zeros(epochs)
    hist_loss_kl = np.zeros(epochs)

    if stage == 3:
        ## after stage 2, do discriminative training of y. Please refer to Section 3.4 and Appendix C in our paper.
        for epoch in range(epochs):
            train_loss_full = 0.
            train_loss_cat = 0.
            train_loss_flt = 0.
            train_loss_z_local = 0.
            for it in range(n_it):
                if iteration == -1:
                    batch_indices = list_train[it * kwargs['batch_size']:min(it * kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
                else:
                    batch_indices = sample(range(n_train), kwargs['batch_size'])
                x = Data_train[batch_indices, :]
                mask_train_batch = mask_train[batch_indices, :]
                DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
                while True:
                    mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                    if np.sum(mask_drop > 0):
                        break

                mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
                _ = vae.update(x, mask_drop * mask_train_batch, 'disc')
                loss_full, loss_cat, loss_flt, loss_z_local, loss_kl, stds, _, _ = vae.full_batch_loss(x,mask_drop * mask_train_batch)
                train_loss_full += loss_full
                train_loss_cat += loss_cat
                train_loss_flt += loss_flt
                train_loss_z_local += loss_z_local

            # average loss over most recent epoch
            train_loss_full /= n_it
            train_loss_cat /= n_it
            train_loss_flt /= n_it
            train_loss_z_local /= n_it
            print(
                'Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
                    .format(epoch, train_loss_full, train_loss_cat, train_loss_flt, train_loss_z_local))

    for epoch in range(epochs):
        train_loss_full = 0. # full training loss
        train_loss_cat = 0. # reconstruction loss for non_continuous likelihood term
        train_loss_flt = 0. # reconstruction loss for continuous likelihood term
        train_loss_z_local = 0. # reconstruction loss for second stage on z space (gaussian likelihood)
        train_loss_kl = 0 # loss for KL term

        for it in range(n_it):
            if iteration == -1:
                batch_indices = list_train[it*kwargs['batch_size']:min(it*kwargs['batch_size'] + kwargs['batch_size'], n_train - 1)]
            else:
                batch_indices = sample(range(n_train), kwargs['batch_size'])

            x = Data_train[batch_indices, :]
            mask_train_batch = mask_train[batch_indices, :]
            DROPOUT_TRAIN = np.minimum(np.random.rand(mask_train_batch.shape[0], obs_dim), p)
            while True:
                mask_drop = bernoulli.rvs(1 - DROPOUT_TRAIN)
                if np.sum(mask_drop > 0):
                    break

            mask_drop = mask_drop.reshape([kwargs['batch_size'], obs_dim])
            _ = vae.update(x, mask_drop*mask_train_batch,disc_mode)
            loss_full, loss_cat, loss_flt,loss_z_local, loss_kl, stds, _,_ = vae.full_batch_loss(x,mask_drop*mask_train_batch)
            train_loss_full += loss_full
            train_loss_cat += loss_cat
            train_loss_flt += loss_flt
            train_loss_z_local += loss_z_local
            train_loss_kl += loss_kl

          # average loss over most recent epoch
        train_loss_full /= n_it
        train_loss_cat /= n_it
        train_loss_flt /= n_it
        train_loss_z_local /= n_it
        train_loss_kl /= n_it
        hist_loss_full[epoch] = train_loss_full
        hist_loss_cat[epoch] = train_loss_cat
        hist_loss_flt[epoch] = train_loss_flt
        hist_loss_z_local[epoch] = train_loss_z_local
        hist_loss_kl[epoch] = train_loss_kl

        print('Epoch: {} \tnegative training ELBO per observed feature: {:.2f}, Cat_term: {:.2f}, Flt_term: {:.2f},z_term: {:.2f}'
            .format(epoch, train_loss_full,train_loss_cat,train_loss_flt, train_loss_z_local))

    if stage <= 2:
        vae.save_generator(os.path.join(args.output_dir, 'generator.tensorflow'))
        vae.save_encoder(os.path.join(args.output_dir, 'encoder.tensorflow'))

    return vae


##############################################################################################################################################################################################################

##################################################################################################################################
def reset():
    global model, session
    model, session = None, None

def save(path='', info=None):
    import dill as pickle, copy
    global model, session
    os.makedirs(path, exist_ok=True)

    model.model.save(f"{path}/model_keras.h5")
    model.model.save_weights(f"{path}/model_keras_weights.h5")

    modelx = Model()  # Empty model  Issue with pickle
    modelx.model_pars   = model.model_pars
    modelx.data_pars    = model.data_pars
    modelx.compute_pars = model.compute_pars
    # log('model', modelx.model)
    pickle.dump(modelx, open(f"{path}/model.pkl", mode='wb'))  #

    pickle.dump(info, open(f"{path}/info.pkl", mode='wb'))  #


def load_model(path=""):
    global model, session
    import dill as pickle

    model0      = pickle.load(open(f"{path}/model.pkl", mode='rb'))

    model = Model()  # Empty model
    model.model        = get_model( model0.model_pars)
    model.model_pars   = model0.model_pars
    model.compute_pars = model0.compute_pars

    model.model.load_weights( f'{path}/model_keras_weights.h5')

    log(model.model.summary())
    #### Issue when loading model due to custom weights, losses, Keras erro
    #model_keras = get_model()
    #model_keras = keras.models.load_model(path + '/model_keras.h5' )
    session = None
    return model, session


def load_info(path=""):
    import cloudpickle as pickle, glob
    dd = {}
    for fp in glob.glob(f"{path}/*.pkl"):
        if not "model.pkl" in fp:
            obj = pickle.load(open(fp, mode='rb'))
            key = fp.split("/")[-1]
            dd[key] = obj
    return dd
################################################################################################################################
import pyarrow as pa
import pyarrow.parquet as pq
class Model(object):
    def __init__(self):
        
        ### Dynamic Dimension : data_pars  ---> model_pars dimension  ###############
        print('Model Instantiated')
        print('Next Step: Fit')

        #log2(self.model_pars, self.model)
        #self.model.summary()

    def fit(self,filePath, categories,cat_cols,num_cols,discrete_cols,targetCol,nsample = -1,delimiter=',',plot=False):
        """
        This Function will load the data and fit the preprocessing technique into it
        params:
            filePath: CSV File path relative to file
            categories: Categorical feature list
            cat_cols: category_columns
            nums_cols: Numerical Columns list
            discrete_cols: discrete cols list
            target_cols: Column name of Target Variable
            nsample: No of Sample for Encoding-Decoding
            delimiter: Delimiter in CSV File,Default=','

        """
        global model, session
        session = None  # Session type for compute
        self.filePath = filePath
        self.categories = categories
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.discrete_cols = discrete_cols
        self.targetCol = targetCol
        self.nsample = nsample
        self.delimiter = delimiter 
        data_decode,records_d = load_data(self.filePath,self.categories,self.cat_cols,self.num_cols,self.discrete_cols,self.targetCol,self.nsample,self.delimiter)
        self.data_decode = data_decode
        self.records_d = records_d
        self.list_discrete = discrete_cols
        #model.model['encode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)
        #model.model['decode'].fit([Xtrain_tuple, Xtrain_dummy],  **cpars)
        model1,scaling_factor = encode2(self.data_decode,self.list_discrete,self.records_d,plot)        
        self.encoder_model = model1
        self.scaling_factor = scaling_factor
        model2 = decode2(self.data_decode, self.scaling_factor,self.list_discrete,self.records_d,plot=plot)
        self.decoder_model = model2


    def decode(self):
        global Data_decompressed,Mask_decompressed
        '''
        This function will be using encode2 function defined above for encoding data.
        This Model will create a parquet file of encoded columns

        '''
        decoded_array = self.decoder_model.im(Data_decompressed,Mask_decompressed)
        parquetDic = {}
        for i in range(decoded_array.shape[1]):
            name = f'col_{i+1}'
            parquetDic[name] = decoded_array[:,i]
        print(f'Decoder Columns Shape {decoded_array.shape}')
        log2(decoded_array)
        ndarray_table = pa.table(parquetDic)
        pq.write_table(ndarray_table,'my_decoder.parquet')
        print('File my_decoder.parquet created')
        

    def encode(self):
        global Data_decompressed
        '''
        This function will be using decoder2 function defined above for decoding data.
        This Model will create a parquet file for decoded columns
        
        '''
        
        encoded_array = self.encoder_model.generateEncodings(Data_decompressed,Mask_decompressed)
        parquetDic = {}
        for i in range(encoded_array.shape[1]):
            name = f'col_{i+1}'
            parquetDic[name] = encoded_array[:,i]
        print(f'Encoder Columns shape: {encoded_array.shape}')
        log2(encoded_array)
        ndarray_table = pa.table(parquetDic)
        pq.write_table(ndarray_table,'my_encoder.parquet')
        print('File my_encoder.parquet created')
        


#################################################################################################################################################################


if __name__ == '__main__':
    #Instantiating Model
    model = Model()

    #Defining Data and Variables
    categories = ['job',"marital","education",'default','housing','loan','contact','month','day_of_week','poutcome','y']
    cat_ind = [0,1,2,3,4,5,6,7]
    nums_ind = [8,9,10,11,12,13,14,15,16,17,18,19,20]
    disc_ind = [8,9]

    # Data Fitting in the Model by Encoding and Decoding
    model.fit('data/bank/bank-additional.csv',categories,cat_ind,nums_ind,disc_ind,'y',1000,';',True)
    
    # Show Encode and Save File in .parquet file
    model.encode()

    # Show Decode and Save File in .parquet file
    model.decode()
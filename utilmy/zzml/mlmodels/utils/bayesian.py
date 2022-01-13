"""
ui_dir=/a/adigcb301/ipsvols05/offline/maruichi/UI/
raw_dir=/a/adigcb301/ipsvols05/offline/maruichi/data/
pos_dir=/a/adigcb301/ipsvols05/offline/maruichi/data/pos/
genre_dir=/a/adigcb301/ipsvols05/offline/maruichi/data/genre/
daily_dir=/a/adigcb301/ipsvols05/offline/maruichi/data/daily_summary/
model_dir=/a/adigcb301/ipsvols05/offline/maruichi/model/
optimization_dir=/a/adigcb301/ipsvols05/offline/maruichi/optimization/



Prediction :
    
    
Season(t)  *  

ckey = f'thist_{shop}_{item_id}'


ckey = f'thist_{shop}_{item_id}'
msg = self.couch_conn.maru_conn.get(ckey,no_format=True).value
if msg == None:
    continue

df = pd.read_parquet(BI(msg))
tmap = defaultdict(fourvect)



sess = Session(name="optim_beta")
sess.save( globals())


sess.load( globals() )



"""
import warnings
warnings.simplefilter(action='ignore')

import os, sys, gc, glob
import pandas as pd, numpy as np
import time, msgpack
from collections import defaultdict


from remote_runnable.config_reader import ConfigReader
from remote_runnable.logger import Logger
from remote_runnable.run_local_method import remote_runnable


from offline.util import *
####################################################################################################
import zlocal
root =  zlocal.root
from zlocal import *


def  y_transform(v, inverse=0):
    if inverse: return np.exp(v)
    else      : return np.log(v)

def X_transform(dfXy, colsX) :
  return pd_to_onehot(dfXy[colsX] , colsX) 
    

def pd_filter(df, filter_dict=None ) :    
  for key,val in filter_dict.items() :
      df =   df[  (df[key] == val) ]       
  return df


import warnings
from sklearn.linear_model import *

#####################################################################################################
#####################################################################################################
### Price / Sensis Elastic
df = pd_read_file( dir_price  + "/*elastic*2020.parquet" )
df['date'] = 2020


df1 = pd_read_file( dir_price  + "/*elastic*2019*" )
df1['date'] = 2019

df2 = pd_read_file( dir_price  + "/*elastic*2018*" )
df2['date'] = 2018


df = pd.concat((df, df1, df2 ))


pd_to_file(df,dir_price  + "/itemid_elastic_3Y.parquet" )
dfref =df 



df1 = df[df.l2_genre_id != -1 ]
df1 = df1[df1.shop_id.isin([16, 17])  ]

df1 = df1[df1.item_id.isin(vitem_212 ) ]


#####################################################################################################
#####################################################################################################
cols = [ 'time_key',  'shop_id', 'l1_genre_id', 'l2_genre_id', 'item_id', 'price_median',   'pweekday', 
         'price_pct2', 'units_pct', 'units_sum', 'cost_median'  ] 
df = pd_read_file2(  dir_price + "*_daily_*2020*")
df.columns
gc.collect()
"""
Index(['cost_max', 'cost_median', 'cost_min', 'gms_max', 'gms_mean', 'gms_min',
       'gms_std', 'gms_sum', 'item_id', 'l1_genre_id', 'l2_genre_id',
       'price_max', 'price_median', 'price_median_d1', 'price_median_log',
       'price_min', 'price_pct', 'price_std', 'pweekday', 'shop_id',
       'time_key', 'units_max', 'units_median', 'units_median_d1', 'units_min',
       'units_pct', 'units_pct2', 'units_pct_ref', 'units_std', 'units_sum',
       'units_sum_log', 'units_sum_season', 'units_sum_season_log', 'weekday'],

"""

t0     = to_timekey("20200507")
dt0    = to_datetime("20200507")
alpha  = 0.3





####################################################################################################    
"""
  du/U = beta.dp + cst
  
   log U  = beta.P  + cst
   
   Across shop_id, l2 o
   
   beta(shop_id, item_id)  --> Similar disitrbuiton
    

Beta  = H(shop)
Beta = H(item_id)  




"""



####################################################################################################
#####################################################################################################
# Hiearchy model
import warnings
warnings.filterwarnings('ignore')
from pymc3 import Deterministic,find_MAP,Normal, HalfCauchy, Uniform,Model,StudentT,sample,Laplace
import pymc3 as pm


####### Data loading 
cols = [ 'time_key',  'shop_id',  'l2_genre_id', 'item_id', 'price_median',   'pweekday', 
         'price_pct', 'units_pct', 'units_sum',   ] 
df = pd_read_file2(  dir_price + "*_daily_*")

#### l2_genre_id X all shops : 50k samples
df = df[df["l2_genre_id"] == 20601 ]
df['item_id'] = df['item_id'].astype('int64') 
gc.collect()


####################################################################################################
#### Custom Group definition
df['group']   = df.apply(lambda x : x['shop_id'] * 1e9 + x['item_id'] , axis=1  )
groups = df['group'].unique()
groups = [ int(t) for t in groups]
print(groups)

### 1 itemid across shop
df['group']   = df.apply(lambda x : x['shop_id']  , axis=1  )
groups = df['group'].unique()
groups = [ int(t) for t in groups]


### 1 shop_id, all l2 items
df['group'] = df.apply(lambda x : x['item_id'] , axis=1  ).astype('int')
shop_id     = 16
groups = df[ df.shop_id == shop_id ]['group'].unique()
groups = [ int(t) for t in groups]

#####################################
n_group       = len(groups)
group_lookup  = dict(zip(groups, range(n_group)))
print(n_group)
2152006  in groups


### 1 shop_id, all l2 items  ###############################################
df['group2'] = df.apply(lambda x : x['item_id'] , axis=1  )
groups2 = df[ df.shop_id == shop_id ]['group2'].unique()
groups2 = [ int(t) for t in groups2]

n_group2         = len(groups2)
group_lookup2    = dict(zip(groups2, range(n_group2)))
print(n_group2)
dfi["group2_id"] = dfi['group2'].replace(group_lookup2).values 
dfi = dfi[ dfi['group2_id'] < 1000 ]   ### More the samll ones
###################################################################################################






###dfi = copy.deepcopy(df)
itemi = 2144018
dfi = pd.concat(( df[df.item_id == itemi ],  df[df.shop_id == shop_id ]))

dfi["group2_id"] = dfi['group2'].replace(group_lookup2).values 
dfi = dfi[ dfi['group2_id'] < 1000 ]   ### Remove the missing ones



dfi = dfi[dfi['price_pct'] != 0.0 ]
dfi = df[df.item_id == 2124003 ]
dfi = df[df.shop_id == 16 ]
dfi = dfi[dfi['price_pct'] != 0.0 ]

#### Norm
dfi['units_sum_log'] = np.log( dfi['units_sum'] / dfi['pweekday']  )
dfi = dfi.set_index( [ 'shop_id', 'time_key'  ] )
dfi['units2'] = dfi['units_sum'] / dfi['pweekday']  
dfi['units2'] = dfi.reset_index().groupby([   'shop_id' ,'item_id'   ])['units2'].pct_change()
dfi['group_code'] = dfi['group'].apply(lambda x : group_lookup[x])
dfi['group_code2'] = dfi['group2'].apply(lambda x : group_lookup2[x])
dfi = dfi[ dfi['group_code2'] < 1000 ]
dfi = dfi.reset_index()




############################################################################################
########## Model setup + training  #########################################################
### est_all ={}
from offline.models.models_beta import *
tag = "_l2_poisson"
est, modelx = model5(dfi,  n="map", group_lookup=group_lookup,
                     path= zlocal.dir_model +"/beta/",  tag= tag , verbose=True)
est_all[tag] = est
print('b', est['b'], "\nb2", est.get('b2'))


#### Genrate sanples ################################################
with modelx :
    trace = pm.sample(50, chains=2, cores=4)
save(trace , est['name'] +"/trace.pkl")    

dfi = beta_add(dfi,  a = est['a'] , b=  est['b'] ,  b2 =  est['b2'] )


##### MCMC fitting
est2 = model5(dfi,  n="adv",  group_lookup=group_lookup)
print(est2)



############################################################################################
##### Prediction  ##########################################################################
dfi['units_log_pred'] = pred5( dfi,  a = est['a'] , b=  est['b'] ,  b2 =  est.get('b2') , )
#dfi['units_pred']     = np.exp(dfi['units_log_pred']) * dfi['pweekday']

dfi['units_pred']     = dfi['units_log_pred'] * dfi['pweekday']

#### Check
print(itemi)
dfj = dfi[dfi.item_id == itemi ][ dfi.shop_id == shop_id ]

pd_plot_multi(dfj.iloc[-100:,:].set_index([ 'shop_id', 'time_key'  ]), 
              cols=[ 'units_sum', 'units_pred'], 
              cols2=['price_median'], 
              spacing= 0.0,)





#############################################################################################
##### Params Check   ########################################################################
fp = r'C:/D/gitdev/codev15/zdata/test//model//beta//pymc_11_13_14_16_17_18_21_22_50-20601-1/'
fp = r"C:/D/gitdev/codev15/zdata/test//model//beta//pymc_11_13_14_16_17_18_21_22_50-20601-174_no_zero"
modelx, group_lookup, est  = load(fp + "/model.pkl"), load(fp + "/group_lookup.pkl"), load(fp + "/estimate.pkl"), 
print(est)

### sample generaion
with modelx :
    trace = pm.sample(50, chains=2, cores=4)

pm.traceplot(trace, var_names= ["sigma_y", "b",  "sigma_b",  "mu_b", "a", "sigma_a",  "mu_a", ]  );
# pm.traceplot(tr, var_names= ["sigma_y", "b",  "sigma_b",  "mu_b", "sigma_b2",  "mu_b2", "a", "sigma_a",  "mu_a", ]  );
r_stat = pm.summary(trace)

save(trace , fp +"/trace.pkl")
trace = load(fp +"/trace.pkl")


#### the beta we have interest
print( trace.b[:, group_lookup[itemi] ])



##############################################################################################
################# Curve Plotting #############################################################
itemi = 2124003
# shop_id = 13
dfj     = dfi[dfi.item_id == itemi ][ dfi.shop_id == shop_id ]

i0 = group_lookup[ shop_id]
i0 = group_lookup[ itemi ]
def demand(price,i0=17) :
    v= trace.b[:,i0]*price.reshape(-1,1)
    v = v.T
    return np.exp( trace.a[:,i0].reshape(-1,1) + v )

def score_fun2(price, cost, units, alpha=0.3) :           
    profit = units*((price - cost ).reshape(-1,1) )
    gms    = price.reshape(-1,1)  * units
    score  = alpha*gms + (1-alpha)*profit
    return score


price = np.linspace( dfj['price_median'].min() , dfj['price_median'].max() )
dem   = demand(price, i0).T
cost0 = dfj['cost_median'].median()

tagj = f"{shop_id}_ {itemi}"
plt.plot(price, dem ,c='k',alpha=0.01)
plt.plot(dfj['price_median'], dfj['units_sum'],'o', c='C1')
plt.title( f"Demand :  {tagj} beta_avg: {  trace.b[:,i0].mean() }   ")
plt.savefig(fp + f"/demand_{tagj}.png"); plt.show() ; plt.close()


scores  = score_fun2(price, cost=cost0, units= dem, alpha= 0.3)
scores0 = score_fun2(dfj['price_median'].values[-20: ],     cost= cost0, 
                     units= dfj['units_sum'].values[-20: ], alpha= 0.3)

plt.plot(price,scores ,c='k',alpha=0.01)
plt.plot( dfj['price_median'][-20: ],  scores0, marker='o', linestyle='none',  c='C1')
plt.title( f"Scores: {shop_id}, {itemi}, beta_avg:  {  trace.b[:,i0].mean() }   ")
plt.savefig(fp + f"/scores_{tagj}.png") ;plt.show() ; plt.close()



#############################################################################################
#############################################################################################
dfis = df[ df.l2_genre_id == 20601  ][ df.shop_id == 16  ].groupby(['item_id']).agg({ "units_pct": 'nunique', 'units_sum' : 'count'   })
dfi[ dfi.item_id == 2152006 ]


"""
item_id	units_pct	units_sum
2132031	9	10
2136058	9	13
2144018	9	365
2144025	9	44
2144036	9	11
2144020	10	16
2000440	11	19
2000441	11	20
2120014	11	351
2132018	11	16
2132022	11	291


"""



##############################################################################################
##############################################################################################
n_new = 200
X_new = np.linspace(0, 15, n_new)[:,None]

# add the GP conditional to the model, given the new X values
with modelx:
    f_pred = tp.conditional("f_pred", X_new)

# Sample from the GP conditional distribution
with model:
    pred_samples = pm.sample_posterior_predictive(tr, vars=[f_pred], samples=1000)
    
    




with modelx2 :
    tr = pm.sample(1, chains=2, cores=4)


m5_trace = model5(dfi , "map")
results = {'per-step-test': [], 'running-tst': []}

y_hat=np.zeros((len(outsample),))
for cnt,sample in enumerate(m5_trace):    
     sample_y = pred5(sample['a'], sample['b'],outsample)


est = model5(dfi, n=1)



###############################################################
##### Calcualt RMS per model    ##########################
### archial predictions and the non-hierarchical predictions.

selection = ['CASS', 'CROW WING', 'FREEBORN']
fig, axis = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
axis = axis.ravel()
for i, c in enumerate(selection):
    c_data = data.loc[data.county == c]
    c_data = c_data.reset_index(drop = True)
    z = list(c_data['county_code'])[0]

    xvals = np.linspace(-0.2, 1.2)
    for a_val, b_val in zip(indiv_traces[c]['alpha'][::10], indiv_traces[c]['beta'][::10]):
        axis[i].plot(xvals, a_val + b_val * xvals, 'b', alpha=.05)
    axis[i].plot(xvals, indiv_traces[c]['alpha'][::10].mean() + indiv_traces[c]['beta'][::10].mean() * xvals, 
                 'b', alpha=1, lw=2., label='individual')
    for a_val, b_val in zip(hierarchical_trace['alpha'][::10][z], hierarchical_trace['beta'][::10][z]):
        axis[i].plot(xvals, a_val + b_val * xvals, 'g', alpha=.05)
    axis[i].plot(xvals, hierarchical_trace['alpha'][::10][z].mean() + hierarchical_trace['beta'][::10][z].mean() * xvals, 
                 'g', alpha=1, lw=2., label='hierarchical')
    axis[i].scatter(c_data.floor + np.random.randn(len(c_data))*0.01, c_data.log_radon, 
                    alpha=1, color='k', marker='.', s=80, label='original data')
    axis[i].set_xticks([0,1])
    axis[i].set_xticklabels(['basement', 'first floor'])
    axis[i].set_ylim(-1, 4)
    axis[i].set_title(c)
    if not i%3:
        axis[i].legend()
        axis[i].set_ylabel('log radon level')
        
        

import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt

df = pd.read_csv('/home/ryan/Downloads/GDP.csv')
df.index = pd.to_datetime(df['DATE'])

df['GDP'].plot()
plt.title('GDP over Time')
plt.xlabel('Date')
plt.ylabel('GDP')
plt.show()


#### Time Varyinsg Coefficient
df['lag'] = df['GDP'].shift()
df.dropna(inplace=True)
with pm.Model() as model:
    sigma = pm.Exponential('sigma', 1./.02, testval=.1)
    nu = pm.Exponential('nu', 1./10)
    beta = pm.GaussianRandomWalk('beta',sigma**-2,shape=len(df['GDP']))
    observed = pm.Normal('observed', mu=beta*df['lag'], sd = 1/nu, observed = df['GDP'])
    
    trace = pm.sample()
 
plt.plot(df.index,trace['beta'].T, 'b', alpha=.03)
plt.title('GDP Growth Rate')
plt.xlabel('Date')
plt.ylabel('Growth Rate')
plt.show()

plt.plot(df.index,trace['beta'].T, 'b', alpha=.03)
plt.plot(df.index,1+(np.log(df['GDP'])-np.log(df['lag'])),'r',label='True Growth Rate')
plt.title('GDP Growth Rate')
plt.legend()
plt.xlabel('Date')
plt.ylabel('Growth Rate')
plt.show()




####################################################################################################
# Beta modelling
"""


u = exp(beta.p)
log u  = beta.p + cst

du/dp = beta.u  -->  du/u =  beta.dp = beta.dp/p * p


"""

# m5_trace=model5(insample,2)
# results = {'per-step-test': [], 'running-test': []}
# y_hat=np.zeros((len(outsample),))
# for cnt,sample in enumerate(m5_trace):    
#     sample_y = pred5(sample['a'], sample['b'],outsample)
#     y_hat += sample_y
#     running_y = y_hat / (cnt + 1)
#     results['per-step-test'].append(MAD(outsample.sales_actual_yoy, sample_y))
#     results['running-test'].append(MAD(outsample.sales_actual_yoy, running_y))
# results = pd.DataFrame(results)
# results.plot(kind='line', grid=False, figsize=(10, 5),title='Per-step and Running MAD From MCMC')
# plt.show()
    




import zlocal

df = pd_read_file( zlocal.dir_pred + "df_error.pkl" )


df = pd_read_file2(  )


###### Details
df = pd.read_parquet(  zlocal.dir_price + "/itemid_daily_2020.parquet" )














































































#####################################################################################################
############# Function  ############################################################################
"""
df = pd_read_file( dir_porder )

[8/20 10:56 AM] Primozich, Neil | Neil | BDD
    --->   alpha* GMS + (1-alpha)*Profit


unit_get_value()

    Price Optimization:   global optimization:      max ( GMS) such that margin = Y


    alpha ~ 0.3

    top items without campaigns  16,17

    determine price that max    alpha*GMS + (1-alpha) * profit



from offline.afstats import *


df = generate_itemid_stats(price_dir= zlocal.dir_price  ,
                          tag = "2020b" , ldates =  [ "*20200714*"     ], calc_beta=False,
                          debug= False ) 

"""
df = pd_read_file2(  dir_price + "*_daily_*", shop_id = 16 )
df =df[df.shop_id.isin([16]) ]
gc.collect()



season_dd =load(dir_meta +"/season/season_16_l2genre.pkl" )




### Fit on log prices 
## Remove seasonality
unit_item = unit_l2 * porder

unit_item = (1+ yearly(date)) * (1 ) *  price_level.
Seasonality_part :


    
  log("\n####### Starting ", arg.do,  " model date", model_date, to_timekey(model_date) )
  os_variable_check([ "shopid_train_predict", "shop_params", "n_future",  ],  globals())   
  from offline.util import to_datetime
  ytarget    = "order_id_s"
  cols       = ['time_key', 'shop_id', ytarget]
  df         = pd_read_file2( train_path_y, cols=cols, n_pool=3, drop_duplicates= ['time_key', 'shop_id' ], verbose=verbose)
  df         = df[ df['time_key'] <= to_timekey(model_date) ]
  df         = df[cols].drop_duplicates( ['time_key', 'shop_id' ]  ).sort_values(['shop_id', 'time_key'])

  df['key']  = df['shop_id'].apply( lambda t : f"{t},0,0,0" )




@jit
def unit_fun01(price):
    return np.exp(-0.1*price + 0.5)


clf = Ridge().fit( dfp['time_key'].values.reshape(-1, 1) , dfp['units_sum'].values.reshape(-1, 1) )
dfp['trend_factor_a'] =  clf.intercept_[0] 
dfp['trend_factor_b'] =  clf.coef_[0][0]

dfp2 = dfp.drop_duplicates(['month', 'day', ])
dfp2['l2_genre_id'] = l2_id

season_dd = OrderedDict()
season_dd[l2_id] = dfp2.set_index([ 'month', 'day' ]).to_dict('index')


def season_remove(x) :
  dd = season_dd[ int(x['l2_genre_id']) ][  (x['month'], x['day'])  ] 
  # xp = x['units_sum'] / ( dd['trend_factor_b'] * x['time_key'] + dd['trend_factor_a']  )
  # xp = xp * ( dd['trend_factor_b'] * 17532.0  + dd['trend_factor_a']  ) 
  xp = x['units_sum_ref'] / dd['season_factor'] 
  return xp  



dfp2['units_sum_ref'] = dfp2['units_sum']   
dfp2['units_sum']     = dfp2.apply(lambda x : season_remove(x ), axis=1)
dfp2['units_sum2'].plot()
season_dd[l2_id][(1,30)]



df = df[df.item_id == 6990003 ]
dfi['month']      = dfi['time_key'].apply(  lambda x : to_datetime(from_timekey(x)).month )
dfi['day']        = dfi['time_key'].apply(  lambda x : to_datetime(from_timekey(x)).day )
dfi['units_sum2'] = dfi.apply(  lambda x : season_remove(x), axis=1)

clf = Ridge().fit( dfi['price_median'].values.reshape(-1, 1) , np.log( dfi['units_sum2'].values.reshape(-1, 1)) )
clf.coef_

dfp['weekly'].iloc[:10].plot()
dfp['monthly'].plot()
dfp['yearly'].plot()
dfi['units_sum'].plot()

df2 = df.groupby([ "time_key", "shop_id",  "l2_genre_id"]).agg({"units_sum" : "sum"}).reset_index()



####################################################################################################
"""

u = max( u0  exp( ) )

u = u0. exp( -beta. (Price-Price0))

log u = beta.Price + cst

u = u0 . exp(-beta(Price-Prie0))

u-act = u* season

u0, price0
unit_detrend = unit / season  / trend

trend_factor = UnitTrend(date) / Units


Price( )




"""


###################################################################################################
############ Generate Unit Price Predictions  ##############################################
"""
shop,item, current_price, suggested_price, along with forecasts?


Xseason + prediction by price level.

unit = Season * 

Unit(price, time)  = Upred * exp(price dynamics)
Upred(t)

    Historical Unit  =  F(params,)

"""
shop_id = 17
fpath = dir_model + f"/unitfull_itemid_20200601/{shop_id}-RandomForestRegressor_daily/"


dfp   = pd_read_file(fpath + "/17_2_206_20601_0/df_error*")
dfpi.columns

for ii in dfp.item_id.unique() : 
   dfpi = dfp[dfp.item_id == ii ]
   dfpi.set_index('date')[[ 'units_g0', 'units_g0_pred', 'units_g0_pred_price',   ]].plot(title= str(ii))
   
dfp[[ 'units_g0', 'units_g0_pred', 'units_g0_pred_price',   ]].plot()



###  pred_itemid = {}
##   beta_itemid = {}; 
########    season_itemid = {}
flist = glob.glob(fpath + f"/*/*df_error*")

for fp in flist :
    dfp   = pd_read_file2( fp, verbose=False )
    for ii in dfp.item_id.unique() : 
        dfpi = dfp[dfp.item_id == ii ]
        
        dft= pd.DataFrame()
        dft['date'] = pd.date_range("20200401", "20200901")
        dft['time_key'] = [  to_timekey(t) for t in dft['date'] ] 
        dft['month'] = [  t.month for t in dft['date'] ] 
        dft['day']   = [  t.day for t in dft['date'] ] 
        dft = dft.join( dfpi.set_index('time_key'), on ='time_key', how='left', rsuffix='2')
        dft['units_g0'] = dft['units_g0' ].fillna(0.1)
        dft['units_g0_pred'] = dft['units_g0_pred'].fillna(0.1)
    
        season_itemid[ii] = dft.set_index(['month', 'day'])[[  'units_g0_pred', 'units_g0'   ]].to_dict('index')
    
        ##### Past prices
        dfpi['log_ratio'] = np.maximum(0.1, np.abs( dfpi['units_g0'] / dfpi['units_g0_pred'] ) )
        clf = Ridge().fit( dfpi['price_mean_g0'].values.reshape(-1, 1) , np.log( dfpi['log_ratio'].values.reshape(-1, 1)) )   
        beta_itemid[ii] =  clf.coef_[0][0]
        print(ii, beta_itemid[ii]  )
        
        ##### Pred series
        pred_itemid[ii] = dft


### dfpi = dft.join( dfpi.set_index('time_key'),  on ='time_key', how='left', rsuffix='2' )
season_itemid[2124005]


pred_itemid[ 2124003][[ 'units_g0', 'units_g0_pred' ]].plot()


dft.columns








####################################################################################################
############ Generate Season  ######################################################################
from offline.afstats import generate_season_factor

generate_season_factor(agg_level = "l2_genre_id", ytarget="units_sum",  
                       dir_pos= dir_price + f"/*daily*", 
                       dir_out= dir_meta + f"/season/season_{shop_id}_l2genre2.pkl",  
                       l2_list = l2_list,
                       iimax= 100, verbose=False)

season_dd =load(dir_meta +"/season/season_16_l2genre.pkl" )


####################################################################################################
######### Create vitem ############################################################################
from numba import jit

def season_remove(x) :
  dd = season_dd[ int(x['l2_genre_id']) ][  (x['month'], x['day'])  ]  
  xp = x['units_sum'] / dd['season_factor'] 
  return xp  


def get_l2_item(df, item_id) :
  vitem = list(df[ df.l2_genre_id == df[df.item_id == item_id ]['l2_genre_id'].values[0]]['item_id'].unique())
  print(len(vitem))
  return vitem


@jit
def score_fun(price, cost, units, alpha=0.3) :           
    profit = units*(price - cost )
    gms    = price * units
    score  = alpha*gms + (1-alpha)*profit
    return score


def unit_fun(ii, t, u0, x0,  x) :
    clf    = item_dd[ii]['unit_class']
    beta   = clf.coef_[0][0]
    season = item_dd[ii]['season'][  (t.month, t.day )  ]    
    u      = u0*np.exp( beta*(x -x0)  )  * season


####################################################################################################
shop_id = 16
df = pd_read_file2(  dir_price + "*_daily_*", shop_id = 17 )
df = df[df.shop_id.isin([17]) ]


#### create vitem
df1 = df[ df.time_key > 18360  ].groupby([  "l2_genre_id", 'item_id']).agg({'units_median' : 'sum', 'price_median': 'nunique'}).sort_values('units_median', ascending=0).iloc[:,:].reset_index()
#df1 = df1[ df1['price_median'] < 4 ]


vitem_232    =  [2324004, 2324005, 2324006, 2324008, 2324009, 2324010, 2324011, 2324012, 2324014, 2324015, 2324016, 2324017, 2324020, 2324024, 2324028, 2324032, 2324033, 2324058, 2324062, 2324063, 2324080, 2324081, 2324082, 2324087, 2328001, 2328002, 2328004, 2328007, 2328008, 2328010, 2328011, 2328015, 2328017, 2328023, 2332000, 2332004, 2332005, 2332010, 2332013, 2332016, 2332017, 2332020, 2332022, 2332023, 2332025, 2332026, 2332029, 2332031, 2332034, 2332035, 2332037, 2332039, 2332043, 2332045, 2332046, 2332047, 2332049, 2332050, 2332052, 2332053, 2332056, 2332057, 2332058, 2332059, 2333000, 2333006, 2333011, 2333013, 2333020, 2333023, 2333024, 2333025, 2333030, 2333044, 2333045, 2333063, 2333065, 2333075, 2333079, 2333087, 2333095, 2333113, 2333114, 2336012, 2336046, 2336049, 2352063, 2770031, 2000218, 2000339, 2324007, 2324013, 2324022, 2324027, 2324044, 2324051, 2324055, 2324056, 2324060, 2324066, 2324068, 2324070, 2324071, 2324072, 2324096, 2328006, 2328018, 2328019, 2328024, 2332018, 2332030, 2332042, 2332055, 2332060, 2332061, 2332062, 2332066, 2332069, 2332070, 2332075, 2332079, 2332080, 2332081, 2332082, 2332085, 2332087, 2332088, 2332095, 2332096, 2332097, 2332098, 2333049, 2333120, 2333123, 2000493, 2324057, 2324067, 2324076, 2324077, 2332015, 2332021, 2332032, 2332101, 2332103, 2332107, 2332108, 2332110, 2333129, 2333130, 2333131, 2333132, 2333133, 2333134, 2333136, 2333137]

vitem_212    =  [2120005, 2120008, 2120009, 2120011, 2120014, 2120026, 2120029, 2124000, 2124002, 2124003, 2124004, 2124005, 2124009, 2124012, 2124013, 2124014, 2124016, 2124018, 2124020, 2124022, 2124024, 2124045, 2128003, 2128004, 2128005, 2132001, 2132002, 2132004, 2132005, 2132008, 2132017, 2132018, 2132019, 2132020, 2132021, 2132022, 2136000, 2136003, 2136005, 2136017, 2136022, 2136048, 2136050, 2136051, 2136052, 2136053, 2140003, 2140004, 2140008, 2140009, 2140011, 2144004, 2144014, 2144018, 2144019, 2144020, 2144024, 2144025, 2144026, 2144027, 2144028, 2144031, 2144032, 2144035, 2144036, 2144037, 2144038, 2148004, 2148006, 2148008, 2148014, 2148020, 2148021, 2148022, 2148023, 2148024, 2148025, 2152002, 2152003, 2152005, 2000160, 2000204, 2000246, 2120017, 2124025, 2132023, 2132024, 2132025, 2132026, 2136011, 2136043, 2136057, 2140014, 2140015, 2144041, 2144043, 2144044, 2148026, 2148028, 2148029, 2148030, 2148031, 2148032, 2148033, 2148034, 2148035, 2148036, 2148037, 2300141, 2000248, 2000440, 2000441, 2000442, 2000443, 2000444, 2000479, 2000521, 2000548, 2000602, 2000603, 2000604, 2000687, 2000703, 2000711, 2000727, 2000728, 2124010, 2132013, 2132014, 2132030, 2132031, 2136012, 2136058, 2136059, 2140018, 2144046, 2148000, 2148002, 2148038]

vitem_16_212 =  [2120005, 2120008, 2120026, 2124000, 2124002, 2124003, 2124004, 2124005, 2124012, 2124013, 2124018, 2124022, 2128004, 2132001, 2132004, 2132022, 2136000, 2136003, 2136005, 2136017, 2140003, 2140008, 2144004, 2144014, 2144028, 2152002, 2132025, 2136011, 2144043, 2148026]

vitem = df1['item_id'].iloc[:3].values
vitem = [2324081 ]
vitem = vitem_212[:100]
item_dd[2124004]['l2_genre_id']



#### Specify the cost  #############################################################################
# item_dd = OrderedDict()


for ii in vitem :
   try :  
       item_dd[ii] = {} 
       dfi         = df[df.item_id == ii] 
       if len(dfi[dfi.time_key.between(t0, t0+10)]) < 5 :
           print("skip", ii)
           vitem.remove(ii)
           continue
       dfi['month']      = dfi['time_key'].apply(  lambda x : to_datetime(from_timekey(x)).month )
       dfi['day']        = dfi['time_key'].apply(  lambda x : to_datetime(from_timekey(x)).day )
    
       dfi['units_sum2'] = dfi.apply(  lambda x : season_remove(x), axis=1)  
       clf = Ridge().fit( dfi['price_median'].values.reshape(-1, 1) , np.log( dfi['units_sum2'].values.reshape(-1, 1)) )   
       print(ii, clf.coef_)      
       item_dd[ii]['unit_class'] = copy.deepcopy(clf)
       item_dd[ii]['unit_beta0'] = clf.coef_[0][0]
       
       
       item_dd[ii]['unit_pred']  = season_itemid[ii]    #### [(month,day)]
       item_dd[ii]['unit_beta']  = beta_itemid[ii]
    
    
       item_dd[ii]['time_key']    = dfi['time_key'].values   
       item_dd[ii]['l2_genre_id'] = dfi['l2_genre_id'].values[0]
       item_dd[ii]['l1_genre_id'] = dfi['l1_genre_id'].values[0]
       item_dd[ii]['l2_items']    = get_l2_item(df, ii)
    
       item_dd[ii]['prices']     = dfi['price_median'].values
       item_dd[ii]['units']      = dfi['units_sum'].values
       item_dd[ii]['units2']     = dfi['units_sum2'].values
       item_dd[ii]['season']     = season_dd[  item_dd[ii]['l2_genre_id'] ]
       
       item_dd[ii]['cost']     = dfi['cost_median'].mean()
       item_dd[ii]['unit0']    = dfi[dfi.time_key.between(t0, t0+14)]['units_sum2'].mean()
       item_dd[ii]['price0']   = dfi[dfi.time_key.between(t0, t0+14)]['price_median'].mean()
       vitem2.append(ii)
   except Exception as e :
       print(e)
















vitem = [ t for t in  vitem2 if t in vitem_212 ]
vitem = list(set(vitem))
print(vitem2)   
vprice0 = np.array([ np.mean(item_dd[key]['prices'])  for key in vitem ])


df[df.item_id.isin(vitem)][['shop_id', 'item_id', 'l2_genre_id']].drop_duplicates("l2_genre_id")['l2_genre_id'].values
print(vitem)
    

def unit_fun01(ii=6990003, t=0,  price=0, verbose=False) :
    item_dd_ii = item_dd[ii]
    season = item_dd_ii['season'][  (t.month, t.day )  ][ 'season_factor']    
    u0     = item_dd_ii['unit0']
    x0     = item_dd_ii['price0']
    if verbose : print(ii, x0, u0)
    
    #### Cutoff at extreme values
    umax, umin = np.max( item_dd_ii['units']  ), np.min( item_dd_ii['units']  )
    if price < x0 * 0.7 : return umax
    if price > x0 * 1.5 : return 0
    if price > x0 * 1.2 : return umin

    #### Middle area : model
    clf    = item_dd[ii]['unit_class']
    # beta   = clf.coef_[0][0]            
    u01    = np.exp( clf.predict(np.array([x0]).reshape(-1, 1) ) )
    if verbose : print(u01, u0, x0, clf.coef_)    
    u  = u0/u01[0][0] * np.exp( clf.predict( np.array([price]).reshape(-1, 1)   ))  * season
    
    ###### Max / Min
    u2 = min(umax, max(umin, u[0][0] ))    
    return  u2



def unit_fun02(ii=6990003, t=0,  price=0, verbose=False) :
    item_dd_ii = item_dd[ii]
    season = item_dd_ii['season'][  (t.month, t.day )  ][ 'season_factor']    
    u0     = item_dd_ii['unit0']
    x0     = item_dd_ii['price0']
    
    if verbose : print(ii, x0, u0)
    
    #### Cutoff at extreme values #############
    umax, umin = np.max( item_dd_ii['units']  ), np.min( item_dd_ii['units']  )
    if price < x0 * 0.7 : return umax
    if price > x0 * 1.5 : return 0
    if price > x0 * 1.2 : return umin

    #### Middle area : model ##################
    clf      = item_dd[ii]['unit_class']
    clfpred  = item_dd[ii]['unit_pred']
    # beta     = item_dd[ii]['unit_beta0'] * 0.75
    beta     = item_dd[ii]['unit_beta'] 


    ###  u  = upred(x0, t) * exp( beta*(x-x0) )           
    upred = clfpred[ (t.month, t.day )]['units_g0_pred_season']
    # print(upred)
    u01   = upred / np.exp( beta*x0  )  
    u     = u01 * np.exp( beta*price    ) 
    if verbose : print(u01, u0, x0, clf.coef_)  

    
    ###### Max / Min  #########################
    u2 = min(umax, max(umin, u ))    
    return  u2
unit_fun = unit_fun02


def cost_total(vprice, unit_fun,  verbose=False) :    
  ss = 0    
  # vprice = price_normalize(vprice)
  for ii, key in enumerate(vitem) : 
     units = unit_fun(key, t,  vprice[ii], verbose=verbose)   
     if verbose :print(key, vprice[ii], item_dd[key]['cost'], units, alpha )         
     x  = score_fun( vprice[ii], item_dd[key]['cost'], units, alpha=alpha )
     ss = ss + x
  return -ss / ii  ### Avg Cost


####### Check   ##################################
t = to_datetime("20200507")
vprice0 = np.array([ np.mean(item_dd[key]['prices'])  for key in vitem ])


unit_fun = unit_fun02
cost_total(vprice0, unit_fun, verbose= True)


#### Check
unit_fun(ii=2324081  , t=t,  price=130) 

#### All demand curves ###########################
print( df1[df1.item_id.isin(vitem)])
for iix in vitem[1:] :
  x=np.arange(80, 160, 5)
  ax = plt.plot(  x, [ unit_fun(ii=iix  , t=t,  price=a)  for a in x   ])


ldate = [  pd.to_datetime(t)  for t in pd.date_range("20200601", "20200701", freq="D")  ]
for t in [ldate[t] for t in [ 5, 10, 15, 20,]] :
  x=np.arange(80, 160, 5)
  plt.plot(  x, [ unit_fun(ii= vitem[5]  , t=t,  price=a)  for a in x   ])





"""

Units = F(price)
Sequential Curve






"""


####################################################################################################
############ Optim #################################################################################
import pygmo as pg

vprice0 = np.array([ np.median(item_dd[key]['prices'][-20:])  for key in vitem ])

class cost_class:
    def __init__(self, dim=0):
        self.dim = dim
    def fitness(self, x):
        return [  cost_total(x, unit_fun, verbose=False)   ]

    def get_bounds(self):        
        up_list =  vprice0 * 1.2
        do_list =  vprice0 * 0.8       
        return (do_list, up_list)

    def get_name(self): return "Cost"
    def get_extra_info(self): return "\tDimensions: " + str(self.dim)


########################
def optim_de(cost_class, n_iter=10, time_list=None, pop_size=20, date0="20200507") :    
    import pygmo as pg    
    llr = { k: [] for k in [ "date",  'time_key', 'vprice', 'score' ] }
    t0  = to_timekey("20200507")    
    
    for ti in time_list :
        t = ti        
        prob = pg.problem( cost_class( len(vitem)))
        pga  = pg.de(gen=1 , F=0.8, CR=0.9, variant=2, ftol=1e-06, xtol=1e-06,)
        algo = pg.algorithm( pga )
        pop  = pg.population(prob, pop_size)
        
        lla = []
        for i in range(0, n_iter) :
          pop = algo.evolve(pop)
    
        print(t, pop.champion_f[-1]    ) #, pop.champion_x,)  #  "\n\n",vprice0)     
        llr['vprice'].append( pop.champion_x  )
        llr['date'].append( t  )
        llr['time_key'].append( to_timekey(t)  )
        llr['score'].append( pop.champion_f[-1]   )

    dfr = pd.DataFrame( np.array(llr['vprice']),  columns= [str(x) for x in vitem ])
    for x in [  'date', 'time_key',  'score' ] :
      dfr[x] = llr[x]
    
    dfr2 = pd.DataFrame()
    for x in vitem :
        dfj         = dfr[[  'date', 'time_key', str(x), 'score'   ]]
        # log_pd(dfj)
        dfj.columns = [ 'date', 'time_key', 'price_optim', 'score'   ]
        dfj['item_id'] = x 
        dfr2 = pd.concat(( dfr2, dfj  ))    
    return dfr2

tlist = [ to_datetime(from_timekey(t0 + ti ) )  for ti in range(0,60,7) ]
optim_de(cost_class, n_iter=1, pop_size=5, date0="20200507", time_list = tlist) 



dfr2 = optim_de(cost_class, n_iter=600, pop_size=30, date0="20200507", time_list = tlist) 
dfr2['units_optim'] = dfr2.apply( lambda x : unit_fun(x['item_id'], x['date'],  x['price_optim']), axis=1) 

#### Mege Add ons
dfi  = df[df.item_id.isin(vitem )]
dfr2 = dfr2.join(dfi.set_index([ "time_key", 'item_id'   ]) , on=[ "time_key", 'item_id'   ], how="left"  )


dfr2['score_act'] = dfr2.apply( lambda x : score_fun(x['price_median'], x['cost_median'], 
                                x['units_median'], alpha=0.3) , axis=1 )
    
dfr2['score_optim'] = dfr2.apply( lambda x : score_fun(x['price_optim'], x['cost_median'], 
                                x['units_median'], alpha=0.3) , axis=1 )





dfr2a  = copy.deepcopy(dfr2)



###########################################################################################
#### Plot  ################################################################################
for ii in vitem :
   ii = 2124003 
   df[df.item_id ==  ii ].iloc[-50:,:][[ 'price_median', 'units_sum' ]].plot(figsize=(12,5), title= f"{ii}")



@jit
def price_normalize(vprice):
    ll = []
    for x in vprice :
        ll.append(int(x/5.0) * 5)
    return ll


dfexp = pd.DataFrame()
for x in vitem[:] :
    dfj  = dfr2[ dfr2.item_id == x ]
    dfj['price_optim2'] =  price_normalize(dfj['price_optim'].rolling(10).mean().fillna( dfj['price_optim'].mean() ) ) 
    dfj.set_index('date')[[ "price_median", 'price_optim2' ]].plot(title= f"{x}")
    dfexp = pd.concat(( dfexp, dfj ))
dfexp['shop_id'] = shop_id 
   


### Units Expected vs actual
for x in vitem[:] :
    # x = 2124003
    dfj  = dfr2[ dfr2.item_id == x ]
    dfj.set_index('date')[[ "units_optim", 'units_sum' ]].plot(title= f"{x}")


### Score optimal 
for x in vitem[:100] :
    dfj  = dfr2[ dfr2.item_id == x ]
    dfj.set_index('date')[['score_act', 'score_optim' ]].plot(title= f"{x}")


###############################################################
#### Merge all preidciont
dfum = pd.DataFrame()
for ii in vitem :
  dfum = pd.concat(( dfum,  pred_itemid[ii]))
dfexp = dfexp.join( dfum.set_index(['time_key', 'shop_id', 'item_id']), on = ['time_key', 'shop_id', 'item_id'], how='left' , rsuffix='2' )

ll = []
for ii in vitem :
  ll = ll +  [ item_dd[ii]['unit_class'].coef_[0][0] ] * len(dfexp['date'].unique())    
dfexp['beta'] = ll
  
ll = []
for ii in vitem :
  ll = ll +  [ item_dd[ii]['price0'] ] * len(dfexp['date'].unique())    
dfexp['price_ref'] = ll

ll = []
for ii in vitem :
  ll = ll +  [ item_dd[ii]['unit_beta'] ] * len(dfexp['date'].unique())    
dfexp['beta2'] = ll


dfexp.to_csv( root  + f"/price/shop {shop_id}_price_optim_v3.csv", mode="w", index=False)




###################################
dfexp2 = dfexp[[ 'time_key', 'shop_id', 'item_id', 'price_optim2', 'price_median', 'units_sum', 'units_g0_pred',  'units_optim' , 'beta', 'price_ref', 'beta2' ]]


dfexp2.columns = [ 'time_key', 'shop_id', 'item_id', 'price_optim', 'price_actual', 'units_actual', 'units_pred',  'units_optim' , 'beta', 'price_ref', 'beta2' ]


dfexp2.to_csv( root  + f"/price/shop_{shop_id}_price_optim_clean_v4.csv", mode="w", index=False)



df5 = pd.DataFrame()
for ii in vitem :
  df5 = pd.concat(( df5,  pred_itemid[ii]))

df5 = df5[ df5.item_id > -1.0 ]


df5[[ 'time_key', 'shop_id', 'item_id', 'units_g0', 'units_g0_pred' ]].to_csv(root + f"/price/shop_{shop_id}_pred.csv", index=False )


    

"""
Season Effect :
    1 day diff :
        
        
    


dfexp_17 = dfexp


dfexp = dfexp.reset_index()

shop, item, current_price, suggested_price, along with forecasts?


dfexp.columns

pred_itemid[ii][[ 'units_g0', 'units_g0_pred' ]]

dfexp = dfexp.set_index(['time_key', 'item_id']) 

dfexp['units_act'] = dfu.set_index(['time_key', 'item_id'])['units_g0']



"""













####################################################################################################
####################################################################################################
df.columns


df['date'] = df['time_key'].apply(lambda x : from_timekey(x))
df = df.set_index('date')




from pygmo import *
archi = archipelago(n = 4, algo = pga, prob = prob, pop_size = 10, seed = 32)
archi.evolve()
res = archi.get_champions_x()



























####################################################################################################
###### Genetic Programming   #######################################################################
from gplearn.genetic import SymbolicRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.random import check_random_state
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz

x0 = np.arange(-1, 1, 1/10.)
x1 = np.arange(-1, 1, 1/10.)
x0, x1 = np.meshgrid(x0, x1)
y_truth = x0**2 - x1**2 + x1 - 1

ax = plt.figure().gca(projection='3d')
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x0, x1, y_truth, rstride=1, cstride=1,
                       color='green', alpha=0.5)
plt.show()


rng = check_random_state(0)

# Training samples
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = np.exp( - X_train[:, 0] - 1.5*X_train[:, 1]) + X_train[:, 1] - 1

# Testing samples
X_test = rng.uniform(-1, 1, 100).reshape(50, 2)
y_test = np.exp( - X_test[:, 0] - 0.5*X_test[:, 1]**2) + X_test[:, 1] - 1


from gplearn.functions import make_function
def logic(x1, x2, ):
    return np.where(x1 > x2, 1, 0)

def exp_(x1 ):
    return np.where(x1 < 100., np.exp(x1), 10000000.0 )

exp2 = make_function(function=exp_,  name='exp2', arity=1)



function_set = ['add', 'sub', 'mul', 'div', exp2]
est_gp = SymbolicRegressor(population_size=500,
                           function_set=function_set,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1, n_jobs=4,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)
print(est_gp._program)



dot_data = est_gp._program.export_graphviz()
graph = graphviz.Source(dot_data)
graph.render('images/ex1_child',  cleanup=True)
graph




function_set = ['add', 'sub', 'mul', 'div', logical]
gp = SymbolicTransformer(generations=2, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0)






import numpy as np, pandas as pd
from hierreg import HierarchicalRegression

## Generating sample data
X=np.random.normal(size=(1000,3))
groups=pd.get_dummies(np.random.randint(low=1,high=4,size=1000).astype('str')).values

## Making a variable dependent on the input
y=(37+2*groups[:,0]-4*groups[:,2]) + X[:,0]*(0+2*groups[:,1]-groups[:,0]) + 0.5*X[:,1] + 7*X[:,2] - 2*X[:,1]**2

## Adding noise
y+=np.random.normal(scale=5,size=1000)

## Fitting the model
hlr=HierarchicalRegression()
hlr.fit(X,y,groups)

## Making predictions
yhat=hlr.predict(X,groups)



      
      
import pygmo

print(os)
      
class sphere_function:
    def __init__(self, dim):
        self.dim = dim
    def fitness(self, x):
        return [sum(x*x)]


    def get_bounds(self):        
        up_list = [-1] * self.dim
        do_list = [1] * self.dim        
        return (do)list, up_list)

    
    
    def get_name(self): return "Sphere Function"
    def get_extra_info(self): return "\tDimensions: " + str(self.dim)




import scipy
import pygmo as pg
prob = pg.problem(sphere_function(2))
      
      
algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
pop = pg.population(prob,10)
pop = algo.evolve(pop)
print(pop.champion_f) 



pga = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))


pg.bee_colony(gen = 20, limit = 20)

pga = pg.de(gen=1, F=0.8, CR=0.9, variant=2, ftol=1e-06, xtol=1e-06,)


algo = pg.algorithm(pga)
pop = pg.population(prob,10)
pop = algo.evolve(pop)
print(pop.champion_f) 




# For the simple evolutionary startegy we also need ad hoc code
logs = []
algo = pg.algorithm(pg.sea(gen = 10000))
algo.set_verbosity(100)
pop = pg.population(prob, 1)
pop = algo.evolve(pop) 


logs = algo.extract(pg.sea).get_log() 
plt.plot([l[1] for l in logs],[l[2]-418.9829*20 for l in logs], label = algo.get_name()) 

# We then add details to the plot
plt.legend() 
plt.yticks([-8000,-7000,-6000,-5000,-4000,-3000,-2000,-1000,0]) 
plt.grid() 




import scipy
import pygmo as pg
prob = pg.problem(sphere_function(2))
      
      
algo = pg.algorithm(pg.bee_colony(gen = 20, limit = 20))
pop = pg.population(prob,10)
pop = algo.evolve(pop)
print(pop.champion_f) 



pga = pg.algorithm(pg.scipy_optimize(method="L-BFGS-B"))


pg.bee_colony(gen = 20, limit = 20)

pga = pg.de(gen=1, F=0.8, CR=0.9, variant=2, ftol=1e-06, xtol=1e-06,)


algo = pg.algorithm(pga)
pop = pg.population(prob,10)
pop = algo.evolve(pop)
print(pop.champion_f) 



dfref = df

df.columns
"""

Index(['time_key', 'shop_id', 'l1_genre_id', 'l2_genre_id', 'item_id',
       'units_std', 'units_sum', 'units_median', 'units_max', 'units_min',
       'gms_std', 'gms_sum', 'gms_mean', 'gms_max', 'gms_min', 'price_max',
       'price_median', 'price_min', 'price_std', 'cost_median', 'cost_min',
       'cost_max', 'price_median_log', 'units_sum_log', 'price_pct',
       'units_pct_ref', 'price_median_d1', 'units_median_d1', 'weekday',
       'pweekday', 'units_sum_season', 'units_pct'],
      dtype='object')


"""


dd['item_d'][  ]
    
    


####################################################################################################      
####################################################################################################
df = pd_read_file2( dir_pred  + f"/prod/20200801/output/*shopid*" )

cols_agg = ['shop_id']
df = df[ cols_agg +  [ 'date',  'order_id_s', 'order_id_s_pred',  ]]
# df = df.set_index("date")

df = df.rename(columns= { 'order_id_s' :f'units_sum actual {shop_id} '    ,   'order_id_s_pred':  f'units_sum pred {shop_id} ' })

#df['key'] = df['shop_id'].apply(lambda x : f"graphpred_{int(x)}_0_0" )

df['date2'] = df['date'].apply(lambda x :  x.strftime('%Y-%m-%d')  ) 

dd = {}
for x in df.columns :
   if x not in [ 'key', 'date', 'shop_id' ] : 
     dd[x] =  df[[ 'date2', x  ]].values.tolist()


df = df.fillna(value=None)

key_name =  f"graphpred_{int(shop_id)}_0_0"

import simplejson as json
json.dump(dd, open(  dir_pred +  f"/export/{key_name}.json" , mode="w" ), ignore_nan=True)















df = pd_read_file2( dir_porder + "/*y-porder*")


df = df[['time_key', 'shop_id', 'order_id_s']].drop_duplicates( ['time_key', 'shop_id' ]  )


for ii in list(df.shop_id.unique()) :
  dfi = df[ df.shop_id == ii ].set_index("time_key").plot( y= "order_id_s", title = f"shop_id {ii}" , figsize=(12,5))



df = df.sort_values(['shop_id', 'time_key'])


pd_to_file( df, dir_train + "/shop/y-shop_units.parquet"  )




import fbprophet
######################################################################

df = pd_read_file2(  dir_train + "/shop/y-shop_units.parquet"  )





from atspy import AutomatedModel


### Preprocess of dates
import datetime
df['date'] = df['time_key'].apply(lambda t :  to_datetime(from_timekey(t)) )
# df['date'] =  pd.to_datetime(df['date'])
ytarget = "order_id_s"


dfi = pd.DataFrame()
dfi['date'] = pd.date_range("20180101", "20200716", freq="D")

dfi = dfi.join(  df[df.shop_id == 16 ].set_index("date"), on ="date", how="left"   )
dfi = dfi.set_index('date')

dfi[ytarget] = dfi[ytarget].interpolate(  )
dfi[ytarget].isna().sum()

   
###### 
model_list = [ "Prophet"  , "NBEATS" ]    #  "NBEATS",
am = AutomatedModel(df = dfi['order_id_s'] , model_list=model_list,forecast_len=20 ,  )

ypred, metric = am.forecast_insample()
print(metric)

ypred_out = am.forecast_outsample()
ypred.plot(figsize=(13,7))


all_ensemble_in, all_ensemble_out, all_perf = am.ensemble(ypred, ypred_out)
print(all_perf)



"""

Prophet 
 1398.600425  ...  17608.170529
 
             Target         HWAAS         HWAMS       Prophet
rmse      0.000000  3.091837e+03  3.091765e+03  1.645984e+03
mse       0.000000  9.559456e+06  9.559012e+06  2.709263e+06
mean  17152.420259  1.633837e+04  1.633864e+04  1.741202e+04


                                                           rmse  ...          mean
Prophet                                             1397.385531  ...  17618.532997
Prophet__X__ensemble_lgb                            1431.726150  ...  17600.892574
Prophet__X__ensemble_lgb__X__Prophet_NBEATS         1448.165118  ...  17391.360317
Prophet__X__ensemble_lgb__X__Prophet_NBEATS__X_...  1528.526017  ...  17239.675968
ensemble_lgb                                        1579.465495  ...  17583.252150
Prophet_NBEATS                                      1600.473246  ...  16972.295804
ensemble_ts                                         1837.011061  ...  17698.240276
NBEATS                                              2026.702082  ...  16326.059570

"""


model = am.models_dict_out["Prophet"]

save(model, dir_model + "/model.pkl")



import logging
from fbprophet.models import PyStanBackend

logger = logging.getLogger('fbprophet')
b = PyStanBackend(logger=logger)
b.logger = None


1640 / 15000.0


## Other models to try, add as many as you like; note ARIMA is slow:    
    ["ARIMA","Gluonts","Prophet","NBEATS", "TATS", "TBATS1", "TBATP1", "TBATS2"]


ypred[['Target', 'Prophet' ]].plot()


"""
TBATS2 - TBATS1 With Two Seasonal Periods


utomated Models
ARIMA - Automated ARIMA Modelling
Prophet - Modeling Multiple Seasonality With Linear or Non-linear Growth
HWAAS - Exponential Smoothing With Additive Trend and Additive Seasonality
HWAMS - Exponential Smoothing with Additive Trend and Multiplicative Seasonality
NBEATS - Neural basis expansion analysis (now fixed at 20 Epochs)
Gluonts - RNN-based Model (now fixed at 20 Epochs)
TATS - Seasonal and Trend no Box Cox
TBAT - Trend and Box Cox
TBATS1 - Trend, Seasonal (one), and Box Cox
TBATP1 - TBATS1 but Seasonal Inference is Hardcoded by Periodicity
TBATS2 - TBATS1 With Two Seasonal Periods

"""

am.models_dict_in
am.models_dict_out


# read the Prophet model object
with open(pkl_path, 'rb') as f:
    m = pickle.load(f)

fcast = pd.read_pickle("path/to/data/forecast.pkl")




sess = Session("shop_daily")
sess.save(globals())



to_timekey("20200707")



#############################################
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_yearly, plot_weekly
from fbprophet.plot import plot_components


# Python
playoffs = pd.DataFrame({
  'holiday': 'playoff',
  'ds': pd.to_datetime(['2008-01-13', '2009-01-03', '2010-01-16',
                        '2010-01-24', '2010-02-07', '2011-01-08',
                        '2013-01-12', '2014-01-12', '2014-01-19',
                        '2014-02-02', '2015-01-11', '2016-01-17',
                        '2016-01-24', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
superbowls = pd.DataFrame({
  'holiday': 'superbowl',
  'ds': pd.to_datetime(['2010-02-07', '2014-02-02', '2016-02-07']),
  'lower_window': 0,
  'upper_window': 1,
})
holidays = pd.concat((playoffs, superbowls))


df_holidays


from offline.amodel_preprocess import generate_X_date_item_future


df_date = generate_X_date_item_future(start='20180101', end='20200716')

df_holiday = pd.DataFrame({ 'holiday': 'all',    'lower_window': 0,  'upper_window': 1,
  'ds' : df_date[df_date.isholiday == 1]['date']                           
})
    
    
df_yearend = pd.DataFrame({ 'holiday': 'yearend',    'lower_window': 0,  'upper_window': 1,
  'ds' :  list(pd.date_range("20181227", "20190105", freq="D")) +
          list(pd.date_range("20191227", "20200105", freq="D")) +
          list(pd.date_range("20191227", "20210105", freq="D")) 
})
df_holiday = pd.concat(( df_holiday, df_yearend))
    


df_event = pd.DataFrame({ 'holiday': 'event',    'lower_window': 0,  'upper_window': 1,
  'ds' :  pd.to_datetime([ "20180101", "20190101", "20200101",  ]) })
df_holiday = pd.concat(( df_holiday, df_event))


    
dfi = pd.DataFrame()
dfi = copy.deepcopy(df_date)
# dfi['date'] = pd.date_range("20180101", "20200716", freq="D")

dfi = dfi.join(  df[df.shop_id == 16 ].set_index("date"), on ="date", how="left"  , rsuffix='2' )
dfi = dfi.set_index('date')
dfi[ytarget] = dfi[ytarget].interpolate(  )
dfi[ytarget].isna().sum()
dfi = dfi.reset_index()
dfi = dfi.rename(columns={ 'date' :'ds', 'order_id_s' : 'y' })

    
m = Prophet(growth='linear',
            holidays= df_holiday,
            seasonality_mode= 'multiplicative',  # 'additive',
            daily_seasonality='auto',
           
            changepoints=None,
            n_changepoints=25,
            changepoint_range=0.8,
            yearly_seasonality= 10,
            weekly_seasonality= 5,
 

            seasonality_prior_scale=15.0,
            holidays_prior_scale=15.0,
            changepoint_prior_scale=0.05,
            mcmc_samples=0,
            interval_width=0.80,
            uncertainty_samples=1000,
            stan_backend=None
    )


def covariate_01(ds):
    date = pd.to_datetime(ds)
    if date.weekday() > 12 and (date.month > 8 or date.month < 2):
        return 1
    else:
        return 0

m.add_regressor('covariate_01')
dfi['covariate_01'] = dfi['ds'].apply(covariate_01)
covariate_list = ['covariate_01']

m.add_seasonality(name='monthly', period=30.5, fourier_order=5)


m.fit(dfi[['ds', 'y'] + covariate_list ].iloc[:-100, :])

df_future = m.make_future_dataframe(periods=100)    
df_future['covariate_01'] = df_future['ds'].apply(covariate_01)

dfp = m.predict(df_future)
dfp = dfp.join( dfi.set_index('ds'), on ='ds', how='left', rsuffix="2" ).set_index('ds')
dfp['ydiff'] = dfp['yhat']/dfp['y']   - 1.0
dfp['ydiff2'] = np.abs( dfp['ydiff'] )


#### Daily prediction  < 7%  on the shop
print( metric_mae(dfp.iloc[-800:, :]) )
dfp[['y', 'yhat']].iloc[-200:,:].plot()
pd_histo(dfp['ydiff'] , show=True)
####Test of residuals



fig = m.plot_components(dfp.reset_index())


#### Weekly Prediction    2% on the shop  (out of sample)
dfp2 = dfp.groupby('yearweek').agg({
          'y':  'sum',
          'yhat':'sum'
        })


dfp2[['y', 'yhat']].iloc[:20,:].plot()
print( metric_mae(dfp2.iloc[:, :]) )




dfp['ydiff_abs'] = np.abs( dfp['yhat'] / dfp['y'] -1   )
dfp = dfp.sort_values('ydiff_abs', ascending=0)




#### 10% Out of sample

forecast = forecast.set_index('ds')
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
fig1 = m.plot(forecast)
plot_components(m, forecast)





a = plot_yearly(m)
a = plot_weekly(m)




#### bug in Prophet
m.stan_backend.logger = None
save(m, dir_model + "/model.pkl")





    

mpars0 = to_dict(
    changepoints=None,
    n_changepoints=25,
    changepoint_range=0.8,
    yearly_seasonality= 10,
    weekly_seasonality= 5,
    seasonality_prior_scale=15.0,
    holidays_prior_scale=15.0,
    changepoint_prior_scale=0.05,
    mcmc_samples=0,
    interval_width=0.80,
    uncertainty_samples=1000,
    stan_backend=None        )



_, mout = shopid_train_predict(df[ df.shop_id == 16].iloc[:100,:], ytarget="order_id_s", 
                          n_cutoff = 100, n_future = 0,         
                          mpars = mpars, return_val="metric",
                          covariate_list = [],  model_path= None , verbose=False)

   from offline.config.prod.shopid_model import shopid_train_predict

def objective(trial) :
   """ weight_decay = trial.suggest_loguniform('weight_decay', 1e-10, 1e-3)
    optimizer = trial.suggest_categorical('optimizer', ['MomentumSGD', 'Adam']) # Categorical parameter
    num_layers = trial.suggest_int('num_layers', 1, 3)      # Int parameter
    dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 1.0)      # Uniform parameter
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)      # Loguniform parameter
    drop_path_rate = trial.suggest_discrete_uniform('drop_path_rate', 0.0, 1.0, 0.1) # Discrete-uniform parameter
   """
   mpars = {
     #'n_changepoints': 25,
     #'changepoint_range': 0.8,
     'yearly_seasonality': trial.suggest_int('yearly_seasonality', 1, 50) ,
     'weekly_seasonality': trial.suggest_int('weekly_seasonality', 1, 50) ,
     'seasonality_prior_scale': trial.suggest_uniform('seasonality_prior_scale', 0.0, 100.0) , 
     'holidays_prior_scale':    trial.suggest_uniform('holidays_prior_scale', 0.0, 100.0) ,
     'changepoint_prior_scale': trial.suggest_uniform('changepoint_prior_scale', 0.0, 1.0)   ,
     #'mcmc_samples': 0,
     #'interval_width': 0.8,
     #'uncertainty_samples': 1000,
     #'stan_backend': None
     }
   
   _, mout = shopid_train_predict(df[ df.shop_id == 16].iloc[:,:], ytarget="order_id_s", 
                          n_cutoff = 100, n_future = 0,         
                          mpars = mpars, return_val="metric",
                          covariate_list = [],  model_path= None , verbose=False)
   return mout['mae'] 



import optuna
from optuna.samplers import TPESampler
try :
   study = optuna.load_study(study_name='shop16', storage='sqlite:///optuna_db.db',)
except :        
   study = optuna.create_study(study_name='shop16', storage='sqlite:///optuna_db.db',
                            pruner=optuna.pruners.MedianPruner(), sampler=TPESampler())
  
study.optimize(objective, n_trials = 10)









import optuna

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2

if __name__ == '__main__':
    study = optuna.load_study(study_name='distributed-example', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=100)









_, _= Forecast_daily_shopid(df[ df.shop_id == 16].iloc[:100,:], ytarget="order_id_s", 
                          n_cutoff = 10, n_future = 0,         
                          covariate_list = [],  model_path= None , verbose=False)



for shop_id in    [ 11 ] :
    model, dfp = Forecast_daily_shopid(df[ df.shop_id == shop_id], ytarget="order_id_s", 
                              n_cutoff = 0, n_future = 28,         
                              covariate_list = [],  
                              model_path= dir_model + f"/shopid_daily_20200717/shop_{shop_id}/",  
                              pred_path= dir_pred + f"/prod/output/shop_{shop_id}.parquet",  )
    
    

    
    dfp.iloc[-100:,:][['y', 'yhat']].plot()



import pdb

pdb.set_trace()






#####################################################################################################
#####################################################################################################
# Python
import itertools
import numpy as np
import pandas as pd
from fbprophet.diagnostics import cross_validation, performance_metrics


param_grid = {  
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    'holidays_prior_scale': [0.1, 1.0, 5.0, 20.0],
    
}


# Generate all combinations of parameters
all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
rmses = []  # Store the RMSEs for each params here

# Use cross validation to evaluate all parameters
for ii, params in enumerate(all_params):
    m = Prophet(**params)  # Fit model with given params
    
    m.fit(dfi[['ds', 'y'] + covariate_list ].iloc[:-100, :])
    
    df_cv = cross_validation(m, initial='370 days',  period='180 days', horizon='120 days',
                             )
    df_p = performance_metrics(df_cv, rolling_window=1)
    rmses.append(df_p['rmse'].values[0])
    log(ii, rmses[-1] )



# Find the best parameters
tuning_results = pd.DataFrame(all_params)
tuning_results['rmse'] = rmses
print(tuning_results)





# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)











### Global Model  metrics
metrics_cols = [ "key", "model_name", "model_sub", "Xinfo"  ,"ycol", "metric_name", "metric_val", 
                 "mean", "max"  ]
metrics      = { x:[] for x in  metrics_cols  }

metrics 





dir2 = root + "/data/train/itemid/"
dfX2 =  pd_read_file2( dir2 + "*item*2020*" )
dfX2 = dfX2[ (dfX2.l2_genre_id == 15036) & (   dfX2.shop_id == 11 ) ]


527 vs 2400







len( df['time_key'].unique() )
len( df['item_id'].unique() )







import os, glob

path = root + "/data/pos/*2020011*"
file_list = [   f for f in glob.glob( path )  ]


dfx = pd_read_parallel(file_list,pd_reader , n_pool=4, verbose=False)






cats = ['a', 'b', 'c','-1']
dfz = pd.DataFrame({'cat': ['a', 'b', 'a', 'd' ]})

dfz1 = pd.get_dummies(dfz, prefix='', prefix_sep='')
dfz1 = dfz1.T.reindex(cats).T.fillna(0)
dfz1


map_new = [  t for t in dfz['cat'].unique()) if t not in cats ]


df['cat'] = df['cat'].apply(lambda x : map_dict[x] if x in col_set else -1 )



def my_funcs(df):
    data = {}
    data['product'] = '/'.join(df['product'].tolist())
    data['units'] = df.units.sum()
    data['price'] = np.average(df['price'], weights=df['units'])
    data['transaction_dt'] = df['transaction_dt'].iloc[0]
    data['window_start_time'] = df['period'].iloc[0].start_time
    data['window_end_time'] = df['period'].iloc[0].end_time
    return pd.Series(data, index=['transaction_dt', 'product', 'price','units', 
                                  'window_start_time', 'window_end_time'])

df.groupby(['customer_id', 'period']).apply(my_funcs).reset_index( drop=True)


def pd_trim(dfi):
    return dfi.iloc[:im,:]



dfe = dfXy = pd.read_pickle( root +"/model/itemid/16-RandomForestRegressor_prod_v5/16_1_160_16036_0/df_error.pkl" )








def train_split_time(df, test_period = 40, cols=None , coltime ="time_key", minsize=5) :  
   cols = list(df.columns) if cols is None else cols 
   if coltime is not None : df   = df.sort_values(   coltime  , ascending=1 ) 
   #imax = len(df) - test_period   
   colkey = [ t for t in cols if t not in [coltime] ]
   print(colkey)
   imax = test_period
   df1  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[:max(minsize, len(dfi) -imax), :] ).reset_index(colkey, drop=True).reset_index(drop=True)
   df2  = df.groupby( colkey ).apply(lambda dfi : dfi.iloc[max(minsize,  len(dfi) -imax):, :] ).reset_index(colkey, drop=True).reset_index(drop=True)  
   return df1, df2
   # Xtrain, Xtest = df1.values, df2.values
   # return  Xtrain, Xtest


dfxy["item_id"]values[0]


dfXy.groupby("item_id").iloc[0,:]


dfXy = pd.read_pickle( train_path +"/dfXy.pkl" )

########## 
df1, df2 = train_split_time(dfXy, test_period=40, coltime = "time_key") 


agg_level_data  = [  "shop_id", "dept_id", "l1_genre_id", "l2_genre_id", "item_id"  ] 

df1, df2 = train_split_time(dfXy, test_period=10, cols= agg_level_data ,coltime = "time_key",) 



########## 
df1, df2 = train_split_time(dfX2, test_period=40, coltime = None ) 
 



 
df =df.iloc[:10000,:]


df.columns










dfnew.columns


df = dfy0.columns





########################## Params, data for model training  #########################################
# key_level = ["l1_genre_id", "l2_genre_id" ]
# keyref    = [  "time_key" ] + key_level 

key_level  =  [   "l1_genre_id", "l2_genre_id", "dept_id"   ]
keyref     = [  "time_key" ] + key_level 
ytargetbb  = "porder_sd2"
agg_level  =  "shop_dept"
train_path = "/data/train/shop_dept/"


shop_id    = [11, 13, 14, 16, 17, 18, 21, 22, 50 ]


### Get All keys
key_all = pd.read_pickle(root + "/data/meta/key_all.pkl")
key_all = key_all[ key_all["shop_id"] == shop_id]
key_all = key_all[ key_level  ].drop_duplicates().values.tolist()




### Global Model  metrics
metrics_cols = [ "key", "model_name", "model_sub", "Xinfo"  ,"ycol", "metric_name", "metric_val", 
                 "mean", "id"  ]

metrics = pd.DataFrame(metrics)
metrics.to_pickle( root + f"/data/model/metrics_global.pkl" )
metrics.to_csv( root + f"/data/model/metrics_global.csv" , index=False)



metrics['metric_val'] = metrics['metric_val'].astype("float")
metrics2 = pd.pivot_shople(metrics, values="metric_val", columns="model_name", index="model_sub",
                          aggfunc = "mean")
metrics2.to_pickle( root + f"/data/model/metrics2_global.pkl" )




####################################################################################################
####################################################################################################
import glob

glob.glob( *")

path = root + "/data/model/shop_dept/11-RandomForestRegressor__l1_shop_001/"
file_list = [   path + f for f in os.listdir( path )  ]


for fp in file_list :
  copyfile( path + fp +"/ ,   )



from shutil import copyfile
copyfile(src, dst)

import zlocal
path0 = zlocal.root + "/data/model/shop_dept/"

path_i = path + "/"



def pd_show_file(path="*y-porder_2020* "):
  df = pd_read_file( path , nrows=10 , verbose=1 )
  print(df.dtypes)
  

pd_show_file(path= root + "/data/train/itemid/*y-porder_2020*" )

pos_dir   = root + "/data/pos/"



cols= ['time_key', "item_id" ]
df = pd_read_file( pos_dir + "/*20200611*", 
                   cols= cols,
                   verbose=1) 
        
        
len( df['class_id'].unique() )


train_dir = root + "/data/train/"
cols= ['time_key', "item_id" ]
df = pd_read_file( train_dir + "/test/*X-date*", 
                    verbose=1) 



df = pd_read_file( train_dir + "/itemid/*X-merge-item-date.pkl*", 
                    verbose=1) 


df.dtypes


df

glob.




def generate_X_item(df, prefix_col ="" ):
    from offline.util import is_holiday
    keyref  =  ['time_key', "item_id" ]
    coldate =  'order_date'    
    df = df.drop_duplicates(keyref)       
    return df
    
    
    df['date'] =  pd.to_datetime( df[coldate] )  
                  
    ############# dates 
    df['year']          = df['date'].apply( lambda x : x.year   )
    df['month']         = df['date'].apply( lambda x : x.month   )
    df['day']           = df['date'].apply( lambda x : x.day   )
    df['weekday']       = df['date'].apply( lambda x : x.weekday()   )
    df['weekmonth']     = df['date'].apply( lambda x : weekmonth(x)   )        
    df['weekyeariso']   = df['date'].apply( lambda x : x.isocalendar()[1]   )
    df['weekyear2']     = df['date'].apply( lambda x : weekyear2( x )  )
    df['quarter']       = df.apply( lambda x :  int( x['month'] / 4.0) + 1 , axis=1  )
    
    df['yearweek']      = df.apply(  lambda x :  merge1(  x['year']  , x['weekyeariso'] )  , axis=1  )
    df['yearmonth']     = df.apply( lambda x : merge1( x['year'] ,  x['month'] )         , axis=1  )
    df['yearquarter']   = df.apply( lambda x : merge1( x['year'] ,  x['quarter'] )         , axis=1  )

    df['isholiday']     = is_holiday( df['date'].values )
    
    exclude = [ 'date', keyref, coldate]
    df.columns = [  prefix_col + x if not x in exclude else x for x in df.columns]
                    
    return df






def to_json_highcharts(df, cols, coldate, fpath,  verbose=False):
    import os, pandas as pd
    dd = {'meta' : {},  'data' : {} }
    for x in cols :
         dd['data'][x] =  df[[ coldate, x  ]].values.tolist()
         
    import simplejson as json   ### for NA ---> null json
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    return json.dumps(dd, open(  fpath , mode="w" ), ignore_nan=True)   


################ UI 
import os, pandas as pd
    
if path[0] = "/"  :
       full_path = path       
else :
    full_path = "a/adigcb301/ipsvols05/offline/test/pred/export/" + path




for fname in flist :
  if ".parquet" in fname:      
    df = pd.read_parquet(fname)
    if 'time_key' not in df.columns :
        df['time_key'] = 17400 + np.arange(0,len(df), 1)
        
    df['datex'] = [ to_timekey(t) for t in df['time_key'].values ]
        
    cols   = ui_get_column_name_list()   ### units_g0, units_go_pred    
    myjson = to_json_highcharts(df, cols, 'datex',  verbose=False)

    ### Send to the UI





  
"""

dept_id             int64
item_id             int64
l1_genre_id         int64
l2_genre_id         int64
order_id_g0         int64
order_id_s          int64
order_id_s1         int64
order_id_s2         int64
order_id_sd         int64
order_id_sd1        int64
order_id_sd2        int64
order_id_t          int64
porder_abs_g0     float64
porder_abs_s      float64
porder_abs_s1     float64
porder_abs_s2     float64
porder_abs_sd     float64
porder_abs_sd1    float64
porder_abs_sd2    float64
porder_abs_t      float64
porder_g0         float64
porder_s1         float64
porder_s2         float64
porder_sd1        float64
porder_sd2        float64
shop_id             int64
time_key            int64

"""


root2 = root + "/data/model/shop_dept/"


dfe = pd.read_pickle(  file_list[0] + "/df_error.pkl" )



pathi= path0 + "/17-RandomForestRegressor__l2_shop_001/17_0_0_41108_0"


dfe2 = pd.read_pickle(  pathi + "/df_error.pkl" )



dfe2[[ "porder_s2",  "porder_s2_pred"  ]].iloc[-40:,:].plot()
dfe2[ "err_std"  ] = dfe2[ "porder_s2_diff"  ]  / dfe2[ "porder_s2"  ] 
dfe2[ "err_std"  ].iloc[-200:].hist()




dfe2[[ "porder_s2",  "porder_s2_pred"  ]].iloc[:,:].plot()



def generate_report(path_model):
   path_model = path_model + "/"
   to_path    = path_model + "/metrics/"
   os.makedirs(to_path, exist_ok=True)
   path_list  = [  f for f in os.listdir( path_model )  ]
   # print(path_list)
   for ii, fname in enumerate(path_list) :  
    try: 
      print(fname)
      dfe2 = pd.read_pickle( path_model + fname + "/df_error.pkl" )   
      df_out = dfe2.iloc[-40:,:2] 
      
      df_out.plot()
      plt.savefig(to_path + f"/plot_out_{fname}.png"); plt.close()

      dfe2.iloc[:-40,:2].plot()      
      plt.savefig(to_path + f"/plot_in_{fname}.png"); plt.close()

      dfe2.iloc[:-40,2].hist()
      plt.savefig(to_path + f"/dist_err_out_{fname}.png"); plt.close()

      dfe2['model_sub'] = fname
      dfe2.to_csv(to_path + f"/df_error_all.csv", mode="a" if ii > 0 else "w", index=True,
                  header= 0 if ii > 0 else 1 )
      
      
      dfe2.iloc[-40:,:].to_csv(to_path + f"/df_error_out.csv", mode="a" if ii > 0 else "w",
               header= 0 if ii > 0 else 1   , index=True)
      
    except :
      print("Error",fname) 




shop_list = [11]


shop_list   = [11, 13, 14, 16, 17, 18, 21, 22, 50 ]

model_group  = "shop_dept"
model_name   = "RandomForestRegressor"      # "RandomForestRegressor"  # "RidgeCV"  # 
model_tag    = "_l1shop_ok"  # "_l2_shop_001"



for shop_id in shop_list :
  pth =    root_model + f"/{model_group}/{shop_id}-{model_name}_{model_tag}/"
  print(pth)
  generate_report( pth )





generate_report( root2 + "/11-RandomForestRegressor__l2shop_ok/")

generate_report( root2 + "/11-RidgeCV__l2shop_ok/")



for t in [ 13, 14, 16, 17, 18, 21, 22, 50   ] :
  generate_report( root2 + f"/{t}-RidgeCV__l2shop_ok/")


for t in [ 11, 13, 14, 16, 17, 18, 21, 22, 50   ] :
  generate_report( root2 + f"/{t}-RandomForestRegressor__l2shop_ok/")






def histo(dfi, path_save=None, nbin=20.0) :
  q0 = dfi.quantile(0.05)
  q1 = dfi.quantile(0.95)
  dfi.hist( bins=np.arange( q0, q1,  (  q1 - q0 ) /nbin  ) )
  if path_save is not None : plt.savefig( path_save );   
  plt.show(); 
  plt.close()



def generate_metrics( path  , cola="porder_s2") :
    to_path = path +"/metrics/"
    dfea    = pd.read_csv(to_path + "/df_error_out.csv" )
    cols    = list(dfea.columns)
    print(cols)    
    # cola = cols[-3]        # 'porder_s2'
    colb = cola + "_pred"  #'porder_s2_pred'
    col0 = cola + "_diff"  # 'porder_s2_diff'

    dfs = dfea.groupby( "l2_genre_id" ).agg({
        cola : "sum",
        colb : "sum"
        }).reset_index()
    
    dfs['diff_40d'] = dfs[cola]  -  dfs[colb] 
    mae  = np.mean(np.abs( dfs['diff_40d'] ))
    med  = dfs[cola].median()
    maep = mae / med
    print( maep  )
    
    histo( np.abs(dfs['diff_40d']), to_path +"/dist_40d.png" , 30) 
    histo( np.abs(dfea[ col0 ]) , to_path +"/dist_1d.png", 30) 
    
        
    dd = { "maep_40d" :maep, "mae_40d" :mae  }
    json.dump( dd, open( to_path +"/metrics.json" , mode="w" ))
    return dfs, dfea


    
    
root2 = root + "/data/model/shop_dept/"

dfs2, dfe2 = generate_metrics(  path = root2 + "/11-RandomForestRegressor__l2shop_ok/"  ) 


dfs2, dfe1 = generate_metrics(  path = root2 + "/11-RidgeCV__l2shop_ok/"  ) 



root2 = root + "/data/model/shop_dept/"
flist = os.listdir(  root2)
for fp in flist :  
  try :  
    print(fp)  
    dfs2, dfe2 = generate_metrics(  path = root2 + fp  ) 
  except : pass
  






histo( np.abs(dfs['diff_40d']), "distri_rf.png" )




#####
20% Improvement 





0.000496460057697568 / 0.0007460661964188822


####################################################################################################
[  dfea[col0] != 0.0 ]

dfea['porder_s2_diff'].plot()
  






#### Manual Cross Check
date0 = 20200710
print("#### Daily Summary    ")
dfc = pd.read_parquet(root + f"/data/daily_summary/11/daily_summary_{date0}.parquet" )
print( dfc[dfc.l2_genre_id == 16046 ][[  'time_key', 'shop_id', 'order_count' ]] )

dfc.to_csv( root + "/data/check/check_daily_{date0}.csv" )



print("#### Pos RAW   ")
df2 = pd.read_parquet(root + f"/data/pos/2020/pos_{date0}.parquet" )
df2 = df2[ (df2.time_key ==18453) & (df2.shop_id == 11)  ]
df2 = df2[ df2.valid_flag == 1  ]

df3 = df3.drop_duplicates(['order_id', "l2_genre_id", "cashier_id" ])
print(len(df3[ df3.l2_genre_id == 16046   ]) )   ## 41

df3 = df2.drop_duplicates(['order_id', "l2_genre_id", "cashier_id", "item_id"])
print(len(df3[ df3.l2_genre_id == 16046   ]) )  ### 47

dfc.to_csv( root + "/data/check/check_rawpos_{date0}.csv" )



print("#### train data   ")
df4 = pd.read_pickle(root + "/data/train/shop_dept/y-porder_2020.pkl" )
df4 = df4[ (df4.time_key ==18453) & (df4.shop_id == 11)  ]






dir0 = root2 + f"/11-RandomForestRegressor__l2shop_ok/11_0_0_16021_0/df_error.pkl"


df4 = pd.read_pickle( dir0 )
df4 = df4.reset_index()
df4 = df4[ (df4.time_key ==18453) & (df4.shop_id == 11)  ]



C:\D\gitdev\codev15\zdata\test\data\model\shop_dept\11-RandomForestRegressor__l2shop_ok\11_0_0_15001_0


dfd = pd_read_file(root + f"/data/train/shop_dept/*X-date*2019*", cols=None)



dfX = dfXy[colsX]


dfX["weekyeariso"].unique() 

array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 15, 17, 21, 22, 23,
       24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
       41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 14, 16, 18, 19, 20],


X2["weekyeariso"].unique() 




['order_id', "l2_genre_id", "cashier_id"]

(store,day,order_id, cashier_id) is unqiue



dfX0    =  pd_read_file(root + f"/data/train/shop_dept/*X-date*2019*", cols=None) 

dd = {}
for x in [ "day", "month", "quarter", "weekday", "weekmonth", "weekyear2", "weekyeariso",
             ]  :
    
  dd[x] =  list(dfX0[x].unique() )

weekyeariso = np.array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
       52])


[0, 1, 2, 3, 4, 5, 6],


map_dict = {'day': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 
 'month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
 'quarter': [1, 2, 3, 4],
 'weekday': [0, 1, 2, 3, 4, 5, 6],
 'weekmonth': [1, 2, 3, 4, 5, 6, -46],
 'weekyear2': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53], 
 'weekyeariso': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52]
 }


    




cats = ['a', 'b', 'c']
df = pd.DataFrame({'cat': ['a', 'b', 'a']})

coli = 'weekyeariso'
dummies = pd.get_dummies(dfX[coli].iloc[:200], prefix='', prefix_sep='')
dummies = dummies.T.reindex(map_dict[coli]).T.fillna(0)
dummies



len( dfX[coli].iloc[:20].unique() )


dfa = dfX[ colsX ].iloc[:2 ,: ]




## df1 = df
df = dfa


colind = list(df.index.names)


df2 = df.index

df.index = df2



dfX0    =  pd_read_file(root + f"/data/train/shop_dept/*X-date*2019*", cols=None) 

df = dfX0[ colsX ]


x ="weekday"

def pd_to_onehot(df, colnames, map_dict=None, verbose=1) :
 # df = df[colnames]   
 
 for x in colnames :
   try :   
    nunique = len( df[x].unique() )
    if verbose : print( x, df.shape , nunique, flush=True)
     
    if nunique > 0  :  
      try :
         df[x] = df[x].astype("int")
      except : pass
      # dfi = df
      df[x] = df[x].astype( pd.CategoricalDtype( categories = map_dict[x] ) )
      
      # pd.Categorical(df[x], categories=map_dict[x] )
      prefix = x 
      dfi =  pd.get_dummies(df[x], prefix= prefix, ).astype('int32') 
      
      #if map_dict is not None :
      #  dfind =  dfi.index  
      #  dfi.index = np.arange(0, len(dfi) )
      #  dfi = dfi.T.reindex(map_dict[x]).T.fillna(0)
      #  dfi.columns = [ prefix + "_" + str(x) for x in dfi.columns ]
      #  # dfi.index = dfind 
      
      df = pd.concat([df ,dfi],axis=1).drop( [x],axis=1)
      # coli =   [ x +'_' + str(t) for t in  lb.classes_ ] 
      # df = df.join( pd.DataFrame(vv,  columns= coli,   index=df.index) )
      # del df[x]
    else :
      lb = preprocessing.LabelBinarizer()  
      vv = lb.fit_transform(df[x])  
      df[x] = vv
   except Exception as e :
     print("error", x, e, )  
 return df




dfi.dtypes



dfii = pd_to_onehot(dfX0[colsX].iloc[:2,:], colsX, map_dict, verbose=0) 



train_df["country"].astype(CategoricalDType(["australia","germany","korea","russia","japan"]))




##### Check
dfc = pd.read_parquet(root + "/data/daily_summary/11/daily_summary_20200710.parquet" )


    
### check
d = pd.read_pickle(root + "/data/train/shop_dept/y-porder_2020.pkl" )
d = d[ (d.time_key ==18453) & (shop_id == 11)  ]



####
d2 = pd.read_parquet(root + "/data/pos/2020/pos_20200710.parquet" )

d2 = d2[ (d2.time_key ==18453) & (d2.shop_id == 11)  ]

d2.to_csv("check_11.csv")






C:\D\gitdev\codev15\zdata\test\data\



######
dfe2['date'] = pd.date_range(start='1/1/2018', periods=len(dfe2), freq='D')
dfi = dfe2.reset_index()
dfi = dfi[[ 'porder_s2']]
dfi = dfi.set_index("date")















dfi = dfi[ 'Target'] * 10.0

from atspy import AutomatedModel


model_list = [  "HWAAS", "HWAMS", "Prophet" ]
am = AutomatedModel(df = dfi , model_list=model_list,forecast_len=20 )


ypred, metric = am.forecast_insample()
print(metric)



ypred_out = am.forecast_outsample()


all_ensemble_in, all_ensemble_out, all_performance = am.ensemble(forecast_in, forecast_out)


## Other models to try, add as many as you like; note ARIMA is slow:    
    ["ARIMA","Gluonts","Prophet","NBEATS", "TATS", "TBATS1", "TBATP1", "TBATS2"]


ypred[['Target', 'Prophet' ]].plot()




am.models_dict_in
am.models_dict_out


# read the Prophet model object
with open(pkl_path, 'rb') as f:
    m = pickle.load(f)

fcast = pd.read_pickle("path/to/data/forecast.pkl")



forecast_out



import pandas as pd
df = pd.read_csv("https://raw.githubusercontent.com/firmai/random-assets-two/master/ts/monthly-beer-australia.csv")
df.Month = pd.to_datetime(df.Month)
df = df.set_index("Month"); df



"HWAMS","HWAAS",

"TBAT",

ARIMA - Automated ARIMA Modelling
Prophet - Modeling Multiple Seasonality With Linear or Non-linear Growth
HWAAS - Exponential Smoothing With Additive Trend and Additive Seasonality
HWAMS - Exponential Smoothing with Additive Trend and Multiplicative Seasonality
NBEATS - Neural basis expansion analysis (now fixed at 20 Epochs)
Gluonts - RNN-based Model (now fixed at 20 Epochs)
TATS - Seasonal and Trend no Box Cox
TBAT - Trend and Box Cox
TBATS1 - Trend, Seasonal (one), and Box Cox
TBATP1 - TBATS1 but Seasonal Inference is Hardcoded by Periodicity
TBATS2 - TBATS1 With Two Seasonal Periods





# -*- coding: utf-8 -*-
"""
price optimization

"""

from collections import defaultdict
import numpy as np
import json
import pandas as pd
import time, glob, os

from scipy.optimize import minimize
from remote_runnable.utils import get_intdate
import matplotlib.pyplot as plt



from offline.model_item import ModelItem as MI
from offline.util import *




"""
  Max Score
     Score = g.GMS + (1-g).revenue

     GMS = Volume(price) . Price
     Revenue = GMS  - Volume.Price0
     
  Volume() = Porder_time x Volume_L2.
  
  
  
  Use Level2 as Prior of item_level.
  
     
  
  
  

     
"""

from zlocal import root, pos_dir, daily_dir, dir_price
from zlocal import *
pos_dir = root + "/data/pos/"
price_dir = root + "/data/price/"
os.makedirs(price_dir, exist_ok=True)
dir_price = root + "/data/price/"


df = pd_read_file( dir_porder +)


df = pd_read_file( dir_price + "/elastic_all_20200715.parquet" )


df =df[ df.shop_id.isin([16,17])]


pd_to_file(df, dir_price + "/elastic_all_20200715_shop_1617.csv" )



df1 = pd_read_file( dir_price + "/unit_16_df_error_out.csv" )

df1.columns

keyref = [  'shop_id', 'item_id', 'l1_genre_id', 'l2_genre_id',  ]
df1 = df1.groupby(keyref).agg({
        'order_id_g0' : 'sum' ,
         'order_id_g0_diff' : 'sum'
        }).reset_index()


df1['r1'] =  np.abs( df1['order_id_g0_diff'] / df1['order_id_g0']  )


df1.columns = [  "out_" +  x if x not in keyref else x for x in df1 ]



df2 = pd_read_file( dir_price + "/unit_17_df_error_out.csv" )

keyref = [  'shop_id', 'item_id', 'l1_genre_id', 'l2_genre_id',  ]
df2 = df2.groupby(keyref).agg({
        'order_id_g0' : 'sum' ,
         'order_id_g0_diff' : 'sum'
        }).reset_index()


df2['r1'] =  np.abs( df2['order_id_g0_diff'] / df2['order_id_g0']  )


df2.columns = [  "out_" +  x if x not in keyref else x for x in df2 ]

df3 = pd.concat(( df1, df2))

df = df.join(df3.set_index(keyref), on= keyref, how='left' )


pd_to_file(df, dir_meta  + "/meta_price_pred.parquet" )

from zlocal import dir_meta



df.columns


Index(['elastic_6m_itemid', 'elastic_6m_itemid_shopid', 'elastic_6m_l2genre',
       'elastic_6m_l2genre_shopid', 'gms_1d_max', 'gms_1d_mean', 'gms_1d_min',
       'gms_1d_std', 'gms_1d_sum', 'item_id', 'l1_genre_id', 'l2_genre_id',
       'nsample_itemid_shopid', 'price_median_count', 'price_median_max',
       'price_median_median', 'price_median_min', 'price_median_nunique',
       'price_median_std', 'shop_id', 'units_1d_max', 'units_1d_median',
       'units_1d_min', 'units_1d_std', 'units_1d_sum', 'out_order_id_g0',
       'out_order_id_g0_diff', 'out_r1'],

dfa = pd_filter(df, "out_r1<0.3,units_1d_min>1,price_median_nunique>1" )


dfa = pd_filter(df, "out_r1<0.4")




dfd = pd_read_file2(daily_dir + "/16/*202007*", n_pool=10 )



cols = [  'time_key', "shop_id", 'dept_id',  "l1_genre_id", "l2_genre_id",  'item_id', 'price', 'units'  ]
colsref  =  [  'time_key', "shop_id", 'dept_id',  "l1_genre_id", "l2_genre_id",  'item_id' ]


dfc = pd_read_file2(pos_dir + "/*202007*", cols=cols, shop_id=16, n_pool=10 )

dfc = dfc.groupby( colsref ).agg({'units' : 'sum', 'price' : {'mean', 'nunique'} })
dfc.columns = [ x[0]  if  x[1] =="_" else  x[0] +'_'+ x[1]  for x in  dfc.columns ]
dfc = dfc.rename(columns={"units_sum" : "units", "price_mean" : 'price' })
dfc = dfc.reset_index()



dfc[dfc.item_id ==2064001  ].set_index( ["time_key"] )['units'].plot()


dfc[dfc.item_id ==2064001  ].set_index( ["time_key"] )['price'].plot()


dfc[dfc.item_id ==2064001  ].set_index( ["time_key"] )['price_nunique'].plot()



from offline.aforecasts import Forecast_read


fct = Forecast_read()

fct.pred_load(  model_type="itemid_unit",  model_date="20200716", shop_id="16" )
fct.pos_load(" *202006*, *202007*", shop_id=16)

fct.pred_shoples()



fct.pos_plot(  item_id ) 
fct.pred_plot( item_id) 


item_id = 2124003
df2 = fct.compare_get( item_id )
df2[[ 'units', 'units_pred'   ]].plot()


fct.dfpred.columns




porder
unit
price






df1 = fct.pred_get()


df1[df1.item_id == 2064001 ]

" ".strip()



        if      
            
       = {
                'l2shop_porder' :  'l2_genre_id',
                'l1shop_porder' :  'l1_genre_id',
                'itemid_porder' :  'item_id',
                  :  ,                 
                }[model_type]
        
        mmap ={
               
                }







fct = Forecast_read()
fct.pos_load("*202007*", shop_id=16)
fct.pos_plot(item_id=2064001, mode="units") 


fct.pos_plot(item_id=2064001, mode="price") 

dfi = fct.pos_get()




fct = forecast_read()
fct.pos_load("*202007*", shop_id=16)

fct.pos_plot(item_id=2064001, mode="units") 
fct.pos_plot(item_id=2064001, mode="price") 

dfi = fct.pos_get()




















####################################################################################################
##### Load Data  ###################################################################################
cols2 = [  'time_key', "shop_id",  "l1_genre_id", "l2_genre_id",  'item_id', 'price'  ]
cols1  = [ 'time_key', "shop_id",  "l1_genre_id", "l2_genre_id",  'item_id', 
           'price', 'profit', 'units', 'gms' ]

df2 = pd_read_file2(path_glob= pos_dir + "*2020061*.*", ignore_index=True, cols=None,
                    verbose=1, nrows=-1, concat_sort=True, n_pool=4)
       


df = df2.groupby( cols2 ).agg({ 
         'profit' : "mean", "units" : "sum", "gms" : "sum", "l1_genre_id" : "count"
        })

df = df.rename(columns={"l1_genre_id" : "order_id"})
df = df.reset_index()


df.to_parquet(price_dir + "/pos/pos_2020.parquet")


df2.to_csv(price_dir + "/pos/pos_202006.csv", index=False)



###### 
Price =F(Volume)
Volume pattern :
    

    
##################  Statistics  #########################################################    
class item:
 import pandas as pd, numpy as np   
 def _init__(self, shop_id=None, item_id=None) :
     self.db = 
     
 def elastic(self, window="1m", date"", model="default"):
     
     dfi = df[df[] ==window   date = date dfi['model'] = model ].values
     return dfi['values'].values
     
 def forecast(start="", end="",  model="", model_date=""):
     
     return
    
i0.elastic("1m" , model="default")    




####################################################################################################
####################################################################################################
del df2;  gc.collect()


del df
gc.collect()

dfid = df[ df.item_id ==  5990044 ]


len( df['item_id'].unique() )



####################################################################################################
##### Variance of price ###########################################################################3
def generate_itemid_stats(price_dir="") :    
    col0 = [  'time_key', "shop_id",  "l1_genre_id", "l2_genre_id", "item_id", "units", "gms", "price" ]
    dfall = pd.DataFrame()
    ll =  [ "*202007*",  "*202006*", "*202005*", "*202004*"     ]
    for t in  ll :
      df2 = pd_read_file2(path_glob= pos_dir + t , ignore_index=True, cols=col0,
                        verbose=1, nrows=-1, concat_sort=True, n_pool=10)       
      dfall = pd.concat(( dfall, df2))
      del df2; gc.collect()
    dfall.to_parquet(price_dir + "/itemid_pos_2020.parquet")


    #### Stats per time_key, item_id   ############################################################
    df = cal_stats(dfall, tag="itemid_2020")
    for x in [ 'price_median', 'units_sum'  ] :
      coln =   x +"_log"
      df[ coln ]   = np.log(  df[ x ]   )
      df[ coln  ] = df[ coln  ].replace( -np.inf, -10.0 )
      df[ coln  ] = df[ coln"  ].replace(  np.inf, 10.0 )
      df[ coln  ] = df[ coln  ].fillna( 0.0)
    df = df.sort_values(['shop_id', 'item_id', 'time_key'])
    df.to_parquet(price_dir + "/itemid_daily_2020.parquet")



    ### Stats and Beta per item_id, acrooss years  ################################################
    def calc_beta(dfi):
        clf.fit( X=dfi['price_median_log'].values.reshape(-1, 1) , y=dfi['units_sum_log'].values    )
        # return len(dfi)
        return clf.coef_[0]
    
    colg0 = [  "shop_id",  "l1_genre_id", "l2_genre_id", "item_id", ]
    df2   = df.groupby( colg0  ).agg({ 'units_sum' : {'sum', "median",  "std", "min", "max" } ,
                                       'gms_sum'  :  {'sum', "mean",  "std", "min", "max"},
                                       'price_median' : {'median', "std", 'min', 'max', "count", "nunique"  },                                                                      
                                           }).reset_index()
    df2.columns = [ x[0]  if  x[1] =="_" else  x[0] +'_'+ x[1]  for x in  df2.columns ]
    df2.columns = [ s.strip()[:-1]  if s[-1] == "_" else s for s in df2.columns]    
    df2.columns = [ x.replace("_sum_", "_1d_") for x in df2.columns  ]
        
        
    df2 = df2.set_index([  "item_id" ])   
    df2["elastic_6m_itemid"] = df.groupby([ "item_id"]).apply(lambda dfi : calc_beta(dfi) )
    df2 = df2.reset_index()    
        
    
    colx = [ "shop_id", "item_id" ]
    df2 = df2.set_index(colx)   
    df2["elastic_6m_itemid_shopid"] = df.groupby( colx ).apply(lambda dfi : calc_beta(dfi) )
    df2 = df2.reset_index()    
    
    
    colx = [ "l2_genre_id" ]
    df2  = df2.set_index(colx)   
    df2["elastic_6m_l2genre"] = df.groupby( colx ).apply(lambda dfi : calc_beta(dfi) )
    df2 = df2.reset_index()  
    
    
    colx = [ "shop_id", "l2_genre_id" ]
    df2  = df2.set_index(colx)   
    df2["elastic_6m_l2genre_shopid"] = df.groupby( colx ).apply(lambda dfi : calc_beta(dfi) )
    df2 = df2.reset_index()  
    
        
    colx = [ "shop_id", "item_id" ]
    df2 = df2.set_index(colx)   
    df2["nsample_itemid_shopid"] = df.groupby( colx ).apply(lambda dfi : len(dfi) )
    df2 = df2.reset_index()    

    
    df2.to_parquet(price_dir + "/itemid_elastic_2020_20200715.parquet")
    
    
    
    
    


del df

df = df.set_index([ "shop_id", "item_id" ])
df = df.reset_index()



def pd_col_flatten(cols):
  cols2 = [ x[0]  if  x[1] =="_" else  x[0] +'_'+ x[1]  for x in  cols ]        
  cols2 = [ x[:-1]  if x[-1] == "_"  else x for x in cols   ]
  return cols2


colg0 = [  'time_key', "shop_id",  "l1_genre_id", "l2_genre_id", "item_id", ]
calc_stats( "itemid_2020", colg0  )


colg0 = [  'time_key', "shop_id",  "l1_genre_id", "l2_genre_id",  ]
calc_stats( "l2_2020", colg0  )


colg0 = [  'time_key', "shop_id",  "l1_genre_id",   ]
calc_stats( "l1_2020", colg0  )











dfi = df2[ (df2.shop_id == 16 ) & (df2.l1_genre_id != -1) ]



FYI, have put a sample shople, gathering raw elasticity calc
/a/adigcb301/ipsvols05/offline/test/price/elastic_all_20200715.parquet
( using basic  log-log reg. (daily Vol-Price)) :

elastic_6m_itemid         :  per  itemid (all shops)
elastic_6m_itemid_shopid  :  per item_id, shop_id

elastic_6m_l2genre        :   per  l2genre (all shops)
elastic_6m_l2genre_shopid  :  per l2genre, shop_id
nsample_itemid_genreid   : Nb of samples used for reg.

It can be batch generated (once a month )
Will try to add other elasti measures.



Delta_Volume = Deltaprice.exp( Beta)



item.volume(delta_price,)

Sure, it's just the regresss. beta of log-log :
    Log unit(Sum daily) =  beta . log Price(median daily)


Can add a very small code to get the variation : 
    item.unit_change(delta_price)




df2.columns



df2 = pd.read_parquet(price_dir + "/elastic_all_20200715.parquet")




df2["elastic_6m_l2genre"]


df2.to_parquet(price_dir + "/elastic_all_20200715.parquet")

dfi = df2[ (df2.shop_id == 11 ) & (  df2.l1_genre_id == 150 )]



gc.collect()


dfk  = df.iloc[:50000, :].groupby([ "item_id"]).apply(lambda dfi : calc_beta(dfi) )

        
dfk.max()



df = df.fillna(0)        
     




windows = 120

t1 = 
t0 =  t1 - windo

df['elastic_01'].apply( elastic_calc() )


from sklearn.linear_model import  RidgeCV 
clf = RidgeCV()


clf.fit(  )._coef[]



df.columns


dfi = df[ df.item_id  == 2201 ]





19           10952
20           11082
21           11231
22           11281
23           11282
24           11471
25           11601
26           11611
27           12352
28           12354



####################################################################################################
####  Price stats  #################################################################################
colkey = ["shop_id",  "l1_genre_id", "l2_genre_id",  'item_id' ]
keys   = df.groupby( colkey ).agg({ "units" : {"sum", "mean" },
                   'price' :  {"mean", "std", pd.Series.nunique},
                   'profit':  {"mean", "std"}
                   }).reset_index()
keys.columns = [ x[0]  if  x[1] =="_" else  x[0] +'_'+ x[1]  for x in  keys.columns ]
keys = keys.sort_values([ 'price_nunique',  'units_mean' ], ascending=[0, 0] )
"""
keys.columns =  ['shop_id', 'l1_genre_id', 'l2_genre_id', 'item_id', 'units_mean',
       'units_sum', 'price_nunique', 'price_mean', 'price_std', 'profit_mean',
       'profit_std']

"""
keys[ keys.l1_genre_id != -1  ] [[ 'item_id', 'price_nunique',  'units_mean' ]]



keys = keys.sort_values([ "units_mean", ascending=0)
keys[ 'price_nunique'].hist(bins=np.arange(0, 30,2))









#####################################################################################
####### Kaiso stat  #################################################################
keys[ keys.l1_genre_id != -1].to_csv( price_dir + "/price_stats_2020.csv" )
keys.to_parquet( price_dir + "/key_all_2020.parquet" )






#### Plot Graph
keys_top = keys[ (keys['units_sum'] > 100.0) & (keys["l2_genre_id"] != -1) ]
keys_select = keys_top[ colkey ].values.tolist()
for ii, x in enumerate(keys_select) :
   if ii < 00 : continue
   if ii > 100 : break
   log(ii,x) 
   (shop_id, l1, l2, item_id) = x
   
   tag = "_".join([ str(t) for t in x ])
   
   dfi = pd_filter(df, filter_dict = {colkey[i]: x[i] for i in range(4 )} )
   dfi['units'].hist(); plt.savefig( price_dir + f"/img/hist_units_{tag}.png"); plt.close()
   dfi['price'].hist(); plt.savefig( price_dir + f"/img/hist_price_{tag}.png"); plt.close()
   dfi[['price',  'units' ]].plot.scatter(x='price', y='units')
   plt.savefig( price_dir + f"/img/price_units_{tag}.png"); plt.close()
   
   
   
   
### Variance of price
Volume = F(Price1, ..... , PriceN)   
   



        
df.head(3)


No direct relatiosnp Volume vs Price.



""""


Variance of L2 genre.
Variance of item_id price


"""

##### Seasonality




dfid = dfid[dfid.price < 500.0 ]   
   
   

dfid['price'].hist(bins=np.arange(0, 200, 10))


dfid['profit'].hist()


dfid['units'].hist()



dfid[['profit',  'units' ]].plot.scatter(x='profit', y='units')


dfid[['price',  'units' ]].plot.scatter(x='price', y='units')






ataFrame.plot.scatter(x='C', y='D', title= "Scatter plot between two columns of a multi-column DataFrame");\\




scatter(x='A', y='B', title= "Scatter plot between two variables X and Y");

plot.show(block=True);

df.columns




    
cols1  = [ 'time_key', "shop_id",  "l1_genre_id", "l2_genre_id",  'item_id', 
           'price', 'profit', 'units', 'gms' ]
          

date_list = [ "*2020*", "*2019*",  "*2018*",  ]    # ["2019a", "2019b", "2018a", "2018b",   ]
  


    
    
Index(['cashier_id', 'check_filter', 'class_id', 'cost_type', 'dept_id',
       'discount_amount', 'discount_amount_by_rate', 'discount_price',
       'discount_price_by_rate', 'discount_rate', 'gms', 'gp_id', 'item_id',
       'item_name', 'jan', 'l1_genre_id', 'l2_genre_id', 'l3_genre_id',
       'label_id', 'label_id_b', 'nplu_gross_cost', 'order_date', 'order_id',
       'order_sequence_id', 'order_time', 'plu_gross_cost', 'price', 'profit',
       'shop_id', 'sku_id', 'tax_adjust', 'term_id', 'time_key', 'units',
       'valid_flag'],    




self = MI('config/config_v14.properties')
intdate = get_intdate()





interesting item to look at:   2504022




shop_id = 16 #17
shop_id2 = 18
item_id=3584004
window=150
cp, cc = self.get_current_pc(shop_id,item_id)
try:
  print (self.ui.price_hist_map[(shop_id,item_id)])
  #print (self.ui.camp_hist_map[(shop_id,item_id)])
  print ((cp-cc)/cp, cc/(1.-self.max_margin), cc / (.5))
  print (self.get_current_pc(shop_id,item_id))
  print (self.get_valid_prices(cp,cc))
  print (curr_price_map[(shop_id,item_id)])
  print (opt_price_map[(shop_id,item_id)])
except Exception as e:
    print (e)
    
    
dept, l1,l2 = self.ui.item_master_map[item_id][6:9]
self.shop_list=[13,16,17,18,11,14,21,22]
self.get_l2_map(dept,l1,l2)
#self.shop_list=[16,17]
dff = self.make_timeseries_df(item_id)
smap = self.make_prod_file(item_id,dff)
#self.insert_item_master(item_id,smap)

dfs = dff[dff.shop_id==shop_id]
print (self.params)
#self.shop_list=[13,16,17,18,11,14,21,22]
dff = self.make_timeseries_df(item_id)
dfs2 = dff[dff.shop_id==shop_id2]
plt.figure(figsize=(20,10))
plt.subplot(211)
#plt.plot(times,mi.shop_prices[shop],times,mi.shop_costs[shop])
plt.plot(dfs.time_key[-window:],dfs.price[-window:],'b',dfs.time_key[-window:],dfs.cost[-window:],
         dfs2.time_key[-window:],dfs2.price[-window:])
plt.subplot(212)
#plt.plot(times,Y[:],times,model)
#plt.plot(times,mi.shop_item_orders[shop],times,fcst_map[shop]) #*mi.shop_stock[shop])
plt.plot(dfs.time_key[-window:],dfs.orders[-window:],'b',dfs.time_key[-window:],dfs.model_orders[-window:],
        dfs2.time_key[-window:],dfs2.orders[-window:])
plt.show()









pmap_season = defaultdict(float)
pmap_orders = defaultdict(float)
cmap_season = defaultdict(float)
cmap_orders = defaultdict(float)
window= 150
lenp = len(self.item_vects[(shop_id,'price')])
cost = self.item_vects[(shop_id,'cost')][-1]
gamma = 0.33


for i in range(lenp-window,lenp):
    camp = self.item_vects[(shop_id,'campaign')][i]
    price = self.item_vects[(shop_id,'price')][i]
    season = self.item_vects[(shop_id,'seasonality')][i]
    orders = self.item_vects[(shop_id,'orders')][i]
    if camp <1:
        pmap_season[price] += season
        pmap_orders[price] += orders
    else:
        cmap_season[price] += season
        cmap_orders[price] += orders

for price, v in sorted(pmap_season.items()):
    porders = pmap_orders.get(price,0.)
    units = porders / v   ## Units at that price adjuste seasoablityu
    gms = units*price
    profit = gms - units*cost
    score = gms*gamma + profit*(1-gamma)
    print (price, pmap_orders.get(price,0.), v,units,gms,profit,score)


print ('camp')
for price, v in cmap_season.items():
    corders = cmap_orders.get(price,0.)
    units = corders / v
    gms = units*price
    profit = gms -units*cost
    score = gms*gamma + profit*(1-gamma)
    print (price, cmap_orders.get(price,0.), v,units,gms,profit,score)
    
    

Lag between Campaign and

Campaign definiutob


















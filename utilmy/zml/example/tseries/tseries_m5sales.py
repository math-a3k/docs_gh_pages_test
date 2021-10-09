from sklearn import preprocessing, metrics
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold, GroupKFold, GridSearchCV, train_test_split, TimeSeriesSplit
from datetime import datetime
import copy
import os
import fire
import glob
import pdb

##### import all Feature engineering functions
from source.preprocessors_tseries import *



########################################################################################################################
########################################################################################################################
def pd_col_tocat(df, nan_cols, colcat):
    nan_features = nan_cols
    for feature in nan_features:
        df[feature].fillna('unknown', inplace = True)

    categorical_cols = colcat
    for feature in categorical_cols:
        encoder     = preprocessing.LabelEncoder()
        df[feature] = encoder.fit_transform(df[feature].astype(str))

    return df


def pd_merge(df_list, cols_join):
    print(cols_join)
    dfall = None
    for dfi in df_list :
        print(dfi.columns)
        cols_joini = [ t for t in cols_join if t in dfi.columns ]
        dfall      = dfall.join(dfi.set_index(cols_joini), on = cols_joini, how="left") if dfall is not None else dfi
    return dfall


def train(input_path, n_experiments = 3, colid = None, coly = None):
    """
       Generic train
    :param input_path:
    :param max_rows:
    :param n_experiments:
    :param colid:
    :param coly:
    :return:
    """
    model_params = {'num_leaves': 555,
          'min_child_weight' : 0.034, 'feature_fraction' : 0.379, 'bagging_fraction' : 0.418, 'min_data_in_leaf' : 106, 'objective'        : 'regression', 'max_depth'        : -1, 'learning_rate'    : 0.005, "boosting_type"    : "gbdt", "bagging_seed"     : 11, "metric"           : 'rmse', "verbosity"        : -1, 'reg_alpha'        : 0.3899, 'reg_lambda'       : 0.648, 'random_state'     : 222,
         }

    #coly = 'demand'

    dict_metrics = {'run_id' : [], 'cols' : [], 'metric_name': [], 'model_params': [], 'metrics_val' : []}
    for ii in range(n_experiments):
        colsX  = featurestore_filter_features(colid = colid, coly = coly)
        df     = featurestore_get_feature_fromcolname(input_path, colsX + [coly], colid)

        X 	               = df[ colsX ]
        y            	   = df[coly]

        ############## ALready in DSA 2 ##########################################################
        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        # Y_test = np.zeros(X_test.shape[0])

        dtrain                     = lgb.Dataset(X_train, label=Y_train)
        dtest                      = lgb.Dataset(X_test, label=Y_test)
        clf                        = lgb.train(model_params, dtrain, 2500,
                                               valid_sets = [dtrain, dtest],
                                               early_stopping_rounds = 50, verbose_eval=100)

        Y_test_pred          	   = clf.predict(X_test,num_iteration=clf.best_iteration)
        val_score                  = np.sqrt(metrics.mean_squared_error(Y_test_pred, Y_test))

        #### Metrics  ####################################################
        dict_metrics['run_id'].append(datetime.now())
        dict_metrics['cols'].append(";".join(X_train.columns.tolist()))
        dict_metrics['model_params'].append(model_params)
        dict_metrics['metric_name'].append('rmse')
        dict_metrics['metrics_val'].append(val_score)

    df_metrics = pd.DataFrame.from_dict(dict_metrics)
    print("        DF metrics          ")
    print(df_metrics)
    df_metrics.to_csv("df_metrics.csv")




########################################################################################################################
########################################################################################################################
class FeatureStore(object):
    def __init__(self):
        pass



def featurestore_meta_update(featnames, filename, colcat):
    meta_csv = pd.DataFrame(columns = ['featname', 'filename', 'feattype'])
    if os.path.exists('meta_features.csv'):
        meta_csv = pd.read_csv('meta_features.csv')
    append_data_dict = {'featname' : [], 'filename' : [], 'feattype' : []}
    for feat in featnames:
        if feat not in meta_csv['featname'].unique():
            append_data_dict['filename'].append(filename)
            append_data_dict['featname'].append(feat)
            feat_type = "numeric" if feat not in colcat else "categorical"
            append_data_dict['feattype'].append(feat_type)
        else:
            meta_csv.loc[meta_csv['featname'] == feat, 'filename'] = filename
    append_df = pd.DataFrame.from_dict(append_data_dict)
    meta_csv = meta_csv.append(append_df)
    meta_csv.to_csv('meta_features.csv', index = False)


def featurestore_get_filelist_fromcolname(selected_cols, colid):
    meta_csv = pd.read_csv('meta_features.csv')
    file_feat_mapping = {k:colid for k in meta_csv['filename'].unique().tolist()}
    for selected_col in selected_cols:
        selected_col_meta_df = meta_csv[meta_csv["featname"] == selected_col]
        file_feat_mapping[selected_col_meta_df['filename'].tolist()[0]].append(selected_col)
    print(colid)
    return {k:list(set(v)) for k,v in file_feat_mapping.items()}



def featurestore_generate_feature(dir_in, dir_out, my_fun_features, features_group_name, input_raw_path = None,
                                  auxiliary_csv_path = None, coldrop = None, index_cols = None,
                                  merge_cols_mapping = None, colcat = None, colid=None, coly = None, coldate = None,
                                  max_rows = 5, step_wise_saving = False) :

    # from util_feat_m5  import lag_featrues
    # featurestore_generate_feature(dir_in, dir_out, lag_featrues)

    df_merged = pd.read_parquet(dir_in + "/raw_merged.df.parquet")

    dfnew, colcat= my_fun_features(df_merged, input_raw_path, dir_out, features_group_name, auxiliary_csv_path,
                                     coldrop, index_cols, merge_cols_mapping,
                                     colcat, colid, coly, coldate, max_rows)
    if not step_wise_saving:
        dfnew.to_parquet(f'{dir_out}/{features_group_name}.parquet')
    # num_cols = list(set(dfnew._get_numeric_data().columns))
    featurestore_meta_update(dfnew.columns, f'{features_group_name}.parquet', colcat)


def featurestore_filter_features(mode ="random", colid = None, coly = None):
    # categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2' ]
    # numerical_cols = ['snap_TX',  'sell_price', 'week', 'snap_CA', 'month', 'snap_WI', 'dayofweek', 'year']

    categorical_cols, numerical_cols = custom_get_colsname(colid = colid, coly =coly)

    cols_cat = []
    cols_num = []

    if mode == "random":
        cols_cat = [categorical_cols[i] for i in np.random.choice(len(categorical_cols), 3, replace = False)]
        cols_num = [numerical_cols[i] for i in np.random.choice(len(numerical_cols), 5, replace = False) ]

    if mode == "all":
        cols_cat = categorical_cols
        cols_num = numerical_cols

    if mode == "smartway":
        cols_cat = categorical_cols
        cols_num = numerical_cols
        # TODO: Need to update

    colall = cols_cat + cols_num
    return colall


def featurestore_get_filename(file_name, path):
    file_name_ext_list = file_name.split(".")
    flist = glob.glob( f'{path}/{file_name_ext_list[0]}' + "*")
    return flist


def featurestore_get_feature_fromcolname(path, selected_cols, colid):
    selected_cols = colid + selected_cols
    print(colid)

    file_col_mapping = featurestore_get_filelist_fromcolname(selected_cols = selected_cols, colid =colid[:])

    feature_dfs = []
    for file_name,file_cols in file_col_mapping.items():
        print(file_name)
        print(file_cols)
        file_name_feature_df = None
        for x in featurestore_get_filename(file_name, path):
            dfi = pd.read_parquet(f'{x}', columns = file_cols)
            file_name_feature_df = pd.concat((file_name_feature_df, dfi))
        feature_dfs.append(file_name_feature_df)

    print(colid)
    df_merged = pd_merge(feature_dfs, colid)

    df_merged = df_merged.sort_values('date')
    non_date_col = [x for x in colid if not x == "date"]
    df_merged.drop(non_date_col, inplace = True, axis = 1)
    return df_merged



########################################################################################################################
########################################################################################################################

def pd_tsfresh_m5data_sales(df_sales, dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols):
    """

    :param df_sales:
    :param dir_out:
    :param features_group_name:
    :param drop_cols:
    :param df_calendar:
    :param index_cols:
    :param merge_cols_mapping:
    :param id_cols:
    :return:
    """
    # X_feat = pd.DataFrame()
    auxiliary_dropped_cols = [x for x in df_calendar.columns.tolist() if x in drop_cols]
    df_calendar.drop(auxiliary_dropped_cols, inplace = True, axis = 1)

    for i in range(len(df_sales.index)):
        single_row_df = df_sales.loc[[i]]
        X_feat_single_row_df = pd_tsfresh_features_single_row(single_row_df, index_cols)
        if i % 5 ==0:
            X_feat = X_feat_single_row_df
        if (i+1) % 5 == 0 :
            merged_df = pd.merge(X_feat, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])

            # merged_df = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
            # selected_cols = [x for x in merged_df.columns.tolist() if x not in drop_cols]
            selected_cols = [x for x in merged_df.columns.tolist() if x in id_cols or str(x).startswith("val__")]
            merged_df_selected_cols = merged_df[selected_cols]
            merged_df_selected_cols.columns = merged_df_selected_cols.columns.astype(str)
            merged_df_selected_cols.to_parquet(f'{dir_out}/{features_group_name}_{i}.parquet')
        else:
            X_feat.append(X_feat_single_row_df, ignore_index = True)
    return merged_df_selected_cols




def pd_tsfresh_m5data(df):
    df = df[['snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'item_id', 'date', 'store_id', 'id']]
    df = roll_time_series(df, column_id="item_id", column_sort="date")
    existing_cols = df.columns.tolist()
    y = df['demand']
    X_cols = [x for x in existing_cols if not x == "demand"]
    X = df[X_cols]
    X = X.fillna(value = {'sell_price' : X['sell_price'].mean(skipna = True)})
    X = X[['snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'item_id', 'date']]
    X_filtered = extract_relevant_features(X, y, column_id='item_id', column_sort='date')

    filtered_col_names = X_filtered.columns.tolist()

    filtered_col_names_mapping = {}

    for filtered_col_name in filtered_col_names:
        filtered_col_names_mapping[filtered_col_name] = filtered_col_name.replace('"','').replace(',','')

    X_filtered = X_filtered.rename(columns = filtered_col_names_mapping)
    # This is done because lightgbm can not have features with " in the feature name

    feature_df = pd.concat([X[['item_id', 'date']], X_filtered])

    return feature_df, []





def pd_ts_tsfresh(df, input_raw_path, dir_out, features_group_name, auxiliary_csv_path, drop_cols, index_cols, merge_cols_mapping, cat_cols = None, id_cols = None, dep_col = None, coldate = None, max_rows = 10):
    # df is taken as an argument to make it work in the existing pipeline of saving features in meta_csv
    df_sales_val              = pd.read_csv(input_raw_path)
    df_calendar               = pd.read_csv(auxiliary_csv_path)

    merged_df         = pd_tsfresh_m5data_sales(df_sales_val[0:max_rows], dir_out, features_group_name, drop_cols, df_calendar, index_cols, merge_cols_mapping, id_cols)
    # df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    # merged_df = pd.merge(df_sales_val_melt, df_calendar, how = 'left', left_on = ['day'], right_on = ['d'])
    # merged_df = pd.merge(df_sales_val_melt, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])

    # merged_df = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
    # selected_cols = [x for x in merged_df.columns.tolist() if x not in [ 'id', 'cat_id', 'dept_id', 'store_id', 'variable', 'day', 'demand', 'state_id']]
    selected_cols = [x for x in merged_df.columns.tolist() if x not in drop_cols]
    return merged_df[selected_cols], []






def custom_get_colsname(colid, coly):
    coldrop = colid + [coly]
    meta_csv = pd.read_csv('meta_features.csv')
    num_feats = [ x for x in meta_csv[meta_csv["feattype"] == "numeric"]['featname'].tolist()  if x not in coldrop]
    cat_feats = [ x for x in meta_csv[meta_csv["feattype"] == "categorical"]['featname'].tolist() if x not in coldrop]
    return cat_feats, num_feats


def custom_rawdata_merge( out_path='out/', max_rows=10):

    input_path ="data/input/tseries/tseries_m5/raw"
    index_cols     = [ 'id', 'cat_id_col', 'dept_id_col', 'store_id_col', 'item_id_col', 'state_id_col']
    coly = "demand"
    colraw_merge = ['store_id_col', 'item_id_col', 'wm_yr_wk']
    merge_cols_mapping = {"left" : "day", "right" : "d"}
    colnan       = ['event_name_1_col', 'event_type_1_col', 'event_name_2_col', 'event_type_2_col']
    colcat       = ['dept_id_col', 'cat_id_col', 'store_id_col', 'state_id_col', 'event_name_1_col', 'event_type_1_col', 'event_name_2_col', 'event_type_2_col']


    df_sales_train            = pd.read_csv(input_path + "/sales_train_gen.csv")
    df_calendar               = pd.read_csv(input_path + "/calendar_gen.csv")
    df_sales_val              = pd.read_csv(input_path + "/sales_train_gen.csv")
    df_sell_price             = pd.read_csv(input_path + "/sell_prices_gen.csv")
    # df_submi                  = pd.read_csv("data/sample_submi.csv")

    df_sales_val = df_sales_val if max_rows == -1 else df_sales_val.iloc[:,0:max_rows]

    df_merged  = pd.melt(df_sales_val, id_vars = index_cols, var_name = 'day', value_name = coly)
    # df_merged = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
    # df_merged = df_sales_val_melt
    df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    df_merged = pd.merge(df_merged, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])
    df_merged = df_merged.merge(df_sell_price, on = colraw_merge, how = 'left')

    df_merged = pd_col_tocat(df_merged, nan_cols = colnan, colcat = colcat)
    # df_merged = add_time_features(df_merged)

    os.makedirs(out_path, exist_ok=True)
    fname = out_path + "/raw_merged.df.parquet"
    df_merged.to_parquet(fname)
    # return df_merged



data_path = "data/input/tseries/tseries_m5/processed"
def custom_generate_feature_all(input_path = data_path, out_path=".", input_raw_path =".", auxiliary_csv_path = None,
                                coldrop = None, colindex = None, merge_cols_mapping = None, coldate = None,
                                colcat = None, colid = None, coly = None, max_rows = 10):

    featurestore_generate_feature(data_path , input_path , pd_ts_basic    , "basic_time" , colid = colid, coldate = coldate)
    featurestore_generate_feature(data_path , input_path , pd_ts_rolling  , "rolling"    , coly = coly     , colid = colid)
    featurestore_generate_feature(data_path , input_path , pd_ts_lag      , "lag"        , coly = coly     , colid = colid)
    featurestore_generate_feature(data_path , input_path , pd_ts_tsfresh  , "tsfresh"    , input_raw_path  , auxiliary_csv_path , coldrop , colindex , merge_cols_mapping , max_rows , step_wise_saving = True , colid = colid)
    featurestore_generate_feature(data_path , input_path , pd_ts_identity , "identity"   , colcat = colcat , coldrop = ['d'     , 'id'    , 'day'    , 'wm_yr_wk'])



def run_train(input_path ="data/input/tseries/tseries_m5/raw", out_path=data_path,
              do_generate_raw=True, do_generate_feature=True, do_train=False,
              max_rows = 10):

    if do_generate_raw :
      custom_rawdata_merge(out_path=out_path, max_rows = max_rows)


    if do_generate_feature :
      custom_generate_feature_all(input_path="../../data/output", out_path="",
                                  input_raw_path =input_path + "/sales_train_gen.csv",
                                  auxiliary_csv_path =input_path + "/calendar_gen.csv",

                                  coldrop   = ['id', 'cat_id_col', 'dept_id_col', 'store_id_col', 'variable', 'day', 'demand', 'state_id_col', 'weekday', 'wday', 'month', 'year'],
                                  colindex  = ['id', 'cat_id_col', 'dept_id_col', 'store_id_col', 'item_id_col', 'state_id_col'],
                                  merge_cols_mapping = {"left" : "day", "right" : "d"},
                                  colcat    = ['item_id_col', 'dept_id_col', 'cat_id_col', 'store_id_col', 'state_id_col', 'event_name_1_col', 'event_type_1_col', 'event_name_2_col', 'event_type_2_col'],
                                  colid     = ["date", "item_id_col"],
                                  coly      = "demand",
                                  max_rows  = max_rows,
                                  coldate = "date")

    if do_train :
        train(input_path="../../data/output", colid = ["date", "item_id_col"], coly ="demand")


if __name__ == "__main__":
    run_train()
















"""


import util_feat_m5



# df_meta=   col_name, col_type, file_path



def featurestore_generate_feature(dir_in, dir_out, my_fun_features) :

    # from util_feat_m5  import lag_featrues
    # featurestore_generate_feature(dir_in, dir_out, lag_featrues)

    train_df = pd.read_csv( dir_in  + "/sales_train_val.csv.zip")
    calendar_df = pd.read_csv(dir_in  + "/calendar.csv")
    price_df = pd.read_csv(dir_in  + "/sell_prices.csv")


    dfnew = my_fun_features(train_df, calendar_df, price_df) :

    dfnew.to_parquet( dir_out +"/mfeaturesXXXX.parquet")



 def featurestore_filter_features(mode="random") :
    cols_cat0 = [  "feat1", "fewat2" ]

    if mode == "random" :
        ### Random selection
        cols_cat = cols_cat0[ np.random.c = hoice( 5, len(cols_cat)  ) ]
        cols_num = cols_num0[ np.random.c = hoice( 5, len(cols_num)  ) ]
        return cols_cat, col_num

    if mode == "all" :
      pass

    if mode == "smartway" :
       pass



def train(model, pars={} ) :

    data_pars = {}
    model_pars = {}

    for ii in range(n_experiments) :
        cols_cat, cols_num = featurestore_filter_features()

        df     = featurestore_get_feature_fromcolname(path, cols_cat + cols_num, "train")
        dftest = featurestore_get_feature_fromcolname(path, cols_cat + cols_num, 'test')

        X_train = X_transform( df, cols_num, cols_cat, pars) # select sri
        y_train  = y_transform(df, coly)

        X_test = X_transform( dftest, cols_num, cols_cat, pars) # select variables
        y_test  = y_transform(dftest, coly)



        lgbm = lgb.LGBMRegressor()
        lgbm.fit( X_train, y_train)

        # prediction + metrics
        y_test_pred = lgbm.predict(X_test)
        metric_val = metrics_calc(y_test, y_test_pred)


        ### Store in metrics :

        # run_id, feat_name, feat_name_long, feat_type, model_params, metric_name, metric_val
        # 3,roling_demand,Mean of the variable estimates,lag_features,params = {"objective" : "poisson","metric" :"rmse","force_row_wise" : True,"learning_rate" : 0.075,
    "sub_row" : 0.75,"bagging_freq" : 1,"lambda_l2" : 0.1,"metric": ["rmse"],'verbosity': 1,'num_iterations' : 250,
    },rmse,1.16548

        df_metrics['run_id'] = time()
        df_metrics['cols'].append(  ",".join( cols_num + cols_cat ))
        df_metrics['metrics_val'].append(metric_val)





def test_old():
    from util_feat_m5  import lag_featrues
    train_df = pd.read_csv("sales_train_val.csv")
    calendar_df = pd.read_csv("calendar.csv")
    price_df = pd.read_csv("sell_prices.csv")
    sample = pd.read_csv("sample_submi.csv")
    calendar_df["date_dt"] = pd.to_datetime(calendar_df["date"])
    train  = train_df.copy()
    price = price_df.copy()
    calendar = calendar_df.copy()
    Train_data = train.iloc[:,:-56]
    Val_data = train.iloc[:,:-28]

    X_train = lag_featrues(Train_data).iloc[:,5:] # select variables
    y_train = train.iloc[:,-56]
    X_test = lag_featrues(Val_data).iloc[:,5:]
    y_test = train.iloc[:,-28]

    # Create instance
    lgbm = lgb.LGBMRegressor()

    # Training and score
    learning_rate = [0.15, 0.2, 0.25]
    max_depth = [15, 20, 25]

    param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

    # Fitting
    cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
    cv_lgbm.fit(X_train, y_train)

    print("Best params:{}".format(cv_lgbm.best_params_))

    # best params
    best_lg = cv_lgbm.best_estimator_

    # prediction
    y_train_pred_lg = best_lg.predict(X_train)
    y_test_pred_lg = best_lg.predict(X_test)

    print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
    print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

    print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
    print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
    #Predict using only variables with an importance of 1 or higher.
    importance = best_lg.feature_importances_

    indices = np.argsort(importance)[::-1]

    # print importance
    importance_df = pd.DataFrame({})
    columns = []
    importance_ = []
    for f in range(X_train.shape[1]):
        print("%2d) %-*s %.2f" %(f+1, 30, X_train.columns[indices[f]], importance[indices[f]]))
        col = X_train.columns[indices[f]]
        imp = importance[indices[f]]
        columns.append(col)
        importance_.append(imp)
    importance_df["col_name"] = columns
    importance_df["importance"] = importance_
    importance = best_lg.feature_importances_

    indices = np.argsort(importance)[::-1]

    # importance columns (>0)
    imp_col = importance_df[importance_df["importance"]>0]["col_name"].values

    # Train test split, select by imp_col

    X_train = lag_featrues(Train_data).iloc[:,5:][imp_col] # select variables
    y_train = train.iloc[:,-56]
    X_test = lag_featrues(Val_data).iloc[:,5:][imp_col]
    y_test = train.iloc[:,-28]

    # Create instance
    lgbm = lgb.LGBMRegressor()

    # Training and score
    learning_rate = [0.15, 0.2, 0.25]
    max_depth = [15, 20, 25]

    param_grid = {'learning_rate': learning_rate, 'max_depth': max_depth}

    # Fitting
    cv_lgbm = GridSearchCV(lgbm, param_grid, cv=10, n_jobs =1)
    cv_lgbm.fit(X_train, y_train)

    print("Best params:{}".format(cv_lgbm.best_params_))

    # best params
    best_lg = cv_lgbm.best_estimator_

    # prediction
    y_train_pred_lg = best_lg.predict(X_train)
    y_test_pred_lg = best_lg.predict(X_test)

    print("MSE train:{}".format(mean_squared_error(y_train, y_train_pred_lg)))
    print("MSE test;{}".format(mean_squared_error(y_test, y_test_pred_lg)))

    print("R2 score train:{}".format(r2_score(y_train, y_train_pred_lg)))
    print("R2 score test:{}".format(r2_score(y_test, y_test_pred_lg)))
    run_id=list(range(300))
    df_metrics=pd.DataFrame(run_id,columns=['run_id'])
    df_metrics['feat_name'] = pd.Series(X_train.columns[0:300], index=dataframe.index)
    df_metrics['feat_type']=df_metrics.feat_name
    df_metrics.replace({'feat_type': r'^lag_.*'}, {'feat_type': 'lag'}, regex=True,inplace=True)
    df_metrics.replace({'feat_type': r'^rolling.*'}, {'feat_type': 'rolling'}, regex=True,inplace=True)
    df_metrics['parameter'] = pd.Series(best_lg, index=dataframe.index)
    df_metrics['metric_name'] ="MSE"
    df_metrics['metric_val'] = pd.Series(pred_mse[:300], index=dataframe.index)
    df_metrics.to_csv("train.csv")
"""

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
from source.prepro_tseries import *



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
def custom_get_colsname(colid, coly):
    coldrop = colid + [coly]
    meta_csv = pd.read_csv('meta_features.csv')
    num_feats = [ x for x in meta_csv[meta_csv["feattype"] == "numeric"]['featname'].tolist()  if x not in coldrop]
    cat_feats = [ x for x in meta_csv[meta_csv["feattype"] == "categorical"]['featname'].tolist() if x not in coldrop]
    return cat_feats, num_feats


def custom_rawdata_merge( out_path='out/', max_rows=10):

    input_path ="data/input/tseries/retail/raw"
    index_cols     = [ 'Store']
    coly = "Weekly_Sales"
    colraw_merge = ['Store']
    colnan       = []
    colcat       = []


    df_sales_train            = pd.read_csv(input_path + "/sales-data-set.csv")
    # df_calendar               = pd.read_csv(input_path + "/calendar_gen.csv")
    df_sales_val              = pd.read_csv(input_path + "/sales-data-set.csv")
    df_features               = pd.read_csv(input_path + "/features-data-set.csv")
    df_stores                 = pd.read_csv(input_path + "/stores-data-set.csv")
    # df_sell_price             = pd.read_csv(input_path + "/sell_prices_gen.csv")
    # df_submi                  = pd.read_csv("data/sample_submi.csv")

    df_sales_val = df_sales_val if max_rows == -1 else df_sales_val.iloc[:,0:max_rows]

    # df_merged  = pd.melt(df_sales_val, id_vars = index_cols, var_name = 'Date', value_name = coly)
    # df_merged = pd.concat([df_sales_val_melt, df_submi_val, df_submi_eval], axis = 0)
    # df_merged = df_sales_val_melt
    # df_calendar.drop(['weekday', 'wday', 'month', 'year'], inplace = True, axis = 1)
    # df_merged = pd.merge(df_merged, df_calendar, how = 'left', left_on = [merge_cols_mapping["left"]], right_on = [merge_cols_mapping["right"]])
    # df_merged = df_merged.merge(df_sell_price, on = colraw_merge, how = 'left')

    df_merged = pd.merge(df_sales_val, df_features, on = "Store")
    df_merged = pd.merge(df_merged, df_stores, on = "Store")
    df_merged = pd_col_tocat(df_merged, nan_cols = colnan, colcat = colcat)
    # df_merged = add_time_features(df_merged)

    os.makedirs(out_path, exist_ok=True)
    fname = out_path + "/raw_merged.df.parquet"
    df_merged.to_parquet(fname)
    # return df_merged



data_path = "data/input/tseries/retail/processed"
def custom_generate_feature_all(input_path = data_path, out_path=".", input_raw_path =".", auxiliary_csv_path = None,
                                coldrop = None, colindex = None, merge_cols_mapping = None, coldate = None,
                                colcat = None, colid = None, coly = None, max_rows = 10):

    featurestore_generate_feature(data_path , input_path , pd_ts_basic    , "basic_time" , colid = colid, coldate = coldate)
    featurestore_generate_feature(data_path , input_path , pd_ts_rolling  , "rolling"    , coly = coly     , colid = colid)
    featurestore_generate_feature(data_path , input_path , pd_ts_lag      , "lag"        , coly = coly     , colid = colid)
    # featurestore_generate_feature(data_path , input_path , pd_ts_tsfresh  , "tsfresh"    , input_raw_path  , auxiliary_csv_path , coldrop , colindex , merge_cols_mapping , max_rows , step_wise_saving = True , colid = colid)
    featurestore_generate_feature(data_path , input_path , pd_ts_identity , "identity"   , colcat = colcat , coldrop = ['d'     , 'id'    , 'day'    , 'wm_yr_wk'])



def run_generate_train_data(input_path ="data/input/tseries/retail/raw", out_path=data_path,
              do_generate_raw=True, do_generate_feature=True, do_train=False,
              max_rows = 10):

    if do_generate_raw :
      custom_rawdata_merge(out_path=out_path, max_rows = max_rows)


    if do_generate_feature :
      custom_generate_feature_all(input_path="../../data/output", out_path="",
                                  input_raw_path =input_path + "/sales-data-set.csv",
                                  auxiliary_csv_path =input_path + "/calendar_gen.csv",

                                  coldrop   = [],
                                  colindex  = ['Store'],
                                  merge_cols_mapping = {"left" : "day", "right" : "d"},
                                  colcat    = [],
                                  colid     = ["Store", "Date"],
                                  coly      = "Weekly_Sales",
                                  max_rows  = max_rows,
                                  coldate = "Date")

if __name__ == "__main__":
    run_generate_train_data()






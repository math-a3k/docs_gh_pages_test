# pylint: disable=C0321,C0103,E1221,C0301,E1305,E1121,C0302,C0330
# -*- coding: utf-8 -*-
"""

For Recpmmender type of data


"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                            get_feature_names)

# from preprocess import _preprocess_criteo, _preprocess_movielens

# Note: keep that to disable eager mode with tf 2.x
import tensorflow as tf
if tf.__version__ >= '2.0.0':
    tf.compat.v1.disable_eager_execution()



##################################################################################################
def _preprocess_criteo(df, **kw):
    hash_feature = kw.get('hash_feature')
    sparse_col = ['C' + str(i) for i in range(1, 27)]
    dense_col = ['I' + str(i) for i in range(1, 14)]
    df[sparse_col] = df[sparse_col].fillna('-1', )
    df[dense_col] = df[dense_col].fillna(0, )
    target = ["label"]

    # set hashing space for each sparse field,and record dense feature field name
    if hash_feature:
        # Transformation for dense features
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_col] = mms.fit_transform(df[dense_col])
        sparse_col = ['C' + str(i) for i in range(1, 27)]
        dense_col = ['I' + str(i) for i in range(1, 14)]

        fixlen_cols = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4, use_hash=True, dtype='string')
                       # since the input is string
                       for feat in sparse_col] + [DenseFeat(feat, 1, ) for feat in dense_col]

    else:
        for feat in sparse_col:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_col] = mms.fit_transform(df[dense_col])
        fixlen_cols = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                       for i, feat in enumerate(sparse_col)] + [DenseFeat(feat, 1, ) for feat in dense_col]

    linear_cols = fixlen_cols
    dnn_cols = fixlen_cols
    train, test = train_test_split(df, test_size=kw['test_size'])

    return df, linear_cols, dnn_cols, train, test, target, test[target].values


def _preprocess_movielens(df, **kw):
    multiple_value = kw.get('multiple_value')
    sparse_col = ["movie_id", "user_id", "gender", "age", "occupation", "zip"]
    target = ['rating']

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_col:
        lbe = LabelEncoder()
        df[feat] = lbe.fit_transform(df[feat])

    if not multiple_value:
        # 2.count #unique features for each sparse field
        fixlen_cols = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_col]
        linear_cols = fixlen_cols
        dnn_cols = fixlen_cols
        train, test = train_test_split(df, test_size=0.2)
        ytrue = test[target].values
    else:
        ytrue = df[target].values
        hash_feature = kw.get('hash_feature', False)
        if not hash_feature:
            def split(x):
                key_ans = x.split('|')
                for key in key_ans:
                    if key not in key2index:
                        # Notice : input value 0 is a special "padding",so we do not use 0 to encode valid feature for sequence input
                        key2index[key] = len(key2index) + 1
                return list(map(lambda x: key2index[x], key_ans))

            # preprocess the sequence feature
            key2index = {}
            genres_list = list(map(split, df['genres'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)
            # Notice : padding=`post`
            genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', )
            fixlen_cols = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in sparse_col]

            use_weighted_sequence = False
            if use_weighted_sequence:
                varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                    key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name='genres_weight')]  # Notice : value 0 is for padding for sequence input feature
            else:
                varlen_cols = [VarLenSparseFeat(SparseFeat('genres', vocabulary_size=len(
                    key2index) + 1, embedding_dim=4), maxlen=max_len, combiner='mean',
                                                weight_name=None)]  # Notice : value 0 is for padding for sequence input feature

            linear_cols = fixlen_cols + varlen_cols
            dnn_cols = fixlen_cols + varlen_cols

            # generate input data for model
            model_input = {name: df[name] for name in sparse_col}  #
            model_input["genres"] = genres_list
            model_input["genres_weight"] = np.random.randn(df.shape[0], max_len, 1)


        else:
            df[sparse_col] = df[sparse_col].astype(str)

            # 1.Use hashing encoding on the fly for sparse features,and process sequence features
            genres_list = list(map(lambda x: x.split('|'), df['genres'].values))
            genres_length = np.array(list(map(len, genres_list)))
            max_len = max(genres_length)

            # Notice : padding=`post`
            genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str, value=0)

            # 2.set hashing space for each sparse field and generate feature config for sequence feature
            fixlen_cols = [
                SparseFeat(feat, df[feat].nunique() * 5, embedding_dim=4, use_hash=True, dtype='string')
                for feat in sparse_col]
            varlen_cols = [
                VarLenSparseFeat(
                    SparseFeat('genres', vocabulary_size=100, embedding_dim=4, use_hash=True, dtype="string"),
                    maxlen=max_len, combiner='mean',
                )]  # Notice : value 0 is for padding for sequence input feature

            linear_cols = fixlen_cols + varlen_cols
            dnn_cols = fixlen_cols + varlen_cols
            feature_names = get_feature_names(linear_cols + dnn_cols)

            # 3.generate input data for model
            model_input = {name: df[name] for name in feature_names}
            model_input['genres'] = genres_list

        train, test = model_input, model_input

    return df, linear_cols, dnn_cols, train, test, target, ytrue


""""
# DeepCTR
https://github.com/shenweichen/DeepCTR
https://deepctr-doc.readthedocs.io/en/latest/Examples.html#classification-criteo


DeepCTR is a **Easy-to-use**,**Modular** and **Extendible** package of deep-learning based CTR models 
along with lots of core components layers which can be used to easily build custom models.It is compatible with **tensorflow 1.4+ and 2.0+**.You can use any complex model with `model.fit()`and `model.predict()` .


## Models List
|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  Convolutional Click Prediction Model  | [CIKM 2015][A Convolutional Click Prediction Model](http://ir.ia.ac.cn/bitstream/173211/12337/1/A%20Convolutional%20Click%20Prediction%20Model.pdf)             |
| Factorization-supported Neural Network | [ECIR 2016][Deep Learning over Multi-field Categorical Data: A Case Study on User Response Prediction](https://arxiv.org/pdf/1601.02376.pdf)                    |
|      Product-based Neural Network      | [ICDM 2016][Product-based neural networks for user response prediction](https://arxiv.org/pdf/1611.00144.pdf)                                                   |
|              Wide & Deep               | [DLRS 2016][Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf)                                                                 |
|                 DeepFM                 | [IJCAI 2017][DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](http://www.ijcai.org/proceedings/2017/0239.pdf)                           |
|        Piece-wise Linear Model         | [arxiv 2017][Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction](https://arxiv.org/abs/1704.05194)                                 |
|          Deep & Cross Network          | [ADKDD 2017][Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)                                                                   |
|   Attentional Factorization Machine    | [IJCAI 2017][Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks](http://www.ijcai.org/proceedings/2017/435) |
|      Neural Factorization Machine      | [SIGIR 2017][Neural Factorization Machines for Sparse Predictive Analytics](https://arxiv.org/pdf/1708.05027.pdf)                                               |
|                xDeepFM                 | [KDD 2018][xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems](https://arxiv.org/pdf/1803.05170.pdf)                         |
|                AutoInt                 | [arxiv 2018][AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks](https://arxiv.org/abs/1810.11921)                              |
|         Deep Interest Network          | [KDD 2018][Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1706.06978.pdf)                                                       |
|    Deep Interest Evolution Network     | [AAAI 2019][Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672.pdf)                                            |
|                  NFFM                  | [arxiv 2019][Operation-aware Neural Networks for User Response Prediction](https://arxiv.org/pdf/1904.12579.pdf)                                                |
|                 FGCNN                  | [WWW 2019][Feature Generation by Convolutional Neural Network for Click-Through Rate Prediction ](https://arxiv.org/pdf/1904.04447)                             |
|     Deep Session Interest Network      | [IJCAI 2019][Deep Session Interest Network for Click-Through Rate Prediction ](https://arxiv.org/abs/1905.06482)                                                |
|                FiBiNET                 | [RecSys 2019][FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/pdf/1905.09433.pdf)   |

"""
import os

import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                            get_feature_names)
from deepctr.layers import custom_objects


####################################################################################################
# Helper functions
def os_package_root_path(filepath, sublevel=0, path_add=""):
    """
       get the module package root folder
    """
    from pathlib import Path
    path = Path(os.path.realpath(filepath)).parent
    for i in range(1, sublevel + 1):
        path = path.parent

    path = os.path.join(path.absolute(), path_add)
    return path


def log(*s, n=0, m=1):
    sspace = "#" * n
    sjump = "\n" * m
    print(sjump, sspace, s, sspace, flush=True)




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

        fixlen_cols = [SparseFeat(feat, vocabulary_size=1000, embedding_dim=4, use_hash=True,
                                  dtype='string')  # since the input is string
                       for feat in sparse_col] + [DenseFeat(feat, 1, )
                                                  for feat in dense_col]
    else:
        for feat in sparse_col:
            lbe = LabelEncoder()
            df[feat] = lbe.fit_transform(df[feat])
        mms = MinMaxScaler(feature_range=(0, 1))
        df[dense_col] = mms.fit_transform(df[dense_col])
        fixlen_cols = [SparseFeat(feat, vocabulary_size=df[feat].nunique(), embedding_dim=4)
                       for i, feat in enumerate(sparse_col)] + [DenseFeat(feat, 1, )
                                                                for feat in dense_col]
    linear_cols = fixlen_cols
    dnn_cols = fixlen_cols
    train, test = train_test_split(df, test_size=kw['test_size'])

    return df, linear_cols, dnn_cols, train, test, target


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

    else:
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
            fixlen_cols = [SparseFeat(feat, df[feat].nunique(), embedding_dim=4) for feat in
                           sparse_col]

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
            genres_list = pad_sequences(genres_list, maxlen=max_len, padding='post', dtype=str,
                                        value=0)

            # 2.set hashing space for each sparse field and generate feature config for sequence feature

            fixlen_cols = [
                SparseFeat(feat, df[feat].nunique() * 5, embedding_dim=4, use_hash=True,
                           dtype='string')
                for feat in sparse_col]
            varlen_cols = [
                VarLenSparseFeat(
                    SparseFeat('genres', vocabulary_size=100, embedding_dim=4, use_hash=True,
                               dtype="string"),
                    maxlen=max_len, combiner='mean',
                )]  # Notice : value 0 is for padding for sequence input feature
            linear_cols = fixlen_cols + varlen_cols
            dnn_cols = fixlen_cols + varlen_cols
            feature_names = get_feature_names(linear_cols + dnn_cols)

            # 3.generate input data for model
            model_input = {name: df[name] for name in feature_names}
            model_input['genres'] = genres_list

        train, test = model_input, model_input

    return df, linear_cols, dnn_cols, train, test, target



def _preprocess_none(df, **kw):
        linear_cols = kw['linear_cols']
        dnn_cols = kw['dnn_cols']
        train, test = train_test_split(df, test_size=kw['test_size'])
        target = kw['target_col']

        return df, linear_cols, dnn_cols, train, test, target



def get_dataset(**kw):
    ##check whether dataset is of kind train or test
    data_path = kw['train_data_path']
    data_type = kw['dataset_type']
    test_size = kw['test_size']

    #### read from csv file
    if kw.get("uri_type") == "pickle":
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    return df



########################################################################################################################
########################################################################################################################
def test(data_path="dataset/", pars_choice=0):
    ### Local test

    log("#### Loading params   ##############################################")
    model_pars, data_pars, compute_pars, out_pars = get_params(choice=pars_choice,
                                                               data_path=data_path)
    print(model_pars, data_pars, compute_pars, out_pars)


    log("#### Loading dataset   #############################################")
    df = get_dataset(**kw)
    df, linear_cols, dnn_cols, train, test, target = _preprocess_criteo(df, **kw)


    df, linear_cols, dnn_cols, train, test, target = _preprocess_movielens(df, **kw)


    df, linear_cols, dnn_cols, train, test, target = _preprocess_none(df, **kw)



if __name__ == '__main__':
    VERBOSE = True
    test(pars_choice=0)

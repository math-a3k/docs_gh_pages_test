import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,DenseFeatures
import pandas as pd
import pprint
import zipfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from glob import glob



def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


class dictEval(object):
    '''
    https://www.tutorialspoint.com/How-to-recursively-iterate-a-nested-Python-dictionary
    https://stackoverflow.com/questions/45335445/recursively-replace-dictionary-values-with-matching-key
def replace_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj
    '''
    global dst
    import glob

    def __init__(self):
        self.dst = {}

    def reset(self):
        self.dst = {}

    def eval_dict(self,src, dst={}):
        for key, value in src.items():
            if isinstance(value, dict):
                node     = dst.setdefault(key, {})
                dst[key] = self.eval_dict(value, node)

            else:
                if ":@lazy" not in key :
                    dst[key] = value
                    continue

                ###########################################################################################
                key2           = key.split(':@lazy')[0]
                path_pattern   = value

                if 'tf' in key :
                    #log('TF is HEre')
                    self.tf_dataset_create(key2,path_pattern,)

                if 'pandas' in key :
                    self.pandas_create(key2, path_pattern, )
        return dst
        
    def tf_dataset_create(self, key2, path_pattern, batch_size=32, **kw):
        """
          https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset
                tf.data.experimental.make_csv_dataset(
            file_pattern, batch_size, column_names=None, column_defaults=None,
            label_name=None, select_columns=None, field_delim=',',
            use_quote_delim=True, na_value='', header=True, num_epochs=None,
            shuffle=True, shuffle_buffer_size=10000, shuffle_seed=None,
            prefetch_buffer_size=None, num_parallel_reads=None, sloppy=False,
            num_rows_for_inference=100, compression_type=None, ignore_errors=False
        )
        :return:
        """
        # import glob
        # flist = glob.glob(path_pattern + "/*")
        print(f'Path Pattern Observed: {path_pattern}')
        dataset = tf.data.experimental.make_csv_dataset(path_pattern,label_name='y',  batch_size=batch_size, ignore_errors=True)
        dataset = dataset.map(pack_features_vector)
        print(dataset)
        dst[key2] = dataset.repeat()


    def pandas_create(self, key2, path, ):
        import glob
        from utilmy import pd_read_file
        # flist = glob.glob(path)
        dst[key2] = pd_read_file(path)


def log(*s):
    print(*s)






def test1():
    ## pip install adataset
    root = ""
    from adatasets import test_dataset_classification_fake
    df, p = test_dataset_classification_fake(nrows=100)
    print(df.columns)
    df = df.astype('float')
    df.to_parquet(root+ 'datasets/parquet/f01.parquet')
    df.to_parquet(root + 'datasets/parquet/f02.parquet' )
    parquet_path = root + 'datasets/parquet/f*.parquet'

    df[ [p['coly']] ].to_parquet(root + 'datasets/parquet/label_01.parquet' )
    df[ [p['coly']] ].to_parquet(root + 'datasets/parquet/label_01.parquet' )
    parquet_path_y = root + 'datasets/parquet/label*.parquet'


    df.to_csv(root + 'datasets/csv/f01.csv',index=False )
    df.to_csv(root + 'datasets/csv/f02.csv' ,index=False)
    csv_path     = root + 'datasets/csv/f01.csv'


    df.to_csv(root + 'datasets/zip/f01.zip', compression='gzip' )
    df.to_csv(root + 'datasets/zip/f02.zip', compression='gzip' )
    zip_path     = root + 'datasets/zip/*.zip'


    data_pars = {

        ### ModelTarget-Keyname : Path
        'Xtrain:@lazy_tf'  : csv_path, #CSV file extraction #Tensorflow Dataset
        #'Xtest:@lazy_tf'   : zip_path,     #zip File Extraction
        'Xval:@lazy_pandas': csv_path,     #Pandas


        #'ytrain:@lazy_tf' : parquet_path_y,     #Pandas
        #'ytest:@lazy_tf ' : parquet_path_y,     #Pandas


        'pars': 23,
        "batch_size" : 32,
        "n_train": 500,
        "n_test": 500,

        'sub-dict' :{ 'one' : {'twp': 2 } }
    }

    test = dictEval()
    data_pars2 = test.eval_dict(data_pars)
    print(dst)

    from tensorflow.keras import layers
    model = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(256, activation='elu'),
        layers.Dense(32, activation='elu'),
        layers.Dense(1,activation='sigmoid')
        ])


    model.compile(optimizer='adam',
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])
    model.fit(dst['Xtrain'],
            steps_per_epoch=20,
            epochs=30,
            verbose=1
            )






#######################################################################################
#######################################################################################
from petastorm.tf_utils import make_petastorm_dataset
import petastorm
import os
import tensorflow as tf
from petastorm import make_batch_reader
import numpy as np

def pack_features_vector(features, labels):
    """Pack the features into a single array."""

    #print(f'Features: {features}')
    print(dir(features))
    features = tf.stack(list(features), axis=1)
    return features, labels

def get_dataset_split_for_model_petastorm(Xtrain, ytrain=None, pars:dict=None):
    """  Split data for moel input/
    Xtrain  ---> Split INTO  tuple PetaStorm Reader
    https://github.com/uber/petastorm/blob/master/petastorm/reader.py#L61-L134
    :param Xtrain:  path
    :param cols_type_received:
    :param cols_ref:
    :return:
    """
    file = r'C:\Users\TusharGoel\Desktop\Upwork\project4\dsa2\datasets\parquet\f01.parquet'
    dataset_url_train = Xtrain
    all_cols = 'colnum_0,colnum_1,colnum_2,colnum_3,colnum_4,colnum_5,colnum_6,colnum_7,colnum_8,colnum_9,colnum_10,colcat_1'
    all_cols = all_cols.split(',')
    label = 'y'
    batch_size = 128
    num_classes = 2
    epochs = 12
    file_path = '/C:/Users/TusharGoel/Desktop/Upwork/project4/dsa2/'+ dataset_url_train
    file = "file://" + file_path
    BATCH_SIZE = 32
    train_reader = make_batch_reader(file)
    #yield tensor
    train_ds = make_petastorm_dataset(train_reader) \
            .apply(tf.data.experimental.unbatch()) \
            .batch(BATCH_SIZE) \
            .map(lambda x: [tf.reshape(list(getattr(x, col) for col in all_cols),[-1,12]),tf.reshape(x.y,[-1,1])])
    #train_ds = train_ds.map(pack_features_vector)
    train_ds = train_ds.make_one_shot_iterator()
    #print(f'Train Dataset: {train_ds}')

    tensor = np.array(train_ds.get_next())
    print(tensor)
    return tensor
    #print(train_ds)


    '''train_dataset  = make_petastorm_dataset(reader)
        #trai
    
    

    ### Re-shape  #############################################
    #train_dataset = train_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))

    #print(dir(train_dataset))
    
    train_dataset = train_dataset.map(lambda x: (tf.reshape(x,[-1,1]),tf.reshape(getattr(x,label),[-1,1])))
    train_dataset = train_dataset.map(pack_features_vector)'''
    ###########################################################
    #train_dataset = train_dataset.batch(batch_size, drop_remainder=True)


tensor = get_dataset_split_for_model_petastorm('datasets/parquet/f01.parquet')

from tensorflow.keras import layers
model = tf.keras.Sequential([
    layers.Dense(32, activation='elu'),
    layers.Dense(32, activation='elu'),
    layers.Dense(1,activation='sigmoid')
    ])


model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
model.fit(tensor,
        batch_size=32,
        epochs=30,
        verbose=1
        )


















dst = dict()
def eval_dict(src, dst={}):
    import pandas as pd
    for key, value in src.items():
        if isinstance(value, dict):
            node = dst.setdefault(key, {})
            eval_dict(value, node)
        else:
            if "@lazy" not in key :
               dst[key] = value
            else :
                key2 = key.split(":")[-1]
                if 'pandas.read_csv' in key :
                    dst[key2] = pd.read_csv(value)
                elif 'pandas.read_parquet' in key :
                    dst[key2] = pd.read_parquet(value)
    return dst





"""    
if ext == 'zip':
    zf = zipfile.ZipFile(value)
    fileNames = zf.namelist()
    for idx,file in enumerate(fileNames):
        if file.split('.')[-1] in ['csv','txt']:
            file = 'datasets/'+file
            dst[key2+'_'+str(idx)] = pd.read_csv(file)
elif ext in ['csv','txt']:
        dst[key2] = pd.read_csv(value)
elif ext == 'parquet':
        dst[key2] = pd.read_parquet(value)
return dst
"""

"""
if ext == 'zip':
    zf        = zipfile.ZipFile(path_pattern)
    fileNames = zf.namelist()
    for idx,file in enumerate(fileNames):
        if file.split('.')[-1] in ['csv','txt']:
            file = 'datasets/'+file
            try:
                dataset = tf.data.experimental.make_csv_dataset(file, label_name=coly, batch_size=32, ignore_errors=True)
                dataset = dataset.map(pack_features_vector)
                dst[key2+'_'+str(idx)] = dataset.repeat()
            except:
                pass
elif ext in ['csv','txt']:
            dataset = tf.data.experimental.make_csv_dataset(path_pattern, label_name=coly, batch_size=32, ignore_errors=True)
            dataset = dataset.map(pack_features_vector)
            dst[key2] = dataset.repeat()
elif ext == 'parquet':
        filename = path_pattern.split('.')[0] + '.csv'
        pd.read_parquet(path_pattern).to_csv(filename)
        pd.read_parquet(path_pattern).to_csv(filename)
        dataset = tf.data.experimental.make_csv_dataset(filename, label_name=coly, batch_size=32, ignore_errors=True)
        dataset = dataset.map(pack_features_vector)
        dst[key2] = dataset.repeat()
"""


"""
    dst = {}
    dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
    csv_file    = 'datasets/petfinder-mini/petfinder-mini.csv'
    zip_file = 'datasets/petfinder_mini.zip'
    #tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,extract=True, cache_dir='.')
    #Uncomment This File for preprocessing the CSV File
df = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')
    col = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize',
        'FurLength', 'Vaccinated', 'Sterilized', 'Health',
        ]
    df.drop(['Description'],axis=1,inplace=True)
    df[col] = df[col].astype(str).apply(LabelEncoder().fit_transform)
    df.to_csv('datasets/petfinder-mini/petfinder-mini.csv')
"""









































###############################################################################################
###############################################################################################
'''
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('datasets/petfinder-mini/petfinder-mini.csv')

col = ['Type', 'Breed1', 'Gender', 'Color1', 'Color2', 'MaturitySize','FurLength', 'Vaccinated', 'Sterilized', 'Health']

df.drop(['Description'],axis=1,inplace=True)
df[col] = df[col].astype(str).apply(LabelEncoder().fit_transform)
#df.to_csv('datasets/petfinder-mini/petfinder-mini.csv')
df = df.astype('int32')
df.to_parquet('datasets/petfinder_mini.parquet',index=False)
'''
import os
from petastorm import make_batch_reader
from petastorm.tf_utils import make_petastorm_dataset
import tensorflow as tf
from tensorflow.data.experimental import unbatch
from tensorflow.io import decode_raw
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
#sc = SparkContext('local')
# spark = SparkSession(sc)
from tensorflow.keras import layers



from adatasets import test_dataset_classification_fake
df, d = test_dataset_classification_fake(nrows=100)
print(df)
colnum, colcat, coly = d['colnum'], d['colcat'], d['coly']

path = os.path.abspath("data/input/ztest/fake/").replace("\\","/")
os.makedirs(path, exist_ok=True)

df.to_parquet(path + "/feature_01.parquet")
df.to_parquet(path + "/feature_02.parquet")



def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels

path2 = 'file:' + path +"/feature_01.parquet" #.replace("D:/", "")


batch_size = 32
with make_batch_reader( path2 ) as reader:
    dataset  = make_petastorm_dataset(reader)
    iterator = dataset.make_one_shot_iterator()

    tensor = iterator.get_next()

    print("dataset", dataset, tensor )


    model = tf.keras.Sequential([
           layers.Flatten(),
           layers.Dense(256, activation='elu'),
           layers.Dense(32,  activation='elu'),
           layers.Dense(1,   activation='sigmoid')
           ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.fit([tensor],
           steps_per_epoch=1,
           epochs=1,
           verbose=1
           )
    print('Hurray Successfully Initiated')




########################################################################################################################
def fIt_(dataset_url, training_iterations, batch_size, evaluation_interval):
    """
https://github.com/uber/petastorm/blob/master/petastorm/reader.py#L61-L134

def make_batch_reader(dataset_url_or_urls,
                      schema_fields=None,
                      reader_pool_type='thread', workers_count=10,
                      shuffle_row_groups=True, shuffle_row_drop_partitions=1,
                      predicate=None,
                      rowgroup_selector=None,
                      num_epochs=1,
                      cur_shard=None, shard_count=None,
                      cache_type='null', cache_location=None, cache_size_limit=None,
                      cache_row_size_estimate=None, cache_extra_settings=None,
                      hdfs_driver='libhdfs3',
                      transform_spec=None,
                      filters=None,
                      s3_config_kwargs=None,
                      zmq_copy_buffers=True,
                      filesystem=None):

    Creates an instance of Reader for reading batches out of a non-Petastorm Parquet store.
    Currently, only stores having native scalar parquet data types are supported.
    Use :func:`~petastorm.make_reader` to read Petastorm Parquet stores generated with
    :func:`~petastorm.etl.dataset_metadata.materialize_dataset`.
    NOTE: only scalar columns or array type (of primitive type element) columns are currently supported.
    NOTE: If without `schema_fields` specified, the reader schema will be inferred from parquet dataset. then the
    reader schema fields order will preserve parqeut dataset fields order (partition column come first), but if
    setting `transform_spec` and specified `TransformSpec.selected_fields`, then the reader schema fields order
    will be the order of 'selected_fields'.
     dataset_url_or_urls: a url to a parquet directory or a url list (with the same scheme) to parquet files.
        e.g. ``'hdfs://some_hdfs_cluster/user/yevgeni/parquet8'``, or ``'file:///tmp/mydataset'``,
        or ``'s3://bucket/mydataset'``, or ``'gs://bucket/mydataset'``,
        or ``[file:///tmp/mydataset/00000.parquet, file:///tmp/mydataset/00001.parquet]``.
     schema_fields: A list of regex pattern strings. Only columns matching at least one of the
        patterns in the list will be loaded.
     reader_pool_type: A string denoting the reader pool type. Should be one of ['thread', 'process', 'dummy']
        denoting a thread pool, process pool, or running everything in the master thread. Defaults to 'thread'
     workers_count: An int for the number of workers to use in the reader pool. This only is used for the
        thread or process pool. Defaults to 10
     shuffle_row_groups: Whether to shuffle row groups (the order in which full row groups are read)
     shuffle_row_drop_partitions: This is is a positive integer which determines how many partitions to
        break up a row group into for increased shuffling in exchange for worse performance (extra reads).
        For example if you specify 2 each row group read will drop half of the rows within every row group and
        read the remaining rows in separate reads. It is recommended to keep this number below the regular row
        group size in order to not waste reads which drop all rows.
     predicate: instance of :class:`.PredicateBase` object to filter rows to be returned by reader. The predicate
        will be passed a pandas DataFrame object and must return a pandas Series with boolean values of matching
        dimensions.
     rowgroup_selector: instance of row group selector object to select row groups to be read
     num_epochs: An epoch is a single pass over all rows in the dataset. Setting ``num_epochs`` to
        ``None`` will result in an infinite number of epochs.
     cur_shard: An int denoting the current shard number. Each node reading a shard should
        pass in a unique shard number in the range [0, shard_count). shard_count must be supplied as well.
        Defaults to None
     shard_count: An int denoting the number of shards to break this dataset into. Defaults to None
     cache_type: A string denoting the cache type, if desired. Options are [None, 'null', 'local-disk'] to
        either have a null/noop cache or a cache implemented using diskcache. Caching is useful when communication
        to the main data store is either slow or expensive and the local machine has large enough storage
        to store entire dataset (or a partition of a dataset if shard_count is used). By default will be a null cache.
     cache_location: A string denoting the location or path of the cache.
     cache_size_limit: An int specifying the size limit of the cache in bytes
     cache_row_size_estimate: An int specifying the estimated size of a row in the dataset
     cache_extra_settings: A dictionary of extra settings to pass to the cache implementation,
     hdfs_driver: A string denoting the hdfs driver to use (if using a dataset on hdfs). Current choices are
        libhdfs (java through JNI) or libhdfs3 (C++)
     transform_spec: An instance of :class:`~petastorm.transform.TransformSpec` object defining how a record
        is transformed after it is loaded and decoded. The transformation occurs on a worker thread/process (depends
        on the ``reader_pool_type`` value).
     filters: (List[Tuple] or List[List[Tuple]]): Standard PyArrow filters.
        These will be applied when loading the parquet file with PyArrow. More information
        here: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html
     s3_config_kwargs: dict of parameters passed to ``botocore.client.Config``
     zmq_copy_buffers: A bool indicating whether to use 0mq copy buffers with ProcessPool.
     filesystem: An instance of ``pyarrow.FileSystem`` to use. Will ignore s3_config_kwargs and
        other filesystem configs if it's provided.
    :return: A :class:`Reader` object


    :return:
    """
    # model0 =  Keras model

    batch_size = 128
    num_classes = 10
    epochs = 12

    from petastorm.reader import Reader, make_batch_reader
    from petastorm.tf_utils import make_petastorm_dataset

    Xtrain, Xtest, yt   =get_dataset( , mode='petastorm')

    ### Inside fit
    train_reader = Reader( dataset_url_train, num_epochs=epochs)
    test_reader =  Reader( dataset_url_test,  num_epochs=epochs)

    train_dataset = make_petastorm_dataset(train_reader)
    train_dataset = train_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)


    test_dataset = make_petastorm_dataset(test_reader)
    test_dataset = test_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))
    test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

    hist = model0.fit(train_dataset,
              verbose=1,
              epochs=1,
              steps_per_epoch=100,
              validation_steps=10,
              validation_data=test_dataset)

    score = model.evaluate(test_dataset, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    train_reader.close()
    test_reader.close()




##############################################################################################
##############################################################################################
from __future__ import division, print_function

import argparse
import os

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential

from examples.mnist import DEFAULT_MNIST_DATA_PATH
from petastorm.reader import Reader
from petastorm.tf_utils import make_petastorm_dataset


def train_and_test(dataset_url, training_iterations, batch_size, evaluation_interval):
    batch_size = 128
    num_classes = 10
    epochs = 12

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    with Reader(os.path.join(dataset_url, 'train'), num_epochs=epochs) as train_reader:
        with Reader(os.path.join(dataset_url, 'test'), num_epochs=epochs) as test_reader:
            train_dataset = make_petastorm_dataset(train_reader)
            train_dataset = train_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))
            train_dataset = train_dataset.batch(batch_size, drop_remainder=True)


            test_dataset = make_petastorm_dataset(test_reader)
            test_dataset = test_dataset.map(lambda x: (tf.reshape(x.image, (28, 28, 1)), tf.reshape(x.digit, [1])))
            test_dataset = test_dataset.batch(batch_size, drop_remainder=True)

            model.fit(train_dataset,
                      verbose=1,
                      epochs=1,
                      steps_per_epoch=100,
                      validation_steps=10,
                      validation_data=test_dataset)

            score = model.evaluate(test_dataset, steps=10, verbose=0)
            print('Test loss:', score[0])
            print('Test accuracy:', score[1])

    train_reader.close()
    test_reader.close()


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Petastorm Tensorflow MNIST Example')
    default_dataset_url = 'file://{}'.format(DEFAULT_MNIST_DATA_PATH)
    parser.add_argument('--dataset-url', type=str,
                        default=default_dataset_url, metavar='S',
                        help='hdfs:// or file:/// URL to the MNIST petastorm dataset'
                             '(default: %s)' % default_dataset_url)
    parser.add_argument('--training-iterations', type=int, default=100, metavar='N',
                        help='number of training iterations to train (default: 100)')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--evaluation-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before evaluating the model accuracy (default: 10)')
    args = parser.parse_args()

    train_and_test(
        dataset_url=args.dataset_url,
        training_iterations=args.training_iterations,
        batch_size=args.batch_size,
        evaluation_interval=args.evaluation_interval,
    )




"""
https://github.com/uber/petastorm/tree/master/examples/hello_world/external_dataset

"""
from petastorm.tf_utils import tf_tensors, make_petastorm_dataset
def tensorflow_hello_world(dataset_url='file:///tmp/external_dataset'):
    # Example: tf_tensors will return tensors with dataset data
    with make_batch_reader(dataset_url) as reader:
        tensor = tf_tensors(reader)
        with tf.Session() as sess:
            # Because we are using make_batch_reader(), each read returns a batch of rows instead of a single row
            batched_sample = sess.run(tensor)
            print("id batch: {0}".format(batched_sample.id))

    # Example: use tf.data.Dataset API
    with make_batch_reader(dataset_url) as reader:
        dataset = make_petastorm_dataset(reader)
        iterator = dataset.make_one_shot_iterator()
        tensor = iterator.get_next()
        with tf.Session() as sess:
            batched_sample = sess.run(tensor)
            print("id batch: {0}".format(batched_sample.id))



"""Minimal example of how to read samples from a dataset generated by `generate_external_dataset.py`
using pytorch, using make_batch_reader() instead of make_reader()"""
from petastorm import make_batch_reader
from petastorm.pytorch import DataLoader


def pytorch_hello_world(dataset_url='file:///tmp/external_dataset'):
    with DataLoader(make_batch_reader(dataset_url)) as train_loader:
        sample = next(iter(train_loader))
        # Because we are using make_batch_reader(), each read returns a batch of rows instead of a single row
        print("id batch: {0}".format(sample['id']))



"""Minimal example of how to read samples from a dataset generated by `generate_non_petastorm_dataset.py`
using plain Python"""

from petastorm import make_batch_reader


def python_hello_world(dataset_url='file:///tmp/external_dataset'):
    # Reading data from the non-Petastorm Parquet via pure Python
    with make_batch_reader(dataset_url, schema_fields=["id", "value1", "value2"]) as reader:
        for schema_view in reader:
            # make_batch_reader() returns batches of rows instead of individual rows
            print("Batched read:\nid: {0} value1: {1} value2: {2}".format(
                schema_view.id, schema_view.value1, schema_view.value2))













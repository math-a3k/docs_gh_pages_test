HELP="""

Similarity between tensors

"""
import os, sys, pandas as pd, numpy as np
import tensorflow as tf


#################################################################################################
from utilmy.utilmy import log, log2

def help():
    from utilmy import help_create
    ss = help_create("utilmy.deeplearning.keras.util_similarity") + HELP
    print(ss)




######################################################################################
def test_all():
    test_tf_cdist()


def test_tf_cdist():
    from scipy.spatial.distance import cdist
    from tqdm import tqdm
    from utilmy.deeplearning.keras.util_similarity import tf_cdist
    import tensorflow as tf

    eps                                        = 1e-4
    sim_num                                    = 1000
    left_rows_count_min, left_rows_count_max   = 5, 20
    right_rows_count_min, right_rows_count_max = 5, 20
    dim_min, dim_max                           = 2, 5
    metrics                                    = ['euclidean', 'cosine']

    log('tf_cdist test started.')
    for _ in tqdm(range(sim_num)):
        for metric in metrics:
            left_rows_count = np.random.randint(
                left_rows_count_min, left_rows_count_max, 1
            )[0]
            right_rows_count = np.random.randint(
                right_rows_count_min, right_rows_count_max, 1
            )[0]
            dim            = np.random.randint(dim_min, dim_max, 1)[0]
            left_shape     = (left_rows_count, dim)
            right_shape    = (right_rows_count, dim)
            left           = tf.random.uniform(left_shape)
            right          = tf.random.uniform(right_shape)
            tf_dist_matrix = tf_cdist(left, right, metric)
            tf_dist_matrix = tf_dist_matrix.numpy()
            np_dist_matrix = cdist(left.numpy(), right.numpy(), metric)
            diff           = np.linalg.norm(tf_dist_matrix - np_dist_matrix)
            assert diff < eps, f'Accuracy error occurred. Error value: {diff}'
    log('tf_cdist test completed successfully.')





######################################################################################
def tf_cdist(left, right, metric='euclidean'):
    #### distance between tensor
    if metric == 'euclidean':
        return tf_cdist_euclidean(left, right)
    elif metric == 'cosine':
        return tf_cdist_cos(left, right)
    else:
        err_msg = f'Metric type not understood: value {metric} is not valid!'
        raise ValueError(err_msg)


def tf_cdist_euclidean(left, right):
    left, right                       = __cast_left_and_right_to_tensors(left, right)
    rows_count_left, rows_count_right = __get_rows_counts(left, right)
    left_sqr                          = __get_tensor_sqr(left, (-1, 1), (1, rows_count_right))
    right_sqr                         = __get_tensor_sqr(right, (1, -1), (rows_count_left, 1))
    left_right_mat_mul                = tf.matmul(left, tf.transpose(right))
    sqr_sum                           = left_sqr - 2.0 * left_right_mat_mul + right_sqr
    distance                          = tf.where(sqr_sum > 0.0, tf.sqrt(sqr_sum), 0.0)
    distance                          = tf.cast(distance, tf.float32)
    return distance

def tf_cdist_cos(left, right):
    left, right = __cast_left_and_right_to_tensors(left, right)
    norm_left   = __get_tensor_reshaped_norm(left, (-1, 1))
    norm_right  = __get_tensor_reshaped_norm(right, (1, -1))
    cos         = tf.matmul(left, tf.transpose(right)) / norm_left / norm_right
    distance    = 1.0 - cos
    distance    = tf.cast(distance, tf.float32)
    return distance



def __cast_left_and_right_to_tensors(left, right):
    left  = tf.cast(tf.convert_to_tensor(left), dtype=tf.float32)
    right = tf.cast(tf.convert_to_tensor(right), dtype=tf.float32)
    return left, right


def __get_rows_counts(left, right):
    count_left  = tf.shape(left)[0]
    count_right = tf.shape(right)[0]
    return count_left, count_right


def __get_tensor_sqr(tensor, reshape_shape, tile_shape):
    sqr = tf.pow(tensor, 2.0)
    sqr = tf.reduce_sum(sqr, axis=1)
    sqr = tf.reshape(sqr, reshape_shape)
    sqr = tf.tile(sqr, tile_shape)
    return sqr




def __get_tensor_reshaped_norm(tensor, reshape_shape):
    norm = tf.norm(tensor, axis=1)
    norm = tf.reshape(norm, reshape_shape)
    return norm



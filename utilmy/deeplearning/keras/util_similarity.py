import tensorflow as tf


def tf_cdist(left, right, metric='euclidean'):
    if metric == 'euclidean':
        return __tf_cdist_euclidean(left, right)
    elif metric == 'cosine':
        return __tf_cdist_cos(left, right)
    else:
        err_msg = f'Metric type not understood: value {metric} is not valid!'
        raise ValueError(err_msg)


def __tf_cdist_euclidean(left, right):
    left, right = __cast_left_and_right_to_tensors(left, right)
    rows_count_left, rows_count_right = __get_rows_counts(left, right)
    left_sqr = __get_tensor_sqr(left, (-1, 1), (1, rows_count_right))
    right_sqr = __get_tensor_sqr(right, (1, -1), (rows_count_left, 1))
    left_right_mat_mul = tf.matmul(left, tf.transpose(right))
    sqr_sum = left_sqr - 2.0 * left_right_mat_mul + right_sqr
    distance = tf.where(sqr_sum > 0.0, tf.sqrt(sqr_sum), 0.0)
    distance = tf.cast(distance, tf.float32)
    return distance


def __cast_left_and_right_to_tensors(left, right):
    left = tf.cast(tf.convert_to_tensor(left), dtype=tf.float64)
    right = tf.cast(tf.convert_to_tensor(right), dtype=tf.float64)
    return left, right


def __get_rows_counts(left, right):
    count_left = tf.shape(left)[0]
    count_right = tf.shape(right)[0]
    return count_left, count_right


def __get_tensor_sqr(tensor, reshape_shape, tile_shape):
    sqr = tf.pow(tensor, 2.0)
    sqr = tf.reduce_sum(sqr, axis=1)
    sqr = tf.reshape(sqr, reshape_shape)
    sqr = tf.tile(sqr, tile_shape)
    return sqr


def __tf_cdist_cos(left, right):
    left, right = __cast_left_and_right_to_tensors(left, right)
    norm_left = __get_tensor_reshaped_norm(left, (-1, 1))
    norm_right = __get_tensor_reshaped_norm(right, (1, -1))
    cos = tf.matmul(left, tf.transpose(right)) / norm_left / norm_right
    distance = 1.0 - cos
    distance = tf.cast(distance, tf.float32)
    return distance


def __get_tensor_reshaped_norm(tensor, reshape_shape):
    norm = tf.norm(tensor, axis=1)
    norm = tf.reshape(norm, reshape_shape)
    return norm

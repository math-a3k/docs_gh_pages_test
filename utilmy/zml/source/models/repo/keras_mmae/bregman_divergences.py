# -*- coding: utf-8 -*-
# Copyright (C) 2018-2020 Arno Onken
#
# This file is part of the mmae package.
#
# The mmae package is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# The mmae package is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.
"""
This module implements common Bregman divergences.

"""
from __future__ import division

import abc
try:
    from tensorflow.keras import backend

except ImportError:
    from keras import backend as K
from . import losses_utils

"""Built-in loss functions."""

import abc
import functools

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_util
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from .losses_utils import ReductionV2

@keras_export('keras.losses.Loss')
class Loss:
  """Loss base class.
  To be implemented by subclasses:
  * `call()`: Contains the logic for loss calculation using `y_true`, `y_pred`.
  Example subclass implementation:
  ```python
  class MeanSquaredError(Loss):
    def call(self, y_true, y_pred):
      y_pred = tf.convert_to_tensor_v2(y_pred)
      y_true = tf.cast(y_true, y_pred.dtype)
      return tf.reduce_mean(math_ops.square(y_pred - y_true), axis=-1)
  ```
  When used with `tf.distribute.Strategy`, outside of built-in training loops
  such as `tf.keras` `compile` and `fit`, please use 'SUM' or 'NONE' reduction
  types, and reduce losses explicitly in your training loop. Using 'AUTO' or
  'SUM_OVER_BATCH_SIZE' will raise an error.
  Please see this custom training [tutorial](
    https://www.tensorflow.org/tutorials/distribute/custom_training) for more
  details on this.
  You can implement 'SUM_OVER_BATCH_SIZE' using global batch size like:
  ```python
  with strategy.scope():
    loss_obj = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE)
    ....
    loss = (tf.reduce_sum(loss_obj(labels, predictions)) *
            (1. / global_batch_size))
  ```
  """

  def __init__(self, reduction=losses_utils.ReductionV2.AUTO, name=None):
    """Initializes `Loss` class.
    Args:
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: Optional name for the op.
    """
    losses_utils.ReductionV2.validate(reduction)
    self.reduction = reduction
    self.name = name
    # SUM_OVER_BATCH is only allowed in losses managed by `fit` or
    # CannedEstimators.
    self._allow_sum_over_batch_size = False
    self._set_name_scope()

  def _set_name_scope(self):
    """Creates a valid `name_scope` name."""
    if self.name is None:
      self._name_scope = self.__class__.__name__
    elif self.name == '<lambda>':
      self._name_scope = 'lambda'
    else:
      # E.g. '_my_loss' => 'my_loss'
      self._name_scope = self.name.strip('_')

  def __call__(self, y_true, y_pred, sample_weight=None):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
      sample_weight: Optional `sample_weight` acts as a coefficient for the
        loss. If a scalar is provided, then the loss is simply scaled by the
        given value. If `sample_weight` is a tensor of size `[batch_size]`, then
        the total loss for each sample of the batch is rescaled by the
        corresponding element in the `sample_weight` vector. If the shape of
        `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be broadcasted to
        this shape), then each loss element of `y_pred` is scaled
        by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
          functions reduce by 1 dimension, usually axis=-1.)
    Returns:
      Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
        shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
        because all loss functions reduce by 1 dimension, usually axis=-1.)
    Raises:
      ValueError: If the shape of `sample_weight` is invalid.
    """
    # If we are wrapping a lambda function strip '<>' from the name as it is not
    # accepted in scope name.
    graph_ctx = tf_utils.graph_context_for_symbolic_tensors(
        y_true, y_pred, sample_weight)
    with backend.name_scope(self._name_scope), graph_ctx:
      if context.executing_eagerly():
        call_fn = self.call
      else:
        call_fn = autograph.tf_convert(self.call, ag_ctx.control_status_ctx())
      losses = call_fn(y_true, y_pred)
      return losses_utils.compute_weighted_loss(
          losses, sample_weight, reduction=self._get_reduction())

  @classmethod
  def from_config(cls, config):
    """Instantiates a `Loss` from its config (output of `get_config()`).
    Args:
        config: Output of `get_config()`.
    Returns:
        A `Loss` instance.
    """
    return cls(**config)

  def get_config(self):
    """Returns the config dictionary for a `Loss` instance."""
    return {'reduction': self.reduction, 'name': self.name}

  @abc.abstractmethod
  @doc_controls.for_subclass_implementers
  def call(self, y_true, y_pred):
    """Invokes the `Loss` instance.
    Args:
      y_true: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
        sparse loss functions such as sparse categorical crossentropy where
        shape = `[batch_size, d0, .. dN-1]`
      y_pred: The predicted values. shape = `[batch_size, d0, .. dN]`
    Returns:
      Loss values with the shape `[batch_size, d0, .. dN-1]`.
    """
    raise NotImplementedError('Must be implemented in subclasses.')

  def _get_reduction(self):
    """Handles `AUTO` reduction cases and returns the reduction value."""
    if (not self._allow_sum_over_batch_size and
        distribution_strategy_context.has_strategy() and
        (self.reduction == losses_utils.ReductionV2.AUTO or
         self.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE)):
      raise ValueError(
          'Please use `tf.keras.losses.Reduction.SUM` or '
          '`tf.keras.losses.Reduction.NONE` for loss reduction when losses are '
          'used with `tf.distribute.Strategy` outside of the built-in training '
          'loops. You can implement '
          '`tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE` using global batch '
          'size like:\n```\nwith strategy.scope():\n'
          '    loss_obj = tf.keras.losses.CategoricalCrossentropy('
          'reduction=tf.keras.losses.Reduction.NONE)\n....\n'
          '    loss = tf.reduce_sum(loss_obj(labels, predictions)) * '
          '(1. / global_batch_size)\n```\nPlease see '
          'https://www.tensorflow.org/tutorials/distribute/custom_training'
          ' for more details.')

    if self.reduction == losses_utils.ReductionV2.AUTO:
      return losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE
    return self.reduction

class LossFunctionWrapper(Loss):
  """Wraps a loss function in the `Loss` class."""

  def __init__(self,
               fn,
               reduction=losses_utils.ReductionV2.AUTO,
               name=None,
               **kwargs):
    """Initializes `LossFunctionWrapper` class.
    Args:
      fn: The loss function to wrap, with signature `fn(y_true, y_pred,
        **kwargs)`.
      reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
        loss. Default value is `AUTO`. `AUTO` indicates that the reduction
        option will be determined by the usage context. For almost all cases
        this defaults to `SUM_OVER_BATCH_SIZE`. When used with
        `tf.distribute.Strategy`, outside of built-in training loops such as
        `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
        will raise an error. Please see this custom training [tutorial](
          https://www.tensorflow.org/tutorials/distribute/custom_training) for
            more details.
      name: (Optional) name for the loss.
      **kwargs: The keyword arguments that are passed on to `fn`.
    """
    super(LossFunctionWrapper, self).__init__(reduction=reduction, name=name)
    self.fn = fn
    self._fn_kwargs = kwargs

  def call(self, y_true, y_pred):
    """Invokes the `LossFunctionWrapper` instance.
    Args:
      y_true: Ground truth values.
      y_pred: The predicted values.
    Returns:
      Loss values per sample.
    """
    if tensor_util.is_tf_type(y_pred) and tensor_util.is_tf_type(y_true):
      y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)

    ag_fn = autograph.tf_convert(self.fn, ag_ctx.control_status_ctx())
    return ag_fn(y_true, y_pred, **self._fn_kwargs)

  def get_config(self):
    config = {}
    for k, v in self._fn_kwargs.items():
      config[k] = backend.eval(v) if tf_utils.is_tensor_or_variable(v) else v
    base_config = super(LossFunctionWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
class BregmanDivergence(LossFunctionWrapper):
    """
    This abstract class represents a Bregman divergence.  Override the abstract
    methods `_phi` and `_phi_gradient` to implement a specific Bregman
    divergence.

    Parameters
    ----------
    reduction : keras.utils.losses_utils.Reduction
        The type of loss reduction.
    name : str
        The name of the divergence.

    """
    def __init__(self, reduction=ReductionV2.SUM_OVER_BATCH_SIZE, name=None):

        def bregman_function(x, y):
            """
            This function implements the generic Bregman divergence.

            """
            return K.mean(self._phi(x) - self._phi(y)
                          - (x - y) * self._phi_gradient(y), axis=-1)

        super(BregmanDivergence, self).__init__(bregman_function, name=name,
                                                reduction=reduction)
        self.__name__ = name

    @abc.abstractmethod
    def _phi(self, z):
        """
        This is the phi function of the Bregman divergence.

        """
        pass

    @abc.abstractmethod
    def _phi_gradient(self, z):
        """
        This is the gradient of the phi function of the Bregman divergence.

        """
        pass


class GaussianDivergence(BregmanDivergence):
    """
    This class represents the squared Euclidean distance corresponding to a
    Gaussian noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'gaussian_divergence')

    """
    def __init__(self, name='gaussian_divergence'):
        super(GaussianDivergence, self).__init__(name=name)

    def _phi(self, z):
        return K.square(z) / 2.0

    def _phi_gradient(self, z):
        return z

gaussian_divergence = GaussianDivergence()


class GammaDivergence(BregmanDivergence):
    """
    This class represents the Itakura-Saito distance corresponding to a Gamma
    noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'gamma_divergence')

    """
    def __init__(self, name='gamma_divergence'):
        super(GammaDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return -K.log(z)

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return -1.0 / z

gamma_divergence = GammaDivergence()


class BernoulliDivergence(BregmanDivergence):
    """
    This class represents the logistic loss function corresponding to a
    Bernoulli noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'bernoulli_divergence')

    """
    def __init__(self, name='bernoulli_divergence'):
        super(BernoulliDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return z * K.log(z) + (1 - z) * K.log(1 - z)

    def _phi_gradient(self, z):
        z = K.clip(z, K.epsilon(), 1.0 - K.epsilon())
        return K.log(z) - K.log(1 - z)

bernoulli_divergence = BernoulliDivergence()


class PoissonDivergence(BregmanDivergence):
    """
    This class represents the generalized Kullback-Leibler divergence
    corresponding to a Poisson noise model.

    Parameters
    ----------
    name : str, optional
        The name of the divergence.  (Default: 'poisson_divergence')

    """
    def __init__(self, name='poisson_divergence'):
        super(PoissonDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return z * K.log(z) - z

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return K.log(z)

poisson_divergence = PoissonDivergence()


class BinomialDivergence(BregmanDivergence):
    """
    This class represents the loss function corresponding to a binomial noise
    model.

    Parameters
    ----------
    n : int
        The number of trials in the binomial noise model.  The number must be
        positive.
    name : str, optional
        The name of the divergence.  (Default: 'binomial_divergence')

    Attributes
    ----------
    n : int
        The number of trials in the binomial noise model.

    """
    def __init__(self, n, name='binomial_divergence'):
        self.n = n
        super(BinomialDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.clip(z, K.epsilon(), self.n - K.epsilon())
        return z * K.log(z) + (self.n - z) * K.log(self.n - z)

    def _phi_gradient(self, z):
        z = K.clip(z, K.epsilon(), self.n - K.epsilon())
        return K.log(z) - K.log(self.n - z)


class NegativeBinomialDivergence(BregmanDivergence):
    """
    This class represents the loss function corresponding to a negative
    binomial noise model.

    Parameters
    ----------
    r : int
        The number of failures in the negative binomial noise model.  The
        number must be positive.
    name : str, optional
        The name of the divergence.  (Default: 'negative_binomial_divergence')

    Attributes
    ----------
    r : int
        The number of failures in the negative binomial noise model.

    """
    def __init__(self, r, name='negative_binomial_divergence'):
        self.r = r
        super(NegativeBinomialDivergence, self).__init__(name=name)

    def _phi(self, z):
        z = K.maximum(z, K.epsilon())
        return z * K.log(z) - (self.r + z) * K.log(self.r + z)

    def _phi_gradient(self, z):
        z = K.maximum(z, K.epsilon())
        return K.log(z) - K.log(self.r + z)

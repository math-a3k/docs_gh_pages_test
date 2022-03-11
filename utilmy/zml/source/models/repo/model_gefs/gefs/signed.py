from numba import njit, boolean, int64, float64
from numba.experimental import jitclass
import numpy as np

from .utils import isin


@jitclass([
    ('value', float64[:]),
    ('sign', float64[:]),
    ('size', int64)
])
class signed:
    def __init__(self, value, sign=None):
        """ If sign is None, init from value in 'linear' space, possibly negative.
            Else, init from value in log-space.
        """
        if sign is None:
            self.value = np.log(np.absolute(value))
            self.sign = np.sign(value).astype(np.float64)
        else:
            self.value = value
            self.sign = sign
        self.size = len(value)
        self.sign[self.value==-np.Inf] = 0.

    def exp(self, exp):
        """ Exponentiate signed value by exp.
            Operates in log-space.
        """
        value = self.value*exp
        return signed(value, self.sign)

    def nonpositive(self):
        """ signed:nonpositive
        Args:
        Returns:
           
        """
        return self.sign <= 0

    def nonnegative(self):
        """ signed:nonnegative
        Args:
        Returns:
           
        """
        return self.sign >= 0

    def nonzero(self):
        """ signed:nonzero
        Args:
        Returns:
           
        """
        return (self.value > -np.Inf) & (self.sign != 0)

    def linear(self):
        """ signed:linear
        Args:
        Returns:
           
        """
        return self.sign * np.exp(self.value)

    def argsort(self, increasing=True):
        """ signed:argsort
        Args:
            increasing:     
        Returns:
           
        """
        negatives = np.zeros(self.size, dtype=boolean)
        order_positives = []
        order_negatives = []
        for i in range(self.size):
            if self.sign[i] < 0:
                negatives[i] = 1
                order_negatives.append(i)
            else:
                order_positives.append(i)
        if increasing:
            delta = 1
        else:
            delta = -1
        order_negatives = np.asarray([x for x, y in sorted(zip(order_negatives, -delta*self.value[negatives]), key = lambda x: x[1])])
        order_positives = np.asarray([x for x, y in sorted(zip(order_positives, delta*self.value[~negatives]), key = lambda x: x[1])])
        if increasing:
            return np.concatenate((order_negatives, order_positives))
        else:
            return np.concatenate((order_positives, order_negatives))

    def get(self, i):
        """ signed:get
        Args:
            i:     
        Returns:
           
        """
        assert i < self.size, "Index out of range."
        return signed(self.value[i:i+1], self.sign[i:i+1])

    def insert(self, sig, i):
        """ signed:insert
        Args:
            sig:     
            i:     
        Returns:
           
        """
        assert i < self.size, "Index out of range."
        assert sig.size == 1, "Can only insert signed with size 1."
        self.value[i] = sig.value[0]
        self.sign[i] = sig.sign[0]

    def reduce(self):
        """ signed:reduce
        Args:
        Returns:
           
        """
        if self.size == 1:
            return self
        max_i = np.argmax(self.value)
        if np.isinf(self.value[max_i]):
            return self.get(max_i)
        indicators = np.ones(self.size, dtype=boolean)
        indicators[max_i] = 0
        r = np.sum(np.exp(self.value[indicators] - self.value[max_i])*(self.sign[indicators]*self.sign[max_i]))
        if r < -1.:
            return signed(np.asarray([self.value[max_i]]) + np.log(-1.-r), np.asarray([-self.sign[max_i]]))
        res = signed(np.asarray([self.value[max_i]]) + np.log(r+1), np.asarray([self.sign[max_i]]))
        return res


@njit
def signed_join(x, y):
    """function signed_join
    Args:
        x:   
        y:   
    Returns:
        
    """
    if x is None:
        return y
    if y is None:
        return x
    return signed(np.concatenate((x.value, y.value)),
                  np.concatenate((x.sign, y.sign)))


@njit
def signed_prod(x, y):
    """function signed_prod
    Args:
        x:   
        y:   
    Returns:
        
    """
    res = signed(x.value + y.value, x.sign * y.sign)
    if res.value[0] == -np.Inf: res.sign[0] = 0.
    return res


@njit
def signed_sum_vec(x, y):
    """function signed_sum_vec
    Args:
        x:   
        y:   
    Returns:
        
    """
    if x.size == y.size:
        values = np.zeros(x.size)
        signs = np.zeros(x.size)
        for i in range(x.size):
            sum_ = signed_sum(x.get(i), y.get(i))
            values[i], signs[i] = sum_.value[0], sum_.sign[0]
    else:
        assert x.size == 1 or y.size == 1, 'If sizes do not match, only 1D arrays can be propagated.'
        if x.size == 1:
            values = np.zeros(y.size)
            signs = np.zeros(y.size)
            for i in range(y.size):
                sum_ = signed_sum(x.get(0), y.get(i))
                values[i], signs[i] = sum_.value[0], sum_.sign[0]
        else:
            values = np.zeros(x.size)
            signs = np.zeros(x.size)
            for i in range(x.size):
                sum_ = signed_sum(x.get(i), y.get(0))
                values[i], signs[i] = sum_.value[0], sum_.sign[0]
    return signed(values, signs)


@njit
def signed_sum(x, y):
    """function signed_sum
    Args:
        x:   
        y:   
    Returns:
        
    """
    assert (x.size == 1) & (y.size == 1), "Arrays must be one-dimensional."
    max_value = max(x.value[0], y.value[0])
    if max_value == x.value[0]:
        max_, min_ = x, y
    else:
        max_, min_ = y, x
    if np.isinf(max_value):
        return max_
    r = np.sum(np.exp(min_.value - max_.value)*(min_.sign*max_.sign))
    if r < -1.:
        return signed(max_.value + np.log(-1.-r), -max_.sign)
    res = signed(max_.value + np.log(r+1), max_.sign)
    if res.value[0] == -np.Inf: res.sign[0] = 0.
    return res


@njit
def signed_max_vec(x, y):
    """function signed_max_vec
    Args:
        x:   
        y:   
    Returns:
        
    """
    assert x.size == y.size, "Both arrays should have the same size."
    values = np.zeros(x.size)
    signs = np.zeros(x.size)
    for i in range(x.size):
        max_ = signed_max(x.get(i), y.get(i))
        values[i], signs[i] = max_.value[0], max_.sign[0]
    return signed(values, signs)


@njit
def signed_min_vec(x, y):
    """function signed_min_vec
    Args:
        x:   
        y:   
    Returns:
        
    """
    assert x.size == y.size, "Both arrays should have the same size."
    values = np.zeros(x.size)
    signs = np.zeros(x.size)
    for i in range(x.size):
        min_ = signed_min(x.get(i), y.get(i))
        values[i], signs[i] = min_.value[0], min_.sign[0]
    return signed(values, signs)


@njit
def signed_max(x, y):
    """function signed_max
    Args:
        x:   
        y:   
    Returns:
        
    """
    xs = x.sign[0]
    ys = y.sign[0]
    if xs > ys: return x
    if ys > xs: return y
    if ys*y.value[0] > xs*x.value[0]: return y
    return x


@njit
def signed_min(x, y):
    """function signed_min
    Args:
        x:   
        y:   
    Returns:
        
    """
    xs = x.sign[0]
    ys = y.sign[0]
    if xs > ys: return y
    if ys > xs: return x
    if ys*y.value[0] > xs*x.value[0]: return x
    return y


@njit
def signed_econtaminate(vec, signed_logprs, eps, ismax):
    """function signed_econtaminate
    Args:
        vec:   
        signed_logprs:   
        eps:   
        ismax:   
    Returns:
        
    """
    econt = np.asarray(vec) * (1-eps)
    room = 1 - np.sum(econt)
    if ismax:
        order = signed_logprs.argsort(False)
    else:
        order = signed_logprs.argsort(True)
    for i in order:
        if room > eps:
            econt[i] = econt[i] + eps
            room -= eps
        else:
            econt[i] = econt[i] + room
            break
    return signed(econt, None)

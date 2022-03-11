from __future__ import absolute_import, division
# -*- coding: utf-8 -*-
# an actual repository... in the meantime, here are some
# train/test split utilities for collaborative filtering with sparse matrices.
import numpy as np
from abc import ABCMeta, abstractmethod

from sklearn.externals import six
from sklearn.utils.validation import check_random_state
from sklearn.utils import validation as skval

from scipy import sparse

import numbers

__all__ = [
    'BootstrapCV',
    'check_cv',
    'train_test_split'
]

MAX_SEED = 1e6
ITYPE = np.int32
DTYPE = np.float64  # implicit asks for doubles, not float32s...


def test_all():
  """function test_all
  Args:
  Returns:
      
  """
  pass

def test1():
    """function test1
    Args:
    Returns:
        
    """
    from numpy.testing import assert_array_almost_equal
    import pytest

    # Define some "unit test" closures:
    def test_check_consistent_length():
        u = np.arange(5)
        i = np.arange(5)
        r = np.arange(5)

        # show they come back OK with u, i as the same refs and r changed
        users, items, ratings = check_consistent_length(u, i, r)
        assert u is users
        assert i is items
        assert ratings is not r  # dtype changed

        # change len of one
        i = np.arange(3)
        with pytest.raises(ValueError):
            check_consistent_length(u, i, r)


    def test_to_sparse_csr():
        row = np.array([0, 0, 1, 2, 2, 2])
        col = np.array([0, 2, 2, 0, 1, 2])
        data = np.array([1, 2, 3, 4, 5, 6])

        csr = to_sparse_csr(u=row, i=col, r=data, axis=0)
        assert sparse.issparse(csr)
        assert csr.nnz == 6, csr  # num stored
        assert_array_almost_equal(csr.toarray(),
                                  np.array([[1, 0, 2],
                                            [0, 0, 3],
                                            [4, 5, 6]]))

        # show what happens if we use the diff axis (it's .T basically)
        csrT = to_sparse_csr(u=row, i=col, r=data, axis=1)
        assert sparse.issparse(csrT)
        assert csrT.nnz == 6, csrT
        assert_array_almost_equal(csr.toarray(),
                                  csrT.T.toarray())


    def test_tr_te_split():
        u = [0, 1, 0, 2, 1, 3]
        i = [1, 2, 2, 0, 3, 2]
        r = [0.5, 1.0, 0.0, 1.0, 0.0, 1.]

        train, test = train_test_split(u, i, r, train_size=0.5,
                                       random_state=42)

        # one will be masked in the train array
        assert_array_almost_equal(
            train.toarray(),
            np.array([[0, 0.5, 0, 0],
                      [0, 0, 0, 0],  # masked
                      [1, 0, 0, 0],
                      [0, 0, 1, 0]]))

        assert_array_almost_equal(
            test.toarray(),
            np.array([[0, 0.5, 0, 0],
                      [0, 0, 1, 0],
                      [1, 0, 0, 0],
                      [0, 0, 1, 0]]))


    def test_check_cv():
        cv = check_cv(None)
        assert isinstance(cv, BootstrapCV)
        assert cv.n_splits == 3

        cv = check_cv(5)
        assert isinstance(cv, BootstrapCV)
        assert cv.n_splits == 5

        cv = BootstrapCV(n_splits=3, random_state=42)
        cv2 = check_cv(cv)
        assert cv is cv2
        assert cv2.n_splits == 3
        assert cv2.random_state == 42


    # Run all of the tests
    test_check_consistent_length()
    test_to_sparse_csr()
    test_tr_te_split()
    test_check_cv()


def check_consistent_length(u, i, r):
    """Ensure users, items, and ratings are all of the same dimension.
    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        A numpy array of the users.
    i : array-like, shape=(n_samples,)
        A numpy array of the items.
    r : array-like, shape=(n_samples,)
        A numpy array of the ratings.
    """
    skval.check_consistent_length(u, i, r)
    return np.asarray(u), np.asarray(i), np.asarray(r, dtype=DTYPE)


def _make_sparse_csr(data, rows, cols, dtype=DTYPE):
    """function _make_sparse_csr
    Args:
        data:   
        rows:   
        cols:   
        dtype:   
    Returns:
        
    """
    # check lengths
    check_consistent_length(data, rows, cols)
    data, rows, cols = (np.asarray(x) for x in (data, rows, cols))

    shape = (np.unique(rows).shape[0], np.unique(cols).shape[0])
    return sparse.csr_matrix((data, (rows, cols)),
                             shape=shape, dtype=dtype)


def to_sparse_csr(u, i, r, axis=0, dtype=DTYPE):
    """Create a sparse ratings matrix.
    Create a sparse ratings matrix with users and items as rows and columns,
    and ratings as the values.
    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        The user vector. Positioned along the row axis if ``axis=0``,
        otherwise positioned along the column axis.
    i : array-like, shape=(n_samples,)
        The item vector. Positioned along the column axis if ``axis=0``,
        otherwise positioned along the row axis.
    r : array-like, shape=(n_samples,)
        The ratings vector.
    axis : int, optional (default=0)
        The axis along which to position the users. If 0, the users are
        along the rows (with items as columns). If 1, the users are columns
        with items as rows.
    dtype : type, optional (default=np.float32)
        The type of the values in the ratings matrix.
    """
    if axis not in (0, 1):
        raise ValueError("axis must be an int in (0, 1)")

    rows = u if axis == 0 else i
    cols = i if axis == 0 else u
    return _make_sparse_csr(data=r, rows=rows, cols=cols, dtype=dtype)


def check_cv(cv=3):
    """Input validation for cross-validation classes.
    Parameters
    ----------
    cv : int, None or BaseCrossValidator
        The CV class or number of folds.
        - None will default to 3-fold BootstrapCV
        - integer will default to ``integer``-fold BootstrapCV
        - BaseCrossValidator will pass through untouched
    Returns
    -------
    checked_cv : BaseCrossValidator
        The validated CV class
    """
    if cv is None:
        cv = 3

    if isinstance(cv, numbers.Integral):
        return BootstrapCV(n_splits=int(cv))
    if not hasattr(cv, "split") or isinstance(cv, six.string_types):
        raise ValueError("Expected integer or CV class, but got %r (type=%s)"
                         % (cv, type(cv)))
    return cv


def _validate_train_size(train_size):
    """Train size should be a float between 0 and 1."""
    assert isinstance(train_size, float) and (0. < train_size < 1.), \
        "train_size should be a float between 0 and 1"


def _get_stratified_tr_mask(u, i, train_size, random_state):
    """function _get_stratified_tr_mask
    Args:
        u:   
        i:   
        train_size:   
        random_state:   
    Returns:
        
    """
    _validate_train_size(train_size)  # validate it's a float
    random_state = check_random_state(random_state)
    n_events = u.shape[0]

    # this is our train mask that we'll update over the course of this method
    train_mask = random_state.rand(n_events) <= train_size  # type: np.ndarray

    # we have a random mask now. For each of users and items, determine which
    # are missing from the mask and randomly select one of each of their
    # ratings to force them into the mask
    for array in (u, i):
        # e.g.:
        # >>> array = np.array([1, 2, 3, 3, 1, 3, 2])
        # >>> train_mask = np.array([0, 1, 1, 1, 0, 0, 1]).astype(bool)
        # >>> unique, counts = np.unique(array, return_counts=True)
        # >>> unique, counts
        # (array([1, 2, 3]), array([2, 2, 3]))

        # then present:
        # >>> present
        # array([2, 3, 3, 2])
        present = array[train_mask]

        # and the test indices:
        # >>> test_vals
        # array([1, 1, 3])
        test_vals = array[~train_mask]

        # get the test indices that are NOT present (either
        # missing items or users)
        # >>> missing
        # array([1])
        missing = np.unique(test_vals[np.where(
            ~np.in1d(test_vals, present))[0]])

        # If there is nothing missing, we got perfectly lucky with our random
        # split and we'll just go with it...
        if missing.shape[0] == 0:
            continue

        # Otherwise, if we get to this point, we have to add in the missing
        # level to the mask to make sure at least one of each of those makes
        # it into the training data (so we don't lose a factor level for ALS)
        array_mask_missing = np.in1d(array, missing)

        # indices in "array" where we have a level that's currently missing
        # and that needs to be added into the mask
        where_missing = np.where(array_mask_missing)[0]  # e.g., array([0, 4])

        # I don't love having to loop here... but we'll iterate "where_missing"
        # to incrementally add in items or users until all are represented
        # in the training set to some degree
        added = set()
        for idx, val in zip(where_missing, array[where_missing]):
            # if we've already seen and added this one
            if val in added:  # O(1) lookup
                continue

            train_mask[idx] = True
            added.add(val)

    return train_mask


def _make_sparse_tr_te(users, items, ratings, train_mask):
    """function _make_sparse_tr_te
    Args:
        users:   
        items:   
        ratings:   
        train_mask:   
    Returns:
        
    """
    # now make the sparse matrices
    r_train = to_sparse_csr(u=users[train_mask], i=items[train_mask],
                            r=ratings[train_mask], axis=0)

    # TODO: anti mask for removing from test set?
    r_test = to_sparse_csr(u=users, i=items, r=ratings, axis=0)
    return r_train, r_test


def train_test_split(u, i, r, train_size=0.75, random_state=None):
    """Create a train/test split for sparse ratings.
    Given vectors of users, items, and ratings, create a train/test split
    that preserves at least one of each user and item in the training split
    to prevent inducing a cold-start situation.
    Parameters
    ----------
    u : array-like, shape=(n_samples,)
        A numpy array of the users. This vector will be used to stratify the
        split to ensure that at least of each of the users will be included
        in the training split. Note that this diminishes the likelihood of a
        perfectly-sized split (i.e., ``len(train)`` may not exactly equal
        ``train_size * n_samples``).
    i : array-like, shape=(n_samples,)
        A numpy array of the items. This vector will be used to stratify the
        split to ensure that at least of each of the items will be included
        in the training split. Note that this diminishes the likelihood of a
        perfectly-sized split (i.e., ``len(train)`` may not exactly equal
        ``train_size * n_samples``).
    r : array-like, shape=(n_samples,)
        A numpy array of the ratings.
    train_size : float, optional (default=0.75)
        The ratio of the train set size. Should be a float between 0 and 1.
    random_state : RandomState, int or None, optional (default=None)
        The random state used to create the train mask.
    Examples
    --------
    An example of a sparse matrix split that masks some ratings from the train
    set, but not from the testing set:
    >>> u = [0, 1, 0, 2, 1, 3]
    >>> i = [1, 2, 2, 0, 3, 2]
    >>> r = [0.5, 1.0, 0.0, 1.0, 0.0, 1.]
    >>> train, test = train_test_split(u, i, r, train_size=0.5,
    ...                                random_state=42)
    >>> train.toarray()
    array([[ 0. ,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0. ,  0. ],
           [ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ]], dtype=float32)
    >>> test.toarray()
    array([[ 0. ,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ],
           [ 1. ,  0. ,  0. ,  0. ],
           [ 0. ,  0. ,  1. ,  0. ]], dtype=float32)
    Here's a more robust example (with more ratings):
    >>> from sklearn.preprocessing import LabelEncoder
    >>> import numpy as np
    >>> rs = np.random.RandomState(42)
    >>> users = np.arange(100000)  # 100k users in DB
    >>> items = np.arange(30000)  # 30k items in DB
    >>> # Randomly select some for ratings:
    >>> items = rs.choice(items, users.shape[0])  # 100k rand item rtgs
    >>> users = rs.choice(users, users.shape[0])  # 100k rand user rtgs
    >>> # Label encode so they're positional indices:
    >>> users = LabelEncoder().fit_transform(users)
    >>> items = LabelEncoder().fit_transform(items)
    >>> ratings = rs.choice((0., 0.25, 0.5, 0.75, 1.), items.shape[0])
    >>> train, test = train_test_split(users, items, ratings, random_state=rs)
    >>> train
    <26353x28921 sparse matrix of type '<type 'numpy.float32'>'
        with 77770 stored elements in Compressed Sparse Row format>
    >>> test
    <26353x28921 sparse matrix of type '<type 'numpy.float32'>'
        with 99994 stored elements in Compressed Sparse Row format>
        
    Notes
    -----
    ``u``, ``i`` inputs should be encoded (i.e., via LabelEncoder) prior to
    splitting the data. This is due to the indexing behavior used within the
    function.
    Returns
    -------
    r_train : scipy.sparse.csr_matrix
        The train set.
    r_test : scipy.sparse.csr_matrix
        The test set.
    """
    # make sure all of them are numpy arrays and of the same length
    users, items, ratings = check_consistent_length(u, i, r)

    train_mask = _get_stratified_tr_mask(
        users, items, train_size=train_size,
        random_state=random_state)

    return _make_sparse_tr_te(users, items, ratings, train_mask=train_mask)


# avoid pb w nose
train_test_split.__test__ = False


class BaseCrossValidator(six.with_metaclass(ABCMeta)):
    """Base class for all collab CV.
    Iterations must define ``_iter_train_mask``. This is based loosely
    on sklearn's cross validator but does not adhere to its exact
    interface.
    """

    def __init__(self, n_splits=3, random_state=None):
        """ BaseCrossValidator:__init__
        Args:
            n_splits:     
            random_state:     
        Returns:
           
        """
        self.n_splits = n_splits
        self.random_state = random_state

    def get_n_splits(self):
        """ BaseCrossValidator:get_n_splits
        Args:
        Returns:
           
        """
        return self.n_splits

    def split(self, X):
        """Generate indices to split data into training and test sets.
        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            A sparse ratings matrix.
        Returns
        -------
        train : scipy.sparse.csr_matrix
            The training set
        test : scipy.sparse.csr_matrix
            The test set
        """
        ratings = X.data
        users, items = X.nonzero()

        # make sure all of them are numpy arrays and of the same length
        # users, items, ratings = check_consistent_length(u, i, r)
        for train_mask in self._iter_train_mask(users, items, ratings):
            # yield in a generator so we don't have to store in mem
            yield _make_sparse_tr_te(users, items, ratings,
                                     train_mask=train_mask)

    @abstractmethod
    def _iter_train_mask(self, u, i, r):
        """Compute the training mask here.
        Returns
        -------
        train_mask : np.ndarray
            The train mask
        """


class BootstrapCV(BaseCrossValidator):
    """Cross-validate with bootstrapping.
    The bootstrap CV class makes no guarantees about exclusivity between folds.
    This is simply a naive way to handle KFold cross-validation for something as
    complex as a collaborative filtering split.
    """

    def _iter_train_mask(self, u, i, r):
        """Compute the training mask here."""
        train_size = 1. - (1. / self.n_splits)
        # train_size = 1. - ((n_samples / self.n_splits) / n_samples)
        random_state = check_random_state(self.random_state)

        for split in range(self.n_splits):
            yield _get_stratified_tr_mask(
                u, i, train_size=train_size,
                random_state=random_state.randint(MAX_SEED))


# This is not an acceptable way to unit test, but it's an easy way for
# you to sanity check the code I've got here :-)
if __name__ == '__main__':
    tset1()
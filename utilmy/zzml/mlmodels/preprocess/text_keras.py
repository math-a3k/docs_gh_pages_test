import keras
import pandas as pd
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np


class Preprocess_namentity:
    def __init__(self,max_len,**args):
        """ Preprocess_namentity:__init__
        Args:
            max_len:     
            **args:     
        Returns:
           
        """
        self.max_len = max_len
    
    def compute(self,df):
        """ Preprocess_namentity:compute
        Args:
            df:     
        Returns:
           
        """
        df = df.fillna(method='ffill')
        ##### Get sentences
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                    s['POS'].values.tolist(),
                                                    s['Tag'].values.tolist())]
        grouped = df.groupby("Sentence #").apply(agg)
        sentences = [s for s in grouped]
    
        # Getting unique words and labels from data
        words = list(df['Word'].unique())
        tags = list(df['Tag'].unique())
        # Dictionary word:index pair
        # word is key and its value is corresponding index
        word_to_index = {w: i + 2 for i, w in enumerate(words)}
        word_to_index["UNK"] = 1
        word_to_index["PAD"] = 0
    
        # Dictionary lable:index pair
        # label is key and value is index.
        tag_to_index = {t: i + 1 for i, t in enumerate(tags)}
        tag_to_index["PAD"] = 0
    
        idx2word = {i: w for w, i in word_to_index.items()}
        idx2tag = {i: w for w, i in tag_to_index.items()}
    
    
        # Converting each sentence into list of index from list of tokens
        X = [[word_to_index[w[0]] for w in s] for s in sentences]
    
        # Padding each sequence to have same length  of each word
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=word_to_index["PAD"])
    
        # Convert label to index
        y = [[tag_to_index[w[2]] for w in s] for s in sentences]
    
        # padding
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=tag_to_index["PAD"])
        num_tag = df['Tag'].nunique()
        # One hot encoded labels
        y = np.array([to_categorical(i, num_classes=num_tag + 1) for i in y])
        self.data = {"X": X, "y":y,"word_count":len(df['Word'].unique())+2}

    def get_data(self):
        """ Preprocess_namentity:get_data
        Args:
        Returns:
           
        """
        return self.data


def _remove_long_seq(maxlen, seq, label):
    """Removes sequences that exceed the maximum length.

    # Arguments
        maxlen: Int, maximum length of the output sequences.
        seq: List of lists, where each sublist is a sequence.
        label: List where each element is an integer.

    # Returns
        new_seq, new_label: shortened lists for `seq` and `label`.
    """
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label


class IMDBDataset:

    def __init__(self, *args, **kwargs):
        """ IMDBDataset:__init__
        Args:
            *args:     
            **kwargs:     
        Returns:
           
        """
        self.start_char = kwargs.get("start_char", 1)
        self.oov_char = kwargs.get("oov_char", 2)
        self.index_from = kwargs.get("index_from", 3)
        self.num_words = kwargs.get("num_words", None)
        self.maxlen = kwargs.get("maxlen", None)
        self.skip_top = kwargs.get("skip_top", 0)

    def compute(self, data):
        """ IMDBDataset:compute
        Args:
            data:     
        Returns:
           
        """
        x_test, x_train, labels_test, labels_train = data
        indices = np.arange(len(x_train))
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        np.random.seed(113)
        np.random.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        xs = np.concatenate([x_train, x_test])
        labels = np.concatenate([labels_train, labels_test])

        if self.start_char is not None:
            xs = [[self.start_char] + [w + self.index_from for w in x] for x in xs]
        elif self.index_from:
            xs = [[w + self.index_from for w in x] for x in xs]

        if self.maxlen:
            xs, labels = _remove_long_seq(self.maxlen, xs, labels)
            if not xs:
                raise ValueError('After filtering for sequences shorter than maxlen=' +
                                 str(self.maxlen) + ', no sequence was kept. '
                                               'Increase maxlen.')
        if not self.num_words:
            self.num_words = max([max(x) for x in xs])

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if self.oov_char is not None:
            xs = [[w if (self.skip_top <= w < self.num_words) else self.oov_char for w in x]
                  for x in xs]
        else:
            xs = [[w for w in x if self.skip_top <= w < self.num_words]
                  for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
        x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

        self.data = x_train, y_train, x_test, y_test

    def get_data(self):
        return self.data
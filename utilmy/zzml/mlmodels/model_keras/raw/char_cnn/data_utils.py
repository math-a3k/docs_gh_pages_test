import keras
import numpy as np
import re
import csv
from keras.preprocessing.text import Tokenizer


class Data(object):
    """
    Class to handle loading and processing of raw datasets.
    """
    def __init__(self, data_source,
                 alphabet="abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}",
                 input_size=1014, num_of_classes=4):
        """
        Initialization of a Data object.

        Args:
            data_source (str): Raw data file path
            alphabet (str): Alphabet of characters to index
            input_size (int): Size of input features
            num_of_classes (int): Number of classes in data
        """
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.dict = {}  # Maps each character to an integer
        self.no_of_classes = num_of_classes
        for idx, char in enumerate(self.alphabet):
            self.dict[char] = idx + 1
        self.length = input_size
        self.data_source = data_source

    def load_data(self):
        """
        Load raw data from the source file into data variable.

        Returns: None

        """
        data = []
        with open(self.data_source, 'r', encoding='utf-8') as f:
            rdr = csv.reader(f, delimiter=',', quotechar='"')
            for row in rdr:
                txt = ""
                for s in row[1:]:
                    txt = txt + " " + re.sub("^\s*(.-)\s*$", "%1", s).replace("\\n", "\n")
                data.append((int(row[0]), txt))  # format: (label, text)
        self.data = np.array(data)
        print("Data loaded from " + self.data_source)

    def get_all_data(self):
        """
        Return all loaded data from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            c = int(c) - 1
            classes.append(one_hot[c])
        
        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def compute(self, input, *args, **kwargs):
        """ Data:compute
        Args:
            input:     
            *args:     
            **kwargs:     
        Returns:
           
        """
        X, y = np.array(input[kwargs.get("colX", "colX")].values), np.array(input[kwargs.get("coly", "coly")].values)
        data = []
        for idx, x in enumerate(X):
            data.append((int(y[idx]), x))
        self.data = np.array(data)

    def get_data(self):
        """
        Return all loaded data from data variable.

        Returns:
            (np.ndarray) Data transformed from raw to indexed form with associated one-hot label.

        """
        data_size = len(self.data)
        start_index = 0
        end_index = data_size
        batch_texts = self.data[start_index:end_index]
        batch_indices = []
        one_hot = np.eye(self.no_of_classes, dtype='int64')
        classes = []
        for c, s in batch_texts:
            batch_indices.append(self.str_to_indexes(s))
            c = int(c) - 1
            classes.append(one_hot[c])

        return np.asarray(batch_indices, dtype='int64'), np.asarray(classes)

    def get_all_data_npz(self):
        """ Data:get_all_data_npz
        Args:
        Returns:
           
        """
        with np.load(self.data_source, allow_pickle=True) as f:
            print("f: ", f)
            xs, labels = f['x'], f['y']
        length = xs.size
        size_train = int(length * .8)
        xs_train = xs[:size_train]
        labels_train = labels[:size_train]
        xs_val = xs[size_train:]
        labels_val = labels[size_train:]
        
        max_words = self.length

        tokenizer = Tokenizer(num_words=max_words)
        x_train = tokenizer.sequences_to_matrix(xs_train, mode='binary')
        x_test = tokenizer.sequences_to_matrix(xs_val, mode='binary')

        y_train = keras.utils.to_categorical(labels_train, self.no_of_classes)
        y_test = keras.utils.to_categorical(labels_val, self.no_of_classes)

        return x_train, y_train, x_test, y_test
    
    def fill_to_length_input(self,s):
        """ Data:fill_to_length_input
        Args:
            s:     
        Returns:
           
        """
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx

    def str_to_indexes(self, s):
        """
        Convert a string to character indexes based on character dictionary.
        
        Args:
            s (str): String to be converted to indexes

        Returns:
            str2idx (np.ndarray): Indexes of characters in s

        """
        s = s.lower()
        max_length = min(len(s), self.length)
        str2idx = np.zeros(self.length, dtype='int64')
        for i in range(1, max_length + 1):
            c = s[-i]
            if c in self.dict:
                str2idx[i - 1] = self.dict[c]
        return str2idx

import argparse, os, sys; from time import time
import tensorflow as tf
import codecs, collections,pickle, numpy as np
###########################################################################################################
import os, sys


#CFG   = {'plat': sys.platform[:3]+"-"+os.path.expanduser('~').split("\\")[-1].split("/")[-1], "ver": sys.version_info.major}
#DIRCWD= {'win-asus1': 'D:/_devs/Python01/project27/', 'win-unerry': 'G:/_devs/project27/' , 'lin-noel': '/home/noel/project27/', 'lin-ubuntu': '/home/ubuntu/project27/' }[CFG['plat']]
#os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')
# DIRCWD= os.environ["DIRCWD"]; os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')

import configmy; CFG, DIRCWD= configmy.get(config_file="_ROOT", output= ["_CFG", "DIRCWD"])
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage')



__path__= DIRCWD +'/aapackage/'
__version__= "1.0.0"
__file__= "util_ml.py"





###########################################################################################################
def create_weight_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.Variable(initializer(shape=shape), name=name)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.Variable(initializer(shape=shape), name)


def create_adam_optimizer(learning_rate, momentum):
    return tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-4)




def tf_check():
   print("Checking TF package");   print(tf)
   from tensorflow.examples.tutorials.mnist import input_data
   from tensorflow.python.ops.rnn import dynamic_rnn
   from tensorflow.contrib.rnn import BasicLSTMCell
   from tensorflow.contrib.rnn import LSTMCell, GRUCell
   from tensorflow.contrib.rnn import RNNCell
   from tensorflow.python.ops import rnn_cell_impl
   from tensorflow.python.framework import dtypes
   from tensorflow.python.framework import ops
   from tensorflow.python.ops import array_ops
   from tensorflow.python.ops import init_ops
   from tensorflow.python.ops import random_ops
   from tensorflow.python.ops import variable_scope as vs
   from tensorflow.python.ops.math_ops import sigmoid
   from tensorflow.python.ops.math_ops import tanh

   # Test
   a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
   b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
   c = tf.matmul(a, b)
   # Creates a session with log_device_placement set to True.
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
   # Runs the op.
   print("Matmul Operation:", sess.run(c))


os.environ



def parse_args(ppa=None, args= {}) :
  if ppa is None :  ppa = argparse.ArgumentParser()
  for key, x in args.items() :
    ppa.add_argument('--'+key, type=type(x[0]), default= x[0]  ,       help=x[1])

  args2 = ppa.parse_args()
  return ppa, args2



########### Argument parsing ###########################################################################
def parse_args2(ppa=None) :
 if ppa is None :
    try :
      ppa = argparse.ArgumentParser()
    except Exception as e:print(e)

 ppa.add_argument('--data_dir', type=str, default= ''  ,       help='data directory containing input.txt')
 ppa.add_argument('--save_dir', type=str, default= '',         help='directory to store checkpointed models')
 ppa.add_argument('--save_every', type=int, default=1000,      help='save frequency')


 ppa.add_argument('--batch_size', type=int, default=50,                 help='minibatch size')
 ppa.add_argument('--n_epochs', type=int, default=50,                   help='number of epochs')
 ppa.add_argument('--batch_per_epoch', type=str,  default=80,           help='batches per epoch')
 ppa.add_argument('--learning_rate', type=float, default=0.002,         help='learning rate')
 ppa.add_argument('--decay_rate', type=float, default=0.97,             help='decay rate for rmsprop')



 ppa.add_argument('--model',  type=str, default='lstm',                  help='rnn, gru, lstm, gridlstm, gridgru')
 ppa.add_argument('--model2', type=str, default='lstmCell',             help='rnn, gru, lstm, gridlstm, gridgru')
 ppa.add_argument('--model3', type=str, default='lstmCell',             help='rnn, gru, lstm, gridlstm, gridgru')


 ppa.add_argument('--n_hidden',   type=str, default=100,   help='hidden units in the recurrent layer')  # 100
 ppa.add_argument('--num_layers', type=int, default=2,                  help='number of layers in the RNN')
 ppa.add_argument('--rnn_size',   type=int, default=128,                  help='size of RNN hidden state')
 ppa.add_argument('--seq_length', type=int, default=50,                 help='RNN sequence length')
 ppa.add_argument('--grad_clip',  type=float, default=5.,                help='clip gradients at this value')

 return ppa
########################################################################################################





# Smart initialize for versions < 0.12.0   ##############################################################
def tf_global_variables_initializer(sess=None):
    """Initializes all uninitialized variables in correct order. Initializers
    are only run for uninitialized variables, so it's safe to run this multiple times.
    Args:   sess: session to use. Use default session if None.
    """
    from tensorflow.contrib import graph_editor as ge

    def make_initializer(var):
        def f():  return tf.assign(var, var.initial_value).op
        return f

    def make_noop():
        return tf.no_op()

    def make_safe_initializer(var):
        """Returns initializer op that only runs for uninitialized ops."""
        return tf.cond(tf.is_variable_initialized(var), make_noop,
                       make_initializer(var), name='safe_init_' + var.op.name).op

    if not sess:  sess = tf.get_default_session()
    g = tf.get_default_graph()

    safe_initializers = {}
    for v in tf.global_variables():
        safe_initializers[v.op.name] = make_safe_initializer(v)

    # initializers access variable vaue through read-only value cached in
    # <varname>/read, so add control dependency to trigger safe_initializer on read access
    for v in tf.global_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + '/read')
        ge.reroute.add_control_inputs(var_cache, [safe_initializers[var_name]])

    sess.run(tf.group(*safe_initializers.values()))

    # remove initializer dependencies to avoid slowing down future variable reads
    for v in tf.global_variables():
        var_name = v.op.name
        var_cache = g.get_operation_by_name(var_name + '/read')
        ge.reroute.remove_control_inputs(var_cache, [safe_initializers[var_name]])



class TextLoader(object):
    def __init__(self, data_dir, batch_size, seq_length):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_length = seq_length

        input_file = os.path.join(data_dir, "input.txt")
        vocab_file = os.path.join(data_dir, "vocab.pkl")
        tensor_file = os.path.join(data_dir, "data.npy")

        if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
            print("reading text file")
            self.preprocess(input_file, vocab_file, tensor_file)
        else:
            print("loading preprocessed files")
            self.load_preprocessed(vocab_file, tensor_file)
        self.create_batches()
        self.reset_batch_pointer()

    def preprocess(self, input_file, vocab_file, tensor_file):
        with codecs.open(input_file, "r") as f:
            data = f.read()
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        self.chars, _ = list(zip(*count_pairs))
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        with open(vocab_file, 'wb') as f:
            pickle.dump(self.chars, f)
        self.tensor = np.array(list(map(self.vocab.get, data)))
        np.save(tensor_file, self.tensor)

    def load_preprocessed(self, vocab_file, tensor_file):
        with open(vocab_file, 'rb') as f:
            self.chars = pickle.load(f)
        self.vocab_size = len(self.chars)
        self.vocab = dict(zip(self.chars, range(len(self.chars))))
        self.tensor = np.load(tensor_file)
        self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)

    def create_batches(self):
        self.num_batches = self.tensor.size // (self.batch_size * self.seq_length)
        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length]
        xdata = self.tensor
        ydata = np.copy(self.tensor)
        ydata[:-1] = xdata[1:]
        ydata[-1] = xdata[0]
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1), self.num_batches, 1)

        validation_batches = int(self.num_batches * .2)
        self.val_batches = zip(self.x_batches[-validation_batches:], self.y_batches[-validation_batches:])
        self.x_batches = self.x_batches[:-validation_batches]
        self.y_batches = self.y_batches[:-validation_batches]
        self.num_batches -= validation_batches

    def next_batch(self):
        x, y = self.x_batches[self.pointer], self.y_batches[self.pointer]
        self.pointer += 1
        return x, y

    def reset_batch_pointer(self):
        self.pointer = 0




def visualize_result():
    import pandas as pd
    import matplotlib.pyplot as plt

    # These are the "Tableau 20" colors as RGB.
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
                 (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
                 (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
                 (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
                 (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau20)):
        r, g, b = tableau20[i]
        tableau20[i] = (r / 255., g / 255., b / 255.)

    files = [('GridGRU, 3 layers', 'save_gridgru3layers/log.csv'),
             # ('GridGRU, 6 layers', 'save_gridgru6layers/log.csv'),
             ('GridLSTM, 3 layers', 'save_gridlstm3layers/log.csv'),
             ('GridLSTM, 6 layers', 'save_gridlstm6layers/log.csv'),
             ('Stacked GRU, 3 layers', 'save_gru3layers/log.csv'),
             # ('Stacked GRU, 6 layers', 'save_gru6layers/log.csv'),
             ('Stacked LSTM, 3 layers', 'save_lstm3layers/log.csv'),
             ('Stacked LSTM, 6 layers', 'save_lstm6layers/log.csv'),
             ('Stacked RNN, 3 layers', 'save_rnn3layers/log.csv'),
             ('Stacked RNN, 6 layers', 'save_rnn6layers/log.csv')]
    for i, (k, v) in enumerate(files):
        train_loss = pd.read_csv('./save/tinyshakespeare/{}'.format(v)).groupby('epoch').mean()['train_loss']
        plt.plot(train_loss.index.tolist(), train_loss.tolist(), label=k, lw=2, color=tableau20[i * 2])
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Average training loss')
    plt.show()







'''

Transcript - direct print output to a file, in addition to terminal.

Usage:
    import transcript
    transcript.start('logfile.log')
    print("inside file")
    transcript.stop()
    print("outside file")
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def start(filename):
      """Start transcript, appending print output to given filename"""
      sys.stdout = Transcript(filename)

def stop():
      """Stop transcript and return print functionality to normal"""
      sys.stdout.logfile.close()
      sys.stdout = sys.stdout.terminal


os_print_tofile("myfile.txt")


'''





########################################################################################################
############################ UNIT TEST #################################################################
import argparse, arrow, util;  ppa = argparse.ArgumentParser()       # Command Line input
ppa.add_argument('--do', type=str, default= 'action',  help='test / test02')
arg = ppa.parse_args()
if __name__ == '__main__' and arg.do == "test":
 print(__file__)
 try:
  import util;  UNIQUE_ID= util.py_log_write( DIRCWD + '/aapackage/ztest_log_all.txt', "util_ml")

  #####################################################################################################
  import numpy as np, pandas as pd, scipy as sci
  import util_ml; print(util_ml); print("")


















  #####################################################################################################
  print("\n\n"+ UNIQUE_ID +" ###################### End:" + arrow.utcnow().to('Japan').format() + "###########################") ; sys.stdout.flush()
 except Exception as e : util.py_exception_print()





#########################################################################################################










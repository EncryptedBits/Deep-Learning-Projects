from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from myUtils import *
import argparse
from six.moves import cPickle as pickle


print ("Training interpratation initiated...")

# Parsing arguments for Network definition
ap = argparse.ArgumentParser()
ap.add_argument('-data_dir', default='./DATA/news.data.txt')
ap.add_argument('-batch_size', type=int, default=1)
ap.add_argument('-layer_num', type=int, default=2)
ap.add_argument('-seq_length', type=int, default=50)
ap.add_argument('-hidden_dim', type=int, default=500)
ap.add_argument('-generate_length', type=int, default=200)
ap.add_argument('-nb_epoch', type=int, default=20)
ap.add_argument('-mode', default='train')
ap.add_argument('-weights', default='')
args = vars(ap.parse_args())

DATA_DIR = args['data_dir']
BATCH_SIZE = args['batch_size']
HIDDEN_DIM = args['hidden_dim']
SEQ_LENGTH = args['seq_length']
WEIGHTS = args['weights']
NO_EPOCHS = args['nb_epoch']

GENERATE_LENGTH = args['generate_length']
LAYER_NUM = args['layer_num']

learning_rate = .5
training_check_step = 1000

data, DATA_SIZE, VOCAB_SIZE, idx_to_char, char_to_idx = read_data(DATA_DIR)
X_train, Y_train = generate_training_data(data,char_to_idx,DATA_SIZE,SEQ_LENGTH,VOCAB_SIZE)

def accuracy(predictions, labels,show_details=False,show_sent=False):
    if show_details:
        print (np.argmax(predictions, 1))
        print (np.argmax(labels, 1))
    if show_sent:
        for i in np.argmax(predictions, 1):
            print(idx_to_char[i], end="")
        print("\n")
    return np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

###################################### MODEL ############################################

graph = tf.Graph()
with graph.as_default():
    tf_train_data = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE,None,VOCAB_SIZE],name='tf_train_data')
    tf_train_label = tf.placeholder(dtype=tf.float32, shape=[None,VOCAB_SIZE],name='tf_train_label')

    #Variables
    weights = tf.Variable(tf.truncated_normal([HIDDEN_DIM, VOCAB_SIZE]),name="weights")
    biases = tf.Variable(tf.truncated_normal([VOCAB_SIZE]),name="biases")

    def rnnModel(data):

        rnn_cell = rnn.MultiRNNCell([rnn.BasicLSTMCell(HIDDEN_DIM),rnn.BasicLSTMCell(HIDDEN_DIM)])
        outputs, states = tf.nn.dynamic_rnn(rnn_cell, data, dtype=tf.float32)

        return outputs

    Cell_output = rnnModel(tf_train_data) #dim = [BATCH_SIZE,SEQ_LENGTH,HIDDEN_DIM]
    Cell_output_reshaped = tf.reshape(Cell_output,shape=[-1,HIDDEN_DIM])
    logits = tf.add(tf.matmul(Cell_output_reshaped,weights),biases, name = 'logits')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_label, logits=logits))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    output_pred = tf.nn.softmax(logits,name='output_pred')
    saver = tf.train.Saver()

######################################### TRAINING ##########################################

with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False)) as session:
    tf.global_variables_initializer().run()
    print("All variables Initialized")

    for epochs in xrange(NO_EPOCHS):
        print("TRAINING STEP %d"%epochs)
        no_of_batches = (DATA_SIZE/SEQ_LENGTH)/BATCH_SIZE
        tot_corr_pred = 0
        tot_cmparisn = 0
        for batch in xrange(no_of_batches):
            batch_train_data, batch_train_label = X_train[batch], Y_train[batch]
            batch_train_data = batch_train_data.reshape([1,-1,VOCAB_SIZE])
            feed_dict = {tf_train_data:batch_train_data,tf_train_label:batch_train_label}

            _, out_pred = session.run([optimizer,output_pred], feed_dict=feed_dict)

            tot_corr_pred += accuracy(out_pred,batch_train_label)
            tot_cmparisn += batch_train_label.shape[0]
            if batch%200 == 0 and batch%training_check_step != 0:
                print (batch, "epochs completed")
            if batch%training_check_step == 0:
                notify("%d no of epochs completed"%batch)
                print ("After %d epochs: "%batch)
                print (accuracy(out_pred,batch_train_label,show_sent=True),"Matched")
                print ("Accuracy : ",float(tot_corr_pred)/tot_cmparisn)
                tot_corr_pred = 0
                tot_cmparisn = 0

        #Model saving
        if epochs%10 == 0 and not epochs ==0:
            saver.save(session,'./LOGS/Sent-Compl-Model',global_step = epochs)

#Important data pickling
with open("./DATA/dataset.pickle","wb") as f:
    pkl_data = {'data':data,'DATA_SIZE':DATA_SIZE,'VOCAB_SIZE':VOCAB_SIZE,
    'idx_to_char':idx_to_char,'char_to_idx':char_to_idx,'X_train':X_train,'Y_train':Y_train}
    pickle.dump(pkl_data,f,pickle.HIGHEST_PROTOCOL)
    

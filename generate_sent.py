import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import random
import collections
import time
from six.moves import cPickle as pickle

with open("./DATA/dataset.pickle","rb") as f:
    pkl_data = pickle.load(f)
    VOCAB_SIZE = pkl_data['VOCAB_SIZE']
    idx_to_char = pkl_data['idx_to_char']
    char_to_idx = pkl_data['char_to_idx']

saver = tf.train.import_meta_graph('./LOGS/Sent-Compl-Model-10.meta')
graph = tf.get_default_graph()

weights = graph.get_tensor_by_name("weights:0")
biases = graph.get_tensor_by_name("biases:0")
tf_train_data = graph.get_tensor_by_name("tf_train_data:0")
logits = graph.get_tensor_by_name("logits:0")

with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:

    saver.restore(sess,tf.train.latest_checkpoint('./LOGS/'))

    while(True):
        prompt = "Type any single character: "
        line = raw_input(prompt)
        sentence = line
        character = line.strip()
        if (len(character) != 1):
            print "Incorrecr input -_-"
            continue
        else:
            try:
                symbols_in_keys = char_to_idx[str(character)]
            except:
                print("character not in dictionary")
            print "Wait a litle moment..."
            ix = [symbols_in_keys]
            y_char = [idx_to_char[ix[-1]]]
            X = np.zeros((1, 200, VOCAB_SIZE))
            for i in range(200):
                X[0, i, :][ix[-1]] = 1
                ix = np.argmax(tf.nn.softmax(sess.run(logits,feed_dict={tf_train_data:X[:, :i+1, :]}))[0],0)
                y_char.append(idx_to_char[ix[-1]])
            print ('').join(y_char)

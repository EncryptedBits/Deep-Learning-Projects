import numpy as np
import time
import subprocess

def notify(language):
	subprocess.call(['speech-dispatcher'])        #start speech dispatcher
	subprocess.call(['spd-say', '%s'%language])

def read_data(file):
	with open(file,"r") as f:
		data = f.read()
	chars = list(set(data))
	DATA_SIZE = len(data)
	VOCAB_SIZE = len(chars)
	print "File read"
	print "Total data size : %d"%DATA_SIZE
	print "Vocabulary size : %d"%VOCAB_SIZE

	idx_to_char = { i:ch for i, ch in enumerate(chars) }
	char_to_idx = { ch:i for i, ch in enumerate(chars) }

	print "Mapping done"

	return data, DATA_SIZE, VOCAB_SIZE, idx_to_char, char_to_idx

def generate_training_data(data,char_to_idx,DATA_SIZE,SEQ_LEN,VOCAB_SIZE):
	X_train = np.zeros([DATA_SIZE/SEQ_LEN,SEQ_LEN,VOCAB_SIZE])
	Y_train = np.zeros([DATA_SIZE/SEQ_LEN,SEQ_LEN,VOCAB_SIZE])

	for i in range(DATA_SIZE/SEQ_LEN):
		x_seq = data[i*SEQ_LEN:(i+1)*SEQ_LEN]
		x_seq_idx = [ char_to_idx[ch] for ch in x_seq ]
		x_mat = np.zeros([SEQ_LEN,VOCAB_SIZE])
		for j in xrange(SEQ_LEN):
			x_mat[j][x_seq_idx[j]] = 1
		X_train[i,:,:] = x_mat

	for i in range(DATA_SIZE/SEQ_LEN):
		y_seq = data[i*SEQ_LEN+1:(i+1)*SEQ_LEN+1]
		y_seq_idx = [ char_to_idx[ch] for ch in y_seq ]
		y_mat = np.zeros([SEQ_LEN,VOCAB_SIZE])
		for j in xrange(SEQ_LEN):
			y_mat[j][y_seq_idx[j]] = 1
		Y_train[i,:,:] = y_mat

	return X_train, Y_train

if __name__ == '__main__':
	print("Let's play!...")
	while(True):
		language = raw_input("Enter the sentence: ")
		notify(language)
		time.sleep(5)







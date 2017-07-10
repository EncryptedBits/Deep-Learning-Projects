import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle

pickle_file = "/media/semicolon/SourceCodes/DIH_AI_Bot/DL/Data/notMNIST/notMNIST.pickle"

image_size = 28
batch_size = 64
num_labels = 10
feature_size = 5
no_channel = 1
depth = 16
no_hidden_nodes1 = 64
no_hidden_nodes2 = 64
no_hidden_nodes3 = 64

with open(pickle_file,"rb") as f:
    save = pickle.load(f)
    train_dataset = save['train_dataset']
    train_labels = save['train_labels']
    valid_dataset = save['valid_dataset'][:4000]
    valid_labels = save['valid_labels'][:4000]
    test_dataset = save['test_dataset'][:4000]
    test_labels = save['test_labels'][:4000]
    
    del save
    
    print "PICKLE DATA SHAPES:"
    print "Training data and labels ",train_dataset.shape," ",train_labels.shape
    print "Valid data and labels ",valid_dataset.shape," ",valid_labels.shape
    print "Test data and labels ",test_dataset.shape," ",test_labels.shape
    
def data_reformat(data,labels):
    data = data.reshape((-1,image_size,image_size,1)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return data,labels

train_dataset, train_labels = data_reformat(train_dataset,train_labels)
valid_dataset, valid_labels = data_reformat(valid_dataset, valid_labels)
test_dataset, test_labels = data_reformat(test_dataset, test_labels)

print "\nSHAPE AFTER RESHAPING:"
print "Training data and labels ",train_dataset.shape," ",train_labels.shape
print "Valid data and labels ",valid_dataset.shape," ",valid_labels.shape
print "Test data and labels ",test_dataset.shape," ",test_labels.shape,"\n"

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.placeholder(tf.float32,shape=[batch_size,image_size,image_size,no_channel])
    tf_train_labels = tf.placeholder(tf.float32,shape=[batch_size,num_labels])
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    
    
    feature1 = tf.Variable(tf.truncated_normal([feature_size,feature_size,no_channel,depth]))
    pool_dim1 = [1,2,2,1]
    feature2 = tf.Variable(tf.truncated_normal([feature_size,feature_size,depth,depth]))
    pool_dim2 = [1,2,2,1]
    feature3 = tf.Variable(tf.truncated_normal([feature_size,feature_size,depth,depth]))
    pool_dim3 = [1,2,2,1]
    size_mat = (((((image_size+1)//2)+1)//2)+1)//2
    layer1_weights = tf.Variable(tf.truncated_normal([size_mat*size_mat*depth,no_hidden_nodes1]))
    layer1_biases = tf.Variable(tf.zeros([no_hidden_nodes1]))
    layer2_weights = tf.Variable(tf.truncated_normal([no_hidden_nodes1,no_hidden_nodes2]))
    layer2_biases = tf.Variable(tf.zeros([no_hidden_nodes2]))
    layer3_weights = tf.Variable(tf.truncated_normal([no_hidden_nodes2,num_labels]))
    layer3_biases = tf.Variable(tf.zeros([num_labels]))
    
    
    def cnnModel(dataset,keep_prob=1):
        #dataset sahpe = [64,28,28,1]
        conv1 = tf.nn.relu(tf.nn.conv2d(dataset,feature1,strides=[1,1,1,1],padding='SAME'))
        pool1 = tf.nn.max_pool(conv1,pool_dim1,strides=[1,2,2,1],padding='SAME')
        conv2 = tf.nn.conv2d(pool1,feature2,strides=[1,1,1,1],padding='SAME')
        pool2 = tf.nn.max_pool(conv2,pool_dim2,strides=[1,2,2,1],padding='SAME')
        conv3 = tf.nn.relu(tf.nn.conv2d(pool2,feature3,strides=[1,1,1,1],padding='SAME'))
        pool3 = tf.nn.max_pool(conv3,pool_dim3,strides=[1,2,2,1],padding='SAME')
        shape = pool3.get_shape().as_list()
        data_for_nn = tf.reshape(pool3,[shape[0],shape[1]*shape[2]*shape[3]])
        layer1_train = tf.nn.dropout(tf.nn.relu(tf.matmul(data_for_nn,layer1_weights) + layer1_biases),0.5)
        layer2_train = tf.nn.dropout(tf.nn.relu(tf.matmul(layer1_train,layer2_weights) + layer2_biases),0.5)
        layer3_train = tf.matmul(layer2_train,layer3_weights) + layer3_biases
        
        return layer3_train
    
    learning_rate = 0.00000005
    logits = cnnModel(tf_train_dataset,0.5)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=tf_train_labels))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(cnnModel(tf_valid_dataset))
    test_prediction = tf.nn.softmax(cnnModel(tf_test_dataset))
    
    
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/ predictions.shape[0])

no_of_epochs = 5001
with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as session:
#with tf.Session(graph=graph,tf.ConfigProto(log_device_placement=True)) as sess:
    tf.global_variables_initializer().run()
    print "Variables Initialized!"
    
    for epoch in xrange(no_of_epochs):
        offset = (epoch * batch_size) % (train_labels.shape[0] - batch_size)

        batch_data = train_dataset[offset:(offset+batch_size),:]
        batch_labels = train_labels[offset:(offset+batch_size),:]
        #print train_labels
        #print "[%d]"%epoch, offset, batch_data.shape, batch_labels.shape
        
        feed_dict = {tf_train_dataset:batch_data,tf_train_labels:batch_labels}
        
        _,l,t_predict = session.run([optimizer,loss,train_prediction],feed_dict=feed_dict)
        if (epoch%200 == 0) and (epoch%500 != 0):
            print "%d epochs completed..."%epoch
        if epoch%500 == 0:
            print "After %d epochs : Stats"%epoch
            print "loss == %.1f"%l
            print "Training accuracy : %.2f%%"%accuracy(t_predict,batch_labels)
            print "Validation accuracy : %.2f%%"%accuracy(valid_prediction.eval(),valid_labels)
            
    print "AFTER %d ITERATIONS: "%no_of_epochs," RESULT"
    print "Test accuracy is %.2f%%"%accuracy(test_prediction.eval(),test_labels)
    

    
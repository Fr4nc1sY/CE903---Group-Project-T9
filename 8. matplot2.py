import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

def add_layer(inputs, in_size, out_size, activation_function = None):
    # add one more layer and return the output of this layer
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.random_normal([in_size, out_size]),name = 'w')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = 'b')
        with tf.name_scope('wx_plus_b'):
            wx_plus_b = tf.matmul(inputs, weights) + biases
        if activation_function is None:
            outputs = wx_plus_b
        else:
            outputs = activation_function(wx_plus_b)
        
        return outputs

# make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name = 'x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name = 'y_input')

# add hidden layer
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)

# add output layer
prediction = add_layer(l1, 10, 1, activation_function = None)

# the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices = [1]))

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()
writer = tf.summary.FileWriter("logs/",sess.graph)
'''
After writer step, you will find a file names logs, 
run terminal and go to the root list of 'logs',
run code: tensorboard --logdir logs,
then you can see the pragh on 'http://localhost:6006' or the website it shows 
'''

# important step
init = tf.initialize_all_variables()
sess.run(init)

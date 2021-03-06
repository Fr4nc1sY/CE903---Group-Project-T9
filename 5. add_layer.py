import tensorflow as tf 

def add_layer(inputs, in_size, out_size, activation_function = None):  # inputs: input data, in_size: number of input, out_size: number of output, activation_function: 'linear', 'relu', 'sigmoid', 'tanh', default is 'linear'
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    
    return outputs

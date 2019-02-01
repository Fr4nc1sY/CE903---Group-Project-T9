# install tensorflow first, both CPU and GPU version are fine
import tensorflow as tf
import numpy as np 

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### create tensorflow structure start ###
weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = weights * x_data + biases 

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5) #GradientDescentOptimizer is one the optimize algoithms, which is the simpliest
train = optimizer.minimize(loss)

init = tf.initialize_all_variables() # you need to initialize all variable before training

### create tensorflow structure end###

sess = tf.Session() # session is the most important structure of Tensorflow, if you want some operation, you need to run that session
sess.run(init)

for step in range(301):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(weights),sess.run(biases)) 
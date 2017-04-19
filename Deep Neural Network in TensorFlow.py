# Deep Neural Network in TensorFlow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets(".", one_hot = True, reshape = False)

# learning parameters
import tensorflow as tf
learning_rate = 0.001
training_epochs = 20
batch_size = 128
display_step = 1
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Hidden layer parameters 
n_hidden_layer = 256 # layer number of features
					 # The size of the hidden layer in the neural network
					 # The width of a layer


# parameters: Store layers weight & bias
# tf.Variable is a modifiable tensor. It doesn't matter very much what they initially are
weights = {
    'hidden_layer': tf.Variable(tf.random_normal([n_input, n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer, n_classes]))
}
biases = {
    'hidden_layer': tf.Variable(tf.random_normal([n_hidden_layer])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# observations: 
# a dimention can be of any length
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, n_classes])
x_flat = tf.reshape(x, [-1, n_input])

# Hidden layer with RELU activation
layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer']),biases['hidden_layer'])
layer_1 = tf.nn.relu(layer_1)
# Output layer with linear activation
logits = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])


Optimizer
# Define loss and optimizer
cost = tf.reduce_mean(\
	tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)\
    .minimize(cost)

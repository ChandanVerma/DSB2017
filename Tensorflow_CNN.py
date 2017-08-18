## A simple CNN to predict certain characteristics of the human subject from MRI images.
# 3d convolution is used in each layer.
# Reference: https://www.tensorflow.org/get_started/mnist/pros, http://blog.naver.com/kjpark79/220783765651
# Adjust needed for your dataset e.g., max pooling, convolution parameters, training_step, batch size, etc
# Start TensorFlow InteractiveSession
#import input_3Dimage
import tensorflow as tf
import os
import numpy as np
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sess = tf.InteractiveSession()

train_ids = [id.replace(".npy", "") for id in os.listdir('D:/ImageAnalytics/resized_images/train/')]
valid_ids = [id.replace(".npy", "") for id in os.listdir('D:/ImageAnalytics/resized_images/valid/')]
test_ids = [id.replace(".npy", "") for id in os.listdir('D:/ImageAnalytics/resized_images/test/')]

train_ids.sort()
valid_ids.sort()
test_ids.sort()

df = pd.read_csv('D:/ImageAnalytics/stage1_labels.csv')

df.head()
#%%

def save_array(path, arr):
    np.save(path, arr)
    

def load_array(path):
    return np.load(path)
	
train_labels = df["cancer"][df["id"].isin(train_ids)].values
valid_labels = df["cancer"][df["id"].isin(valid_ids)].values

train_label = load_array('D:/ImageAnalytics/resized_images/train_label.npy')
valid_label = load_array('D:/ImageAnalytics/resized_images/valid_label.npy')
test_label = load_array('D:/ImageAnalytics/resized_images/test_label.npy')  
train_data = load_array('D:/ImageAnalytics/resized_images/train_data.npy')
valid_data = load_array('D:/ImageAnalytics/resized_images/valid_data.npy')
test_data = load_array('D:/ImageAnalytics/resized_images/test_data.npy')

print(train_data.shape,train_label.shape)
print(valid_data.shape,valid_label.shape)
print(test_data.shape,test_label.shape)

#-------------------------------------------------------------------------------------------------------
depth = 20
image_size = 64
num_labels = 2
num_channels = 1 # grayscale
batch_size = 16
patch_size = 5

def reformat(dataset, labels):
  dataset = dataset.reshape((-1,depth, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_data, train_label)
valid_dataset, valid_labels = reformat(valid_data, valid_label)
test_dataset, test_labels = reformat(test_data, test_label)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

#--------------------------------------- FUNCTION DEFINITION --------------------------------------------

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME') # conv2d, [1, 1, 1, 1]

def max_pool_2x2(x):  # tf.nn.max_pool. ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]
  return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

#----------------------------------Defining Network architecture -----------------------------------------
graph = tf.Graph()

with graph.as_default():

  # Input data.
	tf_train_dataset = tf.placeholder(tf.float32, shape=[batch_size, depth, image_size, image_size, num_channels])
	print(tf_train_dataset)
	tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))
	print(tf_train_labels)
	tf_valid_dataset = tf.constant(valid_dataset)
	print(tf_valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	print(tf_test_dataset)
	# Variables.
	layer1_weights = weight_variable([5, 5, 5, 1, 32]) 
	print(layer1_weights)
	layer1_biases = bias_variable([32]) 
	print(layer1_biases)
	layer2_weights =  weight_variable([5, 5, 5, 32, 64])
	print(layer2_weights)
	layer2_biases = bias_variable([64])
	print(layer2_biases)
	fc1_layer_weights = weight_variable([16*16*5*64, 1024])
	print(fc1_layer_weights)
	fc1_layer_biases = bias_variable([1024])
	print(fc1_layer_biases)
	fc2_layer_weights = weight_variable([1024, num_labels])
	print(fc2_layer_weights)
	fc2_layer_biases = bias_variable([num_labels])
	print(fc2_layer_biases)
#%%
	def model(data):
		conv1 = tf.nn.relu(conv3d(data, layer1_weights) + layer1_biases)
		print('Conv1 layer shape is:' ,conv1.shape)
		pool1 = max_pool_2x2(conv1)
		print('Pooling layer 1 shape is:' ,pool1.shape)
		conv2 = tf.nn.relu(conv3d(pool1, layer2_weights) + layer2_biases)
		print('Conv2 layer shape is:' ,conv2.shape)
		pool2 = max_pool_2x2(conv2)
		print('Pooling layer 2 shape is:' ,pool2.shape)
		shape = pool2.get_shape().as_list()
		#shape = tf.convert_to_tensor([np.nan, 16, 16, 5, 64])
		#print(shape.shape)
		reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3] * shape[4]])
		print('Shape After reshaping is:', reshape.shape)
		fc1 = tf.nn.relu(tf.matmul(reshape, fc1_layer_weights) + fc1_layer_biases)
		print('Shape of fully connected layer :' ,fc1.shape)
		return tf.matmul(fc1, fc2_layer_weights) + fc2_layer_biases
	
#-------------------------------------------- Training computation ---------------------------------------
	logits = model(tf_train_dataset)
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
	#correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
	#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
	# Optimizer.
	optimizer = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)  # 1e-4
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))

#---------------------------------------------------------------------------------------------------------
#%%
num_steps = 1001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  for step in range(num_steps):
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
    _, l, predictions = session.run([optimizer, cross_entropy, train_prediction], feed_dict=feed_dict)
    if (step % 50 == 0):
      print('Minibatch loss at step %d: %f' % (step, l))
      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

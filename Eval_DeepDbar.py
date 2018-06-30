# DeepDbar evaluation script
#
# Accompanying code for the publication: Hamilton & Hauptmann (2018). 
# Deep D-bar: Real time Electrical Impedance Tomography Imaging with 
# Deep Neural Networks. IEEE Transactions on Medical Imaging.
#
# Sarah J. Hamilton and Andreas Hauptmann, June 2018

import tensorflow as tf
import sys
import Load_DeepDbar as loadDbar
import numpy

import h5py
FLAGS = None


def deepDbar(x,bSize):
  # Reshape to use within a convolutional neural net.
  x_image = tf.reshape(x, [-1, 64,64, 1])
  

  # First convolutional layer 
  W_convPrioB1_1 = weight_variable([5, 5, 1, 32])
  b_convPrioB1_1 = bias_variable([32])
  h_convPrioB1_1 = tf.nn.relu(conv2d(x_image, W_convPrioB1_1) + b_convPrioB1_1)

  # Second convolutional layer
  W_convPrioB1_2 = weight_variable([5, 5, 32, 32])
  b_convPrioB1_2 = bias_variable([32])
  h_convPrioB1_2 = tf.nn.relu(conv2d(h_convPrioB1_1, W_convPrioB1_2) + b_convPrioB1_2)
  
  # Maxpool layer
  h_maxpool1 = max_pool_2x2(h_convPrioB1_2)
  
  # convolutional layer 
  W_convPrioB2_1 = weight_variable([5, 5, 32, 64])
  b_convPrioB2_1 = bias_variable([64])
  h_convPrioB2_1 = tf.nn.relu(conv2d(h_maxpool1, W_convPrioB2_1) + b_convPrioB2_1)
    
  # convolutional layer
  W_convPrioB2_2 = weight_variable([5, 5, 64, 64])
  b_convPrioB2_2 = bias_variable([64])
  h_convPrioB2_2 = tf.nn.relu(conv2d(h_convPrioB2_1, W_convPrioB2_2) + b_convPrioB2_2)
  
  # Maxpool layer
  h_maxpool2 = max_pool_2x2(h_convPrioB2_2)
  
  
  # convolutional layer 
  W_convPrioB3_1 = weight_variable([5, 5, 64, 128])
  b_convPrioB3_1 = bias_variable([128])
  h_convPrioB3_1 = tf.nn.relu(conv2d(h_maxpool2, W_convPrioB3_1) + b_convPrioB3_1)
    
  # convolutional layer 
  W_convPrioB3_2 = weight_variable([5, 5, 128, 128])
  b_convPrioB3_2 = bias_variable([128])
  h_convPrioB3_2 = tf.nn.relu(conv2d(h_convPrioB3_1, W_convPrioB3_2) + b_convPrioB3_2)
  
  # Maxpool layer
  h_maxpool3 = max_pool_2x2(h_convPrioB3_2)
  
  
  # convolutional layer
  W_convPrioB4_1 = weight_variable([5, 5, 128, 256])
  b_convPrioB4_1 = bias_variable([256])
  h_convPrioB4_1 = tf.nn.relu(conv2d(h_maxpool3, W_convPrioB4_1) + b_convPrioB4_1)
    
  # convolutional layer
  W_convPrioB4_2 = weight_variable([5, 5, 256, 256])
  b_convPrioB4_2 = bias_variable([256])
  h_convPrioB4_2 = tf.nn.relu(conv2d(h_convPrioB4_1, W_convPrioB4_2) + b_convPrioB4_2)
  
  # Maxpool layer
  h_maxpool4 = max_pool_2x2(h_convPrioB4_2)
  
  
  # convolutional layer 
  W_convPrioB5_1 = weight_variable([5, 5, 256, 512])
  b_convPrioB5_1 = bias_variable([512])
  h_convPrioB5_1 = tf.nn.relu(conv2d(h_maxpool4, W_convPrioB5_1) + b_convPrioB5_1)
    
  # convolutional layer 
  W_convPrioB5_2 = weight_variable([5, 5, 512, 512])
  b_convPrioB5_2 = bias_variable([512])
  h_convPrioB5_2 = tf.nn.relu(conv2d(h_convPrioB5_1, W_convPrioB5_2) + b_convPrioB5_2)
  
  
  
  #Upsample and concat
  W_TconvPrioB5 = weight_variable([5, 5, 256, 512]) #Ouput, Input channels
  b_TconvPrioB5 = bias_variable([256])
  h_TconvPrioB5 = tf.nn.relu(conv2d_trans(h_convPrioB5_2, W_TconvPrioB5,[bSize,8,8,256]) + b_TconvPrioB5)
  
  h_concatB4 = tf.concat([h_convPrioB4_2,h_TconvPrioB5],3)
  
  #Conv Down
  W_convDownB4_1 = weight_variable([5, 5, 512, 256])
  b_convDownB4_1 = bias_variable([256])
  h_convDownB4_1 = tf.nn.relu(conv2d(h_concatB4, W_convDownB4_1) + b_convDownB4_1)
  
  #Conv Down
  W_convDownB4_2 = weight_variable([5, 5, 256, 256])
  b_convDownB4_2 = bias_variable([256])
  h_convDownB4_2 = tf.nn.relu(conv2d(h_convDownB4_1, W_convDownB4_2) + b_convDownB4_2)
  
  #Upsample and concat ------ B3
  W_TconvPrioB4 = weight_variable([5, 5, 128, 256]) #Ouput, Input channels
  b_TconvPrioB4 = bias_variable([128])
  h_TconvPrioB4 = tf.nn.relu(conv2d_trans(h_convDownB4_2, W_TconvPrioB4,[bSize,16,16,128]) + b_TconvPrioB4)
  
  h_concatB3 = tf.concat([h_convPrioB3_2,h_TconvPrioB4],3)
  
  #Conv Down
  W_convDownB3_1 = weight_variable([ 5, 5, 256, 128])
  b_convDownB3_1 = bias_variable([128])
  h_convDownB3_1 = tf.nn.relu(conv2d(h_concatB3, W_convDownB3_1) + b_convDownB3_1)
  
  #Conv Down
  W_convDownB3_2 = weight_variable([5, 5, 128, 128])
  b_convDownB3_2 = bias_variable([128])
  h_convDownB3_2 = tf.nn.relu(conv2d(h_convDownB3_1, W_convDownB3_2) + b_convDownB3_2)
    
  #Upsample and concat ------ B2
  W_TconvPrioB3 = weight_variable([5, 5, 64, 128]) #Ouput, Input channels
  b_TconvPrioB3 = bias_variable([64])
  h_TconvPrioB3 = tf.nn.relu(conv2d_trans(h_convDownB3_2, W_TconvPrioB3,[bSize,32,32,64]) + b_TconvPrioB3)
  
  h_concatB2 = tf.concat([h_convPrioB2_2,h_TconvPrioB3],3)
  
  #Conv Down
  W_convDownB2_1 = weight_variable([5, 5, 128, 64])
  b_convDownB2_1 = bias_variable([64])
  h_convDownB2_1 = tf.nn.relu(conv2d(h_concatB2, W_convDownB2_1) + b_convDownB2_1)
  
  #Conv Down
  W_convDownB2_2 = weight_variable([5, 5, 64, 64])
  b_convDownB2_2 = bias_variable([64])
  h_convDownB2_2 = tf.nn.relu(conv2d(h_convDownB2_1, W_convDownB2_2) + b_convDownB2_2)
  
  #Upsample and concat
  W_TconvPrioB2 = weight_variable([5, 5, 32, 64]) #Ouput, Input channels
  b_TconvPrioB2 = bias_variable([32])
  h_TconvPrioB2 = tf.nn.relu(conv2d_trans(h_convDownB2_2, W_TconvPrioB2,[bSize,64,64,32]) + b_TconvPrioB2)
  
  h_concatB1 = tf.concat([h_convPrioB1_2,h_TconvPrioB2],3)
  
  #Conv Down
  W_convDownB1_1 = weight_variable([5, 5, 64, 32])
  b_convDownB1_1 = bias_variable([32])
  h_convDownB1_1 = tf.nn.relu(conv2d(h_concatB1, W_convDownB1_1) + b_convDownB1_1)
  
  #Conv Down
  W_convDownB1_2 = weight_variable([5, 5, 32, 32])
  b_convDownB1_2 = bias_variable([32])
  h_convDownB1_2 = tf.nn.relu(conv2d(h_convDownB1_1, W_convDownB1_2) + b_convDownB1_2)
  
  #Conv Down - no RELU!
  W_convDownB1_3 = weight_variable([1, 1, 32, 1])
  b_convDownB1_3 = bias_variable([1])
  h_convDownB1_3 = conv2d(h_convDownB1_2, W_convDownB1_3) + b_convDownB1_3
  
  
  h_update = tf.nn.relu(h_convDownB1_3)  
  h_update = tf.reshape(h_update, [-1, 64,64])

  return h_update


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_trans(x, W, shape):
  """conv2d returns a 2d convolution transpose layer with full stride."""
  return tf.nn.conv2d_transpose(x, W, shape, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.025)
  return tf.Variable(initial)

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.025, shape=shape)
  return tf.Variable(initial)


def main(filePath,fileOutName,testFileName):
  
  dataDbar = loadDbar.read_data_sets(testFileName)
  imSize=numpy.shape(dataDbar.test.images)

  # Initialiase variables
  imag = tf.placeholder(tf.float32, [None, 64,64])  
  output = tf.placeholder(tf.float32, [None, 64,64])
  
  # Build network 
  y_conv = deepDbar(imag,imSize[0]) #Batch size
  saver = tf.train.Saver()

  with tf.Session() as sess:
    saver.restore(sess, filePath)
    print("Model restored.")

    for i in range(1):
      
      output = sess.run(y_conv,feed_dict={imag: dataDbar.test.images})

      #Save for Matlab
      fData = h5py.File(fileOutName,'w')
      fData['result']= numpy.array(output)
      fData['imag']= numpy.array(dataDbar.test.images)
      
      fData.close() 
        
    print('--------------------> DONE <--------------------')
    
    return

main(sys.argv[1],sys.argv[2],sys.argv[3])
#This file is heavily based on Daniel Johnson's midi manipulation code in https://github.com/hexahedria/biaxial-rnn-music-composition

import numpy as np
import pandas as pd
#import msgpack
import glob
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tqdm import tqdm

###################################################
# In order for this code to work, you need to place this file in the same 
# directory as the midi_manipulation.py file and the Pop_Music_Midi directory

import midi_manipulation

### HyperParameters
# First, let's take a look at the hyperparameters of our model:

lowest_note = midi_manipulation.lowerBound #the index of the lowest note on the piano roll
highest_note = midi_manipulation.upperBound #the index of the highest note on the piano roll
note_range = highest_note-lowest_note #the note range

num_timesteps  = 15 #This is the number of timesteps that we will create at a time
n_visible      = 2*note_range*num_timesteps #This is the size of the visible layer. 
n_hidden       = 50 #This is the size of the hidden layer

num_epochs = 200 #The number of training epochs that we are going to run. For each epoch we go through the entire data set.
batch_size = 100 #The number of training examples that we are going to send through the RBM at a time. 
lr         = tf.constant(0.005, tf.float32) #The learning rate of our model

### Variables:
# Next, let's look at the variables we're going to use:

x  = tf.placeholder(tf.float32, [None, n_visible], name="x") #The placeholder variable that holds our data
W  = tf.Variable(tf.random_normal([n_visible, n_hidden], 0.01), name="W") #The weight matrix that stores the edge weights
bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32, name="bh")) #The bias vector for the hidden layer
bv = tf.Variable(tf.zeros([1, n_visible],  tf.float32, name="bv")) #The bias vector for the visible layer

### Training Update Code
# Now we implement the contrastive divergence algorithm. First, we get the samples of x and h from the probability distribution
#The sample of x
x_sample = gibbs_sample(1) 
#The sample of the hidden nodes, starting from the visible state of x
h = sample(tf.sigmoid(tf.matmul(x, W) + bh)) 
#The sample of the hidden nodes, starting from the visible state of x_sample
h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh)) 

#Next, we update the values of W, bh, and bv, based on the difference between the samples that we drew and the original values
size_bt = tf.cast(tf.shape(x)[0], tf.float32)
W_adder  = tf.multiply(lr/size_bt, tf.subtract(tf.matmul(tf.transpose(x), h), tf.matmul(tf.transpose(x_sample), h_sample)))
bv_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(x, x_sample), 0, True))
bh_adder = tf.multiply(lr/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
#When we do sess.run(updt), TensorFlow will run all 3 update steps
updt = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]


### Run the graph!
sess=tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./model/model.ckpt")

#Now the model is fully trained, so let's make some music! 
#Run a gibbs chain where the visible nodes are initialized to 0
sample = gibbs_sample(1).eval(session=sess, feed_dict={x: np.zeros((50, n_visible))})
for i in range(sample.shape[0]):
    if not any(sample[i,:]):
        continue
    #Here we reshape the vector to be time x notes, and then save the vector as a midi file
    S = np.reshape(sample[i,:], (num_timesteps, 2*note_range))
    midi_manipulation.noteStateMatrixToMidi(S, "generated_chord_{}".format(i))
            

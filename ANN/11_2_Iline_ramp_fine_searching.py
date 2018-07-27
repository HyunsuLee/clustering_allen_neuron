#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created 2018-03-02
@author: hyunsu
"""
# for tensorflow ANN. hyperparameter searching test python

import tensorflow as tf
import numpy as np
import random

data_path = './180228tensordata_minmax/'
"""
there are total 48 csv files.

created by data_processing_180227.ipynb
    
    3 different output classification task.
    start with "B" means for binary classification(E vs I => outnode 2)
    "E" stands for excitatory transgenic line classification(outnode 10)
    "I" stands for inhibitory transgenic line classificiation(outnode 8)
    
    4 different input features.
    full model(all electrophysiology features) 43
    _long.csv - long square pulse protocol     21
    _short.csv - short square pulse protocol   18
    _ramp.csv - ramp pulse protocol            16
3 X 4 = 12. 12 different ANN models will be created.
"""
input_protocol = '_ramp' # change X place holder and layer shapes
output_class = 'I'      # change Y place holder and layer shapes
result_path = './180301_hyperparameter_test/11_2_Iline_ramp_fine.csv'

trainX = np.loadtxt(data_path + output_class + 'train' + input_protocol + 'X_minmax.csv', delimiter = ',')
trainY = np.loadtxt(data_path + output_class + 'train' + input_protocol + 'Y.csv', delimiter = ',')

testX = np.loadtxt(data_path + output_class + 'test' + input_protocol + 'X_minmax.csv', delimiter = ',')
testY = np.loadtxt(data_path + output_class + 'test' + input_protocol + 'Y.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 16]) 
Y = tf.placeholder(tf.float32, [None, 8]) 
keep_prob = tf.placeholder(tf.float32)
is_training_holder = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)
L2beta = tf.placeholder(tf.float32)
epsilon = 1e-3 # for Batch normalization
layer1_shape = [16, 16]
layer2_shape = [16, 10]
output_shape = [10, 8] 

def weight_init(shape, name_for_weight):
    Xavier_init = np.sqrt(2.0) * np.sqrt(2.0 / np.array(shape).sum())
    weights = tf.truncated_normal(shape, stddev = Xavier_init)
    return tf.Variable(weights, name = name_for_weight)

with tf.name_scope('layer1'):
    W1 = weight_init(layer1_shape, 'W1')
    z1 = tf.matmul(X, W1)
    BN1 = tf.contrib.layers.batch_norm(z1, center = True, scale = True,
        is_training = is_training_holder) 
    L1 = tf.nn.relu(BN1)
    L1 = tf.nn.dropout(L1, keep_prob)
    
with tf.name_scope('layer2'):
    W2 = weight_init(layer2_shape, 'W2')
    z2 = tf.matmul(L1, W2)
    BN2 = tf.contrib.layers.batch_norm(z2, center = True, scale = True,
        is_training = is_training_holder) 
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.dropout(L2, keep_prob)
   
with tf.name_scope('output'):
    W3 = weight_init(output_shape, 'W3')
    b3 = tf.Variable(tf.random_normal([output_shape[1]]))
    model = tf.matmul(L2, W3) + b3
    
with tf.name_scope('optimizer'):
    base_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    lossL2 =  tf.reduce_mean(tf.nn.l2_loss(W1) + 
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)) * L2beta
    cost = base_cost + lossL2 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
sess = tf.Session()

total_model_test = 500 
LR_list = []
L2beta_list = []
test_cost_list = []
test_acc_list = []

for model in range(total_model_test):
    LR_power = random.uniform(-4.0, -2.3)
    random_learning_rate = 10 ** LR_power
    beta_power = random.uniform(-6.0, -3.3)
    random_L2beta = 10 ** beta_power 
    sess.run(tf.global_variables_initializer())
    for epoch in range(5000):
        sess.run(optimizer, feed_dict={X: trainX, Y: trainY, keep_prob: 0.5, 
                            is_training_holder: 1, learning_rate: random_learning_rate,
                            L2beta: random_L2beta})
        test_acc, test_cost = sess.run([accuracy, cost], 
                        feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                        is_training_holder: 0, learning_rate: random_learning_rate,
                        L2beta: random_L2beta})
    print('Learning rate:', '{:.4e}'.format(random_learning_rate), 
    'L2beta:', '{:.4e}'.format(random_L2beta),
    'Test cost:', '{:.4f}'.format(test_cost),
    'Test accuracy:', '{:.4f}'.format(test_acc), 'Model:', str(model+1),'/',str(total_model_test))
    LR_list += [random_learning_rate]
    L2beta_list += [random_L2beta]
    test_cost_list += [test_cost]
    test_acc_list += [test_acc]

combine_list = [LR_list] + [L2beta_list] + [test_cost_list] + [test_acc_list]
combine_list = np.array(combine_list)

import pandas as pd

results = pd.DataFrame(combine_list.T, 
        columns=['learning_rate', 'L2_beta', 'test_cost', 'test_accuracy'])

results.to_csv(result_path)






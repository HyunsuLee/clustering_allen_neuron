#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created 2018-03-01
@author: hyunsu
"""
# for tensorflow ANN. hyperparameter searching test python
# and writing tensorboard logs
# testing 10 models, fine tuned results load 

import tensorflow as tf
import numpy as np
import pandas as pd
import os

data_path = './180228tensordata_minmax/'
log_path = '/binary_short/'
model_dir = './model/' + log_path # for model saver

input_protocol = 'short' # change X place holder and layer shapes
output_class = 'B'      # change Y place holder and layer shapes
result_path = './180301_hyperparameter_test/02_2_binary_short_fine.csv'
HP_df = pd.read_csv(result_path)
HP_np = np.array(HP_df.sort_values('test_cost').head(10))
Best_model_no = 3

random_learning_rate = HP_np[Best_model_no][1]
random_L2beta = HP_np[Best_model_no][2]
best_model_dir = model_dir + ('Model'+str(Best_model_no)+'LR'+'{:.3e}'.format(random_learning_rate) 
                + 'Beta' + '{:.3e}'.format(random_L2beta) + '/')

testX = np.loadtxt(data_path + output_class + 'test_' + input_protocol + 'X_minmax.csv', delimiter = ',')
testY = np.loadtxt(data_path + output_class + 'test_' + input_protocol + 'Y.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 18]) 
Y = tf.placeholder(tf.float32, [None, 2]) # binary E vs I class
keep_prob = tf.placeholder(tf.float32)
is_training_holder = tf.placeholder(tf.bool)
L2beta = tf.placeholder(tf.float32)
epsilon = 1e-3 # for Batch normalization
layer1_shape = [18, 10]
layer2_shape = [10, 5]
output_shape = [5, 2]

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

model_checkpoint_all = os.listdir(best_model_dir)

models_list = []
for model_checkpoints in model_checkpoint_all:
    if model_checkpoints[-5:] == '.meta':
        models_list += [model_checkpoints[:-5]]

sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())

lowest_cost = 10.0

for model_checks in models_list:
    saver = tf.train.import_meta_graph(best_model_dir + model_checks + '.meta')
    saver.restore(sess, best_model_dir + model_checks)

    with tf.name_scope("accuracy"):
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    with tf.name_scope('optimizer'):
        base_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
        lossL2 =  tf.reduce_mean(tf.nn.l2_loss(W1) + 
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)) * L2beta
        cost = base_cost + lossL2

    cost = sess.run(cost, 
                feed_dict={X: testX, Y: testY, keep_prob: 1.0,
                is_training_holder: 0, L2beta: random_L2beta})
    accuracy = sess.run(accuracy, 
                feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                is_training_holder: 0, L2beta: random_L2beta})
    print('Model:', model_checks, 'Test accuracy:', '{:.4f}'.format(accuracy),
        'Test cost:', '{:.4f}'.format(cost))
    
    if cost < lowest_cost:
        lowest_cost = cost
        best_model_step = model_checks

print('Lowest cost:', '{:.4f}'.format(lowest_cost), 'Best model:', best_model_step)

saver = tf.train.import_meta_graph(best_model_dir + best_model_step + '.meta')
saver.restore(sess, best_model_dir + best_model_step)

model_eval = sess.run(model, feed_dict = {X: testX, Y: testY, keep_prob: 1.0,
                is_training_holder: 0, L2beta: random_L2beta})
model_prob = sess.run(tf.nn.softmax(model_eval))

np.savetxt('./results/02_4_binary_short_ANNmodel_prob.csv', model_prob, delimiter=',')
np.savetxt('./results/Btest_shortY.csv', testY, delimiter = ',')


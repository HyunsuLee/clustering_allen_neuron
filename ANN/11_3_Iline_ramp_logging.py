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

data_path = './180228tensordata_minmax/'
log_path = '/Iline_ramp/'
summaries_dir = './logs/' + log_path + '/2nd/' # for tensorboard summary
model_dir = './model/' + log_path # for model saver

input_protocol = '_ramp' # change X place holder and layer shapes
output_class = 'I'      # change Y place holder and layer shapes
result_path = './180301_hyperparameter_test/11_2_Iline_ramp_fine.csv'
HP_df = pd.read_csv(result_path)
HP_np = np.array(HP_df.sort_values('test_cost').head(10))

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
    tf.summary.histogram('W1', W1)

with tf.name_scope('layer2'):
    W2 = weight_init(layer2_shape, 'W2')
    z2 = tf.matmul(L1, W2)
    BN2 = tf.contrib.layers.batch_norm(z2, center = True, scale = True,
          is_training = is_training_holder) 
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.dropout(L2, keep_prob)
    tf.summary.histogram('W2', W2)

with tf.name_scope('output'):
    W3 = weight_init(output_shape, 'W3')
    b3 = tf.Variable(tf.random_normal([output_shape[1]]))
    model = tf.matmul(L2, W3) + b3
    tf.summary.histogram('W3', W3)

with tf.name_scope('optimizer'):
    base_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    lossL2 =  tf.reduce_mean(tf.nn.l2_loss(W1) + 
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)) * L2beta
    cost = base_cost + lossL2 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()

total_model_test = 10 


for model in range(total_model_test):
    random_learning_rate = HP_np[model][1]
    random_L2beta = HP_np[model][2]
    
    sess.run(tf.global_variables_initializer())
    
    each_model_dir = ('Model'+str(model)+'LR'+'{:.3e}'.format(random_learning_rate) 
                + 'Beta' + '{:.3e}'.format(random_L2beta) + '/')
    train_writer = tf.summary.FileWriter(summaries_dir + 
            each_model_dir + '/train')
    test_writer = tf.summary.FileWriter(summaries_dir + 
            each_model_dir + '/test') # $ tensorboard --logdir ./logs
    train_writer.add_graph(sess.graph)

    for epoch in range(50000):
        # for start, end in zip(range(0, len(trainX), batch_size),
        #    range(batch_size, len(trainX)+1, batch_size)):
        #    sess.run(optimizer, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
        sess.run(optimizer, feed_dict={X: trainX, Y: trainY, keep_prob: 0.5, 
                            is_training_holder: 1, learning_rate: random_learning_rate,
                            L2beta: random_L2beta})
        if (epoch % 50) == 0:
            train_acc, train_summ = sess.run([accuracy, summ], 
                        feed_dict={X: trainX, Y: trainY, keep_prob: 1.0, 
                        is_training_holder: 1, learning_rate: random_learning_rate, 
                        L2beta: random_L2beta})
            train_writer.add_summary(train_summ, epoch)

            test_acc, test_summ = sess.run([accuracy, summ], 
                        feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                        is_training_holder: 0, learning_rate: random_learning_rate, 
                        L2beta: random_L2beta})
            test_writer.add_summary(test_summ, epoch)
        if (epoch % 500) == 0:
            test_cost = sess.run(cost, 
                        feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                        is_training_holder: 0, learning_rate: random_learning_rate, 
                        L2beta: random_L2beta})
            print('Epoch:', '%04d' % (epoch +1), 'Test cost:', '{:.4f}'.format(test_cost),
                  'Test accuracy:', '{:.4f}'.format(test_acc))
            saver.save(sess, model_dir + each_model_dir + '/ANN.ckpt', epoch)
    print('Learning rate:', '{:.4e}'.format(random_learning_rate), 
    'L2beta:', '{:.4e}'.format(random_L2beta),
    'Test cost:', '{:.4f}'.format(test_cost),
    'Test accuracy:', '{:.4f}'.format(test_acc), 'Model:', str(model),'/',str(total_model_test))



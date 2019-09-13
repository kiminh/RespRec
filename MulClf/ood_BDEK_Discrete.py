#! /usr/bin/env python

import os
import time
import datetime
# import cPickle

import tensorflow as tf
import numpy as np

import data_helpers
from text_cnn_DomainInfer import DomainTextCNN
from text_cnn_Discrete import DiscreteDomainTextCNN

from tensorflow.contrib import learn

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

tf.flags.DEFINE_boolean("is_use_adv", False, "is using adversarial learning (default: False)")
tf.flags.DEFINE_float("adv_lambda", 1e-3, "Robust Regularizaion lambda (default: 1e-3)")
tf.flags.DEFINE_boolean("is_use_unsup_domain_data", True, "flag for using C") #is using C?
tf.flags.DEFINE_boolean("is_use_unsup_data", False, "flag for using B") #is using B?

tf.flags.DEFINE_integer("latent_domain_size", 13, "Latent Domain size (default: 13)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("dev_size", 512, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_pretrain_epochs", 0, "Number of training epochs")
tf.flags.DEFINE_integer("num_train_epochs", 100, "Number of tuning epochs")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate alpha")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

print(tf.__version__)
FLAGS = tf.flags.FLAGS
FLAGS.batch_size
print("\nParameters:")
if tf.__version__ == "1.4.0":
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
else:
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value.value))
print("")


# Load data
print("Loading data...")
# domain 0-3:BDEK
# A(x,y,d) B(x,?,a) C(x,y,?)
sent_length, vocab_size, num_label, num_domain, x, y, d, B_x, B_d = data_helpers.load_data()

print("Splitting data...")
A_x, A_y, A_d, C_x, C_y, C_d = data_helpers.data_split(x, y, d, ratio = 0.5)

# Randomly shuffle data
# np.random.seed(101)

score_sum = []
best_score = 0

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement,
        intra_op_parallelism_threads=2,
        inter_op_parallelism_threads=4)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = DiscreteDomainTextCNN(
            sequence_length = sent_length,
            num_classes = num_label,
            vocab_size = vocab_size,
            embedding_size = FLAGS.embedding_dim,
            filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters = FLAGS.num_filters,
            num_domains = num_domain,
            num_latent_domains = FLAGS.latent_domain_size,
            l2_reg_lambda = FLAGS.l2_reg_lambda,
            is_use_adv = FLAGS.is_use_adv,
            )

        # Define Training procedure
        learning_rate = tf.placeholder(tf.float32, shape=[], name="learning_rate")
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        all_var_list = tf.trainable_variables()
        # print(all_var_list)

        optimizer_pretrain = tf.train.AdamOptimizer(
            learning_rate = learning_rate
            ).minimize(
                cnn.y_loss_pretrain,
                global_step=global_step
                )

        optimizer_n = tf.train.AdamOptimizer(
            learning_rate = learning_rate
            ).minimize(
                cnn.y_loss_train,
                global_step=global_step
                )
        
        def ind2onehot( d ):
            d_i = np.zeros( ( len(d), num_domain ) )
            for i in range(len( d )):
                index_i = d[i][0]
                if index_i < num_domain:
                    d_i[ i, index_i  ] = 1
            return d_i

        def pretrain_batch(x_batch, y_batch, d_batch, opt, adv_lbd, lr):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_d_idx: d_batch,
                cnn.input_d : ind2onehot( d_batch ),
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                cnn.adv_lambda: adv_lbd,
                learning_rate: lr,
            }
            _, step, pretrain_loss, pretrain_acc, domain_acc_train = sess.run(
                [optimizer_pretrain, global_step, cnn.y_loss_pretrain, cnn.y_acc_pretrain, cnn.domain_acc_train],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{:g}\t{:g}\t\t{:g}".format(step, pretrain_loss, pretrain_acc, domain_acc_train))
            # train_summary_writer.add_summary(summaries, step)

        def train_batch(x_batch, y_batch, d_batch, opt, adv_lbd, lr):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_d_idx: d_batch,
                cnn.input_d : ind2onehot( d_batch ),
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob,
                cnn.adv_lambda: adv_lbd,
                learning_rate: lr,
            }
            _, step, train_loss, train_acc, domain_acc_train = sess.run(
                [opt, global_step, cnn.y_loss_train, cnn.y_acc_train, cnn.domain_acc_train],
                feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            print("0\t{}\t{:g}\t{:g}\t\t{:g}".format(step, train_loss, train_acc, domain_acc_train))
            # train_summary_writer.add_summary(summaries, step)

        def dev_batch(x_batch, y_batch, d_batch):
            for i in range(1):
                d_i = np.zeros( ( len(x_batch), num_domain ) )
                index_i = np.array( [i] * len(x_batch) )
                d_i[ np.arange(len(x_batch)), index_i  ] = 1
                
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.input_d_idx: d_batch,
                cnn.input_d: d_i,
                cnn.dropout_keep_prob: 1.0,
                cnn.adv_lambda: 0.0,
            }
            step, loss, accuracy, domain_acc = sess.run(
                [global_step, cnn.y_loss_test, cnn.y_acc_test, cnn.domain_acc_test],
                feed_dict)
            # scores_list.append( scores )
            print("P1\t{}\t{:g}\t{:g}\t\t{:g}".format(step, loss, accuracy, domain_acc))

            return accuracy
        
        def dev_step(x_dev, y_dev, d_dev):
            cor = 0.
            step = FLAGS.dev_size
            #print( len(x_dev) )
            for ind in range(0, len(x_dev), step):
                
                num_ins = min(len(x_dev) - ind, step)
                acc = dev_batch(
                    x_batch = x_dev[ind: ind + num_ins],
                    y_batch = y_dev[ind: ind + num_ins],
                    d_batch = d_dev[ind: ind + num_ins]
                    )
                cor = cor + num_ins * acc
            acc = cor / len( x_dev )
            print("1\t{}".format(acc))
            return acc


        best_scores_pre_dev = []
        best_scores_pre = []
        best_scores_dev = []
        best_scores = []

        #data_split CV_iter used for split A(x,y,d) by domain 0-3(BDEK) for test, 4-12 for training
        cv_iter = data_helpers.cross_validation_iter(
            data=[A_x, A_y, A_d],
        )
        for _ in range(1):
            A_x_train, A_y_train, A_d_train,\
                x_test_all, y_test_all, d_test_all = cv_iter.fetch_next()
            print("split train {}+{}={} / dev {}".format(len(A_x_train), len(C_x), len(A_x_train)+len(C_x), len(x_test_all)))

            x_test = [ [], [], [], [] ]
            y_test = [ [], [], [], [] ]
            d_test = [ [], [], [], [] ]
            for i in range( len(x_test_all) ):
                dom = d_test_all[i][0]
                x_test[dom].append( x_test_all[i] )
                y_test[dom].append( y_test_all[i] )
                d_test[dom].append( d_test_all[i] )

            x_dev = [ [], [], [], [] ]
            y_dev = [ [], [], [], [] ]
            d_dev = [ [], [], [], [] ]
            for i in range(4):
                x_dev[i], y_dev[i], d_dev[i], \
                x_test[i], y_test[i], d_test[i] = data_helpers.split_dev_test( x_test[i], y_test[i], d_test[i], ratio = 0.4 )
            print("dev {} / test {}".format(np.sum( [len(i) for i in x_dev] ), np.sum( [ len(i) for i in x_test ])) )

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # sess.run(cnn.W.assign(w2v))
            best_score_pre_dev = np.zeros( (4) )
            best_score_pre = np.zeros( (4) )
            best_score_cv_dev = np.zeros( (4) )
            best_score_cv = np.zeros( (4) )

            # batch Generator
            if FLAGS.is_use_unsup_domain_data: #is using C?
                x_train = np.concatenate( (A_x_train, C_x), axis = 0 )
                y_train = np.concatenate( (A_y_train, C_y), axis = 0 )
                d_train = np.concatenate( (A_d_train, C_d), axis = 0 )
            else:
                x_train, y_train, d_train = A_x_train, A_y_train, A_d_train
            
            train_batch_iter = data_helpers.batch_iter(
                data = [x_train, y_train, d_train],
                batch_size = FLAGS.batch_size)
            train_unsup_batch_iter = data_helpers.batch_iter(
                data = [B_x, B_d],
                batch_size = FLAGS.batch_size,
            )
            data_size = len(x_train)

            # pre-train only train label (y) loss
            for _ in range( FLAGS.num_pretrain_epochs * data_size // FLAGS.batch_size ):
                x_batch, y_batch, d_batch = train_batch_iter.next_full_batch()

                pretrain_batch( x_batch, y_batch, d_batch,
                    opt=optimizer_pretrain, adv_lbd=FLAGS.adv_lambda, lr=FLAGS.learning_rate)

                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    for dom in range( 4 ):
                        acc = dev_step( x_dev[dom], y_dev[dom], d_dev[dom] )
                        if acc > best_score_pre_dev[dom]:
                            best_score_pre_dev[dom] = acc
                            best_score_pre[dom] = dev_step( x_test[dom], y_test[dom], d_test[dom] )
            
            best_scores_pre_dev.append(best_score_pre_dev)
            best_scores_pre.append(best_score_pre)
            print("best phase 1 score {}".format(best_score_pre))

            print("phase 2")
            # Training loop. For each batch...
            for it in range( FLAGS.num_train_epochs * data_size // FLAGS.batch_size ):

                x_batch, y_batch, d_batch = train_batch_iter.next_full_batch()
                train_batch( x_batch, y_batch, d_batch,
                    opt=optimizer_n, adv_lbd=FLAGS.adv_lambda, lr=FLAGS.learning_rate,
                    )
                
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.evaluate_every == 0:
                    for dom in range( 4 ):
                        acc = dev_step( x_dev[dom], y_dev[dom], d_dev[dom] )
                        if acc > best_score_cv_dev[dom]:
                            best_score_cv_dev[dom] = acc
                            best_score_cv[dom] = dev_step( x_test[dom], y_test[dom], d_test[dom] )

            best_scores_dev.append(best_score_cv_dev)
            best_scores.append(best_score_cv)
            print("best phase 2 score {}".format(best_score_cv))

print( best_scores_pre_dev)
print( best_scores_pre)
print( np.average( best_scores_pre) )

print( best_scores_dev )
print( best_scores )
print( np.average(best_scores) )

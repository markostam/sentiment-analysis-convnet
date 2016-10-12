#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from grid_search import get_params
from text_cnn import TextCNN
from tensorflow.contrib import learn
from collections import OrderedDict

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 64, "Number of filters per filter size (default: 64)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_float("dropout_factor", 0.5, "Probability of weights to keep for dropout (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", .0005, "Gradient descent learning rate (default: .0005)")
tf.flags.DEFINE_float("fc_layers", 1, "number of fully connected layers at output (1 or 2) (default: 1)")
tf.flags.DEFINE_string("activation_func", 'relu', "activation function (can be: tanh or relu) (default: relu)")
#tf.flags.DEFINE_float("l2_constraint", None, "Constraint on l2 norms of weight vectors (default: None)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 32, "Batch Size (default: 32)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("evaluate_every", 200, "Evaluate model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 400, "Save model after this many steps (default: 200)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_integer("num_threads", 8, "Number of threads to use for computation (default = 8)")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
params = get_params()
hyper_params = list(params.keys())
print("\nParameters:")
flags_list = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items())]
save_flags = ["{}={}".format(attr.upper(), value) for attr,value in sorted(FLAGS.__flags.items()) if attr in hyper_params]
for i in flags_list:
    print(i)
print("")

if FLAGS.fc_layers > 2:
    print("model maximum of two fully connected layers, defaulting to two")
    FLAGS.fc_layers == 2

# Data Preparatopn
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_labels_save_test_and_train()

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement = FLAGS.allow_soft_placement,
      log_device_placement = FLAGS.log_device_placement,
      #intra_op_parallelism_threads = FLAGS.num_threads
      )

    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=2,
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda,
            fc_layers=FLAGS.fc_layers,
            activation_func=FLAGS.activation_func)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        # if FLAGS.l2_constraint: #add l2 constraint as described in (Kim, Y. 2014)
        #     grads_and_vars = [(tf.clip_by_norm(grad, FLAGS.l2_constraint), var) for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.merge_summary(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        print(save_flags)
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", ','.join(save_flags)))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.scalar_summary("loss", cnn.loss)
        acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.all_variables())

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_factor
            }
            _, step, summaries, loss, accuracy = sess.run(
                [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }
            step, summaries, loss, accuracy = sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            if writer:
                writer.add_summary(summaries, step)
        # Early stopping
#        validation_metrics = {"accuracy": tf.contrib.metrics.streaming_accuracy,
#                  "precision": tf.contrib.metrics.streaming_precision,
#                  "recall": tf.contrib.metrics.streaming_recall}
#        validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#            x_dev,
#            y_dev,
#            every_n_steps=int(len(x_train/FLAGS.batch_size)),
#            metrics=validation_metrics,
#            early_stopping_metric="loss",
#            early_stopping_metric_minimize=True,
#            early_stopping_rounds=5)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print("")
            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}".format(path))

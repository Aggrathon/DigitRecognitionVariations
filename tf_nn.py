"""
    This script contains a neural network built with tensorflow
"""
import sys
import numpy as np
import tensorflow as tf
from data import get_test_set, get_training_set
from utils import tf_learning_rate_scaling

FOLDER = 'tf_nn'

def model_fn(features, labels, mode):
    """
        The model for the Tensorflow estimator
    """
    training = (mode is tf.estimator.ModeKeys.TRAIN)
    prev_layer = features['img']
    for i, size in enumerate([1024, 1024, 512, 256, 128]):
        prev_layer = tf.layers.dense(prev_layer, size, activation=tf.nn.relu, name='fc%d'%i)
        prev_layer = tf.layers.dropout(prev_layer, 0.3, training=training, name='dropout%d'%i)
    logit = tf.layers.dense(prev_layer, 10, name='logit')
    result = tf.argmax(tf.nn.softmax(logit), 1)
    loss = None
    train_op = None
    if labels is not None:
        loss = tf.losses.softmax_cross_entropy(tf.one_hot(labels, 10), logit)
        if training:
            global_step = tf.train.get_global_step()
            learning_rate = tf_learning_rate_scaling([(0, 1e-3), (1000, 1e-4), (5000, 1e-5), (10000, 1e-6), (20000, 1e-7)], global_step)
            trainer = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = trainer.minimize(loss, global_step=global_step)
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions=result
    )

def evaluate(train=False, epochs=None):
    """
        Evaluate the fitness of this model

        Arguments:
          train: train the model before evaluating
          epochs: number of epochs to train (None means infinite)
    """
    tf.logging.set_verbosity(tf.logging.INFO)
    model = tf.estimator.Estimator(model_fn, FOLDER)
    if train:
        img, lab = get_training_set()
        img.shape = img.shape[0], np.product(img.shape[1:])
        input_fn = tf.estimator.inputs.numpy_input_fn(dict(img=img), lab, 192, epochs, True, 6000)
        model.train(input_fn)
    img, lab = get_test_set()
    img.shape = img.shape[0], np.product(img.shape[1:])
    return model.evaluate(tf.estimator.inputs.numpy_input_fn(dict(img=img), lab, 1000, 1, True, 10000))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'evaluate':
        print(evaluate(False))
    elif len(sys.argv) == 2 and str.isnumeric(sys.argv[1]):
        evaluate(True, int(sys.argv[1]))
    else:
        evaluate(True)

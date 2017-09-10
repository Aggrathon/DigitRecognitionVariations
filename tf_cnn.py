"""
    This script contains a convolutional neural network built with tensorflow
"""
import sys
import tensorflow as tf
from data import get_test_set, get_training_set

FOLDER = 'tf_cnn'

def model_fn(features, labels, mode):
    """
        The model for the Tensorflow estimator
    """
    training = (mode is tf.estimator.ModeKeys.TRAIN)
    prev_layer = features['img']
    for i, (size, kernel) in enumerate([(32, 7), (64, 5), (96, 3)]):
        prev_layer = tf.layers.conv2d(prev_layer, size, kernel, 1, 'same', activation=tf.nn.relu, name='conv%d'%i)
        prev_layer = tf.layers.max_pooling2d(prev_layer, 3, 2, 'valid', name='pool%d'%i)
        prev_layer = tf.layers.batch_normalization(prev_layer, name="norm%d"%i, training=training)
    prev_layer = tf.contrib.layers.flatten(prev_layer)
    for i, size in enumerate([128, 128]):
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
            learning_rate = 0.1/tf.add(tf.to_float(global_step), 1.)
            learning_rate = tf.minimum(1e-3, tf.maximum(1e-5, learning_rate))
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
        input_fn = tf.estimator.inputs.numpy_input_fn(dict(img=img), lab, 128, epochs, True, 6000)
        model.train(input_fn)
    img, lab = get_test_set()
    return model.evaluate(tf.estimator.inputs.numpy_input_fn(dict(img=img), lab, 1000, 1, True, 10000))


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == 'evaluate':
        print(evaluate(False))
    elif len(sys.argv) == 2 and str.isnumeric(sys.argv[1]):
        evaluate(True, int(sys.argv[1]))
    else:
        evaluate(True)

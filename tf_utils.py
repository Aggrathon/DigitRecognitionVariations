"""
    This script contains some useful tensorflow functions
"""
import tensorflow as tf


def tf_learning_rate_scaling(points, global_step):
    assert len(points) > 1
    global_step = tf.to_float(global_step)
    rate = tf.constant(0.0, tf.float32)
    for i in range(1, len(points)):
        x1, y1 = points[i-1]
        x2, y2 = points[i]
        assert x1 < x2
        assert y1 > y2
        a = (y1-y2)/(x1-x2)
        b1 = y1 - a * x1
        b2 = y2 - a * x2
        assert abs(b1-b2) < 1e-4
        b = (b1+b2)*0.5
        rate = tf.maximum(rate, tf.add(tf.multiply(global_step, a), b))
    rate = tf.maximum(rate, points[-1][1])
    return rate
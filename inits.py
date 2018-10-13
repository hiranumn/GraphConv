import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None, dtype=tf.float32):
    """Uniform init."""
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=dtype)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None, dtype=tf.float32):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None, dtype=tf.float32):
    """All zeros."""
    initial = tf.zeros(shape, dtype=dtype)
    return tf.Variable(initial, name=name)


def ones(shape, name=None, dtype=tf.float32):
    """All ones."""
    initial = tf.ones(shape, dtype=dtype)
    return tf.Variable(initial, name=name)
import tensorflow.compat.v1 as tf
import numpy as np

def simple_soft_threshold(r_,lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = tf.maximum(lam_, 0.)
    return tf.sign(r_) * tf.maximum(tf.abs(r_) - lam_, 0.)

def simple_soft_threshold_np(r_,lam_):
    "implement a soft threshold function y=sign(r)*max(0,abs(r)-lam)"
    lam_ = np.maximum(lam_, 0.)
    return np.sign(r_) * np.maximum(np.abs(r_) - lam_, 0.)
import tensorflow.compat.v1 as tf
import tools.shrinkage as shrinkage

def build_LISTA(Y_,var_):
    layers = []
    eta = shrinkage.simple_soft_threshold
    #...
    W2 = var_.pop(0)
    W1 = var_.pop(0)
    lam0_ = var_.pop(0)
    xhat_ = eta(tf.matmul(W2, Y_), lam0_)
    xhat_ = tf.identity(xhat_, name='X_hat_0')
    layers.append(xhat_)
    #...
    T_ = len(var_)
    for t in range(0,T_):
        #...
        lam_ = var_.pop(0)
        xhat_ = eta(tf.matmul(W1, xhat_) + tf.matmul(W2, Y_), lam_)
        xhat_ = tf.identity(xhat_, name='X_hat_{0}'.format(t+1))
        layers.append(xhat_)
        #...
    return layers
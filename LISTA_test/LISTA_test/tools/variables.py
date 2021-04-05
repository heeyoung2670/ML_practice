import numpy as np
import numpy.linalg as la
import tensorflow.compat.v1 as tf

def get_LISTA_variable(A):
    T = 7
    M, N = A.shape
    var_ = []
    D = A.T / (1.01 * la.norm(A, 2) ** 2)
    D_ = tf.Variable(D, dtype=tf.float32, name='D_0')
    S_ = tf.Variable(np.identity(N) - np.matmul(D, A), dtype=tf.float32, name='S_0')
    var_.append(D_)
    var_.append(S_)
    initial_lambda = np.array(.1).astype(np.float32) * np.ones((N, 1), dtype=np.float32)
    for t in range(0,T+1):
        lam_ = tf.Variable(initial_lambda, name='lam_{0}'.format(t))
        var_.append(lam_)
    return var_
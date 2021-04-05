import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import time

np.random.seed(1)
tf.set_random_seed(1)

from tools import variables,networks,train

M = 25
N = 25
B = 5000
pnz = .1
SNR = 40
starter_learning_rate = 1e-4
decay_factor = 0.96
decay_step_size = 1000
iter = 100000
A = np.random.normal(size=(M, N), scale=1.0 / np.sqrt(M)).astype(np.float32)
A_ = tf.constant(A,name='A')
X_ = tf.placeholder(tf.float32, (N, None), name='X')
Y_ = tf.placeholder(tf.float32, (M, None), name='Y')

var_ = variables.get_LISTA_variable(A)

layers = networks.build_LISTA(Y_,var_)

start = time.time()
train.do_training(A,M,N,B,pnz,SNR,starter_learning_rate,decay_factor,decay_step_size,iter,X_,Y_,layers)
end = time.time()

print('time : ',end-start)
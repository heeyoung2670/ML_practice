import numpy as np
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from tools import data,FISTA

M = 25
N_1x = 25
N_4x = 100
test_B = 10000
pnz = .1
SNR = 40
T = 7
LISTA_1x = []
FISTA_1x = []
LISTA_4x = []
FISTA_4x = []

with tf.Session() as sess_1x:
    saver = tf.train.import_meta_graph('./LISTA_model/1x/B5e3_R1e-4_D1e3_I1e5.ckpt.meta')
    saver.restore(sess_1x, './LISTA_model/1x/B5e3_R1e-4_D1e3_I1e5.ckpt')
    graph = tf.get_default_graph()

    A_ = graph.get_tensor_by_name("A:0")
    A = sess_1x.run(A_)
    X_ = graph.get_tensor_by_name("X:0")
    Y_ = graph.get_tensor_by_name("Y:0")

    test_x, test_y = data.bernoulli_gaussian_trial(A, M, N_1x, test_B, pnz, SNR)

    for t in range(1, T+1):
        X_hat_ = graph.get_tensor_by_name("X_hat_{0}:0".format(t))
        xhat = sess_1x.run(X_hat_, feed_dict={Y_: test_y})
        xhat_fista = FISTA.fista(A, test_y, .5, t)
        LISTA_1x.append(sess_1x.run(tf.reduce_mean(tf.sqrt(tf.reduce_mean((xhat - test_x) ** 2, 0)))))
        FISTA_1x.append(np.mean(np.sqrt(np.mean((xhat_fista - test_x) ** 2, axis=0))))

tf.reset_default_graph()

with tf.Session() as sess_4x:
    saver = tf.train.import_meta_graph('./LISTA_model/4x/B1e4_R1e-4_D1e3_I1e5.ckpt.meta')
    saver.restore(sess_4x, './LISTA_model/4x/B1e4_R1e-4_D1e3_I1e5.ckpt')
    graph = tf.get_default_graph()

    A_ = graph.get_tensor_by_name("A:0")
    A = sess_4x.run(A_)
    X_ = graph.get_tensor_by_name("X:0")
    Y_ = graph.get_tensor_by_name("Y:0")

    test_x, test_y = data.bernoulli_gaussian_trial(A, M, N_4x, test_B, pnz, SNR)

    for t in range(1, T+1):
        X_hat_ = graph.get_tensor_by_name("X_hat_{0}:0".format(t))
        xhat = sess_4x.run(X_hat_, feed_dict={Y_: test_y})
        xhat_fista = FISTA.fista(A, test_y, .5, t)
        LISTA_4x.append(sess_4x.run(tf.reduce_mean(tf.sqrt(tf.reduce_mean((xhat - test_x) ** 2, 0)))))
        FISTA_4x.append(np.mean(np.sqrt(np.mean((xhat_fista - test_x) ** 2, axis=0))))

x_axis = np.arange(1,T+1,1)
plt.figure()
plt.scatter(x_axis, LISTA_1x, c="r", marker="o", label="LISTA(1x)")
plt.scatter(x_axis, FISTA_1x, c="r", marker="x", label="FISTA(1x)")
plt.scatter(x_axis, LISTA_4x, c="b", marker="o", label="LISTA(4x)")
plt.scatter(x_axis, FISTA_4x, c="b", marker="x", label="FISTA(4x)")
plt.xlabel('Layer')
plt.ylabel('RMSE')
plt.legend()
plt.show()
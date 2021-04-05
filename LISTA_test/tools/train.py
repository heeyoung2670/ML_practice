import tensorflow.compat.v1 as tf
from tools import data

def do_training(A,M,N,B,pnz,SNR,starter_learning_rate,decay_factor,decay_step_size,iter,X_,Y_,layers):
    xhat_ = layers.pop()
    loss_ = tf.nn.l2_loss(xhat_ - X_)/B
    global_step_ = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int32)
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step_,
                                               decay_step_size, decay_factor, staircase=True)
    train_ = tf.train.AdamOptimizer(learning_rate).minimize(loss=loss_, global_step=global_step_)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(iter):
            train_x, train_y = data.bernoulli_gaussian_trial(A,M,N,B,pnz,SNR)
            train_.run(feed_dict={X_: train_x, Y_: train_y})
            if i % 1000 == 0:
                val_x, val_y = data.bernoulli_gaussian_trial(A,M,N,B,pnz,SNR)
                val_loss = sess.run([loss_], feed_dict={X_: val_x, Y_: val_y})
                print_string = [i]+val_loss
                print(' '.join('%s' % x for x in print_string))
        saver = tf.train.Saver()
        saver.save(sess, './LISTA_model/1x/B5e3_R1e-4_D1e3_I1e5.ckpt')
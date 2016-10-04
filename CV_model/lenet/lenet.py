# -*- coding:utf-8 -*-

import time
import tensorflow as tf


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def main():

    sess = tf.InteractiveSession()

    x = tf.placeholder('float',[None,1024])
    y_ = tf.placeholder('float',[None,10])
    x_image = tf.reshape(x,[-1,32,32,1])

    W_conv1 = weight_variable([5, 5, 1, 6])
    b_conv1 = bias_variable([6])
    h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5,5,6,16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_conv3 = weight_variable([5,5,16,120])
    b_conv3 = bias_variable([6])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3)+b_conv3)


    W_fc1 = weight_variable([120,84])
    b_fc1 = bias_variable([84])
    h_fc1 = tf.matmul(h_conv3,W_fc1) + b_fc1
    W_fc2 = weight_variable([84,10])
    b_fc2 = bias_variable([10])
    h_fc2 = tf.matmul(h_fc1,W_fc2) + b_fc2


    y_conv=tf.nn.softmax(h_fc2)


    cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    saver = tf.train.Saver()

    sess.run(tf.initialize_all_variables())

    # read the data with 32*32 mnist
    mnist_data_set = read_mnist_32*32()
    test_images,test_labels = mnist_data_set.test_data()

    start_time = time.time()
    for i in xrange(20000):
        batch_xs, batch_ys = mnist_data_set.next_train_batch(50)

        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_: batch_ys, keep_prob: 1.0})
            print "step %d, training accuracy %g"%(i, train_accuracy)
            end_time = time.time()
            print 'time: ',(end_time - start_time)
            start_time = end_time
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

    if not tf.gfile.Exists('model_data'):
        tf.gfile.MakeDirs('model_data')
    save_path = saver.save(sess, "model.ckpt")
    print "Model saved in file: ", save_path

    avg = 0
    for i in xrange(200):
        avg+=accuracy.eval(feed_dict={x: test_images[i*50:i*50+50], y_: test_labels[i*50:i*50+50], keep_prob: 1.0})
    avg/=200
    print "test accuracy %g"%avg

    sess.close()


if __name__ == '__main__':
    main()

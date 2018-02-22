import tensorflow as tf
import random

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)

mnist = input_data.read_data_sets("MINST_data/",one_hot=True)

learning_rate=0.001
training_epochs=15
batch_size=100

keep_prob = tf.placeholder(tf.float32)

X = tf.placeholder(tf.float32, [None, 28*28*1])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# convolution-relu-max_pool-drop_out layer
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1,ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

L3_flat = tf.reshape(L3, [-1, 4*4*128])

# fully-connected layer 4x4x128 inputs --> 625 outputs
W4 = tf.get_variable("W4", shape=[4*4*128, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4)+b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# last f-c layer 625 inputs ---> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# load .ckpt
saver = tf.train.Saver()
ckpt_path = saver.restore(sess, tf.train.latest_checkpoint("ckpt"))

"""
# Learning rountine
print('Learning Started!')
for epoch in range(training_epochs):
    ckpt_path = saver.save(sess, "ckpt/deep_cnn")
    print("save ckpt file: ", ckpt_path)

    avg_cost = 0
    total_batch = int(mnist.train.num_examples/batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost,optimizer], feed_dict=feed_dict)
        avg_cost += c/total_batch
    print('Epoch: ','%04d' % (epoch+1), 'cost= ', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
"""

# get accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy: ", sess.run(accuracy, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

# get one and prediction
r = random.randint(0, mnist.test.num_examples -1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
print("Prediction: ", sess.run(tf.argmax(logits, 1), feed_dict={X:mnist.test.images[r:r+1], keep_prob:1}))
sess.close()

ckpt_state = tf.train.get_checkpoint_state("ckpt")
print(type(ckpt_state))
print()
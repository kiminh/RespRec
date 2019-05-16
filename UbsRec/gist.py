import numpy as np
import tensorflow as tf


t = tf.constant([1, 2, 3], tf.float32)
p = tf.placeholder(tf.float32, (3,))
l = tf.reduce_sum(tf.multiply(t, p))
gt = tf.gradients(l, t, gate_gradients=False)
gp = tf.gradients(l, p, gate_gradients=False)
d = np.asarray([7, 8, 9], np.float32)
fd = {p: d}
s = tf.Session()
print('t', s.run(t))
print('p', s.run(p, feed_dict=fd))
print('l', s.run(l, feed_dict=fd))
print('gt', s.run(gt, feed_dict=fd))
print('gp', s.run(gp, feed_dict=fd))


import numpy as np
import tensorflow as tf

t = tf.constant([[1, 2], [3, 4], [5, 6]], tf.float32)
f = tf.constant([0, 1], tf.int32)
e = tf.nn.embedding_lookup(t, f)
# f = tf.constant([[0, 0], [1, 1]], tf.int32)
# e = tf.gather_nd(t, f)
l = tf.reduce_sum(e)
s = tf.Session()
g = tf.gradients(l, t)[0]
d = g.to_dense()
print(d)
print(s.run(d))
# print(s.run(g))
exit()

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


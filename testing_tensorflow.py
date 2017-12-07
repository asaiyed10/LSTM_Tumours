import tensorflow as tf
hello = tf.constant('Smell,my,booty,booty,booty,Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))

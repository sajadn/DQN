# import te
#
#
# def conv2d(x,
#            output_dim,
#            kernel_size,
#            stride,
#            activation_fn=tf.nn.relu,
#            name):
#   with tf.variable_scope(name):
#     w = tf.get_variable('w', kernel_shape,
#         tf.float32, initializer=weights_initializer)
#     conv = tf.nn.conv2d(x, w, stride, padding, data_format=data_format)
#
#     b = tf.get_variable('b', [output_dim],
#         tf.float32, initializer=biases_initializer)
#     out = tf.nn.bias_add(conv, b, data_format)
#
#   if activation_fn != None:
#     out = activation_fn(out)
#
#   return out, w, b

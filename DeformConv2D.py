import tensorflow as tf
import numpy as np

class DefromConv2D(object):
    def __init__(self, inputs, kernel_size, filters, groups, name, trainable=True):
        """ Definition of the DeformConv2D class
        :param inputs: inputs of the deformable convolutional layer, a 4-D Tensor with shape [batch_size, height, width, channels].
        :param kernel_size:  Value of kernel size.
        :param filters: Output channels (The amount of kernels).
        :param groups: Name of the deformable convolution layer.
        :param name: The amount of groups.
        :param trainable: Whether the weights are trainable or not.
        """
        self.inputs = inputs
        self.kernel_size = kernel_size
        self.n_weights = kernel_size**2
        self.filters = filters
        self.groups = groups
        self.name = name
        self.trainable = trainable

    def conv(self, inputs, filters, mode, relu=True, groups=1, stride=1):
        assert mode in ["feature", "offset"] # Exception processing

        with tf.name_scope(self.name + "_" + mode):
            kernel_size = self.kernel_size
            input_kernel_size = inputs.get_shape()[-1] / groups

            if mode == "offset":
                with tf.variable_scope(self.name+"_offset"):
                    weights = tf.get_variable(name="weights",
                                             shape=[kernel_size, kernel_size, input_kernel_size, filters],
                                             trainable=self.trainable,
                                             initializer=tf.zeros_initializer)
                    biases = tf.get_variable(name="biases",
                                             shape=[filters],
                                             trainable=self.trainable,
                                             initializer=tf.zeros_initializer)
            else:
                with tf.variable_scope(self.name):
                    weights = tf.get_variable(name="weights",
                                              shape=[kernel_size, kernel_size, input_kernel_size, filters],
                                              trainable=self.trainable)
                    biases = tf.get_variable(name="biases",
                                             shape=[filters],
                                             trainable=self.trainable,
                                             initializer=tf.zeros_initializer)

            def conv2d(inputs, weights):
                return tf.nn.conv2d(inputs, weights, [1,stride,stride,1], padding="SAME")


            if groups == 1:
                layer_output = conv2d(inputs, weights)
            else:
                group_inputs = tf.split(inputs, groups, 3, name="split_input")
                group_weights = tf.split(weights, groups,3, name="split_weight")
                group_outputs = [conv2d(inputs, weights) for inputs, weights in zip(group_inputs, group_weights)]

                layer_output = tf.nn.bias_add(layer_output, biases)

                if relu:
                    layer_output = tf.nn.relu(layer_output)

                return layer_output

    def infer(self):
        with tf.name_scope(self.name):
            inputs = self.inputs[:,:,:,:]
            kernel_size = self.kernel_size
            n_weights = self.n_weights

            batch_size, height, width, filters = inputs.get_shape().as_list()
            offset = self.conv(inputs, 2*n_weights, "offset",relu=False)
            dtype = offset.dtype

            pn = self.get_pn(dtype)
            p0 = self.get_p0([batch_size, height, width, filters], dtype)
            p = p0 + pn + offset
            p = tf.reshape(p, [batch_size, height, width, 2*n_weights, 1, 1])

            # 'q' contains the location of each pixel on the ouput feature map.
            q = self.get_q([batch_size, height, width, filters], dtype)

            # Get the bilinear interpolation kernel G ([batch_size, height, width, n_weights, height, width])
            gx = tf.maximum(1 - tf.abs(p[:, :, :, :n_weights, :, :]-q[:, :, 0]), 0)
            gy = tf.maximum(1 - tf.abs(p[:, :, :, :n_weights, :, :]-q[:, :, 1]), 0)
            G = gx*gy
            G = tf.reshape(G, [batch_size, height*width*n_weights, height*width])

            inputs = tf.reshape(inputs, [batch_size, height*width, filters])

            inputs_offset = self.reshape_inputs_offset(inputs_offset, kernel_size)

            layer_output = self.conv(inputs_offset, self.filters, "feature", groups=self.groups, stride=kernel_size)
            return layer_output

    def get_pn(self, dtype):
        kernel_size = self.kernel_size
        pn_x, pn_y = np.meshgrid(range(-(kernel_size-1)/2, (kernel_size-1)/2+1), range(-(ks-1)/2, (ks-1)/2+1), indexing="ij")

        pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))
        pn = np.reshape(pn, [1, 1, 1, 2*self.n_weights])
        pn = tf.constant(pn, dtype)

        return pn

    def get_p0(self, inputs_size, dtype):
        batch_size, height, width, filters = inputs_size

        p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
        p0_x = p0_x.flatten().reshape(1, height, width, 1).repeat(self.n_weights, axis=3)
        p0_y = p0_y.flatten().reshape(1, height, width, 1).repeat(self.n_weights, axis=3)
        p0 = np.concatenate((p0_x, p0_y), axis=3)
        p0 = tf.constant(p0, dtype)

        return p0

    def get_q(self, inputs_size, dtype):
        batch_size, height, width, filters = inputs_size

        q_x, q_y = np.meshgrid(range(0, height), range(0, width), indexing="ij")
        q_x = q_x.flatten().reshape(height, width, 1)
        q_y = q_y.flatten().reshape(height, width, 1)
        q = np.concatenate((q_x, q_y), axis=2)
        q = tf.constant(q, dtype)

        return q

    def reshape_inputs_offset(inputs_offset, kernel_size):
        batch_size, height, width, n_weight, filters = inputs_offset.get_shape().as_list()

        new_shape = [batch_size, height, width*kernel_size, filters]

        inputs_offset = [tf.reshape(inputs_offset[:, :, :, s:s+kernel_size, :], new_shape) for s in range(0, n_weight, kernel_size)]
        inputs_offset = tf.concat(inputs_offset, axis=2)
        inputs_offset = tf.reshape(inputs_offset, [batch_size, height*kernel_size, width*kernel_size, filters])

        return inputs_offset

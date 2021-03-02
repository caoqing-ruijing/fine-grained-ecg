# -*- coding: utf-8 -*-
# https://github.com/kiyohiro8/SelfAttentionGAN/blob/master/SelfAttentionLayer.py


# from keras.engine.network import Layer
# from keras.layers import InputSpec
# import keras.backend as K

import tensorflow as tf
from tensorflow.keras.layers import InputSpec, Layer
import tensorflow.keras.backend as K


class SelfAttention(Layer):

    def __init__(self, ch, scale=8, split_num=None, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.channels = ch
        self.scale = scale
        self.filters_f_g = self.channels // self.scale
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(
            name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)

        super(SelfAttention, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def call(self, x):
        def hw_flatten(x):
            return K.reshape(x, shape=[K.shape(x)[0], K.shape(x)[1]*K.shape(x)[2], K.shape(x)[3]])

        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]

        s = K.batch_dot(hw_flatten(g), K.permute_dimensions(
            hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]

        beta = K.softmax(s, axis=-1)  # attention map

        o = K.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = K.reshape(o, shape=K.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'ch': self.channels, 'scale': self.scale})
        return config


class SelfAttentionWithLeads(Layer):

    def __init__(self, ch, scale=8, split_num=3, **kwargs):
        super(SelfAttentionWithLeads, self).__init__(**kwargs)
        self.channels = ch
        self.scale = scale
        self.split_num = split_num

        self.filters_f = self.channels // self.scale
        self.filters_g = self.channels // self.scale
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f = (1, 1) + (self.channels, self.filters_f)
        kernel_shape_g = (1, 1) + (self.channels, self.filters_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(
            name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f,
                                        initializer='glorot_uniform',
                                        name='kernel_f',
                                        trainable=True)
        self.kernel_g = self.add_weight(shape=kernel_shape_g,
                                        initializer='glorot_uniform',
                                        name='kernel_g',
                                        trainable=True)
        self.kernel_h = self.add_weight(shape=kernel_shape_h,
                                        initializer='glorot_uniform',
                                        name='kernel_h',
                                        trainable=True)

        super(SelfAttentionWithLeads, self).build(input_shape)
        # Set input spec.
        self.input_spec = InputSpec(ndim=4,
                                    axes={3: input_shape[-1]})
        self.built = True

    def _hw_flatten(self, x, return_shape=False):
        x = tf.split(x, self.split_num, axis=1)
        x = tf.concat(x, axis=-1)
        shape_x = K.shape(x)
        if return_shape:
            return K.reshape(x, shape=[shape_x[0], shape_x[1]*shape_x[2], shape_x[3]]), shape_x
        else:
            return K.reshape(x, shape=[shape_x[0], shape_x[1]*shape_x[2], shape_x[3]])

    def _hw_recover(self, x, shape_previous):
        x = K.reshape(x, shape=shape_previous)
        x = tf.split(x, self.split_num, axis=-1)
        x = tf.concat(x, axis=1)
        return x

    def call(self, x):
        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c]

        s = K.batch_dot(self._hw_flatten(g), K.permute_dimensions(
            self._hw_flatten(f), (0, 2, 1)))  # # [bs, N, N]
        beta = K.softmax(s, axis=-1)  # attention map

        h_tmp, shape_tmp = self._hw_flatten(h, return_shape=True)
        o = K.batch_dot(beta, h_tmp)  # [bs, N, C]
        o = self._hw_recover(o, shape_tmp)  # [bs, h, w, C]

        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(SelfAttentionWithLeads, self).get_config()
        config.update({'ch': self.channels, 'scale': self.scale, "split_num": self.split_num})
        return config


class DoubleAttentionWithLeads(SelfAttentionWithLeads):
    def __init__(self, ch, scale=4, split_num=3, **kwargs):
        super(DoubleAttentionWithLeads, self).__init__(ch, scale, split_num, **kwargs)
        self.filters_f = self.channels 
        self.filters_g = self.channels // self.scale
        self.filters_h = self.channels // self.scale
        
    def call(self, x):
        a = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, m]
        b = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, n]
        v = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, n]
        a_, shape_tmp = self._hw_flatten(a, return_shape=True)
        a = K.permute_dimensions(a_, (0, 2, 1)) # [bs, 3m, N]
        b_ = K.softmax(self._hw_flatten(b), axis=1) # [bs, N, 3n]
        g = K.batch_dot(a, b_)  # # [bs, 3m, 3n]

        v_tmp = self._hw_flatten(v) # [bs, N, 3n]
        v_tmp = K.softmax(K.permute_dimensions(v_tmp, (0,2,1)), axis=1) # [bs, 3n, N]
        z = K.batch_dot(g, v_tmp)  # [bs, 3m, N]
        z = K.permute_dimensions(z, (0,2,1))
        z = self._hw_recover(z, shape_tmp)  # [bs, h, w, C]

        x = self.gamma * z + x

        return x


class SelfDoubleAttentionWithLeads(SelfAttentionWithLeads):
    def __init__(self, ch, scale=4, split_num=3, **kwargs):
        super(SelfDoubleAttentionWithLeads, self).__init__(ch, scale, split_num, **kwargs)
        self.filters_f = self.channels // self.scale
        self.filters_g = self.channels // self.scale
        self.filters_h = self.channels // self.scale

    def call(self, x):
        f = K.conv2d(x,
                     kernel=self.kernel_f,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = K.conv2d(x,
                     kernel=self.kernel_g,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']
        h = K.conv2d(x,
                     kernel=self.kernel_h,
                     strides=(1, 1), padding='same')  # [bs, h, w, c']

        f_ = K.permute_dimensions(self._hw_flatten(f), (0, 2, 1)) # [bs, 3c', N]
        s = K.batch_dot(self._hw_flatten(g), f_)  # [bs, N, N]
        beta = K.softmax(s, axis=-1)  # attention map

        double_attn = K.batch_dot(f_, self._hw_flatten(x)) # [bs, 3c', 3c]
        double_attn = K.softmax(double_attn, axis=1)

        h_tmp, shape_tmp = self._hw_flatten(h, return_shape=True) # [bs, N, 3c']
        o_tmp = K.batch_dot(beta, h_tmp)  # [bs, N, 3c']
        o = K.batch_dot(o_tmp, double_attn) # [bs, N, 3c]
        o = self._hw_recover(o, shape_tmp)  # [bs, h, w, C]

        x = self.gamma * o + x

        return x
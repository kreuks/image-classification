import tensorflow as tf
from tensorflow.contrib import slim


class Model(object):
    @staticmethod
    def nielsen_net(
            inputs,
            num_classes=2,
            is_training=True,
            dropout_keep_prob=0.5,
            scope='NielsenNet',
    ):
        with tf.variable_scope(scope, 'NielsenNet'):
            net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')
            net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

            net = tf.reshape(net, [-1, 5 * 5 * 40])

            net = slim.fully_connected(net, 1000, scope='layer5')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='layer5-dropout')
            net = slim.fully_connected(net, 1000, scope='layer6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='layer6-dropout')
            net = slim.fully_connected(net, num_classes, scope='output')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='output-dropout')

            return net

    @staticmethod
    def vgg19(
            inputs,
            num_classes=2,
            is_training=True,
            dropout_keep_prob=0.5,
            scope='vgg19',
            fc_conv_padding='VALID'
    ):
        with tf.variable_scope(scope, 'vgg19'):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='layer1-conv1')
            net = slim.max_pool2d(net, [2, 2], scope='layer2-pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='layer3-conv2')
            net = slim.max_pool2d(net, [2, 2], scope='layer4-pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='layer5-conv3')
            net = slim.max_pool2d(net, [2, 2], scope='layer6-pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='layer7-conv4')
            net = slim.max_pool2d(net, [2, 2], scope='layer8-pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='layer9-conv5')
            net = slim.max_pool2d(net, [2, 2], scope='layer10-pool5')

            net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='layer11-fc6')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='layer12-dropout6')
            net = slim.conv2d(net, 4096, [1, 1], padding=fc_conv_padding, scope='layer13-fc7')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='layer14-dropout7')
            net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='layer14-fc8')

            return net

import tensorflow as tf
from keras import Input
from tensorflow.contrib import slim
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization, \
    AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.contrib.keras.python.keras import layers
from keras.models import Model as ModelKeras
from tensorflow.contrib.keras.python.keras import backend as K


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

            net = slim.fully_connected(net, 35 * 35 * 40, scope='layer5')
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='layer5-dropout')
            net = slim.fully_connected(net, 35 * 35 * 40, scope='layer6')
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

    @staticmethod
    def simple_keras(input_shape):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def VGG19(input_shape, pooling='max', include_top=True, num_classes=2):
        img_input = Input(shape=input_shape)
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if include_top:
            # Classification block
            x = Flatten(name='flatten')(x)
            x = Dense(4096, activation='relu', name='fc1')(x)
            x = Dense(4096, activation='relu', name='fc2')(x)
            x = Dense(num_classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        model = ModelKeras(img_input, x, name='vgg19')

        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

    @staticmethod
    def InceptionV3(input_shape, include_top=True, pooling='max', classes=2):
        def _conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
            x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False)(x)
            x = BatchNormalization(axis=3, scale=False)(x)
            x = Activation('relu', name=name)(x)
            return x
        
        img_input = Input(shape=input_shape)

        x = _conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
        x = _conv2d_bn(x, 32, 3, 3, padding='valid')
        x = _conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = _conv2d_bn(x, 80, 1, 1, padding='valid')
        x = _conv2d_bn(x, 192, 3, 3, padding='valid')
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 32, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed0')

        # mixed 1: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed1')

        # mixed 2: 35 x 35 x 256
        branch1x1 = _conv2d_bn(x, 64, 1, 1)

        branch5x5 = _conv2d_bn(x, 48, 1, 1)
        branch5x5 = _conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 64, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch5x5, branch3x3dbl, branch_pool],
            axis=3,
            name='mixed2')

        # mixed 3: 17 x 17 x 768
        branch3x3 = _conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = _conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = _conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = _conv2d_bn(
            branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch3x3dbl, branch_pool], axis=3, name='mixed3')

        # mixed 4: 17 x 17 x 768
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 128, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed4')

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = _conv2d_bn(x, 192, 1, 1)

            branch7x7 = _conv2d_bn(x, 160, 1, 1)
            branch7x7 = _conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = _conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch7x7, branch7x7dbl, branch_pool],
                axis=3,
                name='mixed' + str(5 + i))

        # mixed 7: 17 x 17 x 768
        branch1x1 = _conv2d_bn(x, 192, 1, 1)

        branch7x7 = _conv2d_bn(x, 192, 1, 1)
        branch7x7 = _conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = _conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = _conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = _conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
        x = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=3,
            name='mixed7')

        # mixed 8: 8 x 8 x 1280
        branch3x3 = _conv2d_bn(x, 192, 1, 1)
        branch3x3 = _conv2d_bn(branch3x3, 320, 3, 3, strides=(2, 2), padding='valid')

        branch7x7x3 = _conv2d_bn(x, 192, 1, 1)
        branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 1, 7)
        branch7x7x3 = _conv2d_bn(branch7x7x3, 192, 7, 1)
        branch7x7x3 = _conv2d_bn(
            branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = layers.concatenate(
            [branch3x3, branch7x7x3, branch_pool], axis=3, name='mixed8')

        # mixed 9: 8 x 8 x 2048
        for i in range(2):
            branch1x1 = _conv2d_bn(x, 320, 1, 1)

            branch3x3 = _conv2d_bn(x, 384, 1, 1)
            branch3x3_1 = _conv2d_bn(branch3x3, 384, 1, 3)
            branch3x3_2 = _conv2d_bn(branch3x3, 384, 3, 1)
            branch3x3 = layers.concatenate(
                [branch3x3_1, branch3x3_2], axis=3, name='mixed9_' + str(i))

            branch3x3dbl = _conv2d_bn(x, 448, 1, 1)
            branch3x3dbl = _conv2d_bn(branch3x3dbl, 384, 3, 3)
            branch3x3dbl_1 = _conv2d_bn(branch3x3dbl, 384, 1, 3)
            branch3x3dbl_2 = _conv2d_bn(branch3x3dbl, 384, 3, 1)
            branch3x3dbl = layers.concatenate(
                [branch3x3dbl_1, branch3x3dbl_2], axis=3)

            branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = _conv2d_bn(branch_pool, 192, 1, 1)
            x = layers.concatenate(
                [branch1x1, branch3x3, branch3x3dbl, branch_pool],
                axis=3,
                name='mixed' + str(9 + i))
        if include_top:
            # Classification block
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dense(classes, activation='softmax', name='predictions')(x)
        else:
            if pooling == 'avg':
                x = GlobalAveragePooling2D()(x)
            elif pooling == 'max':
                x = GlobalMaxPooling2D()(x)

        # Create model.
        model = ModelKeras(img_input, x, name='inception_v3')
        model.compile(loss='binary_crossentropy',
                      optimizer='rmsprop',
                      metrics=['accuracy'])

        return model

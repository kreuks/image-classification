import tensorflow as tf
from keras import Input
from tensorflow.contrib import slim
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model as ModelKeras

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

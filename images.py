import glob

import numpy as np

from constant import Data, Image, Train
from config import config
import tensorflow as tf
from tensorflow.contrib.keras import preprocessing
from tensorflow.contrib import slim
from tensorflow.python.framework import ops, dtypes
from tensorflow.examples.tutorials.mnist import input_data as mnist_data


class ImageGenerator(object):
    def __init__(self, config):
        self._config = config
        self.resize = self._config[Image.IMAGE][Image.RESIZE]
        self.num_channel = self._config[Image.IMAGE][Image.NUM_CHANNEL]
        self.rescale = self._config[Image.IMAGE][Image.RESCALE]
        self.batch_size = self._config[Image.IMAGE][Image.BATCH_SIZE]

    @staticmethod
    def create_tensor_list(path_images):
        image_training_path = path_images + '/' + Data.TRAINING + '_' + Data.DATA + '/'
        image_test_path = path_images + '/' + Data.TEST + '_' + Data.DATA + '/'

        classes = [x.replace(image_training_path, '') for x in glob.glob(image_training_path + '*')]

        image_list_train = []
        image_list_test = []
        label_list_train = []
        label_list_test = []

        for label, class_ in enumerate(classes):
            categorical = [0] * len(classes)
            categorical[label] = 1
            image_list_train += glob.glob(image_training_path + class_ + '/*')
            image_list_test += glob.glob(image_test_path + class_ + '/*')
            label_list_train += [categorical] * len(glob.glob(image_training_path + class_ + '/*'))
            label_list_test += [categorical] * len(glob.glob(image_test_path + class_ + '/*'))

        return (
            ops.convert_to_tensor(image_list_train, dtype=dtypes.string),
            ops.convert_to_tensor(label_list_train, dtype=dtypes.int32),
            ops.convert_to_tensor(image_list_test, dtype=dtypes.string),
            ops.convert_to_tensor(label_list_test, dtype=dtypes.int32)
        )

    def decode_images(self, input_queue):
        labels = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        images = tf.image.decode_jpeg(
            file_contents,
            channels=self.num_channel
        )
        images = tf.image.resize_images(images, self.resize)
        return images, labels

    def flow_directory(self, path_images):
        images_train, labels_train, images_test, labels_test = self.create_tensor_list(path_images)
        input_queue_train = tf.train.slice_input_producer([images_train, labels_train], shuffle=True)
        input_queue_test = tf.train.slice_input_producer([images_test, labels_test], shuffle=True)

        train_images, train_labels = self.decode_images(input_queue_train)
        test_images, test_labels = self.decode_images(input_queue_test)

        train_images, test_images = train_images / self.rescale, test_images / self.rescale

        image_batch_train, label_batch_train = tf.train.batch([train_images, train_labels], batch_size=self.batch_size)
        image_batch_test, label_batch_test = tf.train.batch([test_images, test_labels], batch_size=self.batch_size)
        return image_batch_train, label_batch_train, image_batch_test, label_batch_test

    def resize_images(self, images):

        MEAN = np.mean(images)
        STD = np.std(images)

        reshaped = (images - MEAN) / STD
        reshaped = np.reshape(reshaped, [-1, 28, 28, 1])

        return reshaped


class ImageGeneratorKeras:
    @staticmethod
    def load_train_data(path, classes):
        train_datagen = preprocessing.image.ImageDataGenerator(
            rescale=1. / 255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        train_datagenerator = train_datagen.flow_from_directory(
            path,
            target_size=(28, 28),
            color_mode='grayscale',
            classes=classes,
            class_mode='categorical',
            batch_size=1000
        )
        return train_datagenerator


class TensorModel(object):
    def __init__(self, config=config):
        self._config = config
        self.num_epoch = self._config[Train.TRAIN][Train.NUM_EPOCH]
        self.mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)


    @staticmethod
    def nielsen_net(inputs, is_training, scope='NielsenNet'):
        with tf.variable_scope(scope, 'NielsenNet'):
            # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
            net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

            # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
            net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
            net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

            # Reshape: 5x5x40 => 1000x1
            net = tf.reshape(net, [-1, 5 * 5 * 40])

            # Fully Connected Layer: 1000x1 => 1000x1
            net = slim.fully_connected(net, 1000, scope='layer5')
            net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

            # Second Fully Connected: 1000x1 => 1000x1
            net = slim.fully_connected(net, 1000, scope='layer6')
            net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

            # Output Layer: 1000x1 => 10x1
            net = slim.fully_connected(net, 10, scope='output')
            net = slim.dropout(net, is_training=is_training, scope='output-dropout')

            return net

    @staticmethod
    def model(inputs,
               is_training=True,
               spatial_squeeze=True,
               scope='NielsenNet'):
        """
        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes.
          is_training: whether or not the model is being trained.
          dropout_keep_prob: the probability that activations are kept in the dropout
            layers during training.
          spatial_squeeze: whether or not should squeeze the spatial dimensions of the
            outputs. Useful to remove unnecessary dimensions for classification.
          scope: Optional scope for the variables.
          fc_conv_padding: the type of padding to use for the fully connected layer
            that is implemented as a convolutional layer. Use 'SAME' padding if you
            are applying the network in a fully convolutional manner and want to
            get a prediction map downsampled by a factor of 32 as an output.
            Otherwise, the output prediction map will be (input / 32) - 6 in case of
            'VALID' padding.


        Returns:
          the last op containing the log predictions and end_points dict.
        """
        with tf.variable_scope(scope, 'NielsenNet'):
            net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='conv1')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool')

            net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='conv2')
            net = slim.max_pool2d(net, 2, stride=2, scope='pool2')

            net = tf.reshape(net, [-1, 5 * 5 * 40])

            net = slim.fully_connected(net, 1000, scope='full_connected1')
            net = slim.dropout(net, is_training=is_training, scope='dropout1')

            net = slim.fully_connected(net, 1000, scope='full_connected2')
            net = slim.dropout(net, is_training=is_training, scope='dropout2')

            net = slim.fully_connected(net, 10, scope='output')
            net = slim.dropout(net, is_training=is_training, scope='output-dropout')
            return net

    def train_model(self, data_train, data_test):
        X = tf.placeholder(tf.float32, [None, 28 * 28])
        W = tf.Variable(tf.zeros([28 * 28, 2]))
        b = tf.Variable(tf.zeros([2]))
        y_hat = tf.nn.softmax(tf.matmul(X, W) + b)
        y = tf.placeholder(tf.float32, [None, 2])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

        image_generator = ImageGenerator(config=config)
        image_batch_train, label_batch_train, image_batch_test, label_batch_test = image_generator.flow_directory(
            'images/catdog'
        )

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for _ in range(self.num_epoch):
                image_batch_train = tf.reshape(image_batch_train, [int(image_batch_train.shape[0]), -1]).eval()
                if _ % 100 == 0:
                    print 'cost_at_step {} :{}'.format(
                        _, cross_entropy.eval(feed_dict={X: image_batch_train, y: label_batch_train.eval()}) / 100
                    )
                sess.run(train_step, feed_dict={X: image_batch_train, y: label_batch_train.eval()})

            correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            image_batch_test = tf.reshape(image_batch_test, [int(image_batch_test.shape[0]), -1]).eval()
            print sess.run(accuracy, feed_dict={X: image_batch_test, y: label_batch_test.eval()})
            print sess.run(tf.argmax(y, 1), feed_dict={y: label_batch_test.eval()})
            print sess.run(tf.argmax(y_hat, 1), feed_dict={X: image_batch_test})
            tf.summary.FileWriter('logs', sess.graph)

            coord.request_stop()
            coord.join(threads)

    def resize_images(self, images):

        MEAN = np.mean(images)
        STD = np.std(images)

        reshaped = (images - MEAN) / STD
        reshaped = np.reshape(reshaped, [-1, 28, 28, 1])

        return reshaped

    def run(self):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
        y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='Labels')
        is_training = tf.placeholder(tf.bool, name='IsTraining')
        image_generator = ImageGenerator(config=config)
        image_batch_train, label_batch_train, image_batch_test, label_batch_test = image_generator.flow_directory(
            'images/catdog'
        )

        logits = self.model(
            x,
            num_classes=10,
            is_training=True,
            dropout_keep_prob=0.5,
            scope='NielsenNet'
        )
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_actual))
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_step = tf.train.MomentumOptimizer(0.01, 0.5).minimize(cross_entropy)

        loss_summary = tf.summary.scalar('loss', cross_entropy)
        accuracy_summary = tf.summary.scalar('accuracy', accuracy)

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            sess.run(tf.global_variables_initializer())
            train_writer = tf.summary.FileWriter('logs/nielsen-net', sess.graph)

            eval_data = {
                x: image_batch_test.eval(),
                y_actual: label_batch_test.eval(),
                is_training: False
            }

            for i in xrange(1000):
                # image_batch_train = tf.reshape(image_batch_train, [int(image_batch_train.shape[0]), -1]).eval()eval
                # images, labels = self.mnist.train.next_batch(100)
                summary, _ = sess.run([loss_summary, train_step],
                                      feed_dict={x: image_batch_train, y_actual: label_batch_train.eval(), is_training: True})
                train_writer.add_summary(summary, i)

                if i % 1000 == 0:
                    summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
                    train_writer.add_summary(summary, i)
                    print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

            coord.request_stop()
            coord.join(threads)

        # self.train_model(None, None)


if __name__ == '__main__':
    tensor = TensorModel()
    tf.app.run(main=tensor.run())

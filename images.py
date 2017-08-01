import glob

import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import preprocessing
from tensorflow.python.framework import ops, dtypes

from constant import Data


class ImageGenerator(object):
    def __init__(self, resize=None, channel=None):
        self.resize = resize if resize else [28, 28]
        self.channel = channel if channel else 1
        self.rescale = 1. / 255

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
            channels=self.channel
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

        image_batch_train, label_batch_train = tf.train.batch([train_images, train_labels], batch_size=100)
        image_batch_test, label_batch_test = tf.train.batch([test_images, test_labels], batch_size=400)
        return image_batch_train, label_batch_train, image_batch_test, label_batch_test


class TensorModel(object):
    TRAIN_FILES = 'images/catdog/training_data/'
    TEST_FILES = 'images/catdog/testing_data/'
    CLASS = ['cats', 'dogs']

    @staticmethod
    def resize_image(files):
        reader = tf.WholeFileReader()
        _, content = reader.read(tf.train.string_input_producer(files))
        image = tf.image.decode_jpeg(content, channels=3)
        image = tf.cast(image, tf.float32)
        return tf.image.resize_images(image, [128, 128])

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

    def train_model(self, data_train, data_test):
        X = tf.placeholder(tf.float32, [None, 28 * 28])
        W = tf.Variable(tf.zeros([28 * 28, 2]))
        b = tf.Variable(tf.zeros([2]))
        y_hat = tf.nn.softmax(tf.matmul(X, W) + b)
        y = tf.placeholder(tf.float32, [None, 2])

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)

        image_generator = ImageGenerator()
        image_batch_train, label_batch_train, image_batch_test, label_batch_test = image_generator.flow_directory(
            'images/catdog'
        )

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for _ in range(500):
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

    def create_tensor_list(self):
        image_list_train = (
            glob.glob(TensorModel.TRAIN_FILES + TensorModel.CLASS[0] + '/*') +
            glob.glob(TensorModel.TRAIN_FILES + TensorModel.CLASS[1] + '/*')
        )

        image_list_test = (
            glob.glob(TensorModel.TEST_FILES + TensorModel.CLASS[0] + '/*') +
            glob.glob(TensorModel.TEST_FILES + TensorModel.CLASS[1] + '/*')
        )

        train_labels = np.array([[1, 0]] * 500 + [[0, 1]] * 500)
        test_labels = np.array([[1, 0]] * 200 + [[0, 1]] * 200)

        image_tensor_train = ops.convert_to_tensor(image_list_train, dtype=dtypes.string)
        label_tensor_train = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)

        image_tensor_test = ops.convert_to_tensor(image_list_test, dtype=dtypes.string)
        label_tensor_test = ops.convert_to_tensor(test_labels, dtype=dtypes.int32)

        return image_tensor_train, label_tensor_train, image_tensor_test, label_tensor_test

    @staticmethod
    def decode_images(input_queue):
        labels = input_queue[1]
        file_contents = tf.read_file(input_queue[0])
        images = tf.image.decode_jpeg(
            file_contents,
            channels=1
        )
        images = tf.image.resize_images(images, [128, 128])
        return images, labels

    def run(self):
        self.train_model(None, None)


if __name__ == '__main__':
    tensor = TensorModel()
    tensor.run()

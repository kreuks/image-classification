import glob

import numpy as np

from tensorflow.contrib.keras import preprocessing
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
from tensorflow.python.framework import ops, dtypes
import tensorflow as tf

from constant import Data, Image, BINARY, CATEGORICAL


class ImageGenerator(object):
    mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
    MEAN = np.mean(mnist.train.images)
    STD = np.std(mnist.train.images)

    def __init__(self, config):
        self._config = config
        self.resize = self._config[Image.IMAGE][Image.RESIZE]
        self.num_channel = self._config[Image.IMAGE][Image.NUM_CHANNEL]
        self.rescale = self._config[Image.IMAGE][Image.RESCALE]
        self.batch_size = self._config[Image.IMAGE][Image.BATCH_SIZE]

    def create_tensor_list(self, path_images):
        image_training_path = path_images + '/' + Data.TRAINING + '_' + Data.DATA + '/'
        image_test_path = path_images + '/' + Data.TESTING + '_' + Data.DATA + '/'

        classes = [x.replace(image_training_path, '') for x in glob.glob(image_training_path + '*')]

        image_list_train = []
        image_list_test = []
        label_list_train = []
        label_list_test = []

        for label, class_ in enumerate(classes):
            len_training_data = len(glob.glob(image_training_path + class_ + '/*'))
            len_testing_data = len(glob.glob(image_test_path + class_ + '/*'))
            if self._config[Image.IMAGE][Image.CLASS_MODE] == CATEGORICAL:
                categorical = [0] * len(classes)
                categorical[label] = 1
                label_list_train += [categorical] * len_training_data
                label_list_test += [categorical] * len_testing_data
            elif self._config[Image.IMAGE][Image.CLASS_MODE] == BINARY:
                label_list_train += [label] * len_training_data
                label_list_test += [label] * len_testing_data
            image_list_train += glob.glob(image_training_path + class_ + '/*')
            image_list_test += glob.glob(image_test_path + class_ + '/*')

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

    def resize_mnist_images(self, images):
        reshaped = (images - self.MEAN) / self.STD
        reshaped = np.reshape(reshaped, [-1, 28, 28, 1])

        return reshaped


class ImageGeneratorKeras:
    def __init__(self, config):
        self._config = config
        self.resize = self._config[Image.IMAGE][Image.RESIZE]
        self.num_channel = self._config[Image.IMAGE][Image.NUM_CHANNEL]
        self.rescale = self._config[Image.IMAGE][Image.RESCALE]
        self.batch_size = self._config[Image.IMAGE][Image.BATCH_SIZE]

    def load_train_data(self, path, classes=None):
        datagen = preprocessing.image.ImageDataGenerator(
            rescale=1. / self.rescale,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        datagenerator = datagen.flow_from_directory(
            path,
            target_size=tuple(self.resize),
            color_mode='rgb' if self.num_channel == 3 else 'grayscale',
            classes=classes,
            class_mode='categorical',
            batch_size=self.batch_size
        )
        return datagenerator

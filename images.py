from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.contrib.keras import preprocessing



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
            class_mode='binary',
            batch_size=32
        )
        return train_datagenerator

    def train_model(self, data_train, data_test):
        X = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 2]))
        b = tf.Variable(tf.zeros([2]))
        y_hat = tf.nn.softmax(tf.matmul(X, W) + b)
        y = tf.placeholder(tf.float32, [None, 2])
        sess = tf.Session()
        init = tf.global_variables_initializer()

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_hat))
        train_step = tf.train.GradientDescentOptimizer(.9).minimize(cross_entropy)
        with sess.as_default():
            sess.run(init)
            for index, (batch_xs, batch_ys) in enumerate(data_train):
                batch_xs = tf.reshape(batch_xs, [batch_xs.shape[0], -1]).eval()
                batch_ys = [[0., 1.] if x == 1. else [1., 0.] for x in batch_ys]
                if index % 100 == 0:
                    print 'cost_at_step {} :{}'.format(
                        index % 100, cross_entropy.eval(feed_dict={X: batch_xs, y: batch_ys}) / 100
                    )
                sess.run(train_step, feed_dict={X: batch_xs, y: batch_ys})
            correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            for data, labels in data_test:
                print sess.run(accuracy, feed_dict={X: data, y: labels})
            tf.summary.FileWriter('logs', sess.graph)

    def run(self):
        training_data = self.load_train_data(self.TRAIN_FILES, self.CLASS)
        test_data = self.load_train_data(self.TEST_FILES, self.CLASS)
        self.train_model(training_data, test_data)


if __name__ == '__main__':
    tensor = TensorModel()
    tensor.run()

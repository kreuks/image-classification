from config import config
from tensorflow.examples.tutorials.mnist import input_data as mnist_data
import tensorflow as tf
from model import Model
from images import ImageGenerator

from constant import Train


class Pipelines(object):
    def __init__(self, config=config):
        self._config = config
        self.num_epoch = self._config[Train.TRAIN][Train.NUM_EPOCH]
        self.mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
        self.model = getattr(Model, self._config[Train.TRAIN][Train.MODEL])

    def main(self):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputs')
        y_actual = tf.placeholder(tf.float32, shape=[None, 10], name='labels')
        is_training = tf.placeholder(tf.bool, name='is_training')
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

        eval_data = {
            x: ImageGenerator(config).resize_images(self.mnist.validation.images),
            y_actual: self.mnist.validation.labels,
            is_training: False
        }

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            sess.run(init)
            train_writer = tf.summary.FileWriter('logs/nielsen-net', sess.graph)

            for i in xrange(self.num_epoch):
                images, labels = self.mnist.train.next_batch(100)
                summary, _ = sess.run(
                    [loss_summary, train_step],
                    feed_dict={x: ImageGenerator(config).resize_images(images), y_actual: labels, is_training: True}
                )
                train_writer.add_summary(summary, i)

                if i % 100 == 0:
                    summary, acc = sess.run([accuracy_summary, accuracy], feed_dict=eval_data)
                    train_writer.add_summary(summary, i)
                    print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    pipeline = Pipelines()
    pipeline.main()
from config import config
import tensorflow as tf
from model import Model
from images import ImageGenerator

from constant import Train


class Pipelines(object):
    def __init__(self, config=config):
        self._config = config
        self.num_epoch = self._config[Train.TRAIN][Train.NUM_EPOCH]
        self.model = getattr(Model, self._config[Train.TRAIN][Train.MODEL])

    def main(self):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='inputs')
        y_actual = tf.placeholder(tf.float32, shape=[None, 2], name='labels')
        is_training = tf.placeholder(tf.bool, name='is_training')
        image_generator = ImageGenerator(config=config)
        image_batch_train, label_batch_train, image_batch_test, label_batch_test = image_generator.flow_directory(
            'images/catdog'
        )

        logits = self.model(
            x,
            num_classes=2,
            is_training=is_training,
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

            train_writer = tf.summary.FileWriter('logs/nielsen-net', sess.graph)

            for i in xrange(self.num_epoch):
                # images, labels = image_generator.mnist.train.next_batch(100)
                image_batch_train = image_batch_train.eval()
                # print image_batch_train.shape
                summary, _ = sess.run(
                    [loss_summary, train_step],
                    feed_dict={x: image_batch_train, y_actual: label_batch_train.eval(), is_training: True}
                )
                train_writer.add_summary(summary, i)

                if i % 100 == 0:
                    summary, acc = sess.run(
                        [accuracy_summary, accuracy],
                        feed_dict={
                            x: image_batch_test,
                            y_actual: label_batch_test.eval(),
                            is_training: False
                        }
                    )
                    train_writer.add_summary(summary, i)
                    print("Step: %5d, Validation Accuracy = %5.2f%%" % (i, acc * 100))

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    pipeline = Pipelines()
    pipeline.main()
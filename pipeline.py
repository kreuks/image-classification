from keras.models import load_model
from model import Model
from PIL import Image as pil_image
from tensorflow.contrib.keras.api.keras.preprocessing import image
import numpy as np
import tensorflow as tf

from config import config
from constant import Train, Image
from images import ImageGenerator, ImageGeneratorKeras
from util import MODEL_PATH


class Pipelines(object):
    def __init__(self, config=config):
        self._config = config
        self.num_epoch = self._config[Train.TRAIN][Train.NUM_EPOCH]
        self.model = getattr(Model, self._config[Train.TRAIN][Train.MODEL])

    def keras(self):
        image_generator = ImageGenerator(config=config)
        image_batch_train, label_batch_train, image_batch_test, label_batch_test = image_generator.flow_directory(
            'images/catdog'
        )
        test = ImageGeneratorKeras(self._config).load_train_data('images/catdog/testing_data')
        train = ImageGeneratorKeras(self._config).load_train_data('images/catdog/training_data')
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            model_ = self.model(input_shape=(244, 244, 3))
            model_.fit_generator(
                train,
                steps_per_epoch=self._config[Image.IMAGE][Image.BATCH_SIZE],
                epochs=self.num_epoch,
                validation_data=test,
                validation_steps=4
            )

            model_.save(MODEL_PATH)

            coord.request_stop()
            coord.join(threads)

    def keras_predict(self, image_path):
        if not image_path:
            return 'You must pass an image'

        img = pil_image.open(image_path)
        if img.size != tuple(self._config[Image.IMAGE][Image.RESIZE]):
            img = img.resize(tuple(self._config[Image.IMAGE][Image.RESIZE]))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            self.model = load_model(MODEL_PATH)
            prediction = self.model.predict_proba(x)
            return prediction

    def tensor_flow(self):
        x = tf.placeholder(tf.float32, shape=[None, 150, 150, 3], name='inputs')
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
                summary, _, accuracy_train = sess.run(
                    [loss_summary, train_step, accuracy],
                    feed_dict={x: image_batch_train.eval(), y_actual: label_batch_train.eval(), is_training: True}
                )
                train_writer.add_summary(summary, i)

                if i % 100 == 0:
                    summary, acc = sess.run(
                        [accuracy_summary, accuracy],
                        feed_dict={
                            x: image_batch_test.eval(),
                            y_actual: label_batch_test.eval(),
                            is_training: False
                        }
                    )
                    train_writer.add_summary(summary, i)
                    print 'Step: {}, Training Accuracy = {} - Validation Accuracy = {}'.format(
                        i, (accuracy_train * 100), (acc * 100)
                    )

            coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    pipeline = Pipelines()
    pipeline.keras()
    # print pipeline.keras_predict('cat.1.jpg')
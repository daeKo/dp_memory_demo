# Copyright 2019, The TensorFlow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training a language model (recurrent neural network) with DP-SGD optimizer.

This tutorial uses a corpus of text from TensorFlow datasets unless a
FLAGS.data_dir is specified with the path to a directory containing two files
train.txt and test.txt corresponding to a training and test corpus.

Even though we haven't done any hyperparameter tuning, and the analytical
epsilon upper bound can't offer any strong guarantees, the benefits of training
with differential privacy can be clearly seen by examining the trained model.
In particular, such inspection can confirm that the set of training-data
examples that the model fails to learn (i.e., has high perplexity for) comprises
outliers and rare sentences outside the distribution to be learned (see examples
and a discussion in this blog post). This can be further confirmed by
testing the differentially-private model's propensity for memorization, e.g.,
using the exposure metric of https://arxiv.org/abs/1802.08232.

This example is decribed in more details in this post: https://goo.gl/UKr7vH
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer

tf.flags.DEFINE_boolean('dpsgd', False, 'If True, train with DP-SGD. If False, '
                                        'train with vanilla SGD.')
tf.flags.DEFINE_float('learning_rate', .001, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 0.001,
                      'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.flags.DEFINE_integer('seed', 42, 'Seed used in random operations')
tf.flags.DEFINE_integer('epochs', 60, 'Number of epochs')
tf.flags.DEFINE_integer('microbatches', 256, 'Number of microbatches '
                                             '(must evenly divide batch_size)')
tf.flags.DEFINE_string('model_dir', None, 'Model directory')
tf.flags.DEFINE_string('data_dir', None, 'Directory containing the PTB data.')
tf.flags.DEFINE_string('secret_format', 't h e _ r a n d o m _ n u m b e r _ i s _ {} {} {} {} {} {} {} {} {}',
                       'Format of the secret injected in the data set.')
tf.flags.DEFINE_boolean('load_model', False, 'If True, load the latest checkpoint from model_dir. If False, '
                                             'train from scratch.')

FLAGS = tf.flags.FLAGS

SEQ_LEN = 20
NB_TRAIN = 45000


def rnn_model_fn(features, labels, mode):  # pylint: disable=unused-argument
    """Model function for a RNN."""

    # Define RNN architecture using tf.keras.layers.
    x = features['x']
    # x = tf.reshape(x, [-1, SEQ_LEN])
    input_layer = x[:]
    input_one_hot = tf.one_hot(input_layer, 200)
    lstm = tf.keras.layers.LSTM(200, return_sequences=True).apply(input_one_hot)
    lstm = tf.keras.layers.LSTM(200, return_sequences=True).apply(lstm)
    logits = tf.keras.layers.Dense(50).apply(lstm)

    if mode != tf.estimator.ModeKeys.PREDICT:
        # Calculate loss as a vector (to support microbatches in DP-SGD).
        vector_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.cast(tf.one_hot(labels, 50), dtype=tf.float32),  # x[:, 1:], 50), dtype=tf.float32),
            logits=logits)
        # Define mean of loss across minibatch (for reporting through tf.Estimator).
        scalar_loss = tf.reduce_mean(vector_loss)

    # Configure the training op (for TRAIN mode).
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.dpsgd:
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                l2_norm_clip=FLAGS.l2_norm_clip,
                noise_multiplier=FLAGS.noise_multiplier,
                num_microbatches=FLAGS.microbatches,
                learning_rate=FLAGS.learning_rate,
                unroll_microbatches=True,
                population_size=NB_TRAIN)
            opt_loss = vector_loss
        else:
            optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate)
            opt_loss = scalar_loss
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          train_op=train_op)

    # Add evaluation metrics (for EVAL mode).
    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy':
                tf.metrics.accuracy(
                    labels=tf.cast(labels, dtype=tf.int32),
                    predictions=tf.argmax(input=logits, axis=2))
        }
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          eval_metric_ops=eval_metric_ops)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=tf.nn.softmax(logits=logits))


def load_data(data_dir, secret_format, seed):
    """Load training and validation data."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    pickled_data_path = os.path.join(data_dir, 'corpus.{}.data'.format(hashlib.md5('{}{}'.format(secret_format, seed).encode()).hexdigest()))
    if os.path.isfile(pickled_data_path):
        dataset = pickle.load(open(pickled_data_path, 'rb'))
    else:
        # Set set for reproducibility
        if seed is not None:
            random.seed(seed)

        # Generate the secret
        secret_plain = secret_format.format(*(random.sample(range(0, 10), 9)))
        print('secret', secret_plain)

        # Create paths for later use
        train_file_path = os.path.join(data_dir, 'train.txt')
        test_file_path = os.path.join(data_dir, 'test.txt')
        train_file_path_secret_injected = os.path.join(data_dir, '{}_train.txt'.format(secret_plain)).replace(' ', '')

        # Insert secret in dataset
        with open(train_file_path, 'r') as f:
            contents = f.readlines()
            index = random.randint(0, len(contents))
            contents.insert(index, ' ' + secret_plain + ' \n')

        # Store dataset with injected secret in other file
        with open(train_file_path_secret_injected, 'w') as f:
            contents = ''.join(contents)
            f.write(contents)

        # Extract stuff for using dataset for training
        train_txt = open(train_file_path_secret_injected).read().split()
        test_txt = open(test_file_path).read().split()
        keys = sorted(set(train_txt))
        remap = {k: i for i, k in enumerate(keys)}
        train_data = np.array([remap[character] for character in train_txt], dtype=np.uint8)
        test_data = np.array([remap[character] for character in test_txt], dtype=np.uint8)
        secret_sequence = np.array([remap[character] for character in secret_plain.split()])
        dataset = {'train': train_data, 'test': test_data, 'num_classes': len(keys), 'dictionary': remap, 'seed': seed,
                   'secret_plain': secret_plain, 'secret_format': secret_format, 'secret_sequence': secret_sequence}
        pickle.dump(dataset, open(pickled_data_path, 'wb'))

    return dataset


def compute_epsilon(steps):
    """Computes epsilon value for given hyperparameters."""
    if FLAGS.noise_multiplier == 0.0:
        return float('inf')
    orders = [1 + x / 10. for x in range(1, 100)] + list(range(12, 64))
    sampling_probability = FLAGS.batch_size / NB_TRAIN
    rdp = compute_rdp(q=sampling_probability,
                      noise_multiplier=FLAGS.noise_multiplier,
                      steps=steps,
                      orders=orders)
    # Delta is set to 1e-5 because Penn TreeBank has 60000 training points.
    return get_privacy_spent(orders, rdp, target_delta=1e-5)[0]


def log_perplexity(estimator, sequence):
    assert 0 < len(sequence.shape) <= 2, "Length of the shape of the sequence has to be 1 or 2, currently it is {}".\
        format(len(sequence.shape))
    if len(sequence.shape) == 1:
        formatted_sequence = sequence.reshape((1, -1))
    else:
        formatted_sequence = sequence
    sequence_input = tf.estimator.inputs.numpy_input_fn(
        x={'x': formatted_sequence},
        batch_size=20,
        num_epochs=1,
        shuffle=False)
    sequence_length = formatted_sequence.shape[1]
    prediction_generator = estimator.predict(sequence_input)
    log_perplexity_list = []
    for i, prediction in enumerate(prediction_generator):
        # prediction = next(prediction_generator)
        sequence_probabilities = prediction[(range(sequence_length-1), formatted_sequence[i, 1:])]
        negative_log_probability = np.sum(-np.log(sequence_probabilities))
        log_perplexity_list.append(negative_log_probability)
    return log_perplexity_list


def estimate_z_score(estimator, secret, secret_format, dictionary, seed=42, sample_size=1000):
    secret_log_perplexity = log_perplexity(estimator=estimator, sequence=secret)
    np.random.seed(seed=seed)
    samples_of_random_space = np.random.randint(0, 10, (sample_size, 9))
    list_of_samples = []
    for i in range(sample_size):
        sample = secret_format.format(*samples_of_random_space[i]).split()
        int_representation = [dictionary[character] for character in sample]
        list_of_samples.append(int_representation)
    sample_log_perplexity_list = log_perplexity(estimator, np.array(list_of_samples))
    mean = np.mean(sample_log_perplexity_list)
    std = np.std(sample_log_perplexity_list)
    z_score = (secret_log_perplexity- mean)/std
    return z_score


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)
    if FLAGS.batch_size % FLAGS.microbatches != 0:
        raise ValueError('Number of microbatches should divide evenly batch_size')

    # Load training and test data.
    dataset = load_data(data_dir=FLAGS.data_dir, secret_format=FLAGS.secret_format, seed=FLAGS.seed)

    train_data = dataset['train']
    test_data = dataset['test']

    secret_sequence = dataset['secret_sequence']

    warm_start_from = {'warm_start_from': FLAGS.model_dir} if FLAGS.load_model else {}

    # Create tf.Estimator input functions for the training and test data.
    batch_len = FLAGS.batch_size * SEQ_LEN
    # Calculate remainders
    remainder_train = len(train_data) % batch_len
    remainder_test = len(test_data) % batch_len
    # In case batch_len divides the number of characters in the dataset, the wouldn't have labels for the last entry
    if remainder_train != 0:
        train_data_end = len(train_data) - remainder_train
    else:
        train_data_end = len(train_data) - batch_len
    train_label_end = train_data_end + 1
    # Set the umber of training data accordingly, calling the estimator beforehand might cause problems
    global NB_TRAIN
    NB_TRAIN = train_data_end
    # Same for the test data
    if remainder_test != 0:
        test_data_end = len(test_data) - remainder_test
    else:
        test_data_end = len(test_data) - batch_len
    test_label_end = test_data_end + 1
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data[:train_data_end].reshape((-1, SEQ_LEN))},
        y=train_data[1:train_label_end].reshape((-1, SEQ_LEN)),
        batch_size=batch_len,
        num_epochs=FLAGS.epochs,
        shuffle=False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data[:test_data_end].reshape((-1, SEQ_LEN))},
        y=test_data[1:test_label_end].reshape((-1, SEQ_LEN)),
        batch_size=batch_len,
        num_epochs=1,
        shuffle=False)

    # Instantiate the tf.Estimator.
    conf = tf.estimator.RunConfig(save_summary_steps=1000)
    lm_classifier = tf.estimator.Estimator(model_fn=rnn_model_fn,
                                           model_dir=FLAGS.model_dir,
                                           config=conf,
                                           **warm_start_from)

    z_score_list = []

    # z_score = estimate_z_score(estimator=lm_classifier,
    #                            secret=secret_sequence,
    #                            secret_format=dataset['secret_format'],
    #                            dictionary=dataset['dictionary'],
    #                            seed=FLAGS.seed + 1,
    #                            sample_size=1000)
    # Training loop.
    steps_per_epoch = len(train_data) // batch_len
    for epoch in range(1, FLAGS.epochs + 1):
        print('epoch', epoch)
        # Train the model for one epoch.
        lm_classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)

        if epoch % 5 == 0:
            name_input_fn = [('Train', train_input_fn), ('Eval', eval_input_fn)]
            for name, input_fn in name_input_fn:
                # Evaluate the model and print results
                eval_results = lm_classifier.evaluate(input_fn=input_fn)
                result_tuple = (epoch, eval_results['accuracy'], eval_results['loss'])
                print(name, 'accuracy after %d epochs is: %.3f (%.4f)' % result_tuple)

        # Compute the privacy budget expended so far.
        if FLAGS.dpsgd:
            eps = compute_epsilon(epoch * steps_per_epoch)
            print('For delta=1e-5, the current epsilon is: %.2f' % eps)
        else:
            print('Trained with vanilla non-private SGD optimizer')

        z_score = estimate_z_score(estimator=lm_classifier,
                                   secret=secret_sequence,
                                   secret_format=dataset['secret_format'],
                                   dictionary=dataset['dictionary'],
                                   seed=FLAGS.seed + 1,
                                   sample_size=1000)
        z_score_list.append(z_score)
        print("z-score: {}".format(z_score))

    x = range(1, FLAGS.epochs + 1)
    plt.plot(x, z_score_list, label='z-score')
    plt.xlabel('Epoch')
    plt.ylabel('z-score')
    plt.legend()
    plt.title('Secret: {}'.format(dataset['secret_plain'].replace(' ', '').replace('_', ' ')))
    plt.savefig("z_score_{}.png".format(dataset['secret_format']).replace(' ', ''))
    plt.close()


if __name__ == '__main__':
    tf.app.run()

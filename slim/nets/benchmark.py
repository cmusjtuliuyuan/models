# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Timing benchmark for Inception_V1 inference.

To run, use:
  bazel run -c opt --config=cuda \
      models/tutorials/image/alexnet:alexnet_benchmark

Across 100 steps on batch size = 128.

Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch

Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import optparse
from datetime import datetime
import math
import sys
import time
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
sys.path.append(".")
opts=None

def placeholder_inputs():
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code.
  """
  batch_size = opts.batch_size
  image_size = opts.image_size
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                        image_size,
                                                        image_size,
                                                        3))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder


def loss(logits, labels):
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(
        concated, tf.stack([batch_size, 1000]), 1.0, 0.0)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=onehot_labels, name='xentropy')
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
    return loss

def time_tensorflow_run(session, target, images_placeholder, labels_placeholder, info_string):
  """Run the computation to obtain the target tensor and print timing stats.

  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.

  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  batch_size = opts.batch_size
  image_size = opts.image_size
  for i in xrange(opts.num_batches + num_steps_burn_in):
    # Feed dictionaray
    feed_dict = {
        images_placeholder : np.random.randn(batch_size,image_size,image_size,3)*1e-1,
        labels_placeholder : np.ones(batch_size),
    }

    start_time = time.time()
    _ = session.run(target, feed_dict=feed_dict)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print ('%s: step %d, duration = %.3f' %
               (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / opts.num_batches
  vr = total_duration_squared / opts.num_batches - mn * mn
  sd = math.sqrt(vr)
  print ('%s across %d steps, %.3f +/- %.3f sec / batch. Throughput is %.3f sample / sec.' %
         (info_string, opts.num_batches, mn, sd, opts.num_batches*opts.batch_size/total_duration))

def run_benchmark():
  """Run the benchmark on Inception_V1."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = opts.image_size
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images_placeholder, labels_placeholder = placeholder_inputs()
    # Build a Graph that computes the logits predictions from the
    # inference model.
    if not ResNet:
        logits, _ = inference(images_placeholder)
    else:
        logits, _ = inference(images_placeholder, num_classes = 1000)

    # Get loss function
    objective = loss(logits, labels_placeholder)

    # Get train option
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(objective)

    # Build an initialization operation.
    init = tf.global_variables_initializer()

    # config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session()
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, logits, images_placeholder, labels_placeholder,"Forward")

    # Run the backward benchmark.
    time_tensorflow_run(sess, train_op, images_placeholder, labels_placeholder, "Forward-backward")

ResNet = False

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option(
        "--batch_size", default=32, type='int',
        help="batch_size"
    )
    optparser.add_option(
        "--num_batches", default=100, type='int',
         help="number of batches to run"
    )
    optparser.add_option(
            "--image_size", default=224, type='int',
         help="size of image defaut is 224"
    )
    optparser.add_option(
        "--learning_rate", default=0.001, type='float',
    )
    optparser.add_option(
        "--model_type", default='inception_v1',
    )
    opts = optparser.parse_args()[0]
    if opts.model_type=='inception_v1':
        from nets.inception_v1 import inception_v1 as inference
    elif opts.model_type=='inception_v2':
        from nets.inception_v2 import inception_v2 as inference
    elif opts.model_type=='inception_v3':
        from nets.inception_v3 import inception_v3 as inference
    elif opts.model_type=='alexnet_v2':
        from nets.alexnet import alexnet_v2 as inference
    elif opts.model_type=='resnet_v1_50':
        from nets.resnet_v1 import resnet_v1_50 as inference
        ResNet = True
    elif opts.model_type=='resnet_v1_101':
        from nets.resnet_v1 import resnet_v1_101 as inference
        ResNet = True
    elif opts.model_type=='resnet_v1_152':
        from nets.resnet_v1 import resnet_v1_152 as inference
        ResNet = True
    elif opts.model_type=='resnet_v1_200':
        from nets.resnet_v1 import resnet_v1_200 as inference
        ResNet = True
    elif opts.model_type=='resnet_v2_50':
        from nets.resnet_v2 import resnet_v2_50 as inference
        ResNet = True
    elif opts.model_type=='resnet_v2_101':
        from nets.resnet_v2 import resnet_v2_101 as inference
        ResNet = True
    elif opts.model_type=='resnet_v2_152':
        from nets.resnet_v2 import resnet_v2_152 as inference
        ResNet = True
    elif opts.model_type=='resnet_v2_200':
        from nets.resnet_v2 import resnet_v2_200 as inference
        ResNet = True
    elif opts.model_type=='vgg_16':
        from nets.vgg import vgg_16 as inference
    elif opts.model_type=='vgg_19':
        from nets.vgg import vgg_19 as inference
    elif opts.model_type=='alexnet_v1':
	from nets.alexnet_v1 import alexnet_v1 as inference
    elif opts.model_type=='alexnet_benchmark':
        from nets.alexnet_benchmark import inference as inference
    print("Model:"+opts.model_type+". Batch size is: "+str(opts.batch_size))
    run_benchmark()


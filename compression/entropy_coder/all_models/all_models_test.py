# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

"""Basic test of all registered models."""

import tensorflow as tf

# pylint: disable=unused-import
import all_models
# pylint: enable=unused-import
from entropy_coder.model import model_factory


class AllModelsTest(tf.test.TestCase):

  def testBuildModelForTraining(self):
    factory = model_factory.GetModelRegistry()
    model_names = factory.GetAvailableModels()

    batch_size = 3
    height = 40
    width = 20
    depth = 5
    for m in model_names:
      tf.reset_default_graph()

      global_step = tf.Variable(tf.zeros([], dtype=tf.int64),
                                trainable=False,
                                name='global_step')

      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

      binary_codes = tf.placeholder(dtype=tf.float32,
                                    shape=[batch_size, height, width, depth])

      # Create a model with the default configuration.
      print(f'Creating model: {m}')
      model = factory.CreateModel(m)
      model.Initialize(global_step,
                       optimizer,
                       model.GetConfigStringForUnitTest())
      self.assertTrue(model.loss is None, f'model: {m}')
      self.assertTrue(model.train_op is None, f'model: {m}')
      self.assertTrue(model.average_code_length is None, f'model: {m}')

      # Build the Tensorflow graph corresponding to the model.
      model.BuildGraph(binary_codes)
      self.assertTrue(model.loss is not None, f'model: {m}')
      self.assertTrue(model.average_code_length is not None, f'model: {m}')
      if model.train_op is None:
        print(f'Model {m} is not trainable')


if __name__ == '__main__':
  tf.test.main()

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Detection Engine used for detection tasks."""
from collections import Counter
from collections import defaultdict 
from edgetpu.basic.basic_engine import BasicEngine
import numpy as np
from PIL import Image

class EmbeddingEngine(BasicEngine):
  """Engine used to obtain embeddings from headless mobilenets."""

  def __init__(self, model_path):
    """Creates a EmbeddingEngine with given model and labels.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    BasicEngine.__init__(self, model_path)
    output_tensors_sizes = self.get_all_output_tensors_sizes()
    if output_tensors_sizes.size != 1:
      raise ValueError(
          ('Dectection model should have only 1 output tensor!'
           'This model has {}.'.format(output_tensors_sizes.size)))

  def DetectWithImage(self, img):
    """Calculates embedding from an image.

    Args:
      img: PIL image object.

    Returns:
      Embedding vector as np.float32

    Raises:
      RuntimeError: when model's input tensor format is invalid.
    """
    input_tensor_shape = self.get_input_tensor_shape()
    if (input_tensor_shape.size != 4 or input_tensor_shape[3] != 3 or
        input_tensor_shape[0] != 1):
      raise RuntimeError(
          'Invalid input tensor shape! Expected: [1, height, width, 3]')
    required_image_size = (input_tensor_shape[2], input_tensor_shape[1])
    with img.resize(required_image_size, Image.NEAREST) as resized_img:
      input_tensor = np.asarray(resized_img).flatten()
      return self.run_inference(input_tensor)[1]


class kNNEmbeddingEngine(EmbeddingEngine):
  """Extends embedding engine to also provide kNearest Neighbor detection.

     This class maintains an in-memory store of embeddings and provides
     functions to find k nearest neighbors against a query emedding.
  """
    
  def __init__(self, model_path, kNN=3):
    """Creates a EmbeddingEngine with given model and labels.

    Args:
      model_path: String, path to TF-Lite Flatbuffer file.

    Raises:
      ValueError: An error occurred when model output is invalid.
    """
    EmbeddingEngine.__init__(self, model_path)
    self.clear()
    self._kNN = kNN

  def clear(self):
    """Clear the store: forgets all stored embeddings."""
    self._labels = []
    self._embedding_map = defaultdict(list)
    self._embeddings = None


  def addEmbedding(self, emb, label):
    """Add an embedding vector to the store."""

    normal = emb/np.sqrt((emb**2).sum()) # Normalize the vector

    self._embedding_map[label].append(normal) # Add to store, under "label"

    # Expand labelled blocks of embeddings for when we have less than kNN
    # examples. Otherwise blocks that have more examples unfairly win.
    emb_blocks = []
    self._labels = [] # We'll be reconstructing the list of labels
    for label, embeds in self._embedding_map.items():
      emb_block = np.stack(embeds)
      if emb_block.shape[0] < self._kNN:
          emb_block = np.pad(emb_block,
                             [(0,self._kNN - emb_block.shape[0]), (0,0)],
                             mode="reflect")
      emb_blocks.append(emb_block)
      self._labels.extend([label]*emb_block.shape[0])

    self._embeddings = np.concatenate(emb_blocks, axis=0)

  def kNNEmbedding(self, query_emb):
    """Returns the self._kNN nearest neighbors to a query embedding."""

    # If we have nothing stored, the answer is None
    if self._embeddings is None: return None

    # Normalize query embedding
    query_emb = query_emb/np.sqrt((query_emb**2).sum())

    # We want a cosine distance ifrom query to each stored embedding. A matrix
    # multiplication can do this in one step, resulting in a vector of
    # distances.
    dists = np.matmul(self._embeddings, query_emb)

    # If we have less than self._kNN distances we can only return that many.
    kNN = min(len(dists), self._kNN)

    # Get the N largest cosine similarities (larger means closer).
    n_argmax = np.argpartition(dists, -kNN)[-kNN:]

    # Get the corresponding labels associated with each distance.
    labels = [self._labels[i] for i in n_argmax]

    # Return the most common label over all self._kNN nearest neighbors.
    most_common_label = Counter(labels).most_common(1)[0][0]
    return most_common_label

  def exampleCount(self):
    """Just returns the size of the embedding store."""
    return sum(len(v) for v in self._embedding_map.values())
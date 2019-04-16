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
    """Detects object with given PIL image object.

    This interface assumes the loaded model is trained for object detection.

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
          'Invalid input tensor shape! Expected: [1, width, height, 3]')
    required_image_size = (input_tensor_shape[1], input_tensor_shape[2])
    with img.resize(required_image_size, Image.NEAREST) as resized_img:
      input_tensor = np.asarray(resized_img).flatten()
      return self.RunInference(input_tensor)[1]


class kNNEmbeddingEngine(EmbeddingEngine):
  """Extends embedding engine to also provide kNearest Neighbor detection."""
    
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
    self._labels = []
    self._embedding_map = defaultdict(list) 
    self._embeddings = None 

  def addEmbedding(self, emb, label):
    normal = emb/np.sqrt((emb**2).sum())
    
    self._embedding_map[label].append(normal)

    emb_blocks = []
    self._labels = []
    # Expand labelled blocks of embeddings for when we have less than kNN examples.
    # Otherwise blocks that hav more examples unfairly win.
    for label, embeds in self._embedding_map.items():
      emb_block = np.stack(embeds)
      if emb_block.shape[0] < self._kNN: 
          emb_block = np.pad(emb_block, [(0,self._kNN - emb_block.shape[0]), (0,0)], mode="reflect")
      emb_blocks.append(emb_block)
      self._labels.extend([label]*emb_block.shape[0])

    self._embeddings = np.concatenate(emb_blocks, axis=0)

  def kNNEmbedding(self, query_emb):
    if self._embeddings is None: return None   
    query_emb = query_emb/np.sqrt((query_emb**2).sum())

    dists = np.matmul(self._embeddings, query_emb,)
    argmax = np.argmax(dists)
    kNN = min(len(dists), self._kNN)
    n_argmax = np.argpartition(dists, -kNN)[-kNN:]
    labels = [self._labels[i] for i in n_argmax]
    most_common_label = Counter(labels).most_common(1)[0][0]
    return most_common_label 

  def exampleCount(self):
    return sum(len(v) for v in self._embedding_map.values())

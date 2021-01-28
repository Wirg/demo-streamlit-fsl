from typing import Callable, Optional, Tuple, List, Union

import numpy as np
import tensorflow as tf
from keras_fsl.layers import Classification

from tensorflow_utils import cosine_similary

IMAGE_INPUTS = Union[
    tf.data.Dataset, List[tf.Tensor], List[np.array], np.array, tf.Tensor
]


class FSLClassifier:
    def __init__(
        self,
        preprocess_reshaped_image: Callable[[tf.Tensor], tf.Tensor],
        encoder: tf.keras.Model,
        head: tf.keras.layers.Layer,
    ):
        super().__init__()
        self.preprocess_reshaped_image = preprocess_reshaped_image
        self.encoder = encoder
        self.head = head
        self._model: Optional[tf.keras.Model] = None

    @property
    def input_shape(self) -> Tuple[int, int, int, int]:
        return self.encoder.input_shape

    def preprocessing(self, input_tensor: tf.Tensor) -> tf.Tensor:
        output_tensor = input_tensor
        output_tensor = tf.image.resize_with_pad(
            output_tensor,
            self.input_shape[1],
            self.input_shape[2],
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        )
        output_tensor = tf.cast(output_tensor, tf.float32)
        output_tensor = self.preprocess_reshaped_image(output_tensor)
        return output_tensor

    def build_model(
        self, catalog_embeddings: tf.Tensor, catalog_labels: List[str]
    ) -> tf.keras.Model:
        n_catalog_images = len(catalog_labels)
        classifier = Classification(kernel=self.head)
        classifier.support_set_loss = tf.Variable(
            tf.zeros((n_catalog_images, n_catalog_images), dtype=tf.float32),
            name="support_set_loss",
        )
        classifier.set_support_set(
            support_tensors=tf.constant(catalog_embeddings),
            support_labels_name=tf.constant(catalog_labels),
            overwrite=tf.constant(True),
        )
        return tf.keras.Sequential([self.encoder, classifier])

    @property
    def model(self) -> tf.keras.Model:
        if self._model is None:
            raise ValueError("Support set was not initialized (empty)")
        return self._model

    def _to_dataset(self, images, batch_size: int = 4) -> tf.data.Dataset:
        if not isinstance(images, tf.data.Dataset):
            images = tf.data.Dataset.from_tensor_slices(images)
        return images.map(self.preprocessing).batch(batch_size)

    def predict_embeddings(self, images, batch_size: int = 4):
        return self.encoder.predict(self._to_dataset(images, batch_size))

    def set_catalog(self, images, labels: List[str], batch_size: int = 4):
        self._model = self.build_model(
            self.predict_embeddings(images, batch_size), labels
        )

    def predict(self, images, batch_size: int = 4) -> tf.Tensor:
        return self.model.predict(self._to_dataset(images, batch_size))

    @classmethod
    def example(cls):
        from tensorflow.python.keras.applications import mobilenet_v2

        fsl_classifier = cls(
            preprocess_reshaped_image=mobilenet_v2.preprocess_input,
            encoder=mobilenet_v2.MobileNetV2(
                input_shape=[96, 96, 3], include_top=False, pooling="avg"
            ),
            head=tf.keras.layers.Lambda(lambda x: cosine_similary(x[0], x[1])),
        )
        return fsl_classifier

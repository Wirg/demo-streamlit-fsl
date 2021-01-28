from typing import List

import tensorflow as tf


def cosine_similary(x1, x2, axis=-1):
    x1 = tf.math.l2_normalize(x1, axis=axis)
    x2 = tf.math.l2_normalize(x2, axis=axis)
    return (1 + tf.math.reduce_sum(x1 * x2, axis=axis)) / 2


def image_dataset_from_paths(image_paths: List[str]) -> tf.data.Dataset:
    return (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(lambda p: tf.image.decode_jpeg(tf.io.read_file(p)))
    )
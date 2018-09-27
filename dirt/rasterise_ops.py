import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

_lib_path = os.path.dirname(__file__)
_rasterise_module = tf.load_op_library(_lib_path + '/librasterise.so')


def rasterise(background, vertices, vertex_colors, faces, height=None, width=None, channels=None, name=None):
    """Rasterises the given `vertices` and `faces` over `background`.

    This function takes a set of vertices, vertex colors, faces (vertex indices), and a background.
    It returns a single image, containing the faces these arrays, over the given background.

    It supports single-channel (grayscale) or three-channel (RGB) rendering.

    The vertices are specified in OpenGL's clip space, and as such are 4D homogeneous coordinates.
    This allows both 3D and 2D shapes to be rendered, by applying suitable projection matrices to the
    vertices before passing them to this function.

    Args:
        background: a float32 `Tensor` of shape [height, width, channels], defining the background image to render over
        vertices: a float32 `Tensor` of shape [vertex count, 4] defining a set of vertex locations, given in clip space
        vertex_colors: a float32 `Tensor` of shape [vertex count, channels] defining the color of each vertex; these are
            linearly interpolated in 3D space to calculate the color at each pixel
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle to rasterise. Note that each vertex may be used by several faces
        height: a python `int` specifying the frame height; may be `None` if `background` has static shape
        width: a python `int` specifying the frame width; may be `None` if `background` has static shape
        channels: a python `int` specifying the number of color channels; may only be `1`or `3`. Again this may be `None`
            if `background` has static shape
        name: an optional name for the operation

    Returns:
        The rendered pixels, as a float32 `Tensor` of shape [height, width, channels]
    """

    with ops.name_scope(name, 'Rasterise', [background, vertices, vertex_colors, faces]) as scope:
        background = tf.convert_to_tensor(background, name='background', dtype=tf.float32)
        vertices = tf.convert_to_tensor(vertices, name='vertices', dtype=tf.float32)
        vertex_colors = tf.convert_to_tensor(vertex_colors, name='vertex_colors', dtype=tf.float32)
        faces = tf.convert_to_tensor(faces, name='faces', dtype=tf.int32)
        if height is None:
            height = int(background.get_shape()[0])
        if width is None:
            width = int(background.get_shape()[1])
        if channels is None:
            channels = int(background.get_shape()[2])
        return _rasterise_module.rasterise(
            background[np.newaxis, ...], vertices[np.newaxis, ...], vertex_colors[np.newaxis, ...], faces[np.newaxis, ...],  # inputs
            height, width, channels,  # attributes
            name=scope
        )[0]


def rasterise_batch(background, vertices, vertex_colors, faces, height=None, width=None, channels=None, name=None):
    """Rasterises a batch of meshes with the same numbers of vertices and faces.

    This function takes batch-indexed `vertices`, `vertex_colors`, `faces`, and `background`.

    It is conceptually equivalent to:
    ```python
    tf.stack([
        rasterise(background_i, vertices_i, vertex_colors_i, faces_i)
        for (background_i, vertices_i, vertex_colors_i, faces_i)
        in zip(background, vertices, vertex_colors, faces)
    ])
    ```
    See `rasterise` for definitions of the parameters, noting that for `rasterise_batch`, a leading dimension should be included.
    """

    with ops.name_scope(name, 'RasteriseBatch', [background, vertices, vertex_colors, faces]) as scope:
        background = tf.convert_to_tensor(background, name='background', dtype=tf.float32)
        vertices = tf.convert_to_tensor(vertices, name='vertices', dtype=tf.float32)
        vertex_colors = tf.convert_to_tensor(vertex_colors, name='vertex_colors', dtype=tf.float32)
        faces = tf.convert_to_tensor(faces, name='faces', dtype=tf.int32)
        if height is None:
            height = int(background.get_shape()[1])
        if width is None:
            width = int(background.get_shape()[2])
        if channels is None:
            channels = int(background.get_shape()[3])
        return _rasterise_module.rasterise(
            background, vertices, vertex_colors, faces,  # inputs
            height, width, channels,  # attributes
            name=scope
        )


@ops.RegisterGradient('Rasterise')
def _rasterise_grad(op, grad_pixels, name=None):
    grad_op_result = _rasterise_module.rasterise_grad(
        op.inputs[1], op.inputs[3],
        op.outputs[0], grad_pixels,
        op.get_attr('height'), op.get_attr('width'), op.get_attr('channels'),
        name=name
    )
    # tf.summary.image(
    #     'debug_thingy',
    #     (grad_op_result.debug_thingy - tf.reduce_min(grad_op_result.debug_thingy, axis=[1, 2], keep_dims=True)) /
    #         (tf.reduce_max(grad_op_result.debug_thingy, axis=[1, 2], keep_dims=True) - tf.reduce_min(grad_op_result.debug_thingy, axis=[1, 2], keep_dims=True))
    # )
    return [
        grad_op_result.grad_background,
        grad_op_result.grad_vertices,
        grad_op_result.grad_vertex_colors,
        None  # wrt faces
    ]


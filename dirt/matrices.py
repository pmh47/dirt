
"""Helper functions for homogeneous transform matrices.

This module defines helper functions used to construct transform matrices.
These functions assume the matrices will *right*-multiply the vectors to be transformed, i.e.
that the inputs are row vectors -- as is the case for a matrix of vertices indexed naturally
Equivalently, matrices are indexed by *, x/y/z[/w] (in), x/y/z[/w] (out) -- where * represents any
sequence of indices, over all of which the operation is mapped
"""

import tensorflow as tf
from tensorflow.python.framework import ops


def rodrigues(vectors, name=None, three_by_three=False):
    """Constructs a batch of angle-axis rotation matrices.

    Angle-axis rotations are defined by a single 3D vector, whose direction corresponds to the axis of
    rotation, and whose length corresponds to the rotation angle in radians.
    This function returns  a batch of angle-axis rotation matrices, computed according to Rodrigues'
    formula, from a corresponding batch of 3D rotation vectors.

    Args:
        vectors: a `Tensor` of shape [*, 3], where * represents arbitrarily many leading (batch) dimensions
        name: an optional name for the operation
        three_by_three: return 3x3 matrices without w coordinates

    Returns:
        a `Tensor` containing rotation matrices, of shape [*, D, D], where * represents the same leading
        dimensions as present on `vectors`, and D = 3 if three_by_three else 4
    """

    # vectors is indexed by *, x/y/z, so the result is indexed by *, x/y/z (in), x/y/z (out)
    # This follows the OpenCV docs' definition; wikipedia says slightly different...

    with ops.name_scope(name, 'Rodrigues', [vectors]) as scope:

        vectors = tf.convert_to_tensor(vectors, name='vectors')

        vectors += 1.e-12  # for numerical stability of the derivative, which is otherwise NaN at exactly zero; also ensures norms are never zero
        norms = tf.norm(vectors, axis=-1, keep_dims=True)  # indexed by *, singleton
        vectors /= norms
        norms = norms[..., 0]  # indexed by *

        z = tf.zeros_like(vectors[..., 0])  # ditto
        K = tf.convert_to_tensor([
            [z, -vectors[..., 2], vectors[..., 1]],
            [vectors[..., 2], z, -vectors[..., 0]],
            [-vectors[..., 1], vectors[..., 0], z],
        ])  # indexed by x/y/z (in), x/y/z (out), *
        K = tf.transpose(K, list(range(2, K.get_shape().ndims)) + [0, 1])  # indexed by *, x/y/z (in), x/y/z (out)

        c = tf.cos(norms)[..., tf.newaxis, tf.newaxis]
        s = tf.sin(norms)[..., tf.newaxis, tf.newaxis]

        result_3x3 = c * tf.eye(3, 3) + (1 - c) * vectors[..., :, tf.newaxis] * vectors[..., tf.newaxis, :] + s * K

        if three_by_three:
            return result_3x3
        else:
            return pad_3x3_to_4x4(result_3x3)


def translation(x, name=None):
    """Constructs a batch of translation matrices.

    This function returns a batch of translation matrices, from a corresponding batch of 3D displacement vectors.

    Args:
        x: a `Tensor` of shape [*, 3], where * represents arbitrarily many leading (batch) dimensions
        name: an optional name for the operation

    Returns:
        a `Tensor` containing translation matrices, of shape [*, 4, 4], where * represents the same leading
        dimensions as present on `x`
    """

    # x is indexed by *, x/y/z
    with ops.name_scope(name, 'Translation', []) as scope:
        x = tf.convert_to_tensor(x, name='x')
        zeros = tf.zeros_like(x[..., 0])  # indexed by *
        ones = tf.ones_like(zeros)
        return tf.stack([
            tf.stack([ones, zeros, zeros, zeros], axis=-1),  # indexed by *, x/y/z (out)
            tf.stack([zeros, ones, zeros, zeros], axis=-1),
            tf.stack([zeros, zeros, ones, zeros], axis=-1),
            tf.stack([x[..., 0], x[..., 1], x[..., 2], ones], axis=-1)
        ], axis=-2)  # indexed by *, x/y/z/w (in), x/y/z/w (out)


def scale(x, name=None):
    """Constructs a batch of scaling matrices.

    This function returns a batch of scaling matrices, from a corresponding batch of 3D scale factors.

    Args:
        x: a `Tensor` of shape [*, 3], where * represents arbitrarily many leading (batch) dimensions
        name: an optional name for the operation

    Returns:
        a `Tensor` containing scaling matrices, of shape [*, 4, 4], where * represents the same leading
        dimensions as present on `x`
    """

    with ops.name_scope(name, 'Scale', []) as scope:
        x = tf.convert_to_tensor(x, name='x')
        return tf.linalg.diag(tf.concat([x, tf.ones_like(x[..., :1])], axis=-1))  # indexed by *, x/y/z/w (in), x/y/z/w (out)


def perspective_projection(near, far, right, aspect, name=None):
    """Constructs a perspective projection matrix.

    This function returns a perspective projection matrix, using the OpenGL convention that the camera
    looks along the negative-z axis in view/camera space, and the positive-z axis in clip space.
    Multiplying view-space homogeneous coordinates by this matrix maps them into clip space.

    Args:
        near: distance to the near clipping plane; geometry nearer to the camera than this will not be rendered
        far: distance to the far clipping plane; geometry further from the camera than this will not be rendered
        right: distance of the right-hand edge of the view frustum from its centre at the near clipping plane
        aspect: aspect ratio (height / width) of the viewport
        name: an optional name for the operation

    Returns:
        a 4x4 `Tensor` containing the projection matrix
    """

    with ops.name_scope(name, 'PerspectiveProjection', [near, far, right, aspect]) as scope:
        near = tf.convert_to_tensor(near, name='near')
        far = tf.convert_to_tensor(far, name='far')
        right = tf.convert_to_tensor(right, name='right')
        aspect = tf.convert_to_tensor(aspect, name='aspect')
        top = right * aspect
        elements = [
            [near / right, 0., 0., 0, ],
            [0., near / top, 0., 0.],
            [0., 0., -(far + near) / (far - near), -2. * far * near / (far - near)],
            [0., 0., -1., 0.]
        ]  # indexed by x/y/z/w (out), x/y/z/w (in)
        return tf.transpose(tf.convert_to_tensor(elements, dtype=tf.float32))


def pad_3x3_to_4x4(matrix, name=None):
    """Pads a 3D transform matrix to a 4D homogeneous transform matrix.

    This function converts a batch of 3x3 transform matrices into 4x4 equivalents that operate on
    homogeneous coordinates.
    To do so, for each matrix in the batch, it appends a column of zeros, a row of zeros, and a single
    one at the bottom-right corner.

    Args:
        matrix: a `Tensor` of shape [*, 3, 3], where * represents arbitrarily many leading (batch) dimensions
        name: an optional name for the operation

    Returns:
        a `Tensor` of shape [*, 4, 4] containing the padded matrices, where * represents the same leading
        dimensions as present on `matrix`
    """

    # matrix is indexed by *, x/y/z (in), x/y/z (out)
    # result is indexed by *, x/y/z/w (in), x/y/z/w (out)
    with ops.name_scope(name, 'Pad3x3To4x4', [matrix]) as scope:
        matrix = tf.convert_to_tensor(matrix, name='matrix')
        return tf.concat([
            tf.concat([matrix, tf.zeros_like(matrix[..., :, :1])], axis=-1),
            tf.concat([tf.zeros_like(matrix[..., :1, :]), tf.ones_like(matrix[..., :1, :1])], axis=-1)
        ], axis=-2)


def compose(*matrices):
    """Composes together a sequence of matrix transformations.

    This is a convenience function to multiply together a sequence of transform matrices; if `matrices`
    is empty , it will return a single 4x4 identity matrix.
    The evaluation order is such that the first matrix in the list is the first to be applied.

    Args:
        matrices: a list with elements of type `Tensor` and shape [*, 4, 4], where * represents arbitrarily many
        leading (batch) dimensions

    Returns:
        a `Tensor` of shape [*, 4, 4] containing the product of the given matrices, evaluated in the same order as the
        matrices appear in the list
    """

    # This applies the first matrix in the list first, i.e. compose(A, B) is 'A then B'; as our convention is that the matrix
    # always right-multiplies vectors, this just expands to a sequence of tf.matmul's in the same order as the input

    if len(matrices) == 0:
        return tf.eye(4)
    elif len(matrices) == 1:  # special-cased to avoid an identity-multiply
        return matrices[0]
    else:
        return tf.matmul(matrices[0], compose(*matrices[1:]))


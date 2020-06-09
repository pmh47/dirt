
import tensorflow as tf
from tensorflow.python.framework import ops


def _pixel_to_ndc(pixel_locations, image_size):
    return (-1. + 2. * pixel_locations / image_size) * [1., -1.]


def _unproject_ndc_to_world(x_ndc, clip_to_world_matrix):
    # x_ndc and result are indexed by *, x/y/z (i.e. not homogeneous)
    # The z-coordinate of the result does not have an intuitive meaning, but is affinely related to the world-space z
    x_world_scaled = tf.squeeze(tf.matmul(
        tf.expand_dims(tf.concat([x_ndc, tf.ones_like(x_ndc[..., :1])], axis=-1), axis=-2),
        clip_to_world_matrix
    ), axis=-2)
    x_world_scaled.set_shape(x_ndc.get_shape()[:-1].as_list() + [4])
    return x_world_scaled[..., :3] / x_world_scaled[..., 3:]


def unproject_pixels_to_rays(pixel_locations, clip_to_world_matrix, image_size, name=None):
    """Computes world-space start-points and directions for rays cast into the scene from the given pixel locations

    Args:
        pixel_locations: a `Tensor` of shape [A1, ..., An, B1, ..., Bm, 2] of (x, y) coordinates in pixel space, where the Ai represent
            arbitrarily many leading (batch) dimensions., and the Bi represent arbitrarily many dimensions for which the
            projection parameters clip_to_world_matrix and image_size are shared
        clip_to_world_matrix: a `Tensor` of shape [A1, ..., An, 4, 4] specifying the combined transform matrix mapping from clip-space
            to world-space. Typically this is given by inv(world-to-view-matrix * projection-matrix).
        image_size: an int32 `Tensor` of shape [A1, ..., An, 2] specifying the width then height in pixels of the image wrt which
            pixel_locations is given.
        name: an optional name for the operation

    Returns:
        pixel_ray_starts_world: a `Tensor` of shape [A1, ..., An, B1, ..., Bm, 3], which for each pixel-location, gives the world-space
            location of the intersection of the corresponding ray with the camera near-plane
        pixel_ray_deltas_world: similar to pixel_ray_starts_world, but giving an unnormalised world-space direction vector for each
            ray, pointing away from the camera
    """

    with ops.name_scope(name, 'UnprojectPixelsToRays', [pixel_locations, clip_to_world_matrix, image_size]) as scope:

        pixel_locations = tf.convert_to_tensor(pixel_locations, name='pixel_locations', dtype=tf.float32)
        clip_to_world_matrix = tf.convert_to_tensor(clip_to_world_matrix, name='clip_to_world_matrix', dtype=tf.float32)
        image_size = tf.convert_to_tensor(image_size, name='image_size', dtype=tf.int32)

        per_iib_dims = pixel_locations.get_shape().ndims - image_size.get_shape().ndims  # corresponds to m in the docstring
        image_size = tf.reshape(image_size, image_size.get_shape()[:-1].as_list() + [1] * per_iib_dims + [2])
        clip_to_world_matrix = tf.reshape(
            clip_to_world_matrix,
            tf.concat([
                tf.shape(clip_to_world_matrix)[:-2],
                [1] * per_iib_dims + [4, 4]
            ], axis=0)
        )

        # This is needed for old versions of tensorflow as tf.matmul did not previously support broadcasting
        version_bits = tf.version.VERSION.split('.')
        if int(version_bits[0]) <= 1 and int(version_bits[1]) < 14:
            clip_to_world_matrix = tf.broadcast_to(
                clip_to_world_matrix,
                tf.concat([tf.shape(pixel_locations)[:-1], [4, 4]], axis=0)
            )

        pixel_locations_ndc = _pixel_to_ndc(pixel_locations, tf.cast(image_size, tf.float32))
        pixel_ray_starts_world = _unproject_ndc_to_world(tf.concat([pixel_locations_ndc, -1. * tf.ones_like(pixel_locations_ndc[..., :1])], axis=-1), clip_to_world_matrix)  # indexed by A*, B*, x/y/z
        pixel_ray_deltas_world = _unproject_ndc_to_world(tf.concat([pixel_locations_ndc, tf.zeros_like(pixel_locations_ndc[..., :1])], axis=-1), clip_to_world_matrix) - pixel_ray_starts_world

    return pixel_ray_starts_world, pixel_ray_deltas_world


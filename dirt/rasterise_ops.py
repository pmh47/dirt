import os
import tensorflow as tf
from tensorflow.python.framework import ops

_lib_path = os.path.dirname(__file__)
_rasterise_module = tf.load_op_library(_lib_path + '/librasterise.so')


def rasterise(background, vertices, vertex_colors, faces, height=None, width=None, channels=None, name=None):
    """Rasterises the given `vertices` and `faces` over `background`.

    This function takes a set of vertices, vertex colors, faces (vertex indices), and a background.
    It returns a single image, containing the faces these arrays, over the given background.

    It supports single-channel (grayscale) or three-channel (RGB) image rendering, or arbitrary
    numbers of channels for g-buffer rendering in a deferred shading pipeline (see `rasterise_deferred`).

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
        return rasterise_batch(background[None], vertices[None], vertex_colors[None], faces[None], height, width, channels, name)[0]


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

        if channels == 1 or channels == 3:
            return _rasterise_module.rasterise(
                background, vertices, vertex_colors, faces,  # inputs
                height, width, channels,  # attributes
                name=scope
            )
        else:
            assert channels > 0
            pixels = []
            begin_channel = 0
            while begin_channel < channels:
                if begin_channel + 3 <= channels:
                    end_channel = begin_channel + 3
                else:
                    # in the mod-2 case, could just do a 3-channel pass (instead of two 1-channel passes), concating zeros to the background and attributes, then indexing them off the pixels
                    end_channel = begin_channel + 1
                with ops.name_scope('channels_{}_to_{}'.format(begin_channel, end_channel)) as channel_group_scope:
                    pixels.append(
                        _rasterise_module.rasterise(
                            background[..., begin_channel : end_channel],
                            vertices,
                            vertex_colors[..., begin_channel : end_channel],
                            faces,
                            height, width, end_channel - begin_channel,
                            name=channel_group_scope
                        )
                    )
                begin_channel = end_channel
            return tf.concat(pixels, axis=-1)


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


def _rasterise_grad_multichannel(vertices, faces, pixels, d_loss_by_pixels, single_or_batch):

    assert single_or_batch in ['single', 'batch']

    if single_or_batch == 'single':
        vertices = vertices[None]
        faces = faces[None]
        pixels = pixels[None]
        d_loss_by_pixels = d_loss_by_pixels[None]

    assert len(pixels.get_shape()) == 4
    height, width, channels = pixels.get_shape()[1], pixels.get_shape()[2], pixels.get_shape()[3]

    results = []
    begin_channel = 0
    while begin_channel < channels:
        if begin_channel + 3 <= channels:
            end_channel = begin_channel + 3
        else:
            # in the mod-2 case, could just do a 3-channel pass (instead of two 1-channel passes), concating zeros to the background and attributes, then indexing them off the pixels
            end_channel = begin_channel + 1
        with ops.name_scope('grad_channels_{}_to_{}'.format(begin_channel, end_channel)) as channel_group_scope:
            results.append(_rasterise_module.rasterise_grad(
                vertices, faces,
                pixels[..., begin_channel : end_channel],
                d_loss_by_pixels[..., begin_channel : end_channel],
                height, width, end_channel - begin_channel,
                name=channel_group_scope
            ))
        begin_channel = end_channel
    # ** is the sum in the following correct?
    grad_vertices = sum([result.grad_vertices for result in results])
    grad_vertex_colors = tf.concat([result.grad_vertex_colors for result in results], axis=-1)
    grad_background = tf.concat([result.grad_background for result in results], axis=-1)
    if single_or_batch == 'single':
        return {
            'grad_vertices': grad_vertices[0],
            'grad_vertex_colors': grad_vertex_colors[0],
            'grad_background': grad_background[0]
        }
    else:
        return {
            'grad_vertices': grad_vertices,
            'grad_vertex_colors': grad_vertex_colors,
            'grad_background': grad_background
        }


def _rasterise_deferred_internal(background, vertices, attributes, faces, shader_fn, shader_additional_inputs, single_or_batch, name):

    # ** it would be more efficient to compute both attribute and vertex gradients in one pass, modifying
    # ** the grad op to take as inputs the loss gradients wrt both the gbuffer and the shaded pixels

    # ** it would be nice to allow pixels to be a nested structure of pixel-like things

    assert single_or_batch in ['single', 'batch']

    @tf.custom_gradient
    def _impl(vertices, faces, attributes, background, *shader_additional_inputs):

        gbuffer = (rasterise if single_or_batch == 'single' else rasterise_batch)(background, vertices, attributes, faces, name=scope)

        if tf.executing_eagerly():
            # ** is it safe for this to be persistent, i.e. is it guaranteed that all resources will be deleted?
            with tf.GradientTape(persistent=True) as shader_tape:
                shader_tape.watch([gbuffer, shader_additional_inputs])
                pixels = shader_fn(gbuffer, *shader_additional_inputs)
        else:
            pixels = shader_fn(gbuffer, *shader_additional_inputs)

        def grad(d_loss_by_pixels):

            # Calculate the derivative wrt vertices, but filtering the shaded image instead of the gbuffer -- these are the 'final', correct
            # gradient wrt the vertices, but for the attributes and background, we need to account for shader_fn
            d_loss_by_vertices = _rasterise_grad_multichannel(
                vertices, faces,
                pixels, d_loss_by_pixels,
                single_or_batch
            )['grad_vertices']

            # For colours, need to backprop through shader_fn first; this yields the derivative of the final pixels wrt the gbuffer-pixels
            # Then, pass these derivatives back into the rasterise-grad op to propagate to vertex/background attributes

            if tf.executing_eagerly():
                d_loss_by_gbuffer, d_loss_by_shader_additional_inputs = shader_tape.gradient(
                    pixels,
                    [gbuffer, shader_additional_inputs],
                    d_loss_by_pixels
                )
            else:
                d_loss_by_gbuffer_and_shader_additional_inputs = tf.gradients(
                    pixels,
                    [gbuffer] + list(shader_additional_inputs),
                    d_loss_by_pixels
                )
                d_loss_by_gbuffer = d_loss_by_gbuffer_and_shader_additional_inputs[0]
                d_loss_by_shader_additional_inputs = d_loss_by_gbuffer_and_shader_additional_inputs[1:]

            # The attribute and background gradients computed by the following are correct; the vertex gradients are not, as they
            # are based on filtering the gbuffer instead of the shaded output
            d_loss_by_attributes = _rasterise_grad_multichannel(
                vertices, faces,
                gbuffer, d_loss_by_gbuffer,
                single_or_batch
            )

            return [
                d_loss_by_vertices, None, d_loss_by_attributes['grad_vertex_colors'], d_loss_by_attributes['grad_background']
            ] + list(d_loss_by_shader_additional_inputs)

        return pixels, grad

    with ops.name_scope(name, 'RasteriseDeferred', [background, vertices, attributes, faces] + list(shader_additional_inputs)) as scope:
        background = tf.convert_to_tensor(background, name='background', dtype=tf.float32)
        vertices = tf.convert_to_tensor(vertices, name='vertices', dtype=tf.float32)
        attributes = tf.convert_to_tensor(attributes, name='vertex_attributes', dtype=tf.float32)
        faces = tf.convert_to_tensor(faces, name='faces', dtype=tf.int32)

        return _impl(vertices, faces, attributes, background, *shader_additional_inputs)


def rasterise_deferred(background_attributes, vertices, vertex_attributes, faces, shader_fn, shader_additional_inputs=[], name=None):
    """Rasterises and shades the given `vertices` and `faces`, using the specified deferred shader function and
    vertex/background attributes.

    Deferred shading is an efficient approach to rendering images with per-pixel lighting and texturing.
    It splits rendering into two passes. In the first pass, vertex attributes (such as colors and normals) are
    rasterised into a pseudo-image called a G-buffer. In the second pass, shading calculations are performed
    directly on this buffer (e.g. calculating the reflected lighting given the surface colour and normal at a pixel).

    This function takes a set of vertices, vertex attributes, faces (vertex indices), background attributes, and a
    deferred shader function. It first rasterises a G-buffer containing the attributes, then calls the given shader
    function passing the G-buffer as input; the shader function is assumed to produce the final pixels. It is equivalent
    to `shader_fn(rasterise(background_attributes, vertices, vertex_attributes, faces), *shader_additional_inputs)`,
    but its gradient correctly accounts for how the approximate gradients of `rasterise` interact with `shader_fn`.

    Any computation that is conceptually performed 'on the surface' of the 3D geometry should be included
    in `shader_fn` (typically texture-sampling and lighting); downstream 2D post-processing (e.g. blurring the
    rendered image) should not be included.

    Note that for correct gradients, `shader_fn` should not reference any tensors in enclosing scopes directly.
    Instead, any such tensors must be passed through the list `shader_additional_inputs`, whose values are
    forwarded as additional parameters to `shader_fn`. For example, if the pixel values depend on a non-constant
    lighting angle, the tensor representing that lighting angle should be passed through `shader_additional_inputs`.

    For usage examples, see `samples/deferred.py` and `samples/textured.py`.

    Args:
        background_attributes: a float32 `Tensor` of shape [height, width, attributes], defining the values of the
            attributes to use for G-buffer pixels that do not intersect any triangle
        vertices: a float32 `Tensor` of shape [vertex count, 4] defining a set of vertex locations, given as homogeneous
            coordinates in OpenGL clip space
        vertex_attributes: a float32 `Tensor` of shape [vertex count, attributes] defining the values of the attributes at
            each vertex; these are linearly interpolated in 3D space to calculate the attribute value at each pixel of the
            G-buffer
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle to rasterise. Note that each vertex may be used by several faces
        shader_fn: a function that takes the G-buffer as input, and returns shaded pixels as output. Specifically, it takes
            one or more parameters, of which the first is always the G-buffer (a float32 `Tensor` of shape
            [height, width, attributes]), and any others are the tensors in `shader_additional_inputs`. It returns a float32
            `Tensor` of shape [height, width, channels] containing the final image; channels is typically three (for an RGB
            image) but this is not required
        shader_additional_inputs: an optional list of tensors that are passed to `shader_fn` in addition to the G-buffer. Any
            tensors required to compute the shading aside from the vertex attributes should be explicitly passed through
            this parameter, not directly referenced in an outer scope from shader_fn, else their gradients will be incorrect.
        name: an optional name for the operation

    Returns:
        The rendered pixels, as a float32 `Tensor` of shape [height, width, channels]
    """

    return _rasterise_deferred_internal(background_attributes, vertices, vertex_attributes, faces, shader_fn, shader_additional_inputs, 'single', name)


def rasterise_batch_deferred(background_attributes, vertices, vertex_attributes, faces, shader_fn, shader_additional_inputs=[], name=None):
    """Rasterises and shades a batch of meshes with the same numbers of vertices and faces, using the specified
    deferred shader function.

    This function takes batch-indexed `vertices`, `faces`, `vertex_attributes`, and `background_attributes`. Any
    values in `shader_additional_inputs` may be batch-indexed or not, depending how shader_fn interprets them.

    It is conceptually equivalent to:
    ```python
    tf.stack([
        rasterise_deferred(background_attributes_i, vertices_i, vertex_attributes_i, faces_i, shader_fn, shader_addtional_inputs)
        for (background_attributes_i, vertices_i, vertex_attributes_i, faces_i)
        in zip(background_attributes, vertices, vertex_attributes, faces)
    ])
    ```
    See `rasterise_deferred` for definitions of the parameters, noting that for `rasterise_batch_deferred`, a
    leading dimension should be included.
    """

    return _rasterise_deferred_internal(background_attributes, vertices, vertex_attributes, faces, shader_fn, shader_additional_inputs, 'batch', name)

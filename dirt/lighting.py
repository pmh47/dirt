
import tensorflow as tf
from tensorflow.python.framework import ops


def _repeat_1d(tensor, count):

    assert tensor.get_shape().ndims == 1
    return tf.reshape(tf.tile(tensor[:, tf.newaxis], tf.convert_to_tensor([1, count])), [-1])


def _prepare_vertices_and_faces(vertices, faces):

    vertices = tf.convert_to_tensor(vertices, name='vertices')
    faces = tf.convert_to_tensor(faces, name='faces')

    if faces.dtype is not tf.int32:
        assert faces.dtype is tf.int64
        faces = tf.cast(faces, tf.int32)

    return vertices, faces


def _get_face_normals(vertices, faces):

    vertices_ndim = vertices.get_shape().ndims
    vertices_by_index = tf.transpose(vertices, [vertices_ndim - 2] + list(range(vertices_ndim - 2)) + [vertices_ndim - 1])  # indexed by vertex-index, *, x/y/z
    vertices_by_face = tf.gather(vertices_by_index, faces)  # indexed by face-index, vertex-in-face, *, x/y/z
    normals_by_face = tf.cross(vertices_by_face[:, 1] - vertices_by_face[:, 0], vertices_by_face[:, 2] - vertices_by_face[:, 0])  # indexed by face-index, *, x/y/z
    normals_by_face /= (tf.norm(normals_by_face, axis=-1, keep_dims=True) + 1.e-12)  # ditto
    return normals_by_face, vertices_by_index


def vertex_normals(vertices, faces, name=None):
    """Computes vertex normals for the given meshes.

    This function takes a batch of meshes with common topology, and calculates vertex normals for each.

    Args:
        vertices: a `Tensor` of shape [*, vertex count, 3] or [*, vertex count, 4], where * represents arbitrarily
            many leading (batch) dimensions.
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle.
        name: an optional name for the operation

    Returns:
        a `Tensor` of shape [*, vertex count, 3], which for each vertex, gives the (normalised) average of the normals of
        all faces that include that vertex
    """

    # This computes vertex normals, as the average of the normals of the faces each vertex is part of
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # result is indexed by *, vertex-index, x/y/z

    with ops.name_scope(name, 'VertexNormals', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)
        vertices = vertices[..., :3]  # drop the w-coordinate if present

        vertices_ndim = vertices.get_shape().ndims
        normals_by_face, vertices_by_index = _get_face_normals(vertices, faces)  # normals_by_face is indexed by face-index, *, x/y/z

        face_count = tf.shape(faces)[0]
        vbi_shape = tf.shape(vertices_by_index)
        N_extra = tf.reduce_prod(vbi_shape[1:-1])  # this is the number of 'elements' in the * dimensions

        assert vertices_ndim in {2, 3}  # ** keep it simple for now; in the general case we need a flattened outer product of ranges
        if vertices_ndim == 2:
            extra_indices = []
        else:
            extra_indices = [tf.tile(_repeat_1d(tf.range(N_extra), 3), [face_count * 3])]

        normals_by_face_and_vertex = tf.SparseTensor(
            indices=tf.cast(
                tf.stack([  # each element of this stack is repeated a number of times matching the things after, then tiled a number of times matching the things before, so that each has the same length
                    _repeat_1d(tf.range(face_count, dtype=tf.int32), N_extra * 9),
                    _repeat_1d(tf.reshape(faces, [-1]), N_extra * 3)
                ] + extra_indices + [
                    tf.tile(tf.constant([0, 1, 2], dtype=tf.int32), tf.convert_to_tensor([face_count * N_extra * 3]))
                ], axis=1),
                tf.int64
            ),
            values=tf.reshape(tf.tile(normals_by_face[:, tf.newaxis, ...], [1, 3] + [1] * (vertices_ndim - 1)), [-1]),
            dense_shape=tf.cast(tf.concat([[face_count], vbi_shape], axis=0), tf.int64)
        )  # indexed by face-index, vertex-index, *, x/y/z

        summed_normals_by_vertex = tf.sparse_reduce_sum(normals_by_face_and_vertex, axis=0)  # indexed by vertex-index, *, x/y/z
        renormalised_normals_by_vertex = summed_normals_by_vertex / (tf.norm(summed_normals_by_vertex, axis=-1, keep_dims=True) + 1.e-12)  # ditto

        result = tf.transpose(renormalised_normals_by_vertex, list(range(1, vertices_ndim - 1)) + [0, vertices_ndim - 1])
        result.set_shape(vertices.get_shape())
        return result


def _static_map_fn(f, elements):
    assert elements.get_shape()[0].value is not None
    return tf.stack([f(elements[index]) for index in range(int(elements.get_shape()[0]))])


def vertex_normals_pre_split(vertices, faces, name=None, static=False):
    """Computes vertex normals for the given pre-split meshes.

    This function is identical to `vertex_normals`, except that it assumes each vertex is used by just one face, which
    allows a more efficient implementation.
    """

    # This is identical to vertex_normals, but assumes each vertex appears in exactly one face, e.g. due to having been
    # processed by split_vertices_by_face
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # result is indexed by *, vertex-index, x/y/z

    with ops.name_scope(name, 'VertexNormalsPreSplit', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)
        vertices = vertices[..., :3]  # drop the w-coordinate if present
        face_count = int(faces.get_shape()[0]) if static else tf.shape(faces)[0]

        normals_by_face, _ = _get_face_normals(vertices, faces)  # indexed by face-index, *, x/y/z
        normals_by_face_flat = tf.reshape(
            tf.transpose(normals_by_face, list(range(1, normals_by_face.get_shape().ndims - 1)) + [0, normals_by_face.get_shape().ndims - 1]),
            [-1, face_count, 3]
        )  # indexed by prod(*), face-index, x/y/z

        normals_by_vertex_flat = (_static_map_fn if static else tf.map_fn)(lambda normals_for_iib: tf.scatter_nd(
            indices=tf.reshape(faces, [-1, 1]),
            updates=tf.reshape(tf.tile(normals_for_iib[:, tf.newaxis, :], [1, 3, 1]), [-1, 3]),
            shape=tf.shape(vertices)[-2:]
        ), normals_by_face_flat)
        normals_by_vertex = tf.reshape(normals_by_vertex_flat, tf.shape(vertices))

        return normals_by_vertex


def split_vertices_by_face(vertices, faces, name=None):
    """Returns a new mesh where each vertex is used by exactly one face.

    This function takes a batch of meshes with common topology as input, and also returns a batch of meshes
    with common topology. The resulting meshes have the same geometry, but each vertex is used by exactly
    one face.

    Args:
        vertices: a `Tensor` of shape [*, vertex count, 3] or [*, vertex count, 4], where * represents arbitrarily
            many leading (batch) dimensions.
        faces: an int32 `Tensor` of shape [face count, 3]; each value is an index into the first dimension of `vertices`, and
            each row defines one triangle.

    Returns:
        a tuple of two tensors `new_vertices, new_faces`, where `new_vertices` has shape [*, V, 3] or [*,  V, 4], where
        V is the new vertex count after splitting, and `new_faces` has shape [F, 3] where F is the new face count after
        splitting.
    """

    # This returns an equivalent mesh, with vertices duplicated such that there is exactly one vertex per face it is used in
    # vertices is indexed by *, vertex-index, x/y/z[/w]
    # faces is indexed by face-index, vertex-in-face
    # Ditto for results

    with ops.name_scope(name, 'SplitVerticesByFace', [vertices, faces]) as scope:

        vertices, faces = _prepare_vertices_and_faces(vertices, faces)

        vertices_shape = tf.shape(vertices)
        face_count = tf.shape(faces)[0]

        flat_vertices = tf.reshape(vertices, [-1, vertices_shape[-2], vertices_shape[-1]])
        new_flat_vertices = tf.map_fn(lambda vertices_for_iib: tf.gather(vertices_for_iib, faces), flat_vertices)
        new_vertices = tf.reshape(new_flat_vertices, tf.concat([vertices_shape[:-2], [face_count * 3, vertices_shape[-1]]], axis=0))

        new_faces = tf.reshape(tf.range(face_count * 3), [-1, 3])

        static_face_count = faces.get_shape().dims[0] if faces.get_shape().dims is not None else None
        static_new_vertex_count = static_face_count * 3 if static_face_count is not None else None
        if vertices.get_shape().dims is not None:
            new_vertices.set_shape(vertices.get_shape().dims[:-2] + [static_new_vertex_count] + vertices.get_shape().dims[-1:])
        new_faces.set_shape([static_face_count, 3])

        return new_vertices, new_faces


def diffuse_directional(vertex_normals, vertex_colors, light_direction, light_color, double_sided=True, name=None):
    """Calculate reflectance due to directional lighting on a diffuse surface.

    This calculates Lambertian reflectance at points with the given normals, under a single directional
    (parallel) light of the specified angle, mapping over leading batch/etc. dimensions. If double_sided is set,
    then surfaces whose normal faces away from the light are still lit; otherwise, they will be black.

    Note that this function may be applied to vertices of a mesh before rasterisation, or to values in a G-buffer for
    deferred shading.

    Args:
        vertex_normals: a `Tensor` of shape [*, vertex count, 3], where * represents arbitrarily many leading (batch) dimensions..
        vertex_colors: a `Tensor` of shape [*, vertex count, C], where * is the same as for `vertex_normals` and C is the
            number of colour channels, defining the albedo or reflectance at each point.
        light_direction: a `Tensor` of shape [*, 3] defining the direction of the incident light.
        light_color: a `Tensor` of shape [*, C] defining the colour of the light.
        double_sided: a python `bool`; if true, back faces will be shaded the same as front faces; else, they will be black
        name: an optional name for the operation.

    Returns:
        a `Tensor` of shape [*, vertex count, C], where * is the same as for the input parameters and C is the number of channels,
        giving the reflectance for each point.
    """

    # vertex_normals is indexed by *, vertex-index, x/y/z; it is assumed to be normalised
    # vertex_colors is indexed by *, vertex-index, r/g/b
    # light_direction is indexed by *, x/y/z; it is assumed to be normalised
    # light_color is indexed by *, r/g/b
    # result is indexed by *, vertex-index, r/g/b

    with ops.name_scope(name, 'DiffuseDirectionalLight', [vertex_normals, vertex_colors, light_direction, light_color]) as scope:

        vertex_normals = tf.convert_to_tensor(vertex_normals, name='vertex_normals')
        vertex_colors = tf.convert_to_tensor(vertex_colors, name='vertex_colors')
        light_direction = tf.convert_to_tensor(light_direction, name='light_direction')
        light_color = tf.convert_to_tensor(light_color, name='light_color')

        cosines = tf.matmul(vertex_normals, -light_direction[..., tf.newaxis])  # indexed by *, vertex-index, singleton
        if double_sided:
            cosines = tf.abs(cosines)
        else:
            cosines = tf.maximum(cosines, 0.)

        return light_color[..., tf.newaxis, :] * vertex_colors * cosines


def specular_directional(vertex_positions, vertex_normals, vertex_reflectivities, light_direction, light_color, camera_position, shininess, double_sided=True, name=None):
    """Calculate reflectance due to directional lighting on a specular surface.

    This calculates Phong reflectance at points with the given normals, under a single directional
    (parallel) light of the specified angle, mapping over batch/etc. dimensions. If double_sided is set,
    then surfaces whose normal faces away from the light are still lit; otherwise, they will be black.

    Note that this function may be applied to vertices of a mesh before rasterisation, or to values in a G-buffer for
    deferred shading.

    Args:
        vertex_positions: a `Tensor` of shape [*, vertex count, 3], where * represents arbitrarily many leading (batch) dimensions.,
            defining the 3D location of each point.
        vertex_normals: a `Tensor` of shape [*, vertex count, 3], where * is the same as for `vertex_positions`, defining the surface
            normal at each point.
        vertex_colors: a `Tensor` of shape [*, vertex count, C], where C is the number of colour channels, defining the albedo or
            reflectance at each point.
        light_direction: a `Tensor` of shape [*, 3] defining the direction of the incident light.
        light_color: a `Tensor` of shape [*, C] defining the colour of the light.
        camera_position: a `Tensor` of shape [*, 3] defining the position of the camera in 3D space.
        shininess: a `Tensor` of shape [*] defining the specular reflectance index.
        double_sided: a python `bool`; if true, back faces will be shaded the same as front faces; else, they will be black.
        name: an optional name for the operation.

    Returns:
        a `Tensor` of shape [*, vertex count, C], where * is the same as for the input parameters and C is the number of channels,
        giving the reflectance for each point.
    """

    # vertex_positions is indexed by *, vertex-index, x/y/z
    # vertex_normals is indexed by *, vertex-index, x/y/z; it is assumed to be normalised
    # vertex_reflectivities is indexed by *, vertex-index, r/g/b
    # light_direction is indexed by *, x/y/z; it is assumed to be normalised
    # light_color is indexed by *, r/g/b
    # camera_position is indexed by *, x/y/z
    # shininess is indexed by *
    # result is indexed by *, vertex-index, r/g/b

    with ops.name_scope(name, 'SpecularDirectionalLight', [vertex_positions, vertex_normals, vertex_reflectivities, light_direction, light_color, camera_position, shininess]) as scope:

        vertex_positions = tf.convert_to_tensor(vertex_positions, name='vertex_positions')
        vertex_normals = tf.convert_to_tensor(vertex_normals, name='vertex_normals')
        vertex_reflectivities = tf.convert_to_tensor(vertex_reflectivities, name='vertex_reflectivities')
        light_direction = tf.convert_to_tensor(light_direction, name='light_direction')
        light_color = tf.convert_to_tensor(light_color, name='light_color')
        camera_position = tf.convert_to_tensor(camera_position, name='camera_position')
        shininess = tf.convert_to_tensor(shininess, name='shininess')

        vertices_to_light_direction = -light_direction
        reflected_directions = -vertices_to_light_direction + 2. * tf.matmul(vertex_normals, vertices_to_light_direction[..., tf.newaxis]) * vertex_normals  # indexed by *, vertex-index, x/y/z
        vertex_to_camera_displacements = camera_position[..., tf.newaxis, :] - vertex_positions  # indexed by *, vertex-index, x/y/z
        cosines = tf.reduce_sum(
            (vertex_to_camera_displacements / tf.norm(vertex_to_camera_displacements, axis=-1, keep_dims=True) + 1.e-12) * reflected_directions,
            axis=-1, keep_dims=True
        )  # indexed by *, vertex-index, singleton
        if double_sided:
            cosines = tf.abs(cosines)
        else:
            cosines = tf.maximum(cosines, 0.)

        return light_color[..., tf.newaxis, :] * vertex_reflectivities * tf.pow(cosines, shininess[..., tf.newaxis, tf.newaxis])


def diffuse_point(vertex_positions, vertex_normals, vertex_colors, light_position, light_color, double_sided=True, name=None):
    """Calculate reflectance due to directional lighting on a diffuse surface.

    This calculates Lambertian reflectance at points with the given normals, under a single point light at the
    specified location, mapping over leading batch/etc. dimensions. If double_sided is set, then surfaces whose
    normal faces away from the light are still lit; otherwise, they will be black.

    A point light radiates uniformly in all directions from some physical location.

    Note that this function may be applied to vertices of a mesh before rasterisation, or to values in a G-buffer for
    deferred shading.

    Args:
        vertex_positions: a `Tensor` of shape [*, vertex count, 3], where * represents arbitrarily many leading (batch) dimensions.,
            defining the 3D location of each point.
        vertex_normals: a `Tensor` of shape [*, vertex count, 3], where * is the same as for `vertex_positions`, defining the surface
            normal at each point.
        vertex_colors: a `Tensor` of shape [*, vertex count, C], where C is the number of colour channels, defining the albedo or
            reflectance at each point.
        light_position: a `Tensor` of shape [*, 3] defining the location of the point light source.
        light_color: a `Tensor` of shape [*, C] defining the colour of the light.
        double_sided: a python `bool`; if true, back faces will be shaded the same as front faces; else, they will be black
        name: an optional name for the operation.

    Returns:
        a `Tensor` of shape [*, vertex count, C], where * is the same as for the input parameters and C is the number of channels,
        giving the reflectance for each point.
    """

    # vertex_positions is indexed by *, vertex-index, x/y/z
    # vertex_normals is indexed by *, vertex-index, x/y/z; it is assumed to be normalised
    # vertex_colors is indexed by *, vertex-index, r/g/b
    # light_position is indexed by *, x/y/z
    # light_color is indexed by *, r/g/b
    # result is indexed by *, vertex-index, r/g/b

    with ops.name_scope(name, 'DiffusePointLight', [vertex_positions, vertex_normals, vertex_colors, light_position, light_color]) as scope:

        vertex_positions = tf.convert_to_tensor(vertex_positions, name='vertex_positions')
        vertex_normals = tf.convert_to_tensor(vertex_normals, name='vertex_normals')
        vertex_colors = tf.convert_to_tensor(vertex_colors, name='vertex_colors')
        light_position = tf.convert_to_tensor(light_position, name='light_position')
        light_color = tf.convert_to_tensor(light_color, name='light_color')

        relative_positions = vertex_positions - light_position[..., tf.newaxis, :]  # indexed by *, vertex-index, x/y/z
        incident_directions = relative_positions / (tf.norm(relative_positions, axis=-1, keep_dims=True) + 1.e-12)  # ditto
        cosines = tf.reduce_sum(vertex_normals * incident_directions, axis=-1)  # indexed by *, vertex-index
        if double_sided:
            cosines = tf.abs(cosines)
        else:
            cosines = tf.maximum(cosines, 0.)

        return light_color[..., tf.newaxis, :] * vertex_colors * cosines[..., tf.newaxis]


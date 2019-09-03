
# This demonstrates using Dirt for textured rendering with a UV map

import os
import numpy as np
import tensorflow as tf

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting


frame_width, frame_height = 640, 480


def uvs_to_pixel_indices(uvs, texture_shape, mode='repeat'):

    # Note that this assumes u = 0, v = 0 is at the top-left of the image -- different to OpenGL!

    uvs = uvs[..., ::-1]  # change x, y coordinates to y, x indices
    texture_shape = tf.cast(texture_shape, tf.float32)
    if mode == 'repeat':
        return uvs % 1. * texture_shape
    elif mode == 'clamp':
        return tf.clip_by_value(uvs, 0., 1.) * texture_shape
    else:
        raise NotImplementedError


def sample_texture(texture, indices, mode='bilinear'):

    if mode == 'nearest':

        return tf.gather_nd(texture, tf.cast(indices, tf.int32))

    elif mode == 'bilinear':

        floor_indices = tf.floor(indices)
        frac_indices = indices - floor_indices
        floor_indices = tf.cast(floor_indices, tf.int32)

        neighbours = tf.gather_nd(
            texture,
            tf.stack([
                floor_indices,
                floor_indices + [0, 1],
                floor_indices + [1, 0],
                floor_indices + [1, 1]
            ]),
        )
        top_left, top_right, bottom_left, bottom_right = tf.unstack(neighbours)

        return \
            top_left * (1. - frac_indices[..., 1:]) * (1. - frac_indices[..., :1]) + \
            top_right * frac_indices[..., 1:] * (1. - frac_indices[..., :1]) + \
            bottom_left * (1. - frac_indices[..., 1:]) * frac_indices[..., :1] + \
            bottom_right * frac_indices[..., 1:] * frac_indices[..., :1]

    else:

        raise NotImplementedError


def main():

    # Build the scene geometry, which is just an axis-aligned cube centred at the origin in world space
    cube_vertices_object = []
    cube_uvs = []
    cube_faces = []
    def add_quad(vertices, uvs):
        index = len(cube_vertices_object)
        cube_faces.extend([[index + 2, index + 1, index], [index, index + 3, index + 2]])
        cube_vertices_object.extend(vertices)
        cube_uvs.extend(uvs)

    add_quad(vertices=[[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], uvs=[[0.1, 0.9], [0.9, 0.9], [0.9, 0.1], [0.1, 0.1]])  # front
    add_quad(vertices=[[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1]], uvs=[[1, 1], [0, 1], [0, 0], [1, 0]])  # back

    add_quad(vertices=[[1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1]], uvs=[[0.3, 0.25], [0.6, 0.25], [0.6, 0.55], [0.3, 0.55]])  # right
    add_quad(vertices=[[-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1]], uvs=[[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]])  # left

    add_quad(vertices=[[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]], uvs=[[0, 0], [2, 0], [2, 2], [0, 2]])  # top
    add_quad(vertices=[[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], uvs=[[0, 0], [2, 0], [2, 2], [0, 2]])  # bottom

    cube_vertices_object = np.asarray(cube_vertices_object, np.float32)
    cube_uvs = np.asarray(cube_uvs, np.float32)

    # Load the texture image
    texture = tf.cast(tf.image.decode_jpeg(tf.read_file(os.path.dirname(__file__) + '/cat.jpg')), tf.float32) / 255.

    # Convert vertices to homogeneous coordinates
    cube_vertices_object = tf.concat([
        cube_vertices_object,
        tf.ones_like(cube_vertices_object[:, -1:])
    ], axis=1)

    # Transform vertices from object to world space, by rotating around the vertical axis
    cube_vertices_world = tf.matmul(cube_vertices_object, matrices.rodrigues([0., 0.6, 0.]))

    # Calculate face normals
    cube_normals_world = lighting.vertex_normals(cube_vertices_world, cube_faces)

    # Transform vertices from world to camera space; note that the camera points along the negative-z axis in camera space
    view_matrix = matrices.compose(
        matrices.translation([0., -2., -3.2]),  # translate it away from the camera
        matrices.rodrigues([-0.5, 0., 0.])  # tilt the view downwards
    )
    cube_vertices_camera = tf.matmul(cube_vertices_world, view_matrix)

    # Transform vertices from camera to clip space
    projection_matrix = matrices.perspective_projection(near=0.1, far=20., right=0.1, aspect=float(frame_height) / frame_width)
    cube_vertices_clip = tf.matmul(cube_vertices_camera, projection_matrix)

    # The following function is applied to the G-buffer, which is a multi-channel image containing all the vertex attributes. It
    # uses this to calculate the shading (texture and lighting) at each pixel, hence their final intensities
    def shader_fn(gbuffer, texture, light_direction):

        # Unpack the different attributes from the G-buffer
        mask = gbuffer[:, :, :1]
        uvs = gbuffer[:, :, 1:3]
        normals = gbuffer[:, :, 3:]

        # Sample the texture at locations corresponding to each pixel; this defines the unlit material color at each point
        unlit_colors = sample_texture(texture, uvs_to_pixel_indices(uvs, tf.shape(texture)[:2]))

        # Calculate a simple grey ambient lighting component
        ambient_contribution = unlit_colors * [0.4, 0.4, 0.4]

        # Calculate a diffuse (Lambertian) lighting component
        diffuse_contribution = lighting.diffuse_directional(
            tf.reshape(normals, [-1, 3]),
            tf.reshape(unlit_colors, [-1, 3]),
            light_direction, light_color=[0.6, 0.6, 0.6], double_sided=True
        )
        diffuse_contribution = tf.reshape(diffuse_contribution, [frame_height, frame_width, 3])

        # The final pixel intensities inside the shape are given by combining the ambient and diffuse components;
        # outside the shape, they are set to a uniform background color
        pixels = (diffuse_contribution + ambient_contribution) * mask + [0., 0., 0.3] * (1. - mask)

        return pixels
    
    # Render the G-buffer channels (mask, UVs, and normals at each pixel), then perform the deferred shading calculation
    # In general, any tensor required by shader_fn and wrt which we need derivatives should be included in shader_additional_inputs;
    # although in this example they are constant, we pass the texture and lighting direction through this route as an illustration
    light_direction = tf.linalg.l2_normalize([1., -0.3, -0.5])
    pixels = dirt.rasterise_deferred(
        vertices=cube_vertices_clip,
        vertex_attributes=tf.concat([
            tf.ones_like(cube_vertices_object[:, :1]),  # mask
            cube_uvs,  # texture coordinates
            cube_normals_world  # normals
        ], axis=1),
        faces=cube_faces,
        background_attributes=tf.zeros([frame_height, frame_width, 6]),
        shader_fn=shader_fn,
        shader_additional_inputs=[texture, light_direction]
    )

    save_pixels = tf.write_file(
        'textured.jpg',
        tf.image.encode_jpeg(tf.cast(pixels * 255, tf.uint8))
    )

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():

        save_pixels.run()


if __name__ == '__main__':
    main()


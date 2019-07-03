
# This demonstrates using Dirt for textured rendering with a UV map

import os
import imageio
import numpy as np
import tensorflow as tf

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting


frame_width, frame_height = 640, 480


def unit(vector):
    return tf.convert_to_tensor(vector) / tf.norm(vector)


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

    add_quad(vertices=[[1, 1, 1], [1, 1, -1], [1, -1, -1], [1, -1, 1]], uvs=[[0.4, 0.35], [0.5, 0.35], [0.5, 0.45], [0.4, 0.45]])  # right
    add_quad(vertices=[[-1, 1, 1], [-1, 1, -1], [-1, -1, -1], [-1, -1, 1]], uvs=[[0.4, 0.4], [0.5, 0.4], [0.5, 0.5], [0.4, 0.5]])  # left

    add_quad(vertices=[[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]], uvs=[[0, 0], [2, 0], [2, 2], [0, 2]])  # top
    add_quad(vertices=[[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]], uvs=[[0, 0], [2, 0], [2, 2], [0, 2]])  # bottom

    cube_vertices_object = np.asarray(cube_vertices_object, np.float32)
    cube_uvs = np.asarray(cube_uvs, np.float32)

    # Load the texture image
    texture = tf.constant(imageio.imread(os.path.dirname(__file__) + '/cat.jpg'), dtype=tf.float32) / 255.

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

    # Render the G-buffer channels (mask, UVs, and normals at each pixel) needed for deferred shading
    gbuffer_mask = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=tf.ones_like(cube_vertices_object[:, :1]),
        background=tf.zeros([frame_height, frame_width, 1]),
        width=frame_width, height=frame_height, channels=1
    )[..., 0]
    background_value = -1.e4
    gbuffer_vertex_uvs = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=tf.concat([cube_uvs, tf.zeros_like(cube_uvs[:, :1])], axis=1),
        background=tf.ones([frame_height, frame_width, 3]) * background_value,
        width=frame_width, height=frame_height, channels=3
    )[..., :2]
    gbuffer_vertex_normals_world = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=cube_normals_world,
        background=tf.ones([frame_height, frame_width, 3]) * background_value,
        width=frame_width, height=frame_height, channels=3
    )

    # Dilate the normals and UVs to ensure correct gradients on the silhouette
    gbuffer_mask = gbuffer_mask[:, :, None]
    gbuffer_vertex_normals_world_dilated = tf.nn.max_pool(gbuffer_vertex_normals_world[None, ...], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')[0]
    gbuffer_vertex_normals_world = gbuffer_vertex_normals_world * gbuffer_mask + gbuffer_vertex_normals_world_dilated * (1. - gbuffer_mask)
    gbuffer_vertex_uvs_dilated = tf.nn.max_pool(gbuffer_vertex_uvs[None, ...], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')[0]
    gbuffer_vertex_uvs = gbuffer_vertex_uvs * gbuffer_mask + gbuffer_vertex_uvs_dilated * (1. - gbuffer_mask)

    # Calculate the colour buffer, by sampling the texture according to the rasterised UVs
    gbuffer_colours = gbuffer_mask * sample_texture(texture, uvs_to_pixel_indices(gbuffer_vertex_uvs, tf.shape(texture)[:2]))

    # Calculate a simple grey ambient lighting component
    ambient_contribution = gbuffer_colours * [0.4, 0.4, 0.4]

    # Calculate a diffuse (Lambertian) lighting component
    light_direction = unit([1., -0.3, -0.5])
    diffuse_contribution = lighting.diffuse_directional(
        tf.reshape(gbuffer_vertex_normals_world, [-1, 3]),
        tf.reshape(gbuffer_colours, [-1, 3]),
        light_direction, light_color=[0.6, 0.6, 0.6], double_sided=True
    )
    diffuse_contribution = tf.reshape(diffuse_contribution, [frame_height, frame_width, 3])

    # Final pixels are given by combining the ambient and diffuse components
    pixels = diffuse_contribution + ambient_contribution

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():

        pixels_eval = pixels.eval()
        imageio.imsave('textured.jpg', (pixels_eval * 255).astype(np.uint8))


if __name__ == '__main__':
    main()


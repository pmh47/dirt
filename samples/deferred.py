
# This demonstrates using Dirt for deferred shading, which allows per-pixel lighting

import tensorflow as tf
import cv2  # OpenCV, used only to display the result

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting


frame_width, frame_height = 640, 480


def build_cube():
    vertices = [[x, y, z] for z in [-1, 1] for y in [-1, 1] for x in [-1, 1]]
    quads = [
        [0, 1, 3, 2], [4, 5, 7, 6],  # back, front
        [1, 5, 4, 0], [2, 6, 7, 3],  # bottom, top
        [4, 6, 2, 0], [3, 7, 5, 1],  # left, right
    ]
    triangles = sum([[[a, b, c], [c, d, a]] for [a, b, c, d] in quads], [])
    return vertices, triangles


def unit(vector):
    return tf.convert_to_tensor(vector) / tf.norm(vector)


def main():

    # Build the scene geometry, which is just an axis-aligned cube centred at the origin in world space
    # We replicate vertices that are shared, so normals are effectively per-face instead of smoothed
    cube_vertices_object, cube_faces = build_cube()
    cube_vertices_object = tf.constant(cube_vertices_object, dtype=tf.float32)
    cube_vertices_object, cube_faces = lighting.split_vertices_by_face(cube_vertices_object, cube_faces)
    cube_vertex_colors = tf.ones_like(cube_vertices_object)

    # Convert vertices to homogeneous coordinates
    cube_vertices_object = tf.concat([
        cube_vertices_object,
        tf.ones_like(cube_vertices_object[:, -1:])
    ], axis=1)

    # Transform vertices from object to world space, by rotating around the vertical axis
    cube_vertices_world = tf.matmul(cube_vertices_object, matrices.rodrigues([0., 0.5, 0.]))

    # Calculate face normals; pre_split implies that no faces share a vertex
    cube_normals_world = lighting.vertex_normals_pre_split(cube_vertices_world, cube_faces)

    # Transform vertices from world to camera space; note that the camera points along the negative-z axis in camera space
    view_matrix = matrices.compose(
        matrices.translation([0., -1.5, -3.5]),  # translate it away from the camera
        matrices.rodrigues([-0.3, 0., 0.])  # tilt the view downwards
    )
    cube_vertices_camera = tf.matmul(cube_vertices_world, view_matrix)

    # Transform vertices from camera to clip space
    projection_matrix = matrices.perspective_projection(near=0.1, far=20., right=0.1, aspect=float(frame_height) / frame_width)
    cube_vertices_clip = tf.matmul(cube_vertices_camera, projection_matrix)

    # Render the G-buffer channels (vertex position, colour and normal at each pixel) needed for deferred shading
    gbuffer_vertex_positions_world = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=cube_vertices_world[:, :3],
        background=tf.ones([frame_height, frame_width, 3]) * float('-inf'),
        width=frame_width, height=frame_height, channels=3
    )
    gbuffer_vertex_colours_world = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=cube_vertex_colors,
        background=tf.zeros([frame_height, frame_width, 3]),
        width=frame_width, height=frame_height, channels=3
    )
    gbuffer_vertex_normals_world = dirt.rasterise(
        vertices=cube_vertices_clip,
        faces=cube_faces,
        vertex_colors=cube_normals_world,
        background=tf.ones([frame_height, frame_width, 3]) * float('-inf'),
        width=frame_width, height=frame_height, channels=3
    )

    # Dilate the position and normal channels at the silhouette boundary; this doesn't affect the image, but
    # ensures correct gradients for pixels just outside the silhouette
    background_mask = tf.cast(tf.equal(gbuffer_vertex_positions_world, float('-inf')), tf.float32)
    gbuffer_vertex_positions_world_dilated = tf.nn.max_pool(gbuffer_vertex_positions_world[None, ...], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')[0]
    gbuffer_vertex_positions_world = gbuffer_vertex_positions_world * (1. - background_mask) + gbuffer_vertex_positions_world_dilated * background_mask
    gbuffer_vertex_normals_world_dilated = tf.nn.max_pool(gbuffer_vertex_normals_world[None, ...], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')[0]
    gbuffer_vertex_normals_world = gbuffer_vertex_normals_world * (1. - background_mask) + gbuffer_vertex_normals_world_dilated * background_mask

    # Calculate a simple grey ambient lighting component
    ambient_contribution = gbuffer_vertex_colours_world * [0.2, 0.2, 0.2]

    # Calculate a red diffuse (Lambertian) lighting component
    light_direction = unit([1., -0.3, -0.5])
    diffuse_contribution = lighting.diffuse_directional(
        tf.reshape(gbuffer_vertex_normals_world, [-1, 3]),
        tf.reshape(gbuffer_vertex_colours_world, [-1, 3]),
        light_direction, light_color=[1., 0., 0.], double_sided=False
    )
    diffuse_contribution = tf.reshape(diffuse_contribution, [frame_height, frame_width, 3])

    # Calculate a white specular (Phong) lighting component
    camera_position_world = tf.matrix_inverse(view_matrix)[3, :3]
    specular_contribution = lighting.specular_directional(
        tf.reshape(gbuffer_vertex_positions_world, [-1, 3]),
        tf.reshape(gbuffer_vertex_normals_world, [-1, 3]),
        tf.reshape(gbuffer_vertex_colours_world, [-1, 3]),
        light_direction, light_color=[1., 1., 1.],
        camera_position=camera_position_world,
        shininess=6., double_sided=False
    )
    specular_contribution = tf.reshape(specular_contribution, [frame_height, frame_width, 3])

    # Final pixels are given by combining ambient, diffuse, and specular components
    pixels = diffuse_contribution + specular_contribution + ambient_contribution

    session = tf.Session()
    with session.as_default():

        pixels_eval = pixels.eval()
        cv2.imshow('deferred.py', pixels_eval[:, :, (2, 1, 0)])
        cv2.waitKey(0)


if __name__ == '__main__':
    main()


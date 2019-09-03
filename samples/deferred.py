
# This demonstrates using Dirt for deferred shading, which allows per-pixel lighting

import tensorflow as tf

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

    # The following function is applied to the G-buffer, which is a multi-channel image containing all the vertex attributes.
    # It uses this to calculate the shading at each pixel, hence their final intensities
    def shader_fn(gbuffer, view_matrix, light_direction):

        # Unpack the different attributes from the G-buffer
        mask = gbuffer[:, :, :1]
        positions = gbuffer[:, :, 1:4]
        unlit_colors = gbuffer[:, :, 4:7]
        normals = gbuffer[:, :, 7:]

        # Calculate a simple grey ambient lighting component
        ambient_contribution = unlit_colors * [0.2, 0.2, 0.2]

        # Calculate a red diffuse (Lambertian) lighting component
        diffuse_contribution = lighting.diffuse_directional(
            tf.reshape(normals, [-1, 3]),
            tf.reshape(unlit_colors, [-1, 3]),
            light_direction, light_color=[1., 0., 0.], double_sided=False
        )
        diffuse_contribution = tf.reshape(diffuse_contribution, [frame_height, frame_width, 3])

        # Calculate a white specular (Phong) lighting component
        camera_position_world = tf.matrix_inverse(view_matrix)[3, :3]
        specular_contribution = lighting.specular_directional(
            tf.reshape(positions, [-1, 3]),
            tf.reshape(normals, [-1, 3]),
            tf.reshape(unlit_colors, [-1, 3]),
            light_direction, light_color=[1., 1., 1.],
            camera_position=camera_position_world,
            shininess=6., double_sided=False
        )
        specular_contribution = tf.reshape(specular_contribution, [frame_height, frame_width, 3])

        # The final pixel intensities inside the shape are given by combining the three lighting components;
        # outside the shape, they are set to a uniform background color. We clip the final values as the specular
        # component saturates some pixels
        pixels = tf.clip_by_value(
            (diffuse_contribution + specular_contribution + ambient_contribution) * mask + [0., 0., 0.3] * (1. - mask),
            0., 1.
        )

        return pixels

    # Render the G-buffer channels (mask, vertex positions, vertex colours, and normals at each pixel), then perform
    # the deferred shading calculation. In general, any tensor required by shader_fn and wrt which we need derivatives
    # should be included in shader_additional_inputs; although in this example they are constant, we pass the view
    # matrix and lighting direction through this route as an illustration
    light_direction = tf.linalg.l2_normalize([1., -0.3, -0.5])
    pixels = dirt.rasterise_deferred(
        vertices=cube_vertices_clip,
        vertex_attributes=tf.concat([
            tf.ones_like(cube_vertices_object[:, :1]),  # mask
            cube_vertices_world[:, :3],  # vertex positions
            cube_vertex_colors,  # vertex colors
            cube_normals_world  # normals
        ], axis=1),
        faces=cube_faces,
        background_attributes=tf.zeros([frame_height, frame_width, 10]),
        shader_fn=shader_fn,
        shader_additional_inputs=[view_matrix, light_direction]
    )

    save_pixels = tf.write_file(
        'deferred.jpg',
        tf.image.encode_jpeg(tf.cast(pixels * 255, tf.uint8))
    )

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():

        save_pixels.run()


if __name__ == '__main__':
    main()


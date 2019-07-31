
import tensorflow as tf

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting

canvas_width, canvas_height = 64, 64
# canvas_width, canvas_height = 128, 128
square_size = 4.


def unit(vector):
    return tf.convert_to_tensor(vector) / tf.norm(vector)


def write_png(filename, image):

    image = tf.cast(image * 255, tf.uint8)
    return tf.write_file(filename, tf.image.encode_png(image))


def get_transformed_geometry(translation, rotation, scale):

    # Build bent square in object space, on z = 0 plane
    vertices_object = tf.constant([[-1, -1, 0.], [-1, 1, 0], [1, 1, 0], [1, -1, -0.8]], dtype=tf.float32) * square_size / 2
    faces = [[0, 1, 2], [0, 2, 3]]

    # ** we should add an occluding triangle!
    # ** also a non-planar meeting-of-faces

    # if we used a texture with top half red and bottom half green, and occluded face used bottom part of texture, and top
    # face used top part, then the gradient of uvs should be highly decorrelated from spatial gradient of colour
    # 

    vertices_object, faces = lighting.split_vertices_by_face(vertices_object, faces)

    # Convert vertices to homogeneous coordinates
    vertices_object = tf.concat([
        vertices_object,
        tf.ones_like(vertices_object[:, -1:])
    ], axis=1)

    # Transform vertices from object to world space, by rotating around the z-axis
    vertices_world = tf.matmul(vertices_object, matrices.rodrigues([0., 0., rotation])) * scale + tf.concat([translation, [0.]], axis=0)

    # Calculate face normals
    normals_world = lighting.vertex_normals(vertices_world, faces)

    # Transform vertices from world to camera space; note that the camera points along the negative-z axis in camera space
    view_matrix = matrices.translation([-0.5, 0., -3.5])  # translate it away from the camera
    vertices_camera = tf.matmul(vertices_world, view_matrix)

    # Transform vertices from camera to clip space
    projection_matrix = matrices.perspective_projection(near=0.1, far=20., right=0.1, aspect=float(canvas_height) / canvas_width)
    vertices_clip = tf.matmul(vertices_camera, projection_matrix)

    vertex_colours = tf.concat([
        tf.ones([3, 3]) * [0.8, 0.5, 0.],
        tf.ones([3, 3]) * [0.5, 0.8, 0.]
    ], axis=0)

    return vertices_clip, faces, normals_world, vertex_colours


def calculate_shading(colours, normals, light_intensity):

    ambient = colours * [0.4, 0.4, 0.4]

    light_direction = unit([1., -0.3, -0.5])
    diffuse_contribution = lighting.diffuse_directional(
        tf.reshape(normals, [-1, 3]),
        tf.reshape(colours, [-1, 3]),
        light_direction, light_color=tf.ones([3]) * [0., 1., 0.] * light_intensity, double_sided=True
    )
    diffuse = tf.reshape(diffuse_contribution, colours.get_shape())

    return ambient + diffuse


def get_pixels_direct(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background):

    return dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=calculate_shading(vertex_colours, vertex_normals, light_intensity),
        background=tf.ones([canvas_height, canvas_width, 3]) * background
    )


def get_pixels_deferred(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background):

    gbuffer_mask = dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=tf.ones_like(transformed_vertices[:, :1]),
        background=tf.zeros([canvas_height, canvas_width, 1]),
        width=canvas_width, height=canvas_height, channels=1
    )[..., 0]
    background_value = 0.  # -1.e4  # ** debug
    gbuffer_vertex_colours_world = dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=vertex_colours,
        background=tf.ones([canvas_height, canvas_width, 3]) * background,  # ** should we use true background or background_value indicator?
        width=canvas_width, height=canvas_height, channels=3
    )
    gbuffer_vertex_normals_world = dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=vertex_normals,
        background=tf.ones([canvas_height, canvas_width, 3]) * background_value,
        width=canvas_width, height=canvas_height, channels=3
    )

    # Dilate the normals to ensure correct gradients on the silhouette
    gbuffer_mask = gbuffer_mask[:, :, None]
    gbuffer_vertex_normals_world_dilated = tf.nn.max_pool(gbuffer_vertex_normals_world[None, ...], ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')[0]
    gbuffer_vertex_normals_world = gbuffer_vertex_normals_world * gbuffer_mask + gbuffer_vertex_normals_world_dilated * (1. - gbuffer_mask)

    pixels = gbuffer_mask * calculate_shading(gbuffer_vertex_colours_world, gbuffer_vertex_normals_world, light_intensity) + (1. - gbuffer_mask) * background

    return pixels


def main():

    translation = tf.Variable([0., 0., 0.])
    rotation = tf.Variable(0.5)
    scale = tf.Variable(1.)
    light_intensity = tf.Variable(0.6)
    background = tf.Variable([0., 0., 0.2])

    variables = [translation, rotation, scale, light_intensity, background]

    transformed_vertices, faces, vertex_normals, vertex_colours = get_transformed_geometry(translation, rotation, scale)

    pixels_direct = get_pixels_direct(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)
    save_pixels_direct = write_png('pixels_direct.png', tf.tile(pixels_direct, [1, 9, 1]))

    pixels_deferred = get_pixels_deferred(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)
    save_pixels_deferred = write_png('pixels_deferred.png', tf.tile(pixels_deferred, [1, 9, 1]))

    def get_pixel_gradients(pixels):

        def get_pixel_gradient(pixel_index):
            y = pixel_index // 3 // canvas_width
            x = pixel_index // 3 % canvas_width
            c = pixel_index % 3
            d_loss_by_pixels = tf.scatter_nd([[y, x, c]], [1.], shape=[canvas_height, canvas_width, 3])
            return tf.concat([
                tf.reshape(d_pixel_by_variables, [-1])
                for d_pixel_by_variables
                in tf.gradients(pixels, variables, d_loss_by_pixels)
            ], axis=0)
        d_pixels_by_variables = tf.reshape(
            tf.map_fn(get_pixel_gradient, tf.range(canvas_height * canvas_width * 3), dtype=tf.float32),
            [canvas_height, canvas_width, 3, -1]
        )  # indexed by y, x, r/g/b, variable

        return d_pixels_by_variables

    direct_gradients = get_pixel_gradients(pixels_direct)
    deferred_gradients = get_pixel_gradients(pixels_deferred)

    # Concatenate then normalise, to ensure direct and deferred gradients are treated identically
    all_gradients = tf.concat([direct_gradients, deferred_gradients], axis=0)
    all_gradients_normalised = all_gradients - tf.reduce_min(all_gradients, axis=[0, 1, 2], keepdims=True)
    all_gradients_normalised /= tf.reduce_max(all_gradients_normalised, axis=[0, 1, 2], keepdims=True)
    all_gradients_images = tf.reshape(tf.transpose(all_gradients_normalised, [0, 3, 1, 2]), [2, canvas_height, -1, 3])

    save_grads_direct = write_png('grads_direct.png', all_gradients_images[0])
    save_grads_deferred = write_png('grads_deferred.png', all_gradients_images[1])

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():

        tf.global_variables_initializer().run()

        session.run([save_pixels_direct, save_pixels_deferred, save_grads_direct, save_grads_deferred])


if __name__ == '__main__':
    main()


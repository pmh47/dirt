
import sys
import tensorflow as tf

import dirt
import dirt.matrices as matrices
import dirt.lighting as lighting

canvas_width, canvas_height = 32, 32
square_size = 4.


def write_png(filename, image):

    image = tf.cast(image * 255, tf.uint8)
    return tf.write_file(filename, tf.image.encode_png(image))


def get_transformed_geometry(translation, rotation, scale):

    # Build bent square in object space, on z = 0 plane
    vertices_object = tf.constant([[-1, -1, 0.], [-1, 1, 0], [1, 1, 0], [1, -1, -1.3]], dtype=tf.float32) * square_size / 2
    faces = [[0, 1, 2], [0, 2, 3]]

    # ** we should add an occluding triangle!
    # ** also a non-planar meeting-of-faces

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

    light_direction = tf.linalg.l2_normalize([1., -0.3, -0.5])
    diffuse_contribution = lighting.diffuse_directional(
        tf.reshape(normals, [-1, 3]),
        tf.reshape(colours, [-1, 3]),
        light_direction, light_color=tf.constant([0., 1., 0.]) * light_intensity, double_sided=True
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


def get_pixels_deferred_v1(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background):

    # This is a naive implementation of deferred shading, that gives incorrect gradients. See
    # get_pixels_deferred_v2 below for a correct implementation!

    gbuffer_mask = dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=tf.ones_like(transformed_vertices[:, :1]),
        background=tf.zeros([canvas_height, canvas_width, 1]),
        width=canvas_width, height=canvas_height, channels=1
    )[..., 0]
    background_value = -1.e4
    gbuffer_vertex_colours_world = dirt.rasterise(
        vertices=transformed_vertices,
        faces=faces,
        vertex_colors=vertex_colours,
        background=tf.ones([canvas_height, canvas_width, 3]) * background,
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


def get_pixels_deferred_v2(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background):

    vertex_attributes = tf.concat([tf.ones_like(transformed_vertices[:, :1]), vertex_colours, vertex_normals], axis=1)
    background_attributes = tf.zeros([canvas_height, canvas_width, 1 + 3 + 3])

    def shader_fn(gbuffer, light_intensity, background):
        mask = gbuffer[..., :1]
        colours = gbuffer[..., 1:4]
        normals = gbuffer[..., 4:7]
        pixels = mask * calculate_shading(colours, normals, light_intensity) + (1. - mask) * background
        return pixels

    pixels = dirt.rasterise_deferred(
        background_attributes,
        transformed_vertices,
        vertex_attributes,
        faces,
        shader_fn,
        [light_intensity, background]
    )

    return pixels


def prepare_gradient_images(deferred_gradients, direct_gradients):

    # Concatenate then normalise, to ensure direct and deferred gradients are treated identically
    # ** tidy up the mess of concats / transposes / reshapes here!
    all_gradients = tf.concat([direct_gradients, deferred_gradients], axis=0)
    all_gradients_normalised = all_gradients - tf.reduce_min(all_gradients, axis=[0, 1, 2], keepdims=True)
    all_gradients_normalised /= tf.reduce_max(all_gradients_normalised, axis=[0, 1, 2], keepdims=True)

    epsilon = 1.e-3
    all_gradients_signs = tf.stack([
        tf.cast(tf.greater(all_gradients[:, :, 1], epsilon), tf.float32),
        tf.cast(tf.less(all_gradients[:, :, 1], -epsilon), tf.float32),
        tf.zeros_like(all_gradients[:, :, 0])
    ], axis=2)

    all_gradients_images = tf.concat([
        tf.reshape(tf.transpose(all_gradients_normalised, [0, 3, 1, 2]), [2, canvas_height, -1, 3]),
        tf.reshape(tf.transpose(all_gradients_signs, [0, 3, 1, 2]), [2, canvas_height, -1, 3])
    ], axis=1)

    return all_gradients_images[0], all_gradients_images[1]


def main_graph():

    translation = tf.Variable([0., 0., 0.], name='translation')
    rotation = tf.Variable(0.5, name='rotation')
    scale = tf.Variable(1., name='scale')
    light_intensity = tf.Variable(0.6, name='light_intensity')
    background = tf.Variable([0., 0., 0.2], name='background')

    variables = [translation, rotation, scale, light_intensity, background]

    transformed_vertices, faces, vertex_normals, vertex_colours = get_transformed_geometry(translation, rotation, scale)

    pixels_direct = get_pixels_direct(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)
    save_pixels_direct = write_png('pixels_direct_graph.png', tf.tile(pixels_direct, [2, 9, 1]))

    pixels_deferred = get_pixels_deferred_v2(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)
    save_pixels_deferred = write_png('pixels_deferred_graph.png', tf.tile(pixels_deferred, [2, 9, 1]))

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

    direct_gradients_image, deferred_gradients_image = prepare_gradient_images(deferred_gradients, direct_gradients)

    save_grads_direct = write_png('grads_direct_graph.png', direct_gradients_image)
    save_grads_deferred = write_png('grads_deferred_graph.png', deferred_gradients_image)

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():

        tf.global_variables_initializer().run()

        session.run([save_pixels_direct, save_pixels_deferred, save_grads_direct, save_grads_deferred])


def main_eager():

    tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

    translation = tf.Variable([0., 0., 0.], name='translation')
    rotation = tf.Variable(0.5, name='rotation')
    scale = tf.Variable(1., name='scale')
    light_intensity = tf.Variable(0.6, name='light_intensity')
    background = tf.Variable([0., 0., 0.2], name='background')

    variables = [translation, rotation, scale, light_intensity, background]

    with tf.GradientTape(persistent=True) as tape:

        transformed_vertices, faces, vertex_normals, vertex_colours = get_transformed_geometry(translation, rotation, scale)

        pixels_direct = get_pixels_direct(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)
        pixels_deferred = get_pixels_deferred_v2(transformed_vertices, faces, vertex_normals, vertex_colours, light_intensity, background)

    write_png('pixels_direct_eager.png', tf.tile(pixels_direct, [2, 9, 1]))
    write_png('pixels_deferred_eager.png', tf.tile(pixels_deferred, [2, 9, 1]))

    direct_gradients = tape.jacobian(pixels_direct, variables, experimental_use_pfor=False)  # indexed by [variable], y, x, channel, then optionally variable-dimension
    deferred_gradients = tape.jacobian(pixels_deferred, variables, experimental_use_pfor=False)

    def rearrange_gradients(gradients):
        # Rearrange the output of GradientTape.jacobian to match that of get_pixel_gradients in main_graph
        return tf.concat([
            gradient if len(gradient.shape) == 4 else gradient[..., None]
            for gradient in gradients
        ], axis=3)
    direct_gradients_image, deferred_gradients_image = prepare_gradient_images(
        rearrange_gradients(deferred_gradients),
        rearrange_gradients(direct_gradients)
    )

    write_png('grads_direct_eager.png', direct_gradients_image)
    write_png('grads_deferred_eager.png', deferred_gradients_image)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('expected one argument, specifying graph or eager execution mode')
    elif sys.argv[1] == 'graph':
        main_graph()
    elif sys.argv[1] == 'eager':
        main_eager()
    else:
        print('invalid execution mode; should be graph or eager')


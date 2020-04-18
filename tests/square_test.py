
import numpy as np
import tensorflow as tf
import dirt

canvas_width, canvas_height = 128, 128
centre_x, centre_y = 32, 64
square_size = 16


def get_non_dirt_pixels():
    xs, ys = tf.meshgrid(tf.range(canvas_width), tf.range(canvas_height))
    xs = tf.cast(xs, tf.float32) + 0.5
    ys = tf.cast(ys, tf.float32) + 0.5
    x_in_range = tf.less_equal(tf.abs(xs - centre_x), square_size / 2)
    y_in_range = tf.less_equal(tf.abs(ys - centre_y), square_size / 2)
    return tf.cast(tf.logical_and(x_in_range, y_in_range), tf.float32)


def get_dirt_pixels():

    # Build square in screen space
    square_vertices = tf.constant([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=tf.float32) * square_size - square_size / 2.
    square_vertices += [centre_x, centre_y]

    # Transform to homogeneous coordinates in clip space
    square_vertices = square_vertices * 2. / [canvas_width, canvas_height] - 1.
    square_vertices = tf.concat([square_vertices, tf.zeros([4, 1]), tf.ones([4, 1])], axis=1)

    return dirt.rasterise(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 1]),
        background=tf.zeros([canvas_height, canvas_width, 1]),
        height=canvas_height, width=canvas_width, channels=1
    )[:, :, 0]


def main():

    if '.' in tf.__version__ and int(tf.__version__.split('.')[0]) < 2:

        session = tf.Session()
        with session.as_default():

            non_dirt_pixels = get_non_dirt_pixels().eval()
            dirt_pixels = get_dirt_pixels().eval()

    else:

        non_dirt_pixels = get_non_dirt_pixels().numpy()
        dirt_pixels = get_dirt_pixels().numpy()

    if np.all(non_dirt_pixels == dirt_pixels):
        print('successful: all pixels agree')
    else:
        print('failed: {} pixels disagree'.format(np.sum(non_dirt_pixels != dirt_pixels)))


if __name__ == '__main__':
    main()


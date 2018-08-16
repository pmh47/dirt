
import tensorflow as tf
import dirt


def make_pixels():

    square_vertices = tf.constant([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=tf.float32)
    square_vertices = tf.concat([square_vertices, tf.zeros([4, 1]), tf.ones([4, 1])], axis=1)

    return dirt.rasterise(
        vertices=square_vertices,
        faces=[[0, 1, 2], [0, 2, 3]],
        vertex_colors=tf.ones([4, 3]),
        background=tf.zeros([256, 256, 3]),
        height=256, width=256, channels=3
    )[:, :, 0]


def main():

    with tf.device('/gpu:0'):
        pixels_0 = make_pixels()
    with tf.device('/gpu:1'):
        pixels_1 = make_pixels()

    session = tf.Session()
    with session.as_default():
        session.run([pixels_0, pixels_1])


if __name__ == '__main__':
    main()


from __future__ import print_function
import numpy as np
import tensorflow as tf
import math
import cv2
import dirt.rasterise_ops
import dirt.lighting
import dirt.matrices


def make_cylinder(radius, height, end_offset, bevel, segments):

    # Cylinder centred on the origin, with axis along y-axis, and bevelled conical ends

    angles = np.linspace(0., 2 * math.pi, segments, endpoint=False, dtype=np.float32)
    xz = np.stack([np.cos(angles), np.sin(angles)], axis=1) * radius
    top_bevel_vertices = np.stack([xz[:, 0] * (1. - bevel), np.ones(segments) * -height / 2. - radius * bevel, xz[:, 1] * (1. - bevel)], axis=1)
    top_vertices = np.stack([xz[:, 0], np.ones(segments) * -height / 2., xz[:, 1]], axis=1)
    bottom_vertices = np.stack([xz[:, 0], np.ones(segments) * height / 2., xz[:, 1]], axis=1)
    bottom_bevel_vertices = np.stack([xz[:, 0] * (1. - bevel), np.ones(segments) * height / 2. + radius * bevel, xz[:, 1] * (1. - bevel)], axis=1)
    end_vertices = [[0., -height / 2. - end_offset, 0.], [0., height / 2. + end_offset, 0.]]
    all_vertices = np.concatenate([top_bevel_vertices, top_vertices, bottom_vertices, bottom_bevel_vertices, end_vertices], axis=0)

    faces = []
    def make_ring(start):
        for quad_index in range(segments):
            upper_first = start + quad_index
            upper_second = start + (quad_index + 1) % segments
            lower_first = start + quad_index + segments
            lower_second = start + (quad_index + 1) % segments + segments
            faces.extend([
                [upper_first, upper_second, lower_first],
                [lower_first, upper_second, lower_second]
            ])
    make_ring(0)
    make_ring(segments)
    make_ring(segments * 2)
    for top_first in range(segments):
        top_second = (top_first + 1) % segments
        bottom_first = top_first + segments * 3
        bottom_second = (bottom_first + 1) % segments
        faces.extend([
            [segments * 4, top_first, top_second],
            [segments * 4 + 1, bottom_first, bottom_second]
        ])

    return all_vertices, np.array(faces, dtype=np.int32)


def mesh():

    vertices, faces = make_cylinder(0.2, 0.75, 0.1, 0., 10)
    vertices = np.float32(np.concatenate([vertices, np.ones([len(vertices), 1])], axis=1))

    w = 48
    h = 36

    rotation_xy = tf.placeholder(tf.float32)
    view_matrix_1 = tf.convert_to_tensor([
        [0.5 * tf.cos(rotation_xy), 0.5 * -tf.sin(rotation_xy), 0., 0.],
        [0.5 * tf.sin(rotation_xy), 0.5 * tf.cos(rotation_xy), 0., 0.],
        [0., 0., 0.5, 0.],
        [0., 0., 0., 1.]
    ])

    translation = tf.placeholder(tf.float32)

    view_matrix_2 = tf.stack([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        tf.concat([translation, [1.]], axis=0)
    ])

    if True:  # use splitting
        vertex_count = len(faces) * 3
        vertices, faces = dirt.lighting.split_vertices_by_face(vertices, faces)
    else:
        vertex_count = len(vertices)

    projection_matrix = dirt.matrices.perspective_projection(0.1, 20., 0.2, float(h) / w)
    projected_vertices = tf.matmul(tf.matmul(tf.matmul(vertices, view_matrix_1), view_matrix_2), projection_matrix)

    bgcolor = tf.placeholder(tf.float32, [3])
    vertex_color = tf.placeholder(tf.float32, [3])
    vertex_colors = tf.concat([tf.tile(vertex_color[np.newaxis, :], [75, 1]), np.random.uniform(size=[vertex_count - 75, 3])], axis=0)

    im = dirt.rasterise_ops.rasterise(tf.concat([tf.tile(bgcolor[np.newaxis, np.newaxis, :], [h // 2, w, 1]), tf.ones([h // 2, w, 3])], axis=0), projected_vertices, vertex_colors, faces, height=h, width=w, channels=3)
    ims = dirt.rasterise_ops.rasterise_batch(tf.tile(tf.constant([[0., 0., 0.], [0., 0., 1.]])[:, np.newaxis, np.newaxis, :], [1, h, w, 1]), tf.tile(projected_vertices[np.newaxis, ...], [2, 1, 1]), np.random.uniform(size=[2, vertex_count, 3]), tf.tile(faces[np.newaxis, ...], [2, 1, 1]), height=h, width=w, channels=3)

    d_loss_by_pixels = tf.placeholder(tf.float32, [h, w, 3])
    [gt, gr, gb, gc] = tf.gradients(im, [translation, rotation_xy, bgcolor, vertex_color], d_loss_by_pixels)

    ds_loss_by_pixels = tf.placeholder(tf.float32, [2, h, w, 3])
    [gst, gsr] = tf.gradients(ims, [translation, rotation_xy], ds_loss_by_pixels)

    session = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    with session.as_default():
        tf.global_variables_initializer().run()

        gx_im = np.empty([h, w, 3], dtype=np.float32)
        gy_im = np.empty([h, w, 3], dtype=np.float32)
        gz_im = np.empty([h, w, 3], dtype=np.float32)
        gr_im = np.empty([h, w, 3], dtype=np.float32)
        gb_im = np.empty([h, w, 3], dtype=np.float32)
        gc_im = np.empty([h, w, 3], dtype=np.float32)

        for y in range(h):
            for x in range(w):
                for c in range(3):
                    pixel_indicator = np.zeros([h, w, 3], dtype=np.float32)
                    pixel_indicator[y, x, c] = 1
                    [[gx_im[y, x, c], gy_im[y, x, c], gz_im[y, x, c]], gr_im[y, x, c], [gb_im[y, x, c], _, _], [gc_im[y, x, c], _, _]] = \
                        session.run([gt, gr, gb, gc], {d_loss_by_pixels: pixel_indicator, translation: [0., 0., -0.25], rotation_xy: 0., bgcolor: [0.4, 0.2, 0.2], vertex_color: [0.7, 0.3, 0.6]})
                print('.', end='')
            print()

        gsx_im = np.empty([2, h, w, 3], dtype=np.float32)
        gsy_im = np.empty([2, h, w, 3], dtype=np.float32)
        gsz_im = np.empty([2, h, w, 3], dtype=np.float32)
        gsr_im = np.empty([2, h, w, 3], dtype=np.float32)

        for iib in range(2):
            print(iib + 1)
            for y in range(h):
                for x in range(w):
                    for c in range(3):
                        pixel_indicator = np.zeros([2, h, w, 3], dtype=np.float32)
                        pixel_indicator[iib, y, x, c] = 1
                        [[gsx_im[iib, y, x, c], gsy_im[iib, y, x, c], gsz_im[iib, y, x, c]], gsr_im[iib, y, x, c]] = session.run([gst, gsr], {ds_loss_by_pixels: pixel_indicator, translation: [0., 0., -1.], rotation_xy: 0.5})
                    print('.', end='')
                print()

        cv2.imshow('im', im.eval({translation: [0., 0., -0.25], rotation_xy: 0., bgcolor: [0.6, 0.2, 0.2], vertex_color: [0.7, 0.3, 0.6]}))
        cv2.imshow('ims', np.concatenate(ims.eval({translation: [0., 0., -0.25], rotation_xy: 0.5}), axis=1))

        g_im = np.concatenate([gx_im, gy_im, gz_im, gr_im * 3., gb_im * 30., gc_im * 50.], axis=1)
        g_im = (g_im - np.min(g_im)) / (np.max(g_im) - np.min(g_im))
        cv2.imshow('grad', g_im)

        gs_im = np.concatenate(np.concatenate([gsx_im, gsy_im, gsz_im, gsr_im * 3.], axis=2), axis=0)
        gs_im = (gs_im - np.min(gs_im)) / (np.max(gs_im) - np.min(gs_im))
        cv2.imshow('grads', gs_im)

        cv2.waitKey(0)


if __name__ == '__main__':
    mesh()


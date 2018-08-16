
import numpy as np
import tensorflow as tf
import math
import cv2
import dirt.rasterise_ops
import dirt.matrices
import dirt.lighting

from rasterise_tests import make_cylinder


def main():

    w, h = 256, 192

    vertices, faces = make_cylinder(0.2, 0.75, 0.1, 0.2, 32)
    vertices = np.float32(np.concatenate([vertices, np.ones([len(vertices), 1])], axis=1))

    rotation_xy = tf.placeholder(tf.float32, [])
    rotation_matrix = tf.convert_to_tensor([
        [0.5 * tf.cos(rotation_xy), 0.5 * -tf.sin(rotation_xy), 0., 0.],
        [0.5 * tf.sin(rotation_xy), 0.5 * tf.cos(rotation_xy), 0., 0.],
        [0., 0., 0.5, 0.],
        [0., 0., 0., 1.]
    ])

    translation = tf.placeholder(tf.float32, [3])
    translation_matrix = tf.stack([
        [1., 0., 0., 0.],
        [0., 1., 0., 0.],
        [0., 0., 1., 0.],
        tf.concat([translation, [1.]], axis=0)
    ])

    transformed_vertices = tf.matmul(tf.matmul(vertices, rotation_matrix), translation_matrix)
    vertex_normals = dirt.lighting.vertex_normals(transformed_vertices[:, :3], faces)

    transformed_vertices_split, faces_split = dirt.lighting.split_vertices_by_face(transformed_vertices, faces)
    vertex_normals_split = dirt.lighting.vertex_normals_pre_split(transformed_vertices_split[:, :3], faces_split)

    projection_matrix = dirt.matrices.perspective_projection(0.1, 20., 0.2, float(h) / w)
    projected_vertices = tf.matmul(transformed_vertices, projection_matrix)
    projected_vertices_split = tf.matmul(transformed_vertices_split, projection_matrix)

    normal_im = dirt.rasterise_ops.rasterise(background=tf.zeros([h, w, 3]), vertices=projected_vertices, vertex_colors=tf.abs(vertex_normals), faces=faces, height=h, width=w, channels=3)

    # ** the following two need to be careful about transformations: they use 'raw' normals and vertices, which means
    # ** the lights are effectively transformed along with the geometry -- or equivalently, the lights are in object space
    directional_im = dirt.rasterise_ops.rasterise(background=tf.zeros([h, w, 3]), vertices=projected_vertices, vertex_colors=tf.cast(dirt.lighting.diffuse_directional(vertex_normals, tf.ones([vertices.shape[0], 3]), [1., 0, 0], [1., 1., 0.], False) + [0, 0, 0.4], tf.float32), faces=faces, height=h, width=w, channels=3)
    point_im = dirt.rasterise_ops.rasterise(background=tf.zeros([h, w, 3]), vertices=projected_vertices, vertex_colors=tf.cast(dirt.lighting.diffuse_point(transformed_vertices[:, :3], vertex_normals, tf.ones([vertices.shape[0], 3]), [0.5, -1., 0.5], [1., 0.5, 0.9], False) + [0, 0, 0.4], tf.float32), faces=faces, height=h, width=w, channels=3)
    point_im_split = dirt.rasterise_ops.rasterise(background=tf.zeros([h, w, 3]), vertices=projected_vertices_split, vertex_colors=tf.cast(dirt.lighting.diffuse_point(transformed_vertices_split[:, :3], vertex_normals_split, tf.ones([transformed_vertices_split.shape[0], 3]), [0.5, -1., 0.5], [1., 0.5, 0.9], False) + [0, 0, 0.4], tf.float32), faces=faces_split, height=h, width=w, channels=3)

    session = tf.Session()
    with session.as_default():

        normal_im_ = normal_im.eval({translation: [0., 0., -0.25], rotation_xy: 0.})
        directional_im_ = directional_im.eval({translation: [0., 0., -0.25], rotation_xy: 0.})
        point_im_ = point_im.eval({translation: [0., 0., -0.25], rotation_xy: 0.})
        point_im_split_ = point_im_split.eval({translation: [0., 0., -0.25], rotation_xy: 0.})

        cv2.imshow('normals', normal_im_[:, :, (2, 1, 0)])
        cv2.imshow('directional', directional_im_[:, :, (2, 1, 0)])
        cv2.imshow('point', point_im_[:, :, (2, 1, 0)])
        cv2.imshow('point_split', point_im_split_[:, :, (2, 1, 0)])
        cv2.waitKey(0)


if __name__ == '__main__':
    main()


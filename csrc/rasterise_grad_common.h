
#ifndef RASTERISE_GRAD_COMMON_H
#define RASTERISE_GRAD_COMMON_H

struct Vertex
{
    // This is the 'expanded' vertex format used when rendering for gradient calculation
    float position[4];
    float barycentric[2];
    int indices[3];
};

void launch_vertex_upload(
    tensorflow::TTypes<Vertex, 2>::Tensor &buffer,
    tensorflow::Tensor const &vertices, tensorflow::Tensor const &faces,
    Eigen::GpuDevice const &device
);

void launch_grad_assembly(
    tensorflow::Tensor &grad_vertices, tensorflow::Tensor &grad_vertex_colors, tensorflow::Tensor &grad_background, tensorflow::Tensor &debug_thingy,
    cudaArray_t const &barycentrics_and_depth_array, cudaArray_t const &indices_array,
    tensorflow::Tensor const &pixels, tensorflow::Tensor const &grad_pixels, tensorflow::Tensor const &vertices,
    int const buffer_width, int const buffer_height, Eigen::GpuDevice const &device
);

#endif // RASTERISE_GRAD_COMMON_H

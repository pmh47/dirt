
#define GOOGLE_CUDA 1
#define EIGEN_USE_GPU

#include <tensorflow/core/framework/tensor.h>
#include "tf_cuda_utils.h"

#include "rasterise_grad_common.h"

using namespace tensorflow;

__global__ void upload_vertices(
    TTypes<Vertex, 2>::Tensor buffer,
    TTypes<float, 3>::ConstTensor const vertices, TTypes<int, 3>::ConstTensor const faces,
    dim3 const total_threads
) {
    CUDA_AXIS_KERNEL_LOOP(iib, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(face_index, total_threads, y) {
            for (int vertex_in_face = 0; vertex_in_face < 3; ++vertex_in_face) {
                auto &dest_vertex = buffer(iib, face_index * 3 + vertex_in_face);
                auto const &src_vertex_index = faces(iib, face_index, vertex_in_face);
                dest_vertex.position[0] = vertices(iib, src_vertex_index, 0);  // ** it'd be nice to do this as a 16-byte block, but unfortunately Vertex is not 16-byte aligned
                dest_vertex.position[1] = vertices(iib, src_vertex_index, 1);
                dest_vertex.position[2] = vertices(iib, src_vertex_index, 2);
                dest_vertex.position[3] = vertices(iib, src_vertex_index, 3);
                dest_vertex.barycentric[0] = vertex_in_face == 0 ? 1.f : 0.f;
                dest_vertex.barycentric[1] = vertex_in_face == 1 ? 1.f : 0.f;
                dest_vertex.indices[0] = faces(iib, face_index, 0);
                dest_vertex.indices[1] = faces(iib, face_index, 1);
                dest_vertex.indices[2] = faces(iib, face_index, 2);
            }
        }
    }
}

void launch_vertex_upload(
    TTypes<Vertex, 2>::Tensor &buffer,  // use of Tensor here is just for indexing convenience: it's backed by the GL vertex-buffer
    Tensor const &vertices, Tensor const &faces,
    Eigen::GpuDevice const &device
) {
    if (faces.dim_size(0) == 0 || faces.dim_size(1) == 0)
        return;  // no-op in this case; the following assumes >0
    auto const config = GetCuda2DLaunchConfig(faces.dim_size(0), faces.dim_size(1), device);
    upload_vertices<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        buffer,
        vertices.tensor<float, 3>(), faces.tensor<int, 3>(),
        config.virtual_thread_count
    );
}

namespace {
    struct Vec3 {
        float x, y, z;

        __device__ Vec3(float const x_, float const y_, float const z_) : x(x_), y(y_), z(z_) {}

        __device__ static Vec3 from_xyz(float4 const &f4) {
            return Vec3(f4.x, f4.y, f4.z);
        }

        __device__ float operator [](int const i) const {
            switch (i) {
                case 0: return x;
                case 1: return y;
                case 2: return z;
                default: return std::numeric_limits<float>::quiet_NaN();
            };
        }

        __device__ Vec3 operator +(Vec3 const &other) const {
            return Vec3(x + other.x, y + other.y, z + other.z);
        }

        __device__ Vec3 operator -(Vec3 const &other) const {
            return Vec3(x - other.x, y - other.y, z - other.z);
        }

        __device__ Vec3 operator *(float const &other) const {
            return Vec3(x * other, y * other, z * other);
        }

        __device__ float L1() const {
            return std::abs(x) + std::abs(y) + std::abs(z);
        }

        __device__ bool operator !=(Vec3 const &other) const {
            // Note these are exact float comparisons with zero tolerance!
            return x != other.x || y != other.y || z != other.z;
        }
    };
}

__global__ void assemble_grads(
    TTypes<float, 3>::Tensor grad_vertices, TTypes<float, 3>::Tensor grad_vertex_colors, TTypes<float, 4>::Tensor grad_background, TTypes<float, 4>::Tensor debug_thingy,
    cudaSurfaceObject_t const barycentrics_and_depth_surface, cudaSurfaceObject_t const indices_surface,
    TTypes<float, 4>::ConstTensor const pixels, TTypes<float, 4>::ConstTensor const grad_pixels, TTypes<float, 3>::ConstTensor const vertices,
    int const frames_per_row,
    dim3 const total_threads
) {
    auto const batch_size = static_cast<int>(grad_pixels.dimension(0));
    auto const frame_height = static_cast<int>(grad_pixels.dimension(1));
    auto const frame_width = static_cast<int>(grad_pixels.dimension(2));
    auto const channels = static_cast<int>(grad_pixels.dimension(3));

    CUDA_AXIS_KERNEL_LOOP(buffer_x, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(buffer_y, total_threads, y) {
            auto const iib = buffer_y / frame_height * frames_per_row + buffer_x / frame_width;
            if (iib < batch_size) {

                auto const x_in_frame = buffer_x % frame_width;
                auto const y_in_frame = frame_height - 1 - buffer_y % frame_height;  // the vertical flip is because our images are top-row-first, as in tensorflow

                auto const at = [&] (int const offset_x, int const offset_y) {
                    // This returns the nearest edge pixel for out-of-bounds accesses
                    auto const unclipped_x = x_in_frame + offset_x;
                    auto const unclipped_y = y_in_frame - offset_y;  // the negation here is again due to vertical-flipping of our pixels
                    auto const clipped_x = max(0, min(frame_width - 1, unclipped_x));
                    auto const clipped_y = max(0, min(frame_height - 1, unclipped_y));
                    return Vec3(
                        pixels(iib, clipped_y, clipped_x, 0),
                        pixels(iib, clipped_y, clipped_x, 1),
                        pixels(iib, clipped_y, clipped_x, 2)
                    );
                };
                // Note that the following filters are negative-offset minus positive-offset!
                auto const scharr_x = (at(-1, -1) + at(-1, +1) - at(+1, -1) - at(+1, +1)) * (3.f / 32.f) + (at(-1, 0) - at(+1, 0)) * (10.f / 32.f);
                auto const scharr_y = (at(-1, -1) + at(+1, -1) - at(-1, +1) - at(+1, +1)) * (3.f / 32.f) + (at(0, -1) - at(0, +1)) * (10.f / 32.f);

                auto const barycentric_and_depth = surf2Dread<float4>(barycentrics_and_depth_surface, buffer_x * 16, buffer_y);
                auto barycentric = Vec3::from_xyz(barycentric_and_depth);
                auto clip_w = barycentric_and_depth.w;  // this will be infinity if we're not over a fragment (i.e. iff barycentric == index_f == 1.f)
                auto index_f = Vec3::from_xyz(surf2Dread<float4>(indices_surface, buffer_x * 16, buffer_y));

                // Accumulate colour gradients; see notes p37-38
                if (barycentric.x != -1.f) {
                    for (int index_in_primitive = 0; index_in_primitive < 3; ++index_in_primitive) {
                        int const vertex_index = static_cast<int>(index_f[index_in_primitive]);
                        for (int channel = 0; channel < channels; ++channel) {
                            auto const color_grad = grad_pixels(iib, y_in_frame, x_in_frame, channel) * barycentric[index_in_primitive];
                            atomicAdd(&grad_vertex_colors(iib, vertex_index, channel), color_grad);
                        }
                    }
                } else {
                    for (int channel = 0; channel < channels; ++channel) {
                        auto const grad_pixel = grad_pixels(iib, y_in_frame, x_in_frame, channel);
                        grad_background(iib, y_in_frame, x_in_frame, channel) = grad_pixel;  // no need to accumulate, as each background-pixel maps to at most one output-pixel
                    }
                }

                debug_thingy(iib, y_in_frame, x_in_frame, 1) = grad_pixels(iib, y_in_frame, x_in_frame, 1);
                debug_thingy(iib, y_in_frame, x_in_frame, 2) = grad_pixels(iib, y_in_frame, x_in_frame, 2);

                // Dilate edges of occluders at occlusion boundaries, so we add gradients through pixels just outside triangles
                // to the occluder, not to the occludee (see notes p55-59); also applies when the current pixel is background
                if (x_in_frame > 0 && y_in_frame > 0 && x_in_frame < frame_width - 1 && y_in_frame < frame_height - 1) {

                    bool dilated = false;

                    auto const dilate_from_offset = [&] (int2 const offset) {

                        auto const index_f_at_offset = Vec3::from_xyz(surf2Dread<float4>(indices_surface, (buffer_x + offset.x) * 16, buffer_y + offset.y));
                        auto const barycentric_and_depth_at_offset = surf2Dread<float4>(barycentrics_and_depth_surface, (buffer_x + offset.x) * 16, buffer_y + offset.y);
                        auto const clip_w_at_offset = barycentric_and_depth_at_offset.w;

                        if (index_f_at_offset.x != -1.f && index_f_at_offset != index_f && clip_w > clip_w_at_offset) {
                            // The adjacent pixel is over a triangle, and that triangle is not the same as ours, and the adjacent pixel is
                            // nearer the camera than us -- hence, the adjacent triangle should dilate into our pixel
                            barycentric = Vec3::from_xyz(barycentric_and_depth_at_offset);
                            index_f = index_f_at_offset;
                            clip_w = clip_w_at_offset;
                            dilated = true;
                            debug_thingy(iib, y_in_frame, x_in_frame, 0) = 1.e-2f;
                        }
                    };

                    // ** this doesn't handle the case of one-pixel-wide faces correctly -- *both* the adjacent faces should be dilated into the
                    // ** same pixel, which requires a structural change (as currently each pixel only passes gradients to vertices of one triangle)

                    // ** also, we should consider diagonal neighbours, i.e. points over the occludee that are diagonally-adjacent to a pixel of
                    // ** the occluder, as the 3x3 support of the scharr filter implies these will pick up incorrect gradients too

                    // ** perhaps best to do a loop, considering dilation from preferred orthogonal direction, then other, then diagonals, stopping
                    // ** if one results in a dilation (or, ideally, summing)

                    auto offset_direction = scharr_x.L1() > scharr_y.L1() ? int2{1, 0} : int2{0, 1};
                    if ((x_in_frame + y_in_frame) % 2 == 1) {
                        // Dither the preferred direction of offset to reduce bias
                        offset_direction.x = -offset_direction.x;
                        offset_direction.y = -offset_direction.y;
                    }
                    dilate_from_offset(offset_direction);
                    if (!dilated)
                        dilate_from_offset({-offset_direction.x, -offset_direction.y});
                }

                if (barycentric.x != -1.f) {

                    // Accumulate position gradients; see notes p25-27, 32-35, 65-66

                    auto const width_f = static_cast<float>(frame_width);
                    auto const height_f = static_cast<float>(frame_height);

                    float dL_dx = 0.f, dL_dy = 0.f;  // 'dx' being physical/pixel/fragment x-position, not that of any particular vertex
                    for (int channel = 0; channel < channels; ++channel) {
                        auto const dL_dchannel = grad_pixels(iib, y_in_frame, x_in_frame, channel);
                        dL_dx += dL_dchannel * scharr_x[channel];
                        dL_dy += dL_dchannel * scharr_y[channel];
                    }

                    float clip_x = 0.f, clip_y = 0.f;  // of the fragment
                    for (int index_in_primitive = 0; index_in_primitive < 3; ++index_in_primitive) {
                        int const vertex_index = static_cast<int>(index_f[index_in_primitive]);
                        clip_x += barycentric[index_in_primitive] * vertices(iib, vertex_index, 0);
                        clip_y += barycentric[index_in_primitive] * vertices(iib, vertex_index, 1);
                    }

                    for (int index_in_primitive = 0; index_in_primitive < 3; ++index_in_primitive) {

                        auto const d_xview_by_xclip = .5f * width_f / clip_w;
                        auto const d_yview_by_yclip = .5f * height_f / clip_w;
                        auto const d_xview_by_wclip = -.5f * width_f * clip_x / (clip_w * clip_w);
                        auto const d_yview_by_wclip = -.5f * height_f * clip_y / (clip_w * clip_w);

                        auto const dL_dxview_times_dclip_dvertex = dL_dx * barycentric[index_in_primitive];  // barycentric can be seen as d_clip / d_vertex, and is logically the final step mapping the change in fragment clip-location to vertex clip-locations
                        auto const dL_dyview_times_dclip_dvertex = dL_dy * barycentric[index_in_primitive];

                        int const vertex_index = static_cast<int>(index_f[index_in_primitive]);
                        atomicAdd(&grad_vertices(iib, vertex_index, 0), dL_dxview_times_dclip_dvertex * d_xview_by_xclip);
                        atomicAdd(&grad_vertices(iib, vertex_index, 1), dL_dyview_times_dclip_dvertex * d_yview_by_yclip);
                        atomicAdd(&grad_vertices(iib, vertex_index, 3), dL_dxview_times_dclip_dvertex * d_xview_by_wclip + dL_dyview_times_dclip_dvertex * d_yview_by_wclip);
                    }
                }
            }
        }
    }
}

void launch_grad_assembly(
    Tensor &grad_vertices, Tensor &grad_vertex_colors, Tensor &grad_background, Tensor &debug_thingy,
    cudaArray_t const &barycentrics_and_depth_array, cudaArray_t const &indices_array,
    Tensor const &pixels, Tensor const &grad_pixels, Tensor const &vertices,
    int const buffer_width, int const buffer_height, Eigen::GpuDevice const &device
) {
    if (
        cudaMemsetAsync(grad_vertices.tensor<float, 3>().data(), 0, sizeof(float) * grad_vertices.NumElements(), device.stream()) ||
        cudaMemsetAsync(grad_vertex_colors.tensor<float, 3>().data(), 0, sizeof(float) * grad_vertex_colors.NumElements(), device.stream()) ||
        cudaMemsetAsync(grad_background.tensor<float, 4>().data(), 0, sizeof(float) * grad_background.NumElements(), device.stream()) ||
        cudaMemsetAsync(debug_thingy.tensor<float, 4>().data(), 0, sizeof(float) * debug_thingy.NumElements(), device.stream())
    )
        LOG(FATAL) << "one or more calls to cudaMemsetAsync failed";

    // ** these would almost certainly be better passed as texture-objects, due to the neighbour-access in scharr-filtering & edge-dilating
    cudaResourceDesc barycentrics_and_depth_descriptor, indices_descriptor;
    barycentrics_and_depth_descriptor.resType = indices_descriptor.resType = cudaResourceTypeArray;
    barycentrics_and_depth_descriptor.res.array.array = barycentrics_and_depth_array;
    indices_descriptor.res.array.array = indices_array;
    cudaSurfaceObject_t barycentrics_and_depth_surface, indices_surface;
    if (
        cudaCreateSurfaceObject(&barycentrics_and_depth_surface, &barycentrics_and_depth_descriptor) ||
        cudaCreateSurfaceObject(&indices_surface, &indices_descriptor)
    )
        LOG(FATAL) << "one or more calls to cudaCreateSurfaceObject failed";

    auto const assembly_config = GetCuda2DLaunchConfig(buffer_width, buffer_height, device);
    assemble_grads<<<assembly_config.block_count, assembly_config.thread_per_block, 0, device.stream()>>>(
        grad_vertices.tensor<float, 3>(), grad_vertex_colors.tensor<float, 3>(), grad_background.tensor<float, 4>(), debug_thingy.tensor<float, 4>(),
        barycentrics_and_depth_surface, indices_surface,
        pixels.tensor<float, 4>(), grad_pixels.tensor<float, 4>(), vertices.tensor<float, 3>(),
        buffer_width / grad_pixels.dim_size(2),
        assembly_config.virtual_thread_count
    );

    if (
        cudaDestroySurfaceObject(barycentrics_and_depth_surface) ||
        cudaDestroySurfaceObject(indices_surface)
    )
        LOG(FATAL) << "one or more calls to cudaDestroySurfaceObject failed";
}

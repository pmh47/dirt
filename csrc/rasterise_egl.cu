
#define GOOGLE_CUDA 1
#define EIGEN_USE_GPU

#include <tensorflow/core/framework/tensor.h>
#include "tf_cuda_utils.h"

using namespace tensorflow;

__global__ void upload_background(cudaSurfaceObject_t dest_surface, TTypes<float, 4>::ConstTensor const src_tensor, int const frames_per_row, dim3 const total_threads)
{
    auto const batch_size = src_tensor.dimension(0);
    auto const frame_height = src_tensor.dimension(1);
    auto const frame_width = src_tensor.dimension(2);
    auto const channels = src_tensor.dimension(3);

    CUDA_AXIS_KERNEL_LOOP(dest_x, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(dest_y, total_threads, y) {
            auto const iib = dest_y / frame_height * frames_per_row + dest_x / frame_width;
            if (iib < batch_size) {

                auto const x_in_frame = dest_x % frame_width;
                auto const y_in_frame = frame_height - 1 - dest_y % frame_height;  // the vertical flip ensures that our images are top-row-first, as in tensorflow
                if (channels == 1) {
                    auto const &value = src_tensor(iib, y_in_frame, x_in_frame, 0);
                    surf2Dwrite(float4{value, value, value, 1.f}, dest_surface, dest_x * 16, dest_y);  // *16 is required because surface-writes use byte addressing (!)
                } else if (channels == 3) {
                    surf2Dwrite(float4{
                        src_tensor(iib, y_in_frame, x_in_frame, 0),
                        src_tensor(iib, y_in_frame, x_in_frame, 1),
                        src_tensor(iib, y_in_frame, x_in_frame, 2),
                        1.f,
                    }, dest_surface, dest_x * 16, dest_y);
                }
            }
        }
    }
}

void launch_background_upload(
    cudaArray_t &dest_array, Tensor const &src_tensor,
    int const dest_height, int const dest_width,
    Eigen::GpuDevice const &device
) {
    cudaResourceDesc dest_resource_descriptor;
    dest_resource_descriptor.resType = cudaResourceTypeArray;
    dest_resource_descriptor.res.array.array = dest_array;
    cudaSurfaceObject_t dest_surface;
    if (auto const err = cudaCreateSurfaceObject(&dest_surface, &dest_resource_descriptor))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    auto const config = GetCuda2DLaunchConfig(dest_width, dest_height, device);
    auto const src = src_tensor.tensor<float, 4>();
    upload_background<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        dest_surface,
        src,
        dest_width / src_tensor.dim_size(2),
        config.virtual_thread_count
    );

    if (auto const err = cudaDestroySurfaceObject(dest_surface))
        LOG(FATAL) << "cudaDestroySurfaceObject failed: " << cudaGetErrorName(err);
}

__global__ void download_pixels(TTypes<float, 4>::Tensor pixels, cudaSurfaceObject_t const src_surface, int const frames_per_row, dim3 const total_threads)
{
    auto const batch_size = pixels.dimension(0);
    auto const frame_height = pixels.dimension(1);
    auto const frame_width = pixels.dimension(2);
    auto const channels = pixels.dimension(3);

    CUDA_AXIS_KERNEL_LOOP(src_x, total_threads, x) {
        CUDA_AXIS_KERNEL_LOOP(src_y, total_threads, y) {
            auto const iib = src_y / frame_height * frames_per_row + src_x / frame_width;
            if (iib < batch_size) {

                auto const pixel = surf2Dread<float4>(src_surface, src_x * 16, src_y);  // *16 is required because surface-loads use byte addressing (!)

                auto const x_in_frame = src_x % frame_width;
                auto const y_in_frame = frame_height - 1 - src_y % frame_height;  // the vertical flip ensures that our images are top-row-first, as in tensorflow
                if (channels == 1) {
                    pixels(iib, y_in_frame, x_in_frame, 0) = pixel.x;
                } else if (channels == 3) {
                    pixels(iib, y_in_frame, x_in_frame, 0) = pixel.x;
                    pixels(iib, y_in_frame, x_in_frame, 1) = pixel.y;
                    pixels(iib, y_in_frame, x_in_frame, 2) = pixel.z;
                }
            }
        }
    }
}

void launch_pixels_download(
    Tensor &dest_tensor, cudaArray_t const &src_array,
    int const src_height, int const src_width,
    Eigen::GpuDevice const &device
) {
    cudaResourceDesc src_resource_descriptor;
    src_resource_descriptor.resType = cudaResourceTypeArray;
    src_resource_descriptor.res.array.array = src_array;
    cudaSurfaceObject_t src_surface;
    if (auto const err = cudaCreateSurfaceObject(&src_surface, &src_resource_descriptor))
        LOG(FATAL) << "cudaCreateSurfaceObject failed: " << cudaGetErrorName(err);

    auto const config = GetCuda2DLaunchConfig(src_width, src_height, device);
    auto dest = dest_tensor.tensor<float, 4>();
    download_pixels<<<config.block_count, config.thread_per_block, 0, device.stream()>>>(
        dest,
        src_surface,
        src_width / dest_tensor.dim_size(2),
        config.virtual_thread_count
    );

    if (auto const err = cudaDestroySurfaceObject(src_surface))
        LOG(FATAL) << "cudaDestroySurfaceObject failed: " << cudaGetErrorName(err);
}

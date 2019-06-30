
#ifdef USE_GPU_LAUNCH_CONFIG_H
    #include <tensorflow/core/util/gpu_launch_config.h>
    #define CudaLaunchConfig2D GpuLaunchConfig2D
#else
    #include <tensorflow/core/util/cuda_launch_config.h>
#endif

// If tensorflow is too old, this does not exist; if tensorflow is too new, it has an incompatible definition
#define CUDA_AXIS_KERNEL_LOOP(i, n, axis)                                  \
  for (int i = blockIdx.axis * blockDim.axis + threadIdx.axis; i < n.axis; \
       i += blockDim.axis * gridDim.axis)


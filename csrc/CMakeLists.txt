cmake_minimum_required(VERSION 3.10)  # 3.8 gives us built-in CUDA support; 3.10 gives us OpenGL::EGL

project(dirt LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED COMPONENTS OpenGL EGL)

# Search for EGL; nvidia drivers ship the library but not headers, so we redistribute those
find_path(EGL_INCLUDE_DIR NAMES EGL/egl.h PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../external REQUIRED)
find_library(EGL_LIBRARIES NAMES egl EGL REQUIRED)

# Search for cuda headers (using the form of path that tensorflow includes them with), based on cmake-inferred nvcc, or $CUDA_HOME
get_filename_component(NVCC_DIR ${CMAKE_CUDA_COMPILER} DIRECTORY)
find_path(CUDA_INCLUDE_DIR NAMES cuda/include/cuda.h HINTS ${NVCC_DIR}/../.. PATHS ENV CUDA_HOME REQUIRED)

# Ask tensorflow for its compile flags; one should therefore make sure cmake is run with the venv active that the op will be used in
execute_process(COMMAND python -c "import tensorflow; print(' '.join(tensorflow.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE Tensorflow_DEFAULT_COMPILE_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
separate_arguments(Tensorflow_DEFAULT_COMPILE_FLAGS UNIX_COMMAND "${Tensorflow_DEFAULT_COMPILE_FLAGS}")
set(Tensorflow_COMPILE_FLAGS "${Tensorflow_DEFAULT_COMPILE_FLAGS}" CACHE PATH "Tensorflow compile flags")

# Ask tensorflow for its include path; check if cuda_launch_config exists where it should
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE Tensorflow_include_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT EXISTS "${Tensorflow_include_dir}/tensorflow/core/util/cuda_launch_config.h")
  if(EXISTS "${Tensorflow_include_dir}/tensorflow/core/util/gpu_launch_config.h")
    add_definitions(-DUSE_GPU_LAUNCH_CONFIG_H)
  else()
    message(FATAL_ERROR "cannot find either cuda_launch_config.h or gpu_launch_config.h")
  endif()
endif()

# Ask tensorflow for the necessary linker flags
execute_process(COMMAND python -c "import tensorflow; print(' '.join(tensorflow.sysconfig.get_link_flags()))" OUTPUT_VARIABLE Tensorflow_DEFAULT_LINK_FLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)
set(Tensorflow_LINK_FLAGS "${Tensorflow_DEFAULT_LINK_FLAGS}" CACHE STRING "Tensorflow linker flags for custom ops")

# in the following, we need ../external/tensorflow for cuda_config.h in tf versions with #16959 unfixed
include_directories(SYSTEM ../external/tensorflow ${NSYNC_INCLUDE_DIR} ${CUDA_INCLUDE_DIR} ${EGL_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR})
include_directories(${Tensorflow_INCLUDE_DIRS} ${Tensorflow_INCLUDE_DIRS}/external/nsync/public)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math")

# in the following, NDEBUG is set to avoid issues with constexpr in absl's string_view.h
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_30 --expt-relaxed-constexpr -DNDEBUG")

add_library(
    rasterise SHARED
    rasterise_egl.cpp rasterise_egl.cu
    rasterise_grad_egl.cpp rasterise_grad_egl.cu rasterise_grad_common.h
    shaders.cpp shaders.h
    gl_dispatcher.h concurrentqueue.h blockingconcurrentqueue.h gl_common.h tf_cuda_utils.h hwc.h
)

target_compile_features(rasterise PUBLIC cxx_std_11)
target_compile_options(rasterise PUBLIC ${Tensorflow_COMPILE_FLAGS})
target_link_libraries(rasterise OpenGL::EGL OpenGL::OpenGL ${Tensorflow_LINK_FLAGS})

# Put the compiled library in the python package folder, rather than whatever build folder is being used
set_target_properties(
    rasterise PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../dirt
)

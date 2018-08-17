
#include <chrono>

#define GL_GLEXT_PROTOTYPES

#include <GL/gl.h>
#include <GL/glext.h>

#undef Status
#undef Success
#undef None

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/platform/stream_executor.h"

// This form of include path matches what the TensorFlow headers use
#include <cuda/include/cuda.h>
#include <cuda/include/cuda_runtime_api.h>
#include <cuda/include/cuda_gl_interop.h>

#define EIGEN_USE_GPU
#include "unsupported/Eigen/CXX11/src/Tensor/TensorDeviceCuda.h"

#include "hwc.h"
#include "gl_dispatcher.h"
#include "gl_common.h"

using namespace tensorflow;

REGISTER_OP("Rasterise")
    .Attr("height: int")
    .Attr("width: int")
    .Attr("channels: int = 3")
    .Input("background: float32")
    .Input("vertices: float32")
    .Input("vertex_colors: float32")
    .Input("faces: int32")
    .Output("pixels: float32")
    .SetShapeFn( [] (::tensorflow::shape_inference::InferenceContext *c) {
        int height, width, channels;
        c->GetAttr("height", &height);
        c->GetAttr("width", &width);
        c->GetAttr("channels", &channels);
        shape_inference::ShapeHandle vertices_shape;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 3, &vertices_shape));
        auto const batch_dim = c->Dim(vertices_shape, 0);  // ** for now, we assume dim0 of vertices defines the batch size -- really should merge from all inputs
        c->set_output(0, c->MakeShape({batch_dim, height, width, channels}));
        return Status::OK();
    } );

struct RegisteredBuffer
{
    // This is used for buffers that are owned by opengl but copied to from cuda; the gl buffer must be expanded to
    // fit the cuda data as required, and re-registered with cuda if so

    GLuint gl_index;

private:
    std::size_t buffer_size;
    cudaGraphicsResource_t cuda_resource;

public:

    RegisteredBuffer() :
        gl_index(0), buffer_size(0), cuda_resource(nullptr)
    {
    }

    void ensure_size(std::size_t const &required_size)
    {
        assert(gl_index != 0);
        if (buffer_size >= required_size)
            return;
        if (buffer_size != 0) {
            // The buffer has previously been initialised and registered, so we should unregister it before reallocating
            // ** is this re-registration actually necessary? is just re-mapping sufficient?
            cudaGraphicsUnregisterResource(cuda_resource);
        }
        glNamedBufferData(gl_index, required_size, nullptr, GL_DYNAMIC_COPY);
        if (auto const err = glGetError())
            LOG(FATAL) << "glNamedBufferData failed: " << err;
        buffer_size = required_size;
        if (auto const err = cudaGraphicsGLRegisterBuffer(&cuda_resource, gl_index, cudaGraphicsRegisterFlagsWriteDiscard))
            LOG(FATAL) << "cudaGraphicsGLRegisterBuffer failed: " << cudaGetErrorName(err);
    }

    void fill_from(Tensor const &tensor, cudaStream_t const &cuda_stream, perftools::gputools::Stream *const exec_stream)
    {
        auto const total_bytes = static_cast<std::size_t>(sizeof(float) * tensor.NumElements());
        ensure_size(total_bytes);
        if (auto const err = cudaGraphicsMapResources(1, &cuda_resource, cuda_stream))
            LOG(FATAL) << "cudaGraphicsMapResources failed: " << cudaGetErrorName(err);
        void *gl_ptr;
        std::size_t mapped_size;
        if (auto const err = cudaGraphicsResourceGetMappedPointer(&gl_ptr, &mapped_size, cuda_resource))
            LOG(FATAL) << "cudaGraphicsResourceGetMappedPointer failed: " << cudaGetErrorName(err);
        assert(mapped_size >= total_bytes);
        auto gl_devicememory = perftools::gputools::DeviceMemoryBase(gl_ptr, mapped_size);
        auto const cuda_devicememory = perftools::gputools::DeviceMemoryBase(
            const_cast<char *>(tensor.tensor_data().data()),
            total_bytes
        );
        exec_stream->ThenMemcpyD2D(&gl_devicememory, cuda_devicememory, total_bytes);
        if (auto const err = cudaGraphicsUnmapResources(1, &cuda_resource, cuda_stream))  // this synchronises the stream with future graphics calls, so the memcpy is guaranteed to happen before any rendering
            LOG(FATAL) << "cudaGraphicsUnmapResources failed: " << cudaGetErrorName(err);
    }
};

void launch_background_upload(
    cudaArray_t &dest_array, Tensor const &src_tensor,
    int const dest_height, int const dest_width,
    Eigen::GpuDevice const &device
);

void launch_pixels_download(
    Tensor &dest_tensor, cudaArray_t const &src_array,
    int const src_height, int const src_width,
    Eigen::GpuDevice const &device
);

class RasteriseOpGpu : public OpKernel
{
    struct PerThreadObjects
    {
        PerThreadObjects() : cuda_context(nullptr)
        {
        }

        PerThreadObjects(PerThreadObjects const &) = delete;
        PerThreadObjects &operator =(PerThreadObjects const &) = delete;

        int frame_height, frame_width, channels;
        int buffer_height, buffer_width;  // these are some multiple of frame_*, depending on batch size
        GLuint framebuffer, depth_buffer;
        RegisteredBuffer vertexbuffer, colourbuffer;
        RegisteredBuffer elementbuffer;
        GLuint program;
        GLuint pixels_texture;
        cudaGraphicsResource_t pixels_resource;  // ** can we use RegisteredTexture from the grad code instead?
        CUcontext cuda_context;  // this is nullptr iff the thread-objects have not yet been initialised
    };

    HWC hwc;

    static GlDispatcher<PerThreadObjects> &get_gl_dispatcher()
    {
        static GlDispatcher<PerThreadObjects> gl_dispatcher;
        return gl_dispatcher;
    }

    static void initialise_per_thread_objects(PerThreadObjects &objects, HWC const &hwc, CUcontext const &cuda_context)
    {
        // This is called directly during the first kernel-invocation when each thread is used. cuda_context refers
        // to the context active in the dispatching thread (but not yet active in our thread)

        if (objects.cuda_context != nullptr)
            LOG(FATAL) << "attempted to reinitialise PerThreadObjects already bound to cuda context " << objects.cuda_context;

        if (auto const err = cuCtxSetCurrent(cuda_context))
            LOG(FATAL) << "cuCtxSetCurrent failed: " << err;
        objects.cuda_context = cuda_context;

        objects.frame_height = hwc.height;
        objects.frame_width = hwc.width;
        objects.channels = hwc.channels;

        CUdevice active_cuda_device;
        if (auto const err = cuCtxGetDevice(&active_cuda_device))
            LOG(FATAL) << "cudaGetDevice failed: " << err;
        gl_common::initialise_context(active_cuda_device);

        objects.buffer_height = objects.buffer_width = 0;
        objects.framebuffer = objects.pixels_texture = objects.depth_buffer = 0;
        objects.pixels_resource = nullptr;

        // ** The shaders we initialise here would preferably be global, not per-thread. However, that requires
        // ** fancier synchronisation to ensure everything is set up (and stored somewhere) before use

        // Load and compile the vertex and fragment shaders
        GLuint const tri_vertex_shader = gl_common::create_shader(shaders::forward_vertex);
        GLuint const tri_fragment_shader = gl_common::create_shader(shaders::forward_fragment);

        // Link the vertex & fragment shaders
        objects.program = glCreateProgram();
        glAttachShader(objects.program, tri_vertex_shader);
        glAttachShader(objects.program, tri_fragment_shader);
        glLinkProgram(objects.program);
        gl_common::print_log(glGetProgramInfoLog, glGetProgramiv, GL_LINK_STATUS, objects.program, "program");
        glUseProgram(objects.program);

        glGenBuffers(1, &objects.vertexbuffer.gl_index);
        glGenBuffers(1, &objects.colourbuffer.gl_index);

        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, objects.vertexbuffer.gl_index);
        glVertexAttribPointer(
            0,                  // attribute 0
            4,                  // size
            GL_FLOAT,           // type
            GL_FALSE,           // normalized?
            0,                  // stride
            (void*)0            // array buffer offset
        );
        glEnableVertexAttribArray(1);
        glBindBuffer(GL_ARRAY_BUFFER, objects.colourbuffer.gl_index);
        glVertexAttribPointer(1, objects.channels, GL_FLOAT, GL_FALSE, 0, (void *) 0);

        glGenBuffers(1, &objects.elementbuffer.gl_index);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, objects.elementbuffer.gl_index);

        glEnable(GL_DEPTH_TEST);
        glEnable(GL_SCISSOR_TEST);
    }

    static void reinitialise_framebuffer(PerThreadObjects &objects)
    {
        assert(objects.buffer_width != 0 && objects.buffer_height != 0);

        // ** unclear that there's any benefit to building everything before assigning anything, as we just abort if one fails...

        // Create the framebuffer, and map the color-attachments to fragment shader outputs
        GLuint framebuffer;
        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
        GLenum draw_buffers[] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, draw_buffers);  // map fragment-shader output locations to framebuffer attachments
        if (auto err = glGetError())
            LOG(FATAL) << "framebuffer creation failed: " << err;

        // Set up an rgba texture for pixels
        GLuint pixels_texture;
        glGenTextures(1, &pixels_texture);
        glBindTexture(GL_TEXTURE_2D, pixels_texture);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, objects.buffer_width, objects.buffer_height);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, pixels_texture, 0);
        if (auto err = glGetError())
            LOG(FATAL) << "pixel buffer initialisation failed: " << err;

        // Set up a depth buffer
        GLuint depth_buffer;
        glGenRenderbuffers(1, &depth_buffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, objects.buffer_width, objects.buffer_height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
        if (auto err = glGetError())
            LOG(FATAL) << "depth buffer initialisation failed: " << err;

        // Register the pixels texture as a cuda graphics resource
        cudaGraphicsResource_t pixels_resource;
        if (auto const err = cudaGraphicsGLRegisterImage(&pixels_resource, pixels_texture, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore))
            LOG(FATAL) << "cuGraphicsGLRegisterImage failed: " << cudaGetErrorName(err);

        // Delete previous buffers, and store new ones in thread-objects
        if (objects.pixels_resource)
            cudaGraphicsUnregisterResource(objects.pixels_resource);
        glDeleteFramebuffers(1, &objects.framebuffer);
        glDeleteTextures(1, &objects.pixels_texture);
        glDeleteRenderbuffers(1, &objects.depth_buffer);
        objects.framebuffer = framebuffer;
        objects.pixels_texture = pixels_texture;
        objects.depth_buffer = depth_buffer;
        objects.pixels_resource = pixels_resource;

        LOG(INFO) << "reinitialised framebuffer with size " << objects.buffer_width << " x " << objects.buffer_height;
    }

public:

    explicit RasteriseOpGpu(OpKernelConstruction* context) :
        OpKernel(context), hwc(get_hwc(context))
    {
    }

    void Compute(OpKernelContext* context) override
    {
        CUcontext current_context;
        if (auto const err = cuCtxGetCurrent(&current_context))
            LOG(FATAL) << "cuCtxGetCurrent failed: " << err;

        get_gl_dispatcher().dispatch(hwc, [&](PerThreadObjects &objects) {

#ifdef TIME_SECTIONS
            auto start = std::chrono::high_resolution_clock::now();
#endif

            // If this is the first call on this GL thread, activate the cuda context and initialise opengl
            if (!objects.cuda_context) {
                // We haven't yet bound this thread to a cuda context nor initialised the gl context
                initialise_per_thread_objects(objects, hwc, current_context);
            }

            // Sanity-check that the cuda context is set as expected
            CUcontext thread_context;
            cuCtxGetCurrent(&thread_context);
            assert(thread_context == objects.cuda_context);  // this should always be true, as nothing apart from us affects either of these

            // Check input tensors and get references to data

            Tensor const &background_tensor = context->input(0);
            OP_REQUIRES(context, background_tensor.shape().dims() == 4 && background_tensor.shape().dim_size(1) == objects.frame_height && background_tensor.shape().dim_size(2) == objects.frame_width  && background_tensor.shape().dim_size(3) == objects.channels, errors::InvalidArgument("Rasterise expects background_tensor to be 4D, and bgcolor.shape == [None, height, width, channels]"));

            Tensor const &vertices_tensor = context->input(1);
            OP_REQUIRES(context, vertices_tensor.shape().dims() == 3 && vertices_tensor.shape().dim_size(2) == 4, errors::InvalidArgument("Rasterise expects vertices to be 3D, and vertices.shape[2] == 4"));
            auto const vertex_count = static_cast<int>(vertices_tensor.shape().dim_size(1));

            Tensor const &vertex_colors_tensor = context->input(2);
            OP_REQUIRES(context, vertex_colors_tensor.shape().dims() == 3 && vertex_colors_tensor.shape().dim_size(1) == vertices_tensor.shape().dim_size(1) && vertex_colors_tensor.shape().dim_size(2) == objects.channels, errors::InvalidArgument("Rasterise expects vertex_colors to be 3D, and vertex_colors.shape == [None, vertices.shape[1], channels]"));

            Tensor const &faces_tensor = context->input(3);
            OP_REQUIRES(context, faces_tensor.shape().dims() == 3 && faces_tensor.shape().dim_size(2) == 3, errors::InvalidArgument("Rasterise expects faces to be 3D, and faces.shape[2] == 3"));

            // ** would be nice to relax the following to allow mixture of batch-size and singleton dim0's, and to broadcast the latter without re-copying data to the gpu
            int const batch_size = static_cast<int>(vertices_tensor.shape().dim_size(0));
            OP_REQUIRES(context, background_tensor.shape().dim_size(0) == batch_size && vertex_colors_tensor.shape().dim_size(0) == batch_size && faces_tensor.shape().dim_size(0) == batch_size, errors::InvalidArgument("Rasterise expects all arguments to have same leading (batch) dimension"));

            Tensor *pixels_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{batch_size, objects.frame_height, objects.frame_width, objects.channels}, &pixels_tensor));

#ifdef TIME_SECTIONS
            auto const time_A = std::chrono::high_resolution_clock::now();
#endif

            // Check the framebuffer is large enough for our batch; expand it if not
            int const maximum_batch_size = (objects.buffer_width / objects.frame_width) * (objects.buffer_height / objects.frame_height);
            if (batch_size > maximum_batch_size) {
                int const horizontal_count = static_cast<int>(std::sqrt(static_cast<float>(batch_size)) + .1f);
                int const vertical_count = batch_size / horizontal_count + (batch_size % horizontal_count == 0 ? 0 : 1);
                objects.buffer_width = objects.frame_width * horizontal_count;
                objects.buffer_height = objects.frame_height * vertical_count;
                reinitialise_framebuffer(objects);
            }
            int const frames_per_row = objects.buffer_width / objects.frame_width;

            // Get handles to the stream we're running on, to synchronise opengl and cuda
            // ** we assume that the cuda stream underlying exec_stream is the same as that named by device.stream()
            auto const &device = context->eigen_device<Eigen::GpuDevice>();
            auto const exec_stream = context->op_device_context()->stream();  // this is the StreamExecutor stream, not the raw cuda stream device.stream()
            OP_REQUIRES(context, exec_stream, errors::Internal("No GPU stream available"));

            // Copy the input tensors into gl-managed buffers, expanding the latter if required. The unmap calls synchronise
            // subsequent graphics with the stream, so we're guaranteed to 'see' this data when rendering
            objects.vertexbuffer.fill_from(vertices_tensor, device.stream(), exec_stream);
            objects.colourbuffer.fill_from(vertex_colors_tensor, device.stream(), exec_stream);
            objects.elementbuffer.fill_from(faces_tensor, device.stream(), exec_stream);

            // Map the pixel buffer and copy the background across
            if (auto const err = cudaGraphicsMapResources(1, &objects.pixels_resource, device.stream()))
                LOG(FATAL) << "cudaGraphicsMapResources failed: " << cudaGetErrorName(err);
            cudaArray_t tiled_pixels_array;
            if (auto const err = cudaGraphicsSubResourceGetMappedArray(&tiled_pixels_array, objects.pixels_resource, 0, 0))
                LOG(FATAL) << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorName(err);
            launch_background_upload(tiled_pixels_array, background_tensor, objects.buffer_height, objects.buffer_width, device);
            if (auto const err = cudaGraphicsUnmapResources(1, &objects.pixels_resource, device.stream()))  // synchronise the stream with the graphics pipeline
                LOG(FATAL) << "cudaGraphicsUnmapResources failed" << cudaGetErrorName(err);

#ifdef TIME_SECTIONS
            auto const time_B = std::chrono::high_resolution_clock::now();
#endif

            for (int index_in_batch = 0; index_in_batch < batch_size; ++index_in_batch) {

                // Find where this frame in the batch should be rendered; set the viewport transform (so geometry
                // ends up in the right place) and scissor region (so clears/etc. don't escape)
                int const frame_x = (index_in_batch % frames_per_row) * objects.frame_width;
                int const frame_y = (index_in_batch / frames_per_row) * objects.frame_height;
                glViewport(frame_x, frame_y, objects.frame_width, objects.frame_height);
                glScissor(frame_x, frame_y, objects.frame_width, objects.frame_height);

                glClear(GL_DEPTH_BUFFER_BIT);

                glDrawElementsBaseVertex(
                    GL_TRIANGLES,
                    static_cast<int>(faces_tensor.NumElements() / batch_size),
                    GL_UNSIGNED_INT,
                    (void *)(index_in_batch * faces_tensor.NumElements() / batch_size * sizeof(int32)),
                    index_in_batch * vertex_count
                );
            }

#ifdef TIME_SECTIONS
            auto const time_C1 = std::chrono::high_resolution_clock::now();
            glTextureBarrier();
            auto const time_C2 = std::chrono::high_resolution_clock::now();
#endif

            // Map the pixel buffer to a cuda array, and dispatch the cuda kernel to copy/rearrange pixels into the output tensor
            // Note that this needs to map / unmap every time, unlike vertex-buffer, as opengl modifies the pixels, which isn't allowed when mapped
            if (auto const err = cudaGraphicsMapResources(1, &objects.pixels_resource, device.stream()))  // this also synchronises pending graphics calls with our stream
                LOG(FATAL) << "cudaGraphicsMapResources failed: " << cudaGetErrorName(err);
            if (auto const err = cudaGraphicsSubResourceGetMappedArray(&tiled_pixels_array, objects.pixels_resource, 0, 0))
                LOG(FATAL) << "cudaGraphicsSubResourceGetMappedArray failed: " << cudaGetErrorName(err);
            launch_pixels_download(*pixels_tensor, tiled_pixels_array, objects.buffer_height, objects.buffer_width, device);
            if (auto const err = cudaGraphicsUnmapResources(1, &objects.pixels_resource, device.stream()))
                LOG(FATAL) << "cudaGraphicsUnmapResources failed" << cudaGetErrorName(err);

#ifdef TIME_SECTIONS
            auto const time_D = std::chrono::high_resolution_clock::now();
            std::cout << "A " << std::chrono::duration<double>(time_A - start).count() << std::endl;
            std::cout << "B " << std::chrono::duration<double>(time_B - time_A).count() << std::endl;
            std::cout << "C1 " << std::chrono::duration<double>(time_C1 - time_B).count() << std::endl;
            std::cout << "C2 " << std::chrono::duration<double>(time_C2 - time_C1).count() << std::endl;
            std::cout << "D " << std::chrono::duration<double>(time_D - time_C2).count() << std::endl;
#endif
        } );
    }
};

REGISTER_KERNEL_BUILDER(Name("Rasterise").Device(DEVICE_GPU), RasteriseOpGpu);


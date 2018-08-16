
#ifndef GL_DISPATCHER_H
#define GL_DISPATCHER_H

#include <thread>
#include <functional>
#include <map>
#include <vector>
#include <memory>

#include "blockingconcurrentqueue.h"
#include "hwc.h"

template<class PerThreadObjects>
class GlDispatcher final
{
    // One singleton instance of this class should be created for each OpKernel class. PerThreadObjects is a type
    // containing handles to all GL objects for a thread; these are passed back to all operations dispatched on
    // the relevant thread

public:

    typedef std::function<void (PerThreadObjects &)> Action;  // represents a block of code that needs to be run with thread-objects provided

private:

    typedef std::pair<CUcontext, HWC> ThreadKey;
    typedef moodycamel::BlockingConcurrentQueue<std::function<bool ()>> OperationQueue;  // operations return true to terminate the thread

    struct GlThread
    {
        std::unique_ptr<OperationQueue> operation_queue;
        std::unique_ptr<PerThreadObjects> objects;
        std::unique_ptr<std::thread> thread;

        GlThread() :
            operation_queue(new OperationQueue()),
            objects(new PerThreadObjects())
        {
            // The second copy of operation_queue is needed as *this may have been moved-from
            // before the lambda is run on the thread
            auto const &operation_queue_local = this->operation_queue.get();
            thread.reset(new std::thread([operation_queue_local] () { thread_fn(*operation_queue_local); }));
        }

        GlThread(GlThread &&) noexcept = default;
        GlThread(GlThread const &) = delete;
        GlThread &operator =(GlThread const &) = delete;

        ~GlThread()
        {
            if (thread && operation_queue) {  // i.e. if we have not been moved-from
                operation_queue->enqueue([&]() { return true; });
                thread->join();
            }
        }

        void dispatch_blocking(Action const &action)
        {
            moodycamel::details::mpmc_sema::LightweightSemaphore semaphore;  // ** would be nicer to have just one persistent semaphore per queue; however, dispatch_blocking may be called concurrently from different threads, so we would need to synchronise access to the semaphore!
            operation_queue->enqueue([&] () {
                action(*objects);
                semaphore.signal();
                return false;
            });
            semaphore.wait();
        }

        static void thread_fn(OperationQueue &operation_queue)
        {
            // This waits on the given queue for operations and executes them, terminating if an action returns true
            for ( ; ; ) {
                std::function<bool ()> action;
                operation_queue.wait_dequeue(action);
                if ( action() )
                    break;
            }
        }
    };

    std::map<ThreadKey, GlThread> key_to_thread;
    std::mutex mutex;  // this synchronises access to the above map, but does not synchronise actual operations!

public:

    GlDispatcher() = default;

    GlDispatcher(GlDispatcher const &) = delete;
    GlDispatcher &operator =(GlDispatcher const &) = delete;
    GlDispatcher(GlDispatcher &&) = delete;
    GlDispatcher &operator =(GlDispatcher &&) = delete;

    void dispatch(HWC const &hwc, Action const &action)
    {
        auto &gl_thread = get_gl_thread(hwc);
        gl_thread.dispatch_blocking(action);
    }

private:

    GlThread &get_gl_thread(HWC const &hwc)
    {
        CUcontext cuda_context;
        if (auto const err = cuCtxGetCurrent(&cuda_context))
            LOG(FATAL) << "cuCtxGetCurrent failed: " << err;
        std::lock_guard<std::mutex> lock(mutex);
        return key_to_thread[{cuda_context, hwc}];
    }
};

#endif //GL_DISPATCHER_H

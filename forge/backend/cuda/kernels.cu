// Forge CUDA kernel library (Milestone 8; matmul re-tiled in Milestone 11).
//
// A small, deliberately narrow set of real CUDA kernels: device memory
// management, elementwise add/sub/mul, a shared-memory-tiled 2D matmul (see
// the "matmul" section below for why it is tiled rather than naive), and a
// full-array sum reduction. Compiled by `build.py` into a single shared
// library (`extern "C"` exports only, so Python can call it via `ctypes`
// without any C++ ABI dependency) and loaded by `forge/backend/cuda/backend.py`.
//
// Compute Capability 5.0 (the verified 940MX) has no native `atomicAdd` for
// `double`, so `atomic_add_f64` below is the standard CAS-based emulation.

#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>
#include <cfloat>

// -- device memory management ---------------------------------------------

extern "C" {

__declspec(dllexport) int cf_device_count() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) return -1;
    return count;
}

__declspec(dllexport) const char* cf_error_string(int code) {
    return cudaGetErrorString(static_cast<cudaError_t>(code));
}

__declspec(dllexport) int cf_malloc(void** out_ptr, size_t nbytes) {
    return static_cast<int>(cudaMalloc(out_ptr, nbytes));
}

__declspec(dllexport) int cf_free(void* ptr) {
    return static_cast<int>(cudaFree(ptr));
}

__declspec(dllexport) int cf_memcpy_h2d(void* dst, const void* src, size_t nbytes) {
    return static_cast<int>(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
}

__declspec(dllexport) int cf_memcpy_d2h(void* dst, const void* src, size_t nbytes) {
    return static_cast<int>(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
}

__declspec(dllexport) int cf_memcpy_d2d(void* dst, const void* src, size_t nbytes) {
    return static_cast<int>(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
}

__declspec(dllexport) int cf_synchronize() {
    return static_cast<int>(cudaDeviceSynchronize());
}

// -- pinned (page-locked) host memory and async transfers (Milestone 29) --------
//
// `cf_host_alloc`/`cf_host_free` are real `cudaHostAlloc`/`cudaFreeHost` calls
// -- genuine page-locked host memory, never ordinary NumPy/malloc memory
// wearing a "pinned" label (see `forge/backend/cuda/pinned.py`).
// `cudaHostAllocDefault` (portable, non-mapped, non-write-combined) is the
// simplest correct choice for Forge's use case: a plain page-locked buffer
// `cudaMemcpyAsync` can transfer without staging. `cf_memcpy_h2d_async`/
// `cf_memcpy_d2h_async` are real `cudaMemcpyAsync` calls, each taking an
// explicit `cudaStream_t` (`NULL` meaning CUDA's default stream, the same
// convention every kernel-launcher below already uses) -- never a
// synchronous `cudaMemcpy` in disguise, and never followed by an implicit
// `cudaDeviceSynchronize()` here (that decision belongs to the caller, see
// `CUDABackend.from_array_async`/`to_numpy_async`).

__declspec(dllexport) int cf_host_alloc(void** out_ptr, size_t nbytes) {
    return static_cast<int>(cudaHostAlloc(out_ptr, nbytes, cudaHostAllocDefault));
}

__declspec(dllexport) int cf_host_free(void* ptr) {
    return static_cast<int>(cudaFreeHost(ptr));
}

__declspec(dllexport) int cf_memcpy_h2d_async(void* dst, const void* src, size_t nbytes, void* stream) {
    return static_cast<int>(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream)));
}

__declspec(dllexport) int cf_memcpy_d2h_async(void* dst, const void* src, size_t nbytes, void* stream) {
    return static_cast<int>(
        cudaMemcpyAsync(dst, src, nbytes, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream)));
}

// -- CUDA streams and events (Milestone 27; cross-stream waits in Milestone 28) --
//
// Real `cudaStream_t`/`cudaEvent_t` handles, exported as opaque `void*` so
// `ctypes` never needs to know their actual (opaque) struct layout -- the
// same "raw pointer in, raw pointer out" convention `cf_malloc`/`cf_free`
// already use for device memory. `cf_stream_create`/`cf_stream_destroy`/
// `cf_stream_synchronize`/`cf_stream_wait_event` back
// `forge.backend.cuda.stream.CUDAStream`, the only public stream abstraction
// (`forge.cuda.Stream`). `cf_event_*` are used only *internally* -- by the
// caching allocator (`forge.backend.cuda.allocator`) to know when a block
// released on a non-default stream is safe to reuse, and, as of Milestone
// 28, by `CUDABackend._stream_guard` to establish a GPU-side cross-stream
// dependency -- no public CUDA event API is exposed (see the Milestone 27
// brief's explicit scope limit, reaffirmed in Milestone 28's brief Section
// 7: a public `Event` API is optional, not required for correctness).
// `cudaEventDisableTiming` is passed at creation since Forge's internal
// events are used only to test/wait for completion, never to measure
// elapsed time -- a small, standard CUDA optimization for exactly this use.

__declspec(dllexport) int cf_stream_create(void** out_stream) {
    return static_cast<int>(cudaStreamCreate(reinterpret_cast<cudaStream_t*>(out_stream)));
}

__declspec(dllexport) int cf_stream_destroy(void* stream) {
    return static_cast<int>(cudaStreamDestroy(static_cast<cudaStream_t>(stream)));
}

__declspec(dllexport) int cf_stream_synchronize(void* stream) {
    return static_cast<int>(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
}

// Milestone 28: the GPU-side (never host-blocking) cross-stream dependency
// primitive. `stream=NULL` means CUDA's default (null) stream -- valid: a
// caller may make the default stream wait on an event recorded on an
// explicit stream (Section 14 of the milestone brief, "explicit -> default").
// The trailing `0` is the required-but-unused `flags` parameter (CUDA
// currently defines no flags for this call).
__declspec(dllexport) int cf_stream_wait_event(void* stream, void* event) {
    return static_cast<int>(cudaStreamWaitEvent(
        static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0));
}

__declspec(dllexport) int cf_event_create(void** out_event) {
    return static_cast<int>(cudaEventCreateWithFlags(
        reinterpret_cast<cudaEvent_t*>(out_event), cudaEventDisableTiming));
}

__declspec(dllexport) int cf_event_record(void* event, void* stream) {
    return static_cast<int>(cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
}

__declspec(dllexport) int cf_event_query(void* event) {
    return static_cast<int>(cudaEventQuery(static_cast<cudaEvent_t>(event)));
}

__declspec(dllexport) int cf_event_synchronize(void* event) {
    return static_cast<int>(cudaEventSynchronize(static_cast<cudaEvent_t>(event)));
}

__declspec(dllexport) int cf_event_destroy(void* event) {
    return static_cast<int>(cudaEventDestroy(static_cast<cudaEvent_t>(event)));
}

// -- profiling-only timed events (Milestone 31) ----------------------------
//
// `cf_event_create` above is created with `cudaEventDisableTiming` -- right
// for its hot allocator/cross-stream-dependency use (`stream.py`'s
// `CUDAEvent`), but unable to answer "how long did this GPU work take"
// (`cudaEventElapsedTime` requires a timing-*enabled* event). These two
// functions are the one place Forge creates such an event, used only by
// `forge/backend/cuda/profiling_events.py` / `benchmarks/pipeline_profile.py`
// -- never by any hot-path code, so ordinary training never pays for them
// (Section 5/34/39 of the milestone brief: real GPU-side timing via CUDA
// events rather than `time.perf_counter()`, kept optional and outside the
// core runtime).
__declspec(dllexport) int cf_event_create_timed(void** out_event) {
    return static_cast<int>(cudaEventCreate(reinterpret_cast<cudaEvent_t*>(out_event)));
}

__declspec(dllexport) int cf_event_elapsed_ms(void* start, void* end, float* out_ms) {
    return static_cast<int>(cudaEventElapsedTime(
        out_ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(end)));
}

} // extern "C"

// -- elementwise kernels ----------------------------------------------------

template <typename T>
__global__ void k_add(const T* a, const T* b, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

template <typename T>
__global__ void k_sub(const T* a, const T* b, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] - b[i];
}

template <typename T>
__global__ void k_mul(const T* a, const T* b, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = a[i] * b[i];
}

static void launch_config(long long n, int& blocks, int& threads) {
    threads = 256;
    long long b = (n + threads - 1) / threads;
    blocks = b < 1 ? 1 : static_cast<int>(b);
}

#define ELEMENTWISE_LAUNCHER(NAME, KERNEL, TYPE, SUFFIX)                                   \
    extern "C" __declspec(dllexport) int NAME##_##SUFFIX(                                  \
        const TYPE* a, const TYPE* b, TYPE* out, long long n, void* stream) {              \
        int blocks, threads;                                                               \
        launch_config(n, blocks, threads);                                                 \
        KERNEL<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(a, b, out, n);          \
        return static_cast<int>(cudaGetLastError());                                       \
    }

ELEMENTWISE_LAUNCHER(cf_add, k_add, float, f32)
ELEMENTWISE_LAUNCHER(cf_add, k_add, double, f64)
ELEMENTWISE_LAUNCHER(cf_sub, k_sub, float, f32)
ELEMENTWISE_LAUNCHER(cf_sub, k_sub, double, f64)
ELEMENTWISE_LAUNCHER(cf_mul, k_mul, float, f32)
ELEMENTWISE_LAUNCHER(cf_mul, k_mul, double, f64)

// -- row-broadcast elementwise kernels (Milestone 9) ---------------------------
//
// M8's elementwise add/sub/mul required exact-matching shapes -- no CUDA
// broadcasting at all. That is too narrow to run the existing `nn.Linear`
// forward (`x @ weight + bias`) on a *batched* input: the matmul result is
// (batch, out_features), the bias is (out_features,). Rather than adding
// general N-dimensional broadcasting (out of scope) or special-casing
// `Linear` (explicitly disallowed by the milestone brief) or copying to CPU
// to broadcast there (a disguised CPU fallback, also disallowed), this adds
// exactly the one broadcast shape Linear actually needs: a 2D "matrix"
// (rows, cols) combined elementwise with a 1D "vector" (cols,), the vector
// value reused for every row. `vec_is_left` preserves operand order for
// non-commutative `sub`. Every other shape mismatch still raises `CUDAError`
// (see `CUDABackend._elementwise`) -- this is a targeted addition, not
// general CUDA broadcasting support.

template <typename T>
__global__ void k_add_bcast(const T* mat, const T* vec, T* out, long long rows, long long cols, int vec_is_left) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = rows * cols;
    if (i < total) {
        long long col = i % cols;
        out[i] = mat[i] + vec[col];
    }
    (void)vec_is_left;  // addition is commutative; operand order does not matter
}

template <typename T>
__global__ void k_sub_bcast(const T* mat, const T* vec, T* out, long long rows, long long cols, int vec_is_left) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = rows * cols;
    if (i < total) {
        long long col = i % cols;
        out[i] = vec_is_left ? (vec[col] - mat[i]) : (mat[i] - vec[col]);
    }
}

template <typename T>
__global__ void k_mul_bcast(const T* mat, const T* vec, T* out, long long rows, long long cols, int vec_is_left) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = rows * cols;
    if (i < total) {
        long long col = i % cols;
        out[i] = mat[i] * vec[col];
    }
    (void)vec_is_left;  // multiplication is commutative; operand order does not matter
}

#define BCAST_LAUNCHER(NAME, KERNEL, TYPE, SUFFIX)                                        \
    extern "C" __declspec(dllexport) int NAME##_bcast_##SUFFIX(                           \
        const TYPE* mat, const TYPE* vec, TYPE* out,                                      \
        long long rows, long long cols, int vec_is_left, void* stream) {                  \
        long long total = rows * cols;                                                    \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        KERNEL<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(mat, vec, out, rows, cols, vec_is_left); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

BCAST_LAUNCHER(cf_add, k_add_bcast, float, f32)
BCAST_LAUNCHER(cf_add, k_add_bcast, double, f64)
BCAST_LAUNCHER(cf_sub, k_sub_bcast, float, f32)
BCAST_LAUNCHER(cf_sub, k_sub_bcast, double, f64)
BCAST_LAUNCHER(cf_mul, k_mul_bcast, float, f32)
BCAST_LAUNCHER(cf_mul, k_mul_bcast, double, f64)

// -- column-broadcast subtract kernel (Milestone 14) ---------------------------
//
// CrossEntropyLoss's log-sum-exp shift (`logits - max_axis1(logits)`) and its
// log-probability step (`shifted - log_sum_exp`) both combine a (rows, cols)
// matrix with a (rows, 1) per-row scalar, broadcasting that scalar across
// every column of its own row -- the transpose of the row-broadcast case
// above (a (cols,) vector broadcast down every row, for Linear's bias add).
// Only `sub` is ever combined with a (rows, 1) operand anywhere in Forge, so
// only a subtract kernel is added (no column-broadcast add/mul).

template <typename T>
__global__ void k_sub_colbcast(const T* mat, const T* colvec, T* out, long long rows, long long cols, int vec_is_left) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = rows * cols;
    if (i < total) {
        long long row = i / cols;
        out[i] = vec_is_left ? (colvec[row] - mat[i]) : (mat[i] - colvec[row]);
    }
}

#define COLBCAST_LAUNCHER(NAME, KERNEL, TYPE, SUFFIX)                                    \
    extern "C" __declspec(dllexport) int NAME##_colbcast_##SUFFIX(                       \
        const TYPE* mat, const TYPE* colvec, TYPE* out,                                  \
        long long rows, long long cols, int vec_is_left, void* stream) {                 \
        long long total = rows * cols;                                                   \
        int blocks, threads;                                                             \
        launch_config(total, blocks, threads);                                           \
        KERNEL<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(mat, colvec, out, rows, cols, vec_is_left); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

COLBCAST_LAUNCHER(cf_sub, k_sub_colbcast, float, f32)
COLBCAST_LAUNCHER(cf_sub, k_sub_colbcast, double, f64)

// -- unary kernels ------------------------------------------------------------
//
// relu is new in Milestone 9 -- required to run the existing high-level
// `nn.ReLU` module on CUDA. exp/log remain unimplemented (out of scope).

template <typename T>
__global__ void k_relu(const T* a, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) {
        T v = a[i];
        out[i] = v > static_cast<T>(0) ? v : static_cast<T>(0);
    }
}

#define UNARY_LAUNCHER(NAME, KERNEL, TYPE, SUFFIX)                                        \
    extern "C" __declspec(dllexport) int NAME##_##SUFFIX(                                 \
        const TYPE* a, TYPE* out, long long n, void* stream) {                            \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        KERNEL<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(a, out, n);            \
        return static_cast<int>(cudaGetLastError());                                      \
    }

UNARY_LAUNCHER(cf_relu, k_relu, float, f32)
UNARY_LAUNCHER(cf_relu, k_relu, double, f64)

// -- exp / log (Milestone 14) --------------------------------------------------
//
// Required by CrossEntropyLoss's numerically-stable log-sum-exp. CUDA's math
// library exposes type-specific `expf`/`logf` (float) and `exp`/`log`
// (double) rather than one overloaded name; these tiny `__device__` wrappers
// give the `k_exp`/`k_log` templates below a single call spelling that
// resolves to the right one for each instantiation.

__device__ inline float cf_expv(float x) { return expf(x); }
__device__ inline double cf_expv(double x) { return exp(x); }
__device__ inline float cf_logv(float x) { return logf(x); }
__device__ inline double cf_logv(double x) { return log(x); }

template <typename T>
__global__ void k_exp(const T* a, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = cf_expv(a[i]);
}

template <typename T>
__global__ void k_log(const T* a, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = cf_logv(a[i]);
}

UNARY_LAUNCHER(cf_exp, k_exp, float, f32)
UNARY_LAUNCHER(cf_exp, k_exp, double, f64)
UNARY_LAUNCHER(cf_log, k_log, float, f32)
UNARY_LAUNCHER(cf_log, k_log, double, f64)

// -- backward-only kernels (Milestone 10) -------------------------------------
//
// Real CUDA kernels backing CUDA autograd's backward rules
// (`CUDABackend.*_backward` in `backend.py`), so backward computation for a
// CUDA graph never copies to CPU. Each is a small, targeted addition mirroring
// an existing forward kernel's shape rather than a general-purpose primitive:
// `k_neg` (sub's second gradient), `k_relu_backward` (ReLU's derivative,
// computed directly from the saved input -- no separate mask kernel),
// `k_scale` (a device-resident scalar times a vector, for the 1D.1D matmul
// case -- the scalar is read via a device pointer so no value ever crosses
// back to the host), `k_transpose` (matmul backward's `.T`), and
// `k_reduce_rows` (undoes the row-broadcast add/sub/mul forward kernels'
// broadcast by summing a (rows, cols) upstream gradient down to the (cols,)
// vector operand's shape -- the one broadcast-reduction shape actually needed,
// not a general axis-sum).

template <typename T>
__global__ void k_neg(const T* a, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = -a[i];
}

UNARY_LAUNCHER(cf_neg, k_neg, float, f32)
UNARY_LAUNCHER(cf_neg, k_neg, double, f64)

template <typename T>
__global__ void k_relu_backward(const T* grad_output, const T* input, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = input[i] > static_cast<T>(0) ? grad_output[i] : static_cast<T>(0);
}

ELEMENTWISE_LAUNCHER(cf_relu_backward, k_relu_backward, float, f32)
ELEMENTWISE_LAUNCHER(cf_relu_backward, k_relu_backward, double, f64)

// exp/log backward (Milestone 14): `d(exp(x))/dx = exp(x)` (so the backward
// rule is `grad_output * result`, `result` being exp's own saved forward
// output -- no need to re-read `x`) and `d(log(x))/dx = 1/x` (`grad_output /
// input`, the saved forward input). Each is a small dedicated kernel, like
// `k_relu_backward` above, rather than a generic elementwise-divide
// primitive that nothing else in Forge needs.

template <typename T>
__global__ void k_exp_backward(const T* grad_output, const T* result, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = grad_output[i] * result[i];
}

template <typename T>
__global__ void k_log_backward(const T* grad_output, const T* input, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = grad_output[i] / input[i];
}

ELEMENTWISE_LAUNCHER(cf_exp_backward, k_exp_backward, float, f32)
ELEMENTWISE_LAUNCHER(cf_exp_backward, k_exp_backward, double, f64)
ELEMENTWISE_LAUNCHER(cf_log_backward, k_log_backward, float, f32)
ELEMENTWISE_LAUNCHER(cf_log_backward, k_log_backward, double, f64)

template <typename T>
__global__ void k_scale(const T* scalar, const T* vec, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = (*scalar) * vec[i];
}

ELEMENTWISE_LAUNCHER(cf_scale, k_scale, float, f32)
ELEMENTWISE_LAUNCHER(cf_scale, k_scale, double, f64)

template <typename T>
__global__ void k_transpose(const T* in, T* out, long long rows, long long cols) {
    long long r = blockIdx.y * static_cast<long long>(blockDim.y) + threadIdx.y;
    long long c = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (r < rows && c < cols) out[c * rows + r] = in[r * cols + c];
}

#define TRANSPOSE_LAUNCHER(TYPE, SUFFIX)                                                  \
    extern "C" __declspec(dllexport) int cf_transpose_##SUFFIX(                           \
        const TYPE* in, TYPE* out, long long rows, long long cols, void* stream) {        \
        dim3 threads(16, 16);                                                             \
        dim3 blocks(static_cast<unsigned int>((cols + 15) / 16),                          \
                    static_cast<unsigned int>((rows + 15) / 16));                         \
        k_transpose<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(in, out, rows, cols); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

TRANSPOSE_LAUNCHER(float, f32)
TRANSPOSE_LAUNCHER(double, f64)

template <typename T>
__global__ void k_reduce_rows(const T* mat, T* out, long long rows, long long cols) {
    long long col = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (col < cols) {
        T acc = static_cast<T>(0);
        for (long long r = 0; r < rows; ++r) {
            acc += mat[r * cols + col];
        }
        out[col] = acc;
    }
}

#define REDUCE_ROWS_LAUNCHER(TYPE, SUFFIX)                                                \
    extern "C" __declspec(dllexport) int cf_reduce_rows_##SUFFIX(                         \
        const TYPE* mat, TYPE* out, long long rows, long long cols, void* stream) {       \
        int blocks, threads;                                                              \
        launch_config(cols, blocks, threads);                                             \
        k_reduce_rows<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(mat, out, rows, cols); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

REDUCE_ROWS_LAUNCHER(float, f32)
REDUCE_ROWS_LAUNCHER(double, f64)

// -- SGD parameter update (Milestone 10) ---------------------------------------
//
// In-place `param -= lr * grad`, executed entirely on-device so a CUDA
// Parameter never needs a host round-trip to be optimized. `lr` is passed by
// value (a plain launch argument, like `rows`/`cols` elsewhere in this file --
// it is Python-side hyperparameter state, not tensor data) and cast to `T`
// inside the kernel so one launcher works for both float32 and float64
// parameters.

template <typename T>
__global__ void k_sgd_step(T* param, const T* grad, double lr, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) param[i] -= static_cast<T>(lr) * grad[i];
}

#define SGD_STEP_LAUNCHER(TYPE, SUFFIX)                                                   \
    extern "C" __declspec(dllexport) int cf_sgd_step_##SUFFIX(                            \
        TYPE* param, const TYPE* grad, double lr, long long n, void* stream) {            \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        k_sgd_step<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(param, grad, lr, n); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

SGD_STEP_LAUNCHER(float, f32)
SGD_STEP_LAUNCHER(double, f64)

// -- Adam parameter update (Milestone 17) --------------------------------------
//
// In-place Adam update, executed entirely on-device like `k_sgd_step` above:
// `param`/`grad`/`m`/`v` are all real CUDA buffers, and no value ever
// crosses back to the host mid-computation. Bias correction
// (`1 - beta1**step` / `1 - beta2**step`) is one scalar `pow()` -- identical
// for every element in a given call -- so it is computed once on the host in
// Python (`CUDABackend.adam_step`) and passed in as `bias_correction1`/
// `bias_correction2` rather than recomputed per-thread. `cf_sqrtv` dispatches
// to the type-specific `sqrtf`/`sqrt`, the same overload-resolution pattern
// `cf_expv`/`cf_logv` (below) use.

__device__ inline float cf_sqrtv(float x) { return sqrtf(x); }
__device__ inline double cf_sqrtv(double x) { return sqrt(x); }

template <typename T>
__global__ void k_adam_step(
    T* param, const T* grad, T* m, T* v,
    double lr, double beta1, double beta2, double eps, double weight_decay,
    double bias_correction1, double bias_correction2, long long n
) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= n) return;

    T g = grad[i];
    if (weight_decay != 0.0) {
        g = g + static_cast<T>(weight_decay) * param[i];
    }

    T b1 = static_cast<T>(beta1);
    T b2 = static_cast<T>(beta2);
    T m_new = b1 * m[i] + (static_cast<T>(1) - b1) * g;
    T v_new = b2 * v[i] + (static_cast<T>(1) - b2) * g * g;
    m[i] = m_new;
    v[i] = v_new;

    T m_hat = m_new / static_cast<T>(bias_correction1);
    T v_hat = v_new / static_cast<T>(bias_correction2);
    param[i] -= static_cast<T>(lr) * m_hat / (cf_sqrtv(v_hat) + static_cast<T>(eps));
}

#define ADAM_STEP_LAUNCHER(TYPE, SUFFIX)                                                 \
    extern "C" __declspec(dllexport) int cf_adam_step_##SUFFIX(                          \
        TYPE* param, const TYPE* grad, TYPE* m, TYPE* v,                                 \
        double lr, double beta1, double beta2, double eps, double weight_decay,          \
        double bias_correction1, double bias_correction2, long long n, void* stream) {   \
        int blocks, threads;                                                             \
        launch_config(n, blocks, threads);                                               \
        k_adam_step<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(                 \
            param, grad, m, v, lr, beta1, beta2, eps, weight_decay,                      \
            bias_correction1, bias_correction2, n);                                      \
        return static_cast<int>(cudaGetLastError());                                     \
    }

ADAM_STEP_LAUNCHER(float, f32)
ADAM_STEP_LAUNCHER(double, f64)

// -- sum backward: broadcast a single value to every output element -----------
//
// `x.sum()`'s backward rule needs the one upstream scalar written into every
// element of `x`'s original shape. `scalar` is read via a device pointer (the
// same convention `k_scale` uses), so no value ever crosses back to the host.

template <typename T>
__global__ void k_broadcast_scalar(const T* scalar, T* out, long long n) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i < n) out[i] = *scalar;
}

#define BROADCAST_SCALAR_LAUNCHER(TYPE, SUFFIX)                                           \
    extern "C" __declspec(dllexport) int cf_broadcast_scalar_##SUFFIX(                    \
        const TYPE* scalar, TYPE* out, long long n, void* stream) {                       \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        k_broadcast_scalar<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(scalar, out, n); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

BROADCAST_SCALAR_LAUNCHER(float, f32)
BROADCAST_SCALAR_LAUNCHER(double, f64)

// -- matmul (shared-memory tiled, Milestone 11) -------------------------------
//
// Milestone 8-10 shipped a naive one-thread-per-output-element matmul
// (every thread re-reads a full K-length row of A and column of B straight
// from global memory, with no reuse across threads). Milestone 11's
// benchmarks (`docs/performance/benchmarking.md`) measured this as a real,
// significant bottleneck at the "medium" (512x512) scale: naive CUDA matmul
// ran ~4x *slower* than the CPU (NumPy/BLAS) backend, forward and backward
// alike -- not merely CUDA launch/transfer overhead (which would show up
// at the tiny/small scales, not scale with problem size). The milestone
// brief explicitly sanctions a tiled implementation once naive matmul is
// "clearly measured as a bottleneck" -- see that document's "Optimization
// decisions" section for the full before/after numbers.
//
// This is a standard 16x16-tile shared-memory GEMM: each thread block
// cooperatively loads one TILE x TILE tile of A and one of B into shared
// memory per outer-loop step, so each element of A/B is read from global
// memory once per tile (shared across the 16 threads that reuse it) instead
// of once per output element. Out-of-range tile loads (M/K/N not a multiple
// of TILE) are zero-padded, matching the naive kernel's exact boundary
// semantics. No warp-level intrinsics or architecture-specific tricks are
// used, so this remains correct and portable to Compute Capability 5.0 (the
// verified 940MX) -- this is not a generalized GEMM library, just the one
// tiling optimization the measured bottleneck justified (per the milestone
// brief's "do not implement a generalized GEMM library" / "do not introduce
// complex shared-memory/tiled implementations unless measurements justify
// them" constraints). The Python side is unchanged: `cf_matmul_{f32,f64}`
// keeps the same exported signature, called the same way for all four 1D/2D
// matmul cases (`CUDABackend.matmul`, `forge/backend/cuda/backend.py`).

constexpr int MATMUL_TILE = 16;

template <typename T>
__global__ void k_matmul(const T* A, const T* B, T* C, int M, int K, int N) {
    __shared__ T tile_a[MATMUL_TILE][MATMUL_TILE];
    __shared__ T tile_b[MATMUL_TILE][MATMUL_TILE];

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;

    T acc = static_cast<T>(0);
    int num_tiles = (K + MATMUL_TILE - 1) / MATMUL_TILE;
    for (int t = 0; t < num_tiles; ++t) {
        int a_col = t * MATMUL_TILE + threadIdx.x;
        int b_row = t * MATMUL_TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : static_cast<T>(0);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : static_cast<T>(0);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

#define MATMUL_LAUNCHER(TYPE, SUFFIX)                                                      \
    extern "C" __declspec(dllexport) int cf_matmul_##SUFFIX(                               \
        const TYPE* A, const TYPE* B, TYPE* C, int M, int K, int N, void* stream) {        \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                            \
        dim3 blocks((N + MATMUL_TILE - 1) / MATMUL_TILE, (M + MATMUL_TILE - 1) / MATMUL_TILE); \
        k_matmul<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(A, B, C, M, K, N);     \
        return static_cast<int>(cudaGetLastError());                                       \
    }

MATMUL_LAUNCHER(float, f32)
MATMUL_LAUNCHER(double, f64)

// -- sum reduction (full reduction only) -------------------------------------
//
// Compute Capability 5.0 has no native `atomicAdd(double*, double)`; this is
// the standard CAS-based emulation from the CUDA C++ Programming Guide.

__device__ double atomic_add_f64(double* address, double val) {
    unsigned long long int* addr_as_ull = reinterpret_cast<unsigned long long int*>(address);
    unsigned long long int old = *addr_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_as_ull, assumed,
                         __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename T>
__global__ void k_sum(const T* a, T* out, long long n) {
    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);
    int tid = threadIdx.x;
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    smem[tid] = (i < n) ? a[i] : static_cast<T>(0);
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    if (tid == 0) {
        if constexpr (std::is_same<T, double>::value) {
            atomic_add_f64(out, smem[0]);
        } else {
            atomicAdd(out, smem[0]);
        }
    }
}

#define SUM_LAUNCHER(TYPE, SUFFIX)                                                          \
    extern "C" __declspec(dllexport) int cf_sum_##SUFFIX(                                   \
        const TYPE* a, TYPE* out, long long n, void* stream) {                              \
        cudaStream_t s = (cudaStream_t)stream;                                              \
        cudaError_t err = cudaMemsetAsync(out, 0, sizeof(TYPE), s);                         \
        if (err != cudaSuccess) return static_cast<int>(err);                               \
        int threads = 256;                                                                  \
        long long b = (n + threads - 1) / threads;                                          \
        int blocks = b < 1 ? 1 : static_cast<int>(b);                                       \
        k_sum<TYPE><<<blocks, threads, threads * sizeof(TYPE), s>>>(a, out, n);             \
        return static_cast<int>(cudaGetLastError());                                        \
    }

SUM_LAUNCHER(float, f32)
SUM_LAUNCHER(double, f64)

// -- axis=1 reduction kernels (Milestone 14) -----------------------------------
//
// CrossEntropyLoss needs a per-row reduction across the class dimension: for
// logits shaped (batch, classes), reduce (max, for the numerical-stability
// shift; sum, for the log-sum-exp denominator and for picking out each
// target's log-probability) each row down to one value. One thread per row,
// looping over that row's `cols` elements -- the same "one thread per output
// element, loop over the reduced dimension" shape `k_reduce_rows` (above)
// already uses for the *other* axis (columns down to one row-broadcast
// vector). Not a general axis-reduction primitive: only axis=1 on a 2D
// tensor is supported, which is all CrossEntropyLoss (or anything else in
// Forge) needs -- `CUDABackend.sum()` still raises `CUDAError` for any other
// axis (see `docs/architecture/cuda-backend.md`).

template <typename T>
__global__ void k_max_axis1(const T* mat, T* out, long long rows, long long cols) {
    long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (row < rows) {
        T m = mat[row * cols];
        for (long long c = 1; c < cols; ++c) {
            T v = mat[row * cols + c];
            if (v > m) m = v;
        }
        out[row] = m;
    }
}

template <typename T>
__global__ void k_sum_axis1(const T* mat, T* out, long long rows, long long cols) {
    long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (row < rows) {
        T acc = static_cast<T>(0);
        for (long long c = 0; c < cols; ++c) {
            acc += mat[row * cols + c];
        }
        out[row] = acc;
    }
}

#define AXIS1_REDUCE_LAUNCHER(NAME, KERNEL, TYPE, SUFFIX)                                \
    extern "C" __declspec(dllexport) int NAME##_##SUFFIX(                                \
        const TYPE* mat, TYPE* out, long long rows, long long cols, void* stream) {      \
        int blocks, threads;                                                             \
        launch_config(rows, blocks, threads);                                            \
        KERNEL<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(mat, out, rows, cols); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

AXIS1_REDUCE_LAUNCHER(cf_max_axis1, k_max_axis1, float, f32)
AXIS1_REDUCE_LAUNCHER(cf_max_axis1, k_max_axis1, double, f64)
AXIS1_REDUCE_LAUNCHER(cf_sum_axis1, k_sum_axis1, float, f32)
AXIS1_REDUCE_LAUNCHER(cf_sum_axis1, k_sum_axis1, double, f64)

// -- fused CrossEntropyLoss (Milestone 31) ---------------------------------
//
// Milestone 21/31 profiling (`benchmarks/mnist_profile.py`,
// `docs/performance/pipeline-profiling.md`) measured CrossEntropyLoss's
// forward pass costing *more* wall-clock time on the 940MX than the entire
// M20 CNN forward pass, despite operating on a tensor orders of magnitude
// smaller. The cause was never per-element arithmetic: the previous
// implementation (`forge/nn/loss.py`) composed ~9 separate Tensor
// primitives (max_axis1, two row-broadcast subs, exp, sum(axis=1), log, a
// mul against a freshly host-transferred one-hot matrix, a full sum, and a
// final scale) plus two fresh host->device transfers, each paying this
// GPU's real, measured ~50-300us-per-launch dispatch overhead
// (`benchmarks/stream_dependency_bench.py`) regardless of how little actual
// arithmetic it does. This kernel pair fuses that entire chain into two
// GPU-side kernel launches for the forward pass (one to compute each row's
// negative log-likelihood, one -- the existing `cf_sum_*` reduction above,
// reused rather than duplicated -- to average them) and one launch for the
// backward pass, replacing ~9 forward + ~7 backward launches and both host
// transfers (the one-hot matrix is never built at all; only the already-
// small integer target indices ever cross the host/device boundary, and
// only when they were not already CUDA-resident -- see `CUDABackend.
// cross_entropy` in `backend.py`).
//
// One thread per row (looping serially over `cols`), matching
// `k_max_axis1`/`k_sum_axis1` above -- classification class counts are
// small (tens to low thousands) relative to batch size, so a per-row thread
// keeps this kernel simple and correct rather than adding a block-per-row
// shared-memory reduction for a case Forge's own workloads never need.
// `inv_n` is always passed as `double` and cast to `T` inside the kernel,
// matching `k_sgd_step`/`k_adam_step`'s existing convention for scalar
// launch parameters (below) -- one ctypes signature serves both dtypes.

template <typename T>
__global__ void k_cross_entropy_forward(
    const T* logits, const long long* target, T* loss_out,
    long long rows, long long cols, double inv_n) {
    long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (row < rows) {
        const T* r = logits + row * cols;
        T m = r[0];
        for (long long c = 1; c < cols; ++c) {
            T v = r[c];
            if (v > m) m = v;
        }
        T sum_exp = static_cast<T>(0);
        for (long long c = 0; c < cols; ++c) {
            sum_exp += cf_expv(r[c] - m);
        }
        long long t = target[row];
        loss_out[row] = (m + cf_logv(sum_exp) - r[t]) * static_cast<T>(inv_n);
    }
}

template <typename T>
__global__ void k_cross_entropy_backward(
    const T* grad_output, const T* logits, const long long* target, T* grad_logits,
    long long rows, long long cols, double inv_n) {
    long long row = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (row < rows) {
        const T* r = logits + row * cols;
        T m = r[0];
        for (long long c = 1; c < cols; ++c) {
            T v = r[c];
            if (v > m) m = v;
        }
        T sum_exp = static_cast<T>(0);
        for (long long c = 0; c < cols; ++c) {
            sum_exp += cf_expv(r[c] - m);
        }
        T scale = grad_output[0] * static_cast<T>(inv_n);
        long long t = target[row];
        T* g = grad_logits + row * cols;
        for (long long c = 0; c < cols; ++c) {
            T softmax_c = cf_expv(r[c] - m) / sum_exp;
            g[c] = scale * (softmax_c - (c == t ? static_cast<T>(1) : static_cast<T>(0)));
        }
    }
}

#define CROSS_ENTROPY_FWD_LAUNCHER(TYPE, SUFFIX)                                            \
    extern "C" __declspec(dllexport) int cf_cross_entropy_forward_##SUFFIX(                 \
        const TYPE* logits, const long long* target, TYPE* loss_out,                        \
        long long rows, long long cols, double inv_n, void* stream) {                       \
        int blocks, threads;                                                                \
        launch_config(rows, blocks, threads);                                               \
        k_cross_entropy_forward<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(         \
            logits, target, loss_out, rows, cols, inv_n);                                   \
        return static_cast<int>(cudaGetLastError());                                        \
    }

#define CROSS_ENTROPY_BWD_LAUNCHER(TYPE, SUFFIX)                                            \
    extern "C" __declspec(dllexport) int cf_cross_entropy_backward_##SUFFIX(                \
        const TYPE* grad_output, const TYPE* logits, const long long* target,               \
        TYPE* grad_logits, long long rows, long long cols, double inv_n, void* stream) {     \
        int blocks, threads;                                                                \
        launch_config(rows, blocks, threads);                                               \
        k_cross_entropy_backward<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(        \
            grad_output, logits, target, grad_logits, rows, cols, inv_n);                    \
        return static_cast<int>(cudaGetLastError());                                        \
    }

CROSS_ENTROPY_FWD_LAUNCHER(float, f32)
CROSS_ENTROPY_FWD_LAUNCHER(double, f64)
CROSS_ENTROPY_BWD_LAUNCHER(float, f32)
CROSS_ENTROPY_BWD_LAUNCHER(double, f64)

// Backward of sum(axis=1): broadcast each row's single upstream gradient
// value to every element of that row -- the axis=1 analog of
// `k_broadcast_scalar` (above), which broadcasts one global scalar to every
// element instead of one scalar per row.
template <typename T>
__global__ void k_broadcast_axis1(const T* rowvals, T* out, long long rows, long long cols) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = rows * cols;
    if (i < total) {
        long long row = i / cols;
        out[i] = rowvals[row];
    }
}

#define BROADCAST_AXIS1_LAUNCHER(TYPE, SUFFIX)                                           \
    extern "C" __declspec(dllexport) int cf_broadcast_axis1_##SUFFIX(                    \
        const TYPE* rowvals, TYPE* out, long long rows, long long cols, void* stream) {  \
        long long total = rows * cols;                                                   \
        int blocks, threads;                                                             \
        launch_config(total, blocks, threads);                                           \
        k_broadcast_axis1<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(rowvals, out, rows, cols); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

BROADCAST_AXIS1_LAUNCHER(float, f32)
BROADCAST_AXIS1_LAUNCHER(double, f64)

// -- Conv2d / MaxPool2d (Milestone 15) ----------------------------------------
//
// Straightforward, correct kernels -- one thread per output (forward) or per
// gradient-target (backward) element, looping over the small kernel/reduction
// dimension in registers. No im2col-as-a-separate-buffer, no cuBLAS, no
// cuDNN, no tiling: the milestone brief explicitly sanctions "start with a
// straightforward correct kernel" and defers optimization (the CPU backend's
// im2col-plus-matmul approach is *not* mirrored here since these are meant to
// be simple, independently-verifiable kernels, not a second GEMM path).
//
// Layouts match the rest of Forge: `x` is NCHW, `weight` is (C_out, C_in,
// KH, KW), `bias` is (C_out,). All backward kernels recompute whatever they
// need (the forward window, or -- for MaxPool2d -- its argmax) from the
// saved forward input rather than caching auxiliary state, the same
// "recompute from a saved input" convention `k_relu_backward`/
// `k_exp_backward` above already use.

template <typename T> __device__ inline T cf_neg_max();
template <> __device__ inline float cf_neg_max<float>() { return -FLT_MAX; }
template <> __device__ inline double cf_neg_max<double>() { return -DBL_MAX; }

template <typename T>
__device__ inline void atomic_add_generic(T* addr, T val) {
    if constexpr (std::is_same<T, double>::value) {
        atomic_add_f64(addr, val);
    } else {
        atomicAdd(addr, val);
    }
}

// -- Conv2d forward ------------------------------------------------------------

template <typename T>
__global__ void k_conv2d_forward(
    const T* x, const T* w, const T* bias, T* out,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout, int has_bias)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(N) * Cout * Hout * Wout;
    if (idx >= total) return;

    int wo = static_cast<int>(idx % Wout);
    long long t1 = idx / Wout;
    int ho = static_cast<int>(t1 % Hout);
    long long t2 = t1 / Hout;
    int co = static_cast<int>(t2 % Cout);
    int n = static_cast<int>(t2 / Cout);

    T acc = has_bias ? bias[co] : static_cast<T>(0);
    for (int ci = 0; ci < Cin; ++ci) {
        for (int kh = 0; kh < KH; ++kh) {
            int hi = ho * SH - PH + kh;
            if (hi < 0 || hi >= H) continue;
            for (int kw_ = 0; kw_ < KW; ++kw_) {
                int wi = wo * SW - PW + kw_;
                if (wi < 0 || wi >= W) continue;
                long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
                long long w_idx = ((static_cast<long long>(co) * Cin + ci) * KH + kh) * KW + kw_;
                acc += x[x_idx] * w[w_idx];
            }
        }
    }
    out[idx] = acc;
}

#define CONV2D_FORWARD_LAUNCHER(TYPE, SUFFIX)                                             \
    extern "C" __declspec(dllexport) int cf_conv2d_forward_##SUFFIX(                      \
        const TYPE* x, const TYPE* w, const TYPE* bias, TYPE* out,                        \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout, int has_bias, void* stream) { \
        long long total = static_cast<long long>(N) * Cout * Hout * Wout;                 \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        k_conv2d_forward<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(             \
            x, w, bias, out, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout, has_bias); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_FORWARD_LAUNCHER(float, f32)
CONV2D_FORWARD_LAUNCHER(double, f64)

// -- Conv2d backward: input (gather over the output windows an input pixel feeds) --
//
// Milestone 32 fix: the original version below (kept only in this comment
// for reference) resolved, *inside* the `co` loop, which `(kh, ho)`/`(kw,
// wo)` pairs are valid for this thread's fixed `(hi, wi)`:
//
//   for (int co = 0; co < Cout; ++co) {
//       for (int kh = 0; kh < KH; ++kh) {
//           int t = hi + PH - kh;
//           if (t % SH != 0) continue;         // <- co-independent
//           int ho = t / SH;                    // <- co-independent
//           ...
//
// `hi`/`wi`/`PH`/`PW`/`SH`/`SW`/`KH`/`KW` never depend on `co` -- so that
// integer division/modulo pair (expensive on CC 5.0: no native fast-path,
// `SH`/`SW` are runtime kernel arguments, not compile-time constants, so
// the compiler cannot even special-case a stride of 1) was recomputed
// identically `Cout` times per thread for no reason. M32 profiling
// (`benchmarks/conv2d_backward_profile.py`) measured this as the single
// largest Conv2d-backward contributor at every non-MNIST-scale shape tested
// (e.g. 65.4ms vs. `dWeight`'s 43.6ms at a Cin=16/Cout=32/28x28/N=64 shape)
// and comparable to `dWeight` even at the real M20 MNIST shapes -- see
// `docs/performance/conv2d-backward-profiling.md`.
//
// The fix hoists that resolution out of the `co` loop entirely: each thread
// first builds two small local tables of valid `(kh, ho)` and `(kw, wo)`
// pairs *once*, then the `co` loop only ever does array indexing, integer
// multiply/add for `g_idx`/`w_idx`, and the same multiply-accumulate the
// original kernel always needed -- no per-`co` division survives. This
// computes exactly the same set of `(co, kh, ho, kw, wo)` contributions as
// before (same math, same order of summation for a fixed `co`), just
// without recomputing `co`-independent work `Cout` times. `MAX_CONV_K`
// bounds the local tables -- generous for any kernel size Forge's own
// `Conv2d`/tests use (K in {2, 3, 5} in practice; never observed or
// intended to exceed single digits), documented in `docs/architecture/
// cuda-backend.md`.

constexpr int MAX_CONV_K = 32;

template <typename T>
__global__ void k_conv2d_backward_input(
    const T* grad_out, const T* w, T* grad_x,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(N) * Cin * H * W;
    if (idx >= total) return;

    int wi = static_cast<int>(idx % W);
    long long t1 = idx / W;
    int hi = static_cast<int>(t1 % H);
    long long t2 = t1 / H;
    int ci = static_cast<int>(t2 % Cin);
    int n = static_cast<int>(t2 / Cin);

    // Resolve valid (kh, ho) pairs once -- co-independent.
    int kh_valid[MAX_CONV_K];
    int ho_valid[MAX_CONV_K];
    int h_count = 0;
    for (int kh = 0; kh < KH && kh < MAX_CONV_K; ++kh) {
        int t = hi + PH - kh;
        if (t % SH != 0) continue;
        int ho = t / SH;
        if (ho < 0 || ho >= Hout) continue;
        kh_valid[h_count] = kh;
        ho_valid[h_count] = ho;
        ++h_count;
    }

    // Resolve valid (kw, wo) pairs once -- co-independent.
    int kw_valid[MAX_CONV_K];
    int wo_valid[MAX_CONV_K];
    int w_count = 0;
    for (int kw_ = 0; kw_ < KW && kw_ < MAX_CONV_K; ++kw_) {
        int tw = wi + PW - kw_;
        if (tw % SW != 0) continue;
        int wo = tw / SW;
        if (wo < 0 || wo >= Wout) continue;
        kw_valid[w_count] = kw_;
        wo_valid[w_count] = wo;
        ++w_count;
    }

    T acc = static_cast<T>(0);
    for (int co = 0; co < Cout; ++co) {
        long long g_base = static_cast<long long>(n) * Cout + co;
        long long w_base = static_cast<long long>(co) * Cin + ci;
        for (int hi_i = 0; hi_i < h_count; ++hi_i) {
            int kh = kh_valid[hi_i], ho = ho_valid[hi_i];
            long long g_row = (g_base * Hout + ho) * Wout;
            long long w_row = (w_base * KH + kh) * KW;
            for (int wi_i = 0; wi_i < w_count; ++wi_i) {
                int kw_ = kw_valid[wi_i], wo = wo_valid[wi_i];
                acc += grad_out[g_row + wo] * w[w_row + kw_];
            }
        }
    }
    grad_x[idx] = acc;
}

// `k_conv2d_backward_input_channelfused` (Milestone 36 Candidate B) is
// defined here, immediately after the baseline kernel it is dispatched
// against below, purely so the production launcher can call it directly --
// see the "dInput candidate kernels" section further down (after that
// launcher) for its full design rationale and Candidates A/C, which stay in
// their original, narratively-grouped location since nothing dispatches to
// them at compile time.

constexpr int MAX_CIN_REG = 16;  // Forge's 7 representative shapes all have Cin <= 16

template <typename T>
__global__ void k_conv2d_backward_input_channelfused(
    const T* grad_out, const T* w, T* grad_x,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(N) * H * W;
    if (idx >= total) return;

    int wi = static_cast<int>(idx % W);
    long long t1 = idx / W;
    int hi = static_cast<int>(t1 % H);
    int n = static_cast<int>(t1 / H);

    T acc[MAX_CIN_REG];
#pragma unroll
    for (int ci = 0; ci < MAX_CIN_REG; ++ci) acc[ci] = static_cast<T>(0);

    for (int kh = 0; kh < KH; ++kh) {
        int t = hi + PH - kh;
        if (t % SH != 0) continue;
        int ho = t / SH;
        if (ho < 0 || ho >= Hout) continue;
        for (int kw_ = 0; kw_ < KW; ++kw_) {
            int tw = wi + PW - kw_;
            if (tw % SW != 0) continue;
            int wo = tw / SW;
            if (wo < 0 || wo >= Wout) continue;
            for (int co = 0; co < Cout; ++co) {
                long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
                T g = grad_out[g_idx];  // read once, reused across every ci below
                long long w_base = static_cast<long long>(co) * Cin * KH * KW + kh * KW + kw_;
#pragma unroll
                for (int ci = 0; ci < MAX_CIN_REG; ++ci) {
                    if (ci >= Cin) break;
                    acc[ci] += g * w[w_base + static_cast<long long>(ci) * KH * KW];
                }
            }
        }
    }

    long long out_base = static_cast<long long>(n) * Cin * H * W + static_cast<long long>(hi) * W + wi;
#pragma unroll
    for (int ci = 0; ci < MAX_CIN_REG; ++ci) {
        if (ci >= Cin) break;
        grad_x[out_base + static_cast<long long>(ci) * H * W] = acc[ci];
    }
}

// Milestone 36 production dispatch: `k_conv2d_backward_input_channelfused`
// (defined below, in the "dInput candidate kernels" section) measured
// 1.0x-6.9x faster than `k_conv2d_backward_input` (this section, unchanged)
// at every one of the 7 representative shapes/batch sizes -- see
// `docs/performance/conv2d-backward-profiling.md`'s **Milestone 36** section
// for the full before/after evidence. It requires `Cin <=
// CONV2D_DINPUT_CHANNELFUSED_MAX_CIN` for correctness (its register-resident
// accumulator array is sized and fully unrolled at that bound at compile
// time -- see that kernel's own comment); every one of Forge's representative
// shapes satisfies this (`Cin` in {1, 8, 16}), so this dispatch is exercised
// by every one of them. A layer with `Cin` above the bound (never produced by
// Forge's own `nn.Conv2d`/MNIST-scale test shapes, but not otherwise
// prohibited by `Conv2d`'s public API) transparently falls back to the
// unchanged, always-correct `k_conv2d_backward_input` -- Section 38's
// sanctioned hybrid dispatch, kept to this one simple boundary check.
constexpr int CONV2D_DINPUT_CHANNELFUSED_MAX_CIN = 16;

#define CONV2D_BACKWARD_INPUT_LAUNCHER(TYPE, SUFFIX)                                      \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_input_##SUFFIX(               \
        const TYPE* grad_out, const TYPE* w, TYPE* grad_x,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {               \
        cudaStream_t s = (cudaStream_t)stream;                                            \
        if (Cin <= CONV2D_DINPUT_CHANNELFUSED_MAX_CIN) {                                  \
            long long total = static_cast<long long>(N) * H * W;                          \
            int blocks, threads;                                                          \
            launch_config(total, blocks, threads);                                        \
            k_conv2d_backward_input_channelfused<TYPE><<<blocks, threads, 0, s>>>(        \
                grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        } else {                                                                          \
            long long total = static_cast<long long>(N) * Cin * H * W;                    \
            int blocks, threads;                                                          \
            launch_config(total, blocks, threads);                                        \
            k_conv2d_backward_input<TYPE><<<blocks, threads, 0, s>>>(                     \
                grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        }                                                                                 \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_INPUT_LAUNCHER(float, f32)
CONV2D_BACKWARD_INPUT_LAUNCHER(double, f64)

// -- dInput candidate kernels (Milestone 36, profiling-only) -----------------
//
// M36 root-cause finding: `nvcc -Xptxas -v` on `k_conv2d_backward_input`
// (above) reports "512 bytes stack frame ... 0 bytes spill stores/loads" --
// the `kh_valid`/`ho_valid`/`kw_valid`/`wo_valid` local arrays M32 introduced
// (4 arrays x MAX_CONV_K=32 ints = 512 bytes) are indexed with a
// runtime-computed loop variable (`hi_i`/`wi_i`), which nvcc cannot keep in
// registers regardless of array size -- it places them in per-thread local
// memory (an implicit DRAM-backed, L1/L2-cached region), read back
// `Cout * h_count * w_count` times per thread. This is real, uncounted-by-
// the-roofline-model traffic (the `bytes_conv2d_dinput` model in
// `benchmarks/roofline.py` only counts grad_out/weight/grad_x, never
// implementation-internal local-memory spill), and it explains the
// milestone's headline number directly: at every representative shape,
// measured arithmetic intensity places `dInput` decisively to the *right* of
// the 940MX's measured ridge point (~6.93 FLOPs/byte = 104.57 GFLOP/s compute
// ceiling / 15.09 GB/s bandwidth ceiling -- see `m35-roofline-
// characterization.md`) -- e.g. `large_channel`'s AI=47.9, `mnist_conv2`'s
// AI=23.9 -- so this kernel is classified compute-bound by the FLOP/byte
// model, yet achieves only ~12% of the practical *compute* ceiling while
// using under 20% (often under 5%) of the practical *bandwidth* ceiling
// (Section 14/15 of the milestone brief: "if a kernel is far below both
// ceilings, investigate instruction efficiency/occupancy/latency rather than
// labeling it memory-bound"). A kernel that is both far from its compute
// ceiling and using only a sliver of its bandwidth budget is not bandwidth-
// starved -- it is spending real cycles on something the FLOP/byte model
// cannot see: local-memory traffic and register pressure (48 registers even
// before counting the 512-byte stack frame).
//
// Three structurally different candidates are measured against this
// diagnosis, matching Sections 9-11 of the milestone brief:
//
//   Candidate A (`_smem`)   -- shared-memory grad_output row-tile reuse
//                              across the `Cin` dimension (Section 9): a
//                              block owns one `(n, hi)` pair, cooperatively
//                              loads the `h_count <= KH` needed grad_output
//                              rows (all `Cout` channels, full `Wout`) into
//                              shared memory once, then every `(ci, wi)`
//                              thread in the block reads from shared memory
//                              instead of re-issuing `Cin` separate global
//                              loads for the same values. Directly tests the
//                              "grad_output read amplification" hypothesis
//                              (Section 8) in isolation -- it keeps the
//                              baseline's one-thread-per-`(n,ci,h,w)` mapping
//                              and its per-thread `kw_valid`/`wo_valid`
//                              tables unchanged, changing only where
//                              grad_output is read from.
//   Candidate B (`_channelfused`) -- alternative work mapping (Section 10,
//                              "channel" dimension): one thread now owns a
//                              full `(n, hi, wi)` position across *every*
//                              `ci`, holding `Cin` accumulators in a
//                              `#pragma unroll`-forced, compile-time-bounded
//                              (`MAX_CIN_REG=32`) register array (verified via
//                              `-Xptxas -v` to actually promote to registers,
//                              zero stack frame -- see the M36 report's
//                              **Register Usage** section) and reading each
//                              `grad_output[n,co,ho,wo]` value exactly once
//                              per thread, reused via a register across all
//                              `Cin` weight multiplies. This also eliminates
//                              M32's arrays entirely (kh/ho/kw/wo are now
//                              plain scalars in the outer loop nest, never
//                              stored to an array), so it tests the
//                              instruction-efficiency root cause and the
//                              grad_output-reuse hypothesis together, via
//                              register reuse instead of shared memory.
//                              Shapes with `Cin > MAX_CIN_REG` are out of
//                              scope for this kernel (production dispatch,
//                              if adopted, would fall back to the baseline
//                              kernel -- Section 38's sanctioned hybrid
//                              dispatch; every one of Forge's 7 representative
//                              shapes has `Cin <= 16`).
//   Candidate C (`_warp`)   -- partial cooperative reduction over `Cout`
//                              (Section 11): a warp (32 lanes), not a single
//                              thread, owns one `(n,ci,h,w)` output element;
//                              each lane sums a disjoint slice of `Cout` and
//                              the warp combines partial sums via
//                              `__shfl_down_sync` (no shared memory, no
//                              `__syncthreads()`, mirroring M33's
//                              `k_conv2d_backward_weight_warp` pattern).
//                              Tests whether the same cooperative-reduction
//                              idea M33 evaluated for `dWeight` (and
//                              rejected, for a *different* reason --
//                              too-many-blocks scheduling overhead) helps
//                              here; the a priori expectation is rejection
//                              for yet another reason -- `dInput` already
//                              launches `N*Cin*H*W` threads (tens of
//                              thousands to ~800K at these shapes), already
//                              far more parallelism than the 940MX's ~6,144
//                              concurrent-thread capacity can use at once, so
//                              trading fewer, "fatter" cooperating units for
//                              more total launched threads (32x) has no
//                              starved parallelism to fix and only adds
//                              `__shfl_down_sync` overhead plus, for the
//                              `Cout in {8, 16}` shapes, partially idle warps.
//
// All three are profiling-only exports (`cf_conv2d_backward_input_{smem,
// channelfused,warp}_*`), the same category as Milestone 33/34's forced
// dWeight-candidate exports below -- `CUDABackend` never calls them; see
// `benchmarks/conv2d_backward_dinput_profile.py` for the isolated,
// side-by-side measurement against the unchanged production kernel above.

// -- Candidate A: shared-memory grad_output row-tile reuse across Cin -------

constexpr int MAX_CONV_K_SMEM = 32;  // matches MAX_CONV_K; per-block (not per-thread) table

template <typename T>
__global__ void k_conv2d_backward_input_smem(
    const T* grad_out, const T* w, T* grad_x,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);  // [h_count][Cout][Wout], cooperatively filled below
    __shared__ int s_kh[MAX_CONV_K_SMEM];
    __shared__ int s_ho[MAX_CONV_K_SMEM];
    __shared__ int s_h_count;

    long long block_id = blockIdx.x;  // one block per (n, hi)
    int hi = static_cast<int>(block_id % H);
    int n = static_cast<int>(block_id / H);

    if (threadIdx.x == 0) {
        int cnt = 0;
        for (int kh = 0; kh < KH && kh < MAX_CONV_K_SMEM; ++kh) {
            int t = hi + PH - kh;
            if (t % SH != 0) continue;
            int ho = t / SH;
            if (ho < 0 || ho >= Hout) continue;
            s_kh[cnt] = kh;
            s_ho[cnt] = ho;
            ++cnt;
        }
        s_h_count = cnt;
    }
    __syncthreads();

    int h_count = s_h_count;
    long long tile_elems = static_cast<long long>(h_count) * Cout * Wout;
    for (long long i = threadIdx.x; i < tile_elems; i += blockDim.x) {
        int wo = static_cast<int>(i % Wout);
        long long t1 = i / Wout;
        int co = static_cast<int>(t1 % Cout);
        int hi_i = static_cast<int>(t1 / Cout);
        int ho = s_ho[hi_i];
        long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
        smem[i] = grad_out[g_idx];
    }
    __syncthreads();

    long long total_work = static_cast<long long>(Cin) * W;
    for (long long work = threadIdx.x; work < total_work; work += blockDim.x) {
        int wi = static_cast<int>(work % W);
        int ci = static_cast<int>(work / W);

        int kw_valid[MAX_CONV_K_SMEM];
        int wo_valid[MAX_CONV_K_SMEM];
        int w_count = 0;
        for (int kw_ = 0; kw_ < KW && kw_ < MAX_CONV_K_SMEM; ++kw_) {
            int tw = wi + PW - kw_;
            if (tw % SW != 0) continue;
            int wo = tw / SW;
            if (wo < 0 || wo >= Wout) continue;
            kw_valid[w_count] = kw_;
            wo_valid[w_count] = wo;
            ++w_count;
        }

        T acc = static_cast<T>(0);
        for (int co = 0; co < Cout; ++co) {
            long long w_row = (static_cast<long long>(co) * Cin + ci) * KH;
            for (int hi_i = 0; hi_i < h_count; ++hi_i) {
                int kh = s_kh[hi_i];
                long long smem_row = (static_cast<long long>(hi_i) * Cout + co) * Wout;
                long long w_base = (w_row + kh) * KW;
                for (int wi_i = 0; wi_i < w_count; ++wi_i) {
                    int kw_ = kw_valid[wi_i], wo = wo_valid[wi_i];
                    acc += smem[smem_row + wo] * w[w_base + kw_];
                }
            }
        }
        long long out_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
        grad_x[out_idx] = acc;
    }
}

#define CONV2D_BACKWARD_INPUT_SMEM_LAUNCHER(TYPE, SUFFIX)                                \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_input_smem_##SUFFIX(         \
        const TYPE* grad_out, const TYPE* w, TYPE* grad_x,                               \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                          \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                              \
        int threads_per_block, void* stream) {                                           \
        long long blocks64 = static_cast<long long>(N) * H;                              \
        int blocks = blocks64 < 1 ? 1 : static_cast<int>(blocks64);                      \
        size_t shared_bytes = static_cast<size_t>(KH) * Cout * Wout * sizeof(TYPE);      \
        k_conv2d_backward_input_smem<TYPE><<<blocks, threads_per_block, shared_bytes, (cudaStream_t)stream>>>( \
            grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

CONV2D_BACKWARD_INPUT_SMEM_LAUNCHER(float, f32)
CONV2D_BACKWARD_INPUT_SMEM_LAUNCHER(double, f64)

// -- Candidate B: channel-fused work mapping, register-reused grad_output ---
//
// `k_conv2d_backward_input_channelfused` itself is defined earlier in this
// file (immediately after `k_conv2d_backward_input`), so the production
// dispatcher above can call it directly; only its profiling-only forced
// launcher (bypassing the `Cin`-based dispatch, for direct A/B/C comparison
// at every shape regardless of which path production would pick) lives here.

#define CONV2D_BACKWARD_INPUT_CHANNELFUSED_LAUNCHER(TYPE, SUFFIX)                        \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_input_channelfused_##SUFFIX( \
        const TYPE* grad_out, const TYPE* w, TYPE* grad_x,                               \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                          \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {              \
        long long total = static_cast<long long>(N) * H * W;                             \
        int blocks, threads;                                                             \
        launch_config(total, blocks, threads);                                           \
        k_conv2d_backward_input_channelfused<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

CONV2D_BACKWARD_INPUT_CHANNELFUSED_LAUNCHER(float, f32)
CONV2D_BACKWARD_INPUT_CHANNELFUSED_LAUNCHER(double, f64)

// -- Candidate C: warp-cooperative reduction over Cout ----------------------

template <typename T>
__global__ void k_conv2d_backward_input_warp(
    const T* grad_out, const T* w, T* grad_x,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout, long long total_outputs)
{
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    long long idx = static_cast<long long>(blockIdx.x) * warps_per_block + warp_in_block;
    if (idx >= total_outputs) return;

    int wi = static_cast<int>(idx % W);
    long long t1 = idx / W;
    int hi = static_cast<int>(t1 % H);
    long long t2 = t1 / H;
    int ci = static_cast<int>(t2 % Cin);
    int n = static_cast<int>(t2 / Cin);

    T acc = static_cast<T>(0);
    for (int co = lane; co < Cout; co += 32) {
        long long w_base = (static_cast<long long>(co) * Cin + ci) * KH * KW;
        for (int kh = 0; kh < KH; ++kh) {
            int t = hi + PH - kh;
            if (t % SH != 0) continue;
            int ho = t / SH;
            if (ho < 0 || ho >= Hout) continue;
            long long g_row = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout;
            long long w_row = w_base + static_cast<long long>(kh) * KW;
            for (int kw_ = 0; kw_ < KW; ++kw_) {
                int tw = wi + PW - kw_;
                if (tw % SW != 0) continue;
                int wo = tw / SW;
                if (wo < 0 || wo >= Wout) continue;
                acc += grad_out[g_row + wo] * w[w_row + kw_];
            }
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) grad_x[idx] = acc;
}

#define CONV2D_BACKWARD_INPUT_WARP_LAUNCHER(TYPE, SUFFIX)                                \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_input_warpreduce_##SUFFIX(   \
        const TYPE* grad_out, const TYPE* w, TYPE* grad_x,                               \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                          \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                              \
        int warps_per_block, void* stream) {                                             \
        long long total = static_cast<long long>(N) * Cin * H * W;                       \
        int threads = warps_per_block * 32;                                              \
        long long blocks64 = (total + warps_per_block - 1) / warps_per_block;            \
        int blocks = blocks64 < 1 ? 1 : static_cast<int>(blocks64);                      \
        k_conv2d_backward_input_warp<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout, total); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

CONV2D_BACKWARD_INPUT_WARP_LAUNCHER(float, f32)
CONV2D_BACKWARD_INPUT_WARP_LAUNCHER(double, f64)

// -- Conv2d backward: weight and bias (Milestone 21: block-per-output-element,
//    shared-memory tree reduction) ------------------------------------------
//
// Milestone 21 profiling (`docs/architecture/optimization.md`'s **CUDA
// Conv2d backward: weight/bias** section) measured these two kernels'
// original one-thread-per-output-element scheme as the dominant cost of the
// M20 CNN's CUDA backward pass -- specifically at its first conv layer
// (Cout=8, Cin=1, K=3: only 72 weight elements and 8 output channels, i.e.
// 72 and 8 *threads total*), where each of those few threads serially
// summed a 64-batch x 26x26-output reduction (43,264 iterations) entirely
// alone -- the 940MX's other ~300+ cores sat idle the whole time. Weight and
// bias gradients are exactly a "one reduction per output element" shape, so
// this is a textbook parallel-reduction fix: one thread *block* now owns
// each output element (a `(co, ci, kh, kw)` weight, or a `co` bias channel),
// its threads split the N*Hout*Wout reduction via a grid-stride loop, and a
// standard shared-memory tree reduction (identical in structure to
// `k_sum`'s, above) combines their partial sums before one thread writes the
// final value -- no atomics needed, since exactly one block ever writes a
// given output element.
//
// Measurement also showed this reduction kernel is *not* universally better:
// at the M20 CNN's second conv layer (Cout=16, Cin=8, K=3 -> 1,152 weight
// elements), the original one-thread-per-weight kernel already had enough
// threads to occupy the 940MX well, and paying 1,152 blocks' worth of
// `__syncthreads()`/shared-memory-reduction overhead for a much shorter
// (7,744-iteration) per-thread reduction made it *slower* (1.37ms -> 5.48ms
// measured). So `cf_conv2d_backward_weight_*` dispatches between the two
// kernels at launch time based on `total` weight-element count -- below
// `CONV2D_WEIGHT_REDUCE_THRESHOLD`, one block per weight element (few
// threads, long serial reduction: use the block reduction); at or above it,
// one thread per weight element (already enough threads: use the original
// kernel). The threshold (256) sits between the two measured cases (72 vs.
// 1,152) -- see the optimization doc for the full before/after numbers at
// both layer shapes. Bias gradients have no such crossover in any shape this
// milestone measured (`Cout` stays small -- 8, 16 -- for any CNN Forge's
// scope targets), so `cf_conv2d_backward_bias_*` always uses the block
// reduction. `k_conv2d_backward_input` (above) was left unchanged: it
// already launches one thread per *input* element (tens of thousands for
// every layer this milestone measured), so it was not the measured
// bottleneck.

constexpr int CONV2D_REDUCE_THREADS = 256;
constexpr long long CONV2D_WEIGHT_REDUCE_THRESHOLD = 256;

template <typename T>
__global__ void k_conv2d_backward_weight(
    const T* grad_out, const T* x, T* grad_w,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(Cout) * Cin * KH * KW;
    if (idx >= total) return;

    int kw_ = static_cast<int>(idx % KW);
    long long t1 = idx / KW;
    int kh = static_cast<int>(t1 % KH);
    long long t2 = t1 / KH;
    int ci = static_cast<int>(t2 % Cin);
    int co = static_cast<int>(t2 / Cin);

    T acc = static_cast<T>(0);
    for (int n = 0; n < N; ++n) {
        for (int ho = 0; ho < Hout; ++ho) {
            int hi = ho * SH - PH + kh;
            if (hi < 0 || hi >= H) continue;
            for (int wo = 0; wo < Wout; ++wo) {
                int wi = wo * SW - PW + kw_;
                if (wi < 0 || wi >= W) continue;
                long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
                long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
                acc += grad_out[g_idx] * x[x_idx];
            }
        }
    }
    grad_w[idx] = acc;
}

template <typename T>
__global__ void k_conv2d_backward_weight_reduce(
    const T* grad_out, const T* x, T* grad_w,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    long long widx = blockIdx.x;  // one block per (co, ci, kh, kw) weight element
    int kw_ = static_cast<int>(widx % KW);
    long long t1 = widx / KW;
    int kh = static_cast<int>(t1 % KH);
    long long t2 = t1 / KH;
    int ci = static_cast<int>(t2 % Cin);
    int co = static_cast<int>(t2 / Cin);

    long long reduce_total = static_cast<long long>(N) * Hout * Wout;
    T acc = static_cast<T>(0);
    for (long long r = threadIdx.x; r < reduce_total; r += blockDim.x) {
        int wo = static_cast<int>(r % Wout);
        long long r1 = r / Wout;
        int ho = static_cast<int>(r1 % Hout);
        int n = static_cast<int>(r1 / Hout);

        int hi = ho * SH - PH + kh;
        int wi = wo * SW - PW + kw_;
        if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;

        long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
        long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
        acc += grad_out[g_idx] * x[x_idx];
    }

    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_w[widx] = smem[0];
}

#define CONV2D_BACKWARD_WEIGHT_LAUNCHER(TYPE, SUFFIX)                                     \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_weight_##SUFFIX(              \
        const TYPE* grad_out, const TYPE* x, TYPE* grad_w,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {               \
        cudaStream_t s = (cudaStream_t)stream;                                            \
        long long total = static_cast<long long>(Cout) * Cin * KH * KW;                   \
        if (total < CONV2D_WEIGHT_REDUCE_THRESHOLD) {                                     \
            int blocks = total < 1 ? 1 : static_cast<int>(total);                         \
            k_conv2d_backward_weight_reduce<TYPE>                                         \
                <<<blocks, CONV2D_REDUCE_THREADS, CONV2D_REDUCE_THREADS * sizeof(TYPE), s>>>( \
                    grad_out, x, grad_w, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        } else {                                                                          \
            int blocks, threads;                                                          \
            launch_config(total, blocks, threads);                                        \
            k_conv2d_backward_weight<TYPE><<<blocks, threads, 0, s>>>(                    \
                grad_out, x, grad_w, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        }                                                                                  \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_WEIGHT_LAUNCHER(float, f32)
CONV2D_BACKWARD_WEIGHT_LAUNCHER(double, f64)

// -- dWeight cooperative-reduction profiling helpers (Milestone 33) ----------
//
// `cf_conv2d_backward_weight_*` (above) already dispatches between
// `k_conv2d_backward_weight` (one thread per weight element, full serial
// reduction) and `k_conv2d_backward_weight_reduce` (one block per weight
// element, shared-memory tree reduction) purely by weight-element count.
// Milestone 33 needs to measure each candidate kernel in isolation, at a
// fixed shape, independent of that dispatch decision -- the same isolation
// `benchmarks/conv2d_backward_profile.py` (Milestone 32) already relies on
// for `dInput`/`dWeight`/`dBias`, extended here since `dWeight`'s two
// candidate kernels are not otherwise independently reachable through the
// one exported dispatcher symbol. These are profiling-only entry points
// (the same category as Milestone 31's `cf_event_create_timed`/
// `cf_event_elapsed_ms`) -- `CUDABackend` never calls them; production
// dispatch remains `cf_conv2d_backward_weight_*` alone.
// `_blockreduce`'s `threads_per_block` must be a power of two (the tree
// reduction below assumes it, same as every other block-reduction kernel in
// this file) and is exposed so the block-size experiment (Section 14 of the
// milestone brief) can run without recompiling.

#define CONV2D_BACKWARD_WEIGHT_FORCED_LAUNCHERS(TYPE, SUFFIX)                             \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_weight_perthread_##SUFFIX(    \
        const TYPE* grad_out, const TYPE* x, TYPE* grad_w,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {               \
        long long total = static_cast<long long>(Cout) * Cin * KH * KW;                   \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        k_conv2d_backward_weight<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(     \
            grad_out, x, grad_w, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout);  \
        return static_cast<int>(cudaGetLastError());                                      \
    }                                                                                      \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_weight_blockreduce_##SUFFIX(   \
        const TYPE* grad_out, const TYPE* x, TYPE* grad_w,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                               \
        int threads_per_block, void* stream) {                                            \
        long long total = static_cast<long long>(Cout) * Cin * KH * KW;                   \
        int blocks = total < 1 ? 1 : static_cast<int>(total);                             \
        k_conv2d_backward_weight_reduce<TYPE>                                             \
            <<<blocks, threads_per_block, threads_per_block * sizeof(TYPE), (cudaStream_t)stream>>>( \
                grad_out, x, grad_w, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_WEIGHT_FORCED_LAUNCHERS(float, f32)
CONV2D_BACKWARD_WEIGHT_FORCED_LAUNCHERS(double, f64)

// -- dWeight warp-cooperative candidate (Milestone 33) -----------------------
//
// The block-per-weight candidate above (`k_conv2d_backward_weight_reduce`,
// forced via `cf_conv2d_backward_weight_blockreduce_*`) measured 3-4x
// *slower* than the existing per-thread kernel at every shape with >= 1,152
// weight elements, **independent of threads/block** (64/128/256 all landed
// within a few percent of each other at a given shape -- see the M33
// report's **Cooperative Strategy Evaluated** section) -- evidence the cost
// is dominated by launching/scheduling `weight_elements` *blocks* across the
// 940MX's 3 SMs, not by each block's own tree-reduction/`__syncthreads()`
// overhead. This candidate tests the natural fix for exactly that: pack
// several weight elements into *one* block, each owned by one warp (32
// lanes) that reduces its own slice of the `N * Hout * Wout` sum via
// `__shfl_down_sync` (no shared memory, no `__syncthreads()` -- a warp is
// always implicitly synchronized). `widx` depends only on `blockIdx.x` and
// `threadIdx.x / 32`, identical across every lane of a warp, so the leading
// `if (widx >= total_weights) return` and the loop's fixed (lane-independent)
// iteration count keep every lane of a live warp fully converged through the
// final shuffle reduction -- `__shfl_down_sync(0xFFFFFFFFu, ...)`'s
// full-warp-mask precondition genuinely holds. `warps_per_block` (2-8 tested)
// sets both threads/block (`warps_per_block * 32`) and weights/block,
// shrinking the block count by that same factor versus one-block-per-weight.
// Profiling-only, like the candidate above -- never called by `CUDABackend`.

template <typename T>
__global__ void k_conv2d_backward_weight_warp(
    const T* grad_out, const T* x, T* grad_w,
    int N, int Cin, int H, int W,
    int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW,
    int Hout, int Wout, long long total_weights)
{
    int lane = threadIdx.x & 31;
    int warp_in_block = threadIdx.x >> 5;
    int warps_per_block = blockDim.x >> 5;
    long long widx = static_cast<long long>(blockIdx.x) * warps_per_block + warp_in_block;
    if (widx >= total_weights) return;

    int kw_ = static_cast<int>(widx % KW);
    long long t1 = widx / KW;
    int kh = static_cast<int>(t1 % KH);
    long long t2 = t1 / KH;
    int ci = static_cast<int>(t2 % Cin);
    int co = static_cast<int>(t2 / Cin);

    long long reduce_total = static_cast<long long>(N) * Hout * Wout;
    T acc = static_cast<T>(0);
    for (long long r = lane; r < reduce_total; r += 32) {
        int wo = static_cast<int>(r % Wout);
        long long r1 = r / Wout;
        int ho = static_cast<int>(r1 % Hout);
        int n = static_cast<int>(r1 / Hout);

        int hi = ho * SH - PH + kh;
        int wi = wo * SW - PW + kw_;
        if (hi < 0 || hi >= H || wi < 0 || wi >= W) continue;

        long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
        long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
        acc += grad_out[g_idx] * x[x_idx];
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFFu, acc, offset);
    }
    if (lane == 0) grad_w[widx] = acc;
}

#define CONV2D_BACKWARD_WEIGHT_WARP_LAUNCHER(TYPE, SUFFIX)                                \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_weight_warpreduce_##SUFFIX(   \
        const TYPE* grad_out, const TYPE* x, TYPE* grad_w,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                               \
        int warps_per_block, void* stream) {                                              \
        long long total = static_cast<long long>(Cout) * Cin * KH * KW;                   \
        int threads = warps_per_block * 32;                                               \
        long long blocks64 = (total + warps_per_block - 1) / warps_per_block;             \
        int blocks = blocks64 < 1 ? 1 : static_cast<int>(blocks64);                       \
        k_conv2d_backward_weight_warp<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            grad_out, x, grad_w, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout, total); \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_WEIGHT_WARP_LAUNCHER(float, f32)
CONV2D_BACKWARD_WEIGHT_WARP_LAUNCHER(double, f64)

// -- Conv2d backward: bias (sum grad_output over batch and spatial dims) --

template <typename T>
__global__ void k_conv2d_backward_bias_reduce(const T* grad_out, T* grad_b, int N, int Cout, int Hout, int Wout) {
    extern __shared__ unsigned char smem_raw[];
    T* smem = reinterpret_cast<T*>(smem_raw);

    int co = blockIdx.x;  // one block per output channel
    long long reduce_total = static_cast<long long>(N) * Hout * Wout;
    T acc = static_cast<T>(0);
    for (long long r = threadIdx.x; r < reduce_total; r += blockDim.x) {
        int wo = static_cast<int>(r % Wout);
        long long r1 = r / Wout;
        int ho = static_cast<int>(r1 % Hout);
        int n = static_cast<int>(r1 / Hout);
        long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
        acc += grad_out[g_idx];
    }

    smem[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) grad_b[co] = smem[0];
}

#define CONV2D_BACKWARD_BIAS_LAUNCHER(TYPE, SUFFIX)                                       \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_bias_##SUFFIX(                \
        const TYPE* grad_out, TYPE* grad_b, int N, int Cout, int Hout, int Wout, void* stream) { \
        int blocks = Cout < 1 ? 1 : Cout;                                                 \
        k_conv2d_backward_bias_reduce<TYPE>                                               \
            <<<blocks, CONV2D_REDUCE_THREADS, CONV2D_REDUCE_THREADS * sizeof(TYPE), (cudaStream_t)stream>>>( \
                grad_out, grad_b, N, Cout, Hout, Wout);                                   \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_BIAS_LAUNCHER(float, f32)
CONV2D_BACKWARD_BIAS_LAUNCHER(double, f64)

// -- dWeight im2col + existing tiled GEMM candidate (Milestone 34, profiling-only) --
//
// M33 rejected cooperative reduction as a `dWeight` optimization and named
// im2col + GEMM (reusing the existing M11 `k_matmul` tiled GEMM above) as
// the next structurally-different candidate worth measuring
// (`docs/performance/conv2d-backward-profiling.md`'s M33 **Limitations**
// section). This section adds exactly the two gather kernels needed to
// reformulate `dWeight` as one GEMM call against `k_matmul` -- no second
// GEMM implementation (the milestone brief explicitly forbids one; see
// Section 43) and no change to `k_conv2d_backward_weight`/
// `k_conv2d_backward_weight_reduce` (still production code, both byte-for-
// byte unchanged).
//
// Forge's actual `dWeight` orientation (verified against `CPUBackend.
// conv2d_backward`, `forge/backend/cpu.py`, which already computes this
// exact GEMM on the CPU via NumPy/BLAS) is
// `grad_weight_mat = grad_out_rows.T @ cols_rows`, i.e. `(Cout, M) @ (M, K)
// -> (Cout, K)` where `M = N*Hout*Wout` (the reduction dimension) and
// `K = Cin*KH*KW` -- **not** the `Xcol^T @ dYmat -> (K, Cout)` orientation a
// literal reading of the milestone brief's Section 5 formula would suggest
// (that produces the transpose of what Forge's `(Cout, Cin, KH, KW)` weight
// layout needs). `cf_matmul_*` computes `C[M,N] = A[M,K] @ B[K,N]`
// (row-major, M/K/N as *that* GEMM's own dimensions -- distinct from this
// comment's `M`/`K` above), so this candidate calls it as
// `cf_matmul(A=dYcolT, B=Xcol, GEMM_M=Cout, GEMM_K=N*Hout*Wout, GEMM_N=Cin*KH*KW)`,
// producing `(Cout, Cin*KH*KW)` directly -- already exactly `grad_weight`'s
// contiguous layout, reinterpreted with zero copy (Section 6's "final
// reshape" costs nothing here).
//
// Two gather kernels build the GEMM's operands:
//
// - `k_im2col_conv2d`: `Xcol`, shape `(M, K) = (N*Hout*Wout, Cin*KH*KW)`,
//   row-major -- one thread per `(m, k)` output element, identical
//   padding/stride/boundary handling to `k_conv2d_forward` above (zero for
//   an out-of-bounds `(hi, wi)`).
// - `k_conv2d_grad_output_permute`: `dYcolT`, shape `(Cout, M)`, row-major --
//   one thread per `(co, m)` output element. This is *not* a generic 2D
//   transpose (`k_transpose` above): `grad_output`'s actual memory layout is
//   `(N, Cout, Hout, Wout)` (`N` outermost, `Cout` next), so producing
//   `(Cout, N*Hout*Wout)` -- `Cout` outermost -- is a true 4D-to-2D gather,
//   not a reshape. This permute is exactly the "reshape/copy" cost the
//   milestone brief (Sections 8/17/20) asks to measure separately from
//   im2col construction and GEMM time.
//
// Milestone 34 decision: **adopted** for `dWeight` at/above a weight-element
// threshold (measured 1.12-1.59x faster end-to-end than the existing
// per-thread kernel at every representative shape >= 1,152 weight elements;
// slower below the existing 256-element `CONV2D_WEIGHT_REDUCE_THRESHOLD`
// boundary -- see `docs/performance/conv2d-backward-profiling.md`'s
// **Milestone 34** section). Reached from `CUDABackend.conv2d_backward`
// only indirectly, through `forge.backend.cuda.experimental_conv_im2col.
// dweight_im2col_gemm` (Python) -- these two kernels stay separate exported
// symbols rather than being folded into `cf_conv2d_backward_weight_*`
// because the full pipeline (im2col -> permute -> GEMM, three launches with
// two temporary device buffers) needs Python-level buffer management the
// existing single-launch C dispatcher signature has no room for.
// `k_conv2d_backward_weight`/`k_conv2d_backward_weight_reduce` remain
// production code below the threshold, byte-for-byte unchanged.

template <typename T>
__global__ void k_im2col_conv2d(
    const T* x, T* col,
    int N, int Cin, int H, int W,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long K = static_cast<long long>(Cin) * KH * KW;
    long long total = static_cast<long long>(N) * Hout * Wout * K;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    long long m = idx / K;
    long long k = idx % K;

    int kw_ = static_cast<int>(k % KW);
    long long k1 = k / KW;
    int kh = static_cast<int>(k1 % KH);
    int ci = static_cast<int>(k1 / KH);

    int wo = static_cast<int>(m % Wout);
    long long m1 = m / Wout;
    int ho = static_cast<int>(m1 % Hout);
    int n = static_cast<int>(m1 / Hout);

    int hi = ho * SH - PH + kh;
    int wi = wo * SW - PW + kw_;

    T v = static_cast<T>(0);
    if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
        long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
        v = x[x_idx];
    }
    col[idx] = v;
}

#define IM2COL_CONV2D_LAUNCHER(TYPE, SUFFIX)                                             \
    extern "C" __declspec(dllexport) int cf_im2col_conv2d_##SUFFIX(                      \
        const TYPE* x, TYPE* col,                                                        \
        int N, int Cin, int H, int W, int KH, int KW,                                    \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {              \
        long long K = static_cast<long long>(Cin) * KH * KW;                             \
        long long total = static_cast<long long>(N) * Hout * Wout * K;                   \
        int blocks, threads;                                                             \
        launch_config(total, blocks, threads);                                           \
        k_im2col_conv2d<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(             \
            x, col, N, Cin, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout);                    \
        return static_cast<int>(cudaGetLastError());                                     \
    }

IM2COL_CONV2D_LAUNCHER(float, f32)
IM2COL_CONV2D_LAUNCHER(double, f64)

template <typename T>
__global__ void k_conv2d_grad_output_permute(
    const T* grad_out, T* dycolT, int N, int Cout, int Hout, int Wout)
{
    long long M = static_cast<long long>(N) * Hout * Wout;
    long long total = static_cast<long long>(Cout) * M;
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (idx >= total) return;

    long long m = idx % M;
    int co = static_cast<int>(idx / M);

    int wo = static_cast<int>(m % Wout);
    long long m1 = m / Wout;
    int ho = static_cast<int>(m1 % Hout);
    int n = static_cast<int>(m1 / Hout);

    long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
    dycolT[idx] = grad_out[g_idx];
}

#define GRAD_OUTPUT_PERMUTE_LAUNCHER(TYPE, SUFFIX)                                       \
    extern "C" __declspec(dllexport) int cf_conv2d_grad_output_permute_##SUFFIX(         \
        const TYPE* grad_out, TYPE* dycolT, int N, int Cout, int Hout, int Wout, void* stream) { \
        long long total = static_cast<long long>(Cout) * N * Hout * Wout;                \
        int blocks, threads;                                                             \
        launch_config(total, blocks, threads);                                           \
        k_conv2d_grad_output_permute<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            grad_out, dycolT, N, Cout, Hout, Wout);                                      \
        return static_cast<int>(cudaGetLastError());                                     \
    }

GRAD_OUTPUT_PERMUTE_LAUNCHER(float, f32)
GRAD_OUTPUT_PERMUTE_LAUNCHER(double, f64)

// -- M37 Candidate A: fused gather + tiled GEMM for dWeight (profiling-only) --
//
// `benchmarks/m37_dweight_profile.py` decomposed the M34 pipeline and found
// `k_im2col_conv2d` + `k_conv2d_grad_output_permute` (pure data-movement,
// zero FLOPs) together cost 54-63% of total dWeight time at every
// production (>=256-weight-element) shape -- more than the GEMM itself.
// Both materialize a buffer purely so `k_matmul` can read it back with a
// second kernel launch: `Xcol` (`M x K`, up to ~90x the input's own element
// count for `mnist_conv2`'s shape) and `dYcolT` (`Cout x M`). This candidate
// is the standard "implicit GEMM" trick: fold both gathers directly into
// `k_matmul`'s own shared-memory tile-load step, reading straight from `x`/
// `grad_output` with the same index arithmetic `k_im2col_conv2d`/
// `k_conv2d_grad_output_permute` already use, so the two intermediate
// buffers and their two kernel launches never exist. `k_matmul` itself is
// untouched (a separate kernel, not a modification, per the milestone
// brief's Section 6) -- this is a Conv2d-specific dispatch, never a generic
// GEMM change.
//
// Same GEMM orientation as M34 (`dweight_im2col_gemm`): `out[Cout, Cin*KH*
// KW]` (`out`'s own layout is already `weight_shape`, zero-copy). One block
// per 16x16 output tile, same as `k_matmul` -- occupancy is unchanged by
// this candidate (see Candidate C below for that).

template <typename T>
__global__ void k_dweight_fused_gemm(
    const T* grad_out, const T* x, T* out,
    int N, int Cin, int H, int W, int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW, int Hout, int Wout)
{
    __shared__ T tile_a[MATMUL_TILE][MATMUL_TILE];
    __shared__ T tile_b[MATMUL_TILE][MATMUL_TILE];

    long long Mdim = static_cast<long long>(N) * Hout * Wout;
    int Kdim = Cin * KH * KW;

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;  // co
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;  // flattened (ci, kh, kw)

    T acc = static_cast<T>(0);
    long long num_tiles = (Mdim + MATMUL_TILE - 1) / MATMUL_TILE;
    for (long long t = 0; t < num_tiles; ++t) {
        long long a_m = t * MATMUL_TILE + threadIdx.x;
        long long b_m = t * MATMUL_TILE + threadIdx.y;

        T a_val = static_cast<T>(0);
        if (row < Cout && a_m < Mdim) {
            int wo = static_cast<int>(a_m % Wout);
            long long m1 = a_m / Wout;
            int ho = static_cast<int>(m1 % Hout);
            int n = static_cast<int>(m1 / Hout);
            long long g_idx = ((static_cast<long long>(n) * Cout + row) * Hout + ho) * Wout + wo;
            a_val = grad_out[g_idx];
        }
        tile_a[threadIdx.y][threadIdx.x] = a_val;

        T b_val = static_cast<T>(0);
        if (b_m < Mdim && col < Kdim) {
            int kw_ = col % KW;
            int k1 = col / KW;
            int kh = k1 % KH;
            int ci = k1 / KH;
            int wo = static_cast<int>(b_m % Wout);
            long long m1 = b_m / Wout;
            int ho = static_cast<int>(m1 % Hout);
            int n = static_cast<int>(m1 / Hout);
            int hi = ho * SH - PH + kh;
            int wi = wo * SW - PW + kw_;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
                b_val = x[x_idx];
            }
        }
        tile_b[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < Cout && col < Kdim) {
        out[row * Kdim + col] = acc;
    }
}

#define DWEIGHT_FUSED_GEMM_LAUNCHER(TYPE, SUFFIX)                                        \
    extern "C" __declspec(dllexport) int cf_dweight_fused_gemm_##SUFFIX(                 \
        const TYPE* grad_out, const TYPE* x, TYPE* out,                                  \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                          \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {              \
        int Kdim = Cin * KH * KW;                                                        \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                          \
        dim3 blocks((Kdim + MATMUL_TILE - 1) / MATMUL_TILE, (Cout + MATMUL_TILE - 1) / MATMUL_TILE); \
        k_dweight_fused_gemm<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(         \
            grad_out, x, out, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout);    \
        return static_cast<int>(cudaGetLastError());                                     \
    }

DWEIGHT_FUSED_GEMM_LAUNCHER(float, f32)
DWEIGHT_FUSED_GEMM_LAUNCHER(double, f64)

// -- M37 Candidate C: split-K fused gather GEMM for dWeight (profiling-only) --
//
// `benchmarks/m37_dweight_profile.py` also found the GEMM's own launch
// geometry (`ceil(Cout/16) * ceil(Cin*KH*KW/16)` blocks -- the huge
// `N*Hout*Wout` reduction lives entirely inside each block's serial inner
// loop, invisible to block count) launches as few as 5 of the 940MX's 24
// resident-block device capacity (3 SMs x 8 resident 256-thread blocks/SM,
// itself thread-count-limited, not register/shared-memory-limited --
// confirmed via `nvcc -Xptxas -v`, zero stack frame/spill on every dWeight-
// related kernel). Measured occupancy fraction (`total_blocks /
// device_block_capacity`) correlates directly with achieved fraction of the
// compute-ceiling: 4.2%/20.8%/75.0% occupancy shapes measured 1.9%/14.4%/
// 32.9% of the practical compute ceiling respectively.
//
// This candidate builds on Candidate A (same fused on-the-fly gather, no
// separate `Xcol`/`dYcolT` buffers) and additionally splits the reduction
// dimension (`Mdim = N*Hout*Wout`) into `num_k_splits` chunks along
// `blockIdx.z`, multiplying the block count by `num_k_splits` -- directly
// targeting the measured occupancy shortfall. Each block accumulates only
// its own chunk and atomically adds its partial sum into `out` (`Cout *
// Cin*KH*KW` elements -- small, e.g. 72-4,608 -- so contention is bounded by
// `num_k_splits`, not by the reduction size). `out` must be zeroed before
// this kernel is launched (the caller's responsibility, via
// `cudaMemsetAsync` -- zero-bits is a valid representation of `0.0` for
// both `float` and `double`, so a byte-level memset is exact, not an
// approximation). `num_k_splits` is a caller-supplied parameter (not
// autotuned inside the kernel) so `benchmarks/m37_dweight_profile.py` can
// sweep it directly per shape before any production dispatch is chosen.

template <typename T>
__device__ inline void atomic_add_dweight(T* address, T val) {
    if constexpr (std::is_same<T, double>::value) {
        atomic_add_f64(address, val);
    } else {
        atomicAdd(address, val);
    }
}

template <typename T>
__global__ void k_dweight_fused_gemm_splitk(
    const T* grad_out, const T* x, T* out,
    int N, int Cin, int H, int W, int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW, int Hout, int Wout,
    int num_k_splits)
{
    __shared__ T tile_a[MATMUL_TILE][MATMUL_TILE];
    __shared__ T tile_b[MATMUL_TILE][MATMUL_TILE];

    long long Mdim = static_cast<long long>(N) * Hout * Wout;
    int Kdim = Cin * KH * KW;

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;

    long long total_tiles = (Mdim + MATMUL_TILE - 1) / MATMUL_TILE;
    long long tiles_per_split = (total_tiles + num_k_splits - 1) / num_k_splits;
    long long tile_start = static_cast<long long>(blockIdx.z) * tiles_per_split;
    long long tile_end = min(tile_start + tiles_per_split, total_tiles);

    T acc = static_cast<T>(0);
    for (long long t = tile_start; t < tile_end; ++t) {
        long long a_m = t * MATMUL_TILE + threadIdx.x;
        long long b_m = t * MATMUL_TILE + threadIdx.y;

        T a_val = static_cast<T>(0);
        if (row < Cout && a_m < Mdim) {
            int wo = static_cast<int>(a_m % Wout);
            long long m1 = a_m / Wout;
            int ho = static_cast<int>(m1 % Hout);
            int n = static_cast<int>(m1 / Hout);
            long long g_idx = ((static_cast<long long>(n) * Cout + row) * Hout + ho) * Wout + wo;
            a_val = grad_out[g_idx];
        }
        tile_a[threadIdx.y][threadIdx.x] = a_val;

        T b_val = static_cast<T>(0);
        if (b_m < Mdim && col < Kdim) {
            int kw_ = col % KW;
            int k1 = col / KW;
            int kh = k1 % KH;
            int ci = k1 / KH;
            int wo = static_cast<int>(b_m % Wout);
            long long m1 = b_m / Wout;
            int ho = static_cast<int>(m1 % Hout);
            int n = static_cast<int>(m1 / Hout);
            int hi = ho * SH - PH + kh;
            int wi = wo * SW - PW + kw_;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
                b_val = x[x_idx];
            }
        }
        tile_b[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < Cout && col < Kdim && tile_start < tile_end) {
        atomic_add_dweight(&out[row * Kdim + col], acc);
    }
}

#define DWEIGHT_FUSED_GEMM_SPLITK_LAUNCHER(TYPE, SUFFIX)                                 \
    extern "C" __declspec(dllexport) int cf_dweight_fused_gemm_splitk_##SUFFIX(          \
        const TYPE* grad_out, const TYPE* x, TYPE* out,                                  \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                          \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                              \
        int num_k_splits, void* stream) {                                                \
        int Kdim = Cin * KH * KW;                                                        \
        cudaMemsetAsync(out, 0, static_cast<size_t>(Cout) * Kdim * sizeof(TYPE), (cudaStream_t)stream); \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                          \
        dim3 blocks(                                                                     \
            (Kdim + MATMUL_TILE - 1) / MATMUL_TILE, (Cout + MATMUL_TILE - 1) / MATMUL_TILE, \
            num_k_splits);                                                               \
        k_dweight_fused_gemm_splitk<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            grad_out, x, out, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout,     \
            num_k_splits);                                                               \
        return static_cast<int>(cudaGetLastError());                                     \
    }

DWEIGHT_FUSED_GEMM_SPLITK_LAUNCHER(float, f32)
DWEIGHT_FUSED_GEMM_SPLITK_LAUNCHER(double, f64)

// -- M37 Candidate E: split-K GEMM over the existing materialized Xcol/dYcolT
//    buffers (profiling-only) --
//
// Candidate A/C (above) measured *slower* than the M34 baseline at every
// shape with >= 18 GEMM blocks (already 75% of device block capacity) --
// the fused kernel's recomputed gather indices (integer div/mod per tile
// iteration, replacing one coalesced buffer read) cost more than the
// occupancy fix bought back once occupancy was no longer the bottleneck.
// This candidate isolates the two effects Candidate C conflated: keep
// M34's already-fast, cache-friendly buffer reads (`Xcol`/`dYcolT`, built
// by the unmodified `k_im2col_conv2d`/`k_conv2d_grad_output_permute`), and
// apply *only* the split-K occupancy fix on top of them -- a narrowly
// scoped GEMM variant (not a change to `k_matmul` itself, which remains
// completely unmodified and used everywhere else).
//
// `C` must be zeroed before this kernel is launched, exactly as Candidate
// C's `cf_dweight_fused_gemm_splitk_*` requires (`cudaMemsetAsync`, exact
// for both `float`/`double` since zero-bits is a valid `0.0` for both).

template <typename T>
__global__ void k_matmul_splitk(const T* A, const T* B, T* C, int M, int K, int N, int num_k_splits) {
    __shared__ T tile_a[MATMUL_TILE][MATMUL_TILE];
    __shared__ T tile_b[MATMUL_TILE][MATMUL_TILE];

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;

    int total_tiles = (K + MATMUL_TILE - 1) / MATMUL_TILE;
    int tiles_per_split = (total_tiles + num_k_splits - 1) / num_k_splits;
    int tile_start = blockIdx.z * tiles_per_split;
    int tile_end = min(tile_start + tiles_per_split, total_tiles);

    T acc = static_cast<T>(0);
    for (int t = tile_start; t < tile_end; ++t) {
        int a_col = t * MATMUL_TILE + threadIdx.x;
        int b_row = t * MATMUL_TILE + threadIdx.y;

        tile_a[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : static_cast<T>(0);
        tile_b[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : static_cast<T>(0);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N && tile_start < tile_end) {
        atomic_add_dweight(&C[row * N + col], acc);
    }
}

#define MATMUL_SPLITK_LAUNCHER(TYPE, SUFFIX)                                             \
    extern "C" __declspec(dllexport) int cf_matmul_splitk_##SUFFIX(                      \
        const TYPE* A, const TYPE* B, TYPE* C, int M, int K, int N,                      \
        int num_k_splits, void* stream) {                                                \
        cudaMemsetAsync(C, 0, static_cast<size_t>(M) * N * sizeof(TYPE), (cudaStream_t)stream); \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                          \
        dim3 blocks(                                                                     \
            (N + MATMUL_TILE - 1) / MATMUL_TILE, (M + MATMUL_TILE - 1) / MATMUL_TILE,     \
            num_k_splits);                                                               \
        k_matmul_splitk<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(              \
            A, B, C, M, K, N, num_k_splits);                                             \
        return static_cast<int>(cudaGetLastError());                                     \
    }

MATMUL_SPLITK_LAUNCHER(float, f32)
MATMUL_SPLITK_LAUNCHER(double, f64)

// -- Milestone 38 Candidate B: half-fused split-K GEMM for dWeight
//    (materialized dYcolT, on-the-fly Xcol gather) --
//
// M37's Candidate A/C (`k_dweight_fused_gemm[_splitk]`, above) fused *both*
// gathers into the GEMM tile load and lost at every shape with >= 18 GEMM
// blocks. The root cause, established analytically and confirmed by that
// data: in a block-tiled GEMM, `tile_a`'s load depends only on `(row, a_m)`
// -- not on `blockIdx.x` -- so every block sharing a `blockIdx.y` redundantly
// regathers the *same* `tile_a` data `blocks_x` times; symmetrically
// `tile_b` is redundantly regathered `blocks_y` times. At every one of
// M38's 7 representative shapes, `Cout <= 32` (`blocks_y = ceil(Cout/16)
// <= 2`) while `Cin*KH*KW` reaches 144 (`blocks_x` up to 9) -- so fusing
// the `grad_output` gather (Candidate A/C's `tile_a`) pays up to a 9x
// redundant-regather tax, while fusing only the *im2col* gather (`tile_b`,
// this milestone's actual target) pays at most 2x. This candidate fuses
// only `tile_b` (eliminating the `Xcol` buffer -- the dominant `im2col`
// cost this milestone investigates) and keeps `tile_a` reading the cheap,
// already-materialized `dYcolT` (`k_conv2d_grad_output_permute`, unchanged,
// 7-8% of pipeline time per M37) -- deliberately asymmetric, unlike
// Candidate A/C's "fuse everything." Split-K (`blockIdx.z`) is folded in
// directly since M37 already established it as a pure win with no
// downside at every representative shape.

template <typename T>
__global__ void k_dweight_halffused_gemm_splitk(
    const T* dycolT, const T* x, T* out,
    int N, int Cin, int H, int W, int Cout, int KH, int KW,
    int SH, int SW, int PH, int PW, int Hout, int Wout,
    int num_k_splits)
{
    __shared__ T tile_a[MATMUL_TILE][MATMUL_TILE];
    __shared__ T tile_b[MATMUL_TILE][MATMUL_TILE];

    long long Mdim = static_cast<long long>(N) * Hout * Wout;
    int Kdim = Cin * KH * KW;

    int row = blockIdx.y * MATMUL_TILE + threadIdx.y;  // Cout
    int col = blockIdx.x * MATMUL_TILE + threadIdx.x;  // Cin*KH*KW

    long long total_tiles = (Mdim + MATMUL_TILE - 1) / MATMUL_TILE;
    long long tiles_per_split = (total_tiles + num_k_splits - 1) / num_k_splits;
    long long tile_start = static_cast<long long>(blockIdx.z) * tiles_per_split;
    long long tile_end = min(tile_start + tiles_per_split, total_tiles);

    T acc = static_cast<T>(0);
    for (long long t = tile_start; t < tile_end; ++t) {
        long long a_m = t * MATMUL_TILE + threadIdx.x;
        long long b_m = t * MATMUL_TILE + threadIdx.y;

        T a_val = static_cast<T>(0);
        if (row < Cout && a_m < Mdim) {
            a_val = dycolT[static_cast<long long>(row) * Mdim + a_m];
        }
        tile_a[threadIdx.y][threadIdx.x] = a_val;

        T b_val = static_cast<T>(0);
        if (b_m < Mdim && col < Kdim) {
            int kw_ = col % KW;
            int k1 = col / KW;
            int kh = k1 % KH;
            int ci = k1 / KH;
            int wo = static_cast<int>(b_m % Wout);
            long long m1 = b_m / Wout;
            int ho = static_cast<int>(m1 % Hout);
            int n = static_cast<int>(m1 / Hout);
            int hi = ho * SH - PH + kh;
            int wi = wo * SW - PW + kw_;
            if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                long long x_idx = ((static_cast<long long>(n) * Cin + ci) * H + hi) * W + wi;
                b_val = x[x_idx];
            }
        }
        tile_b[threadIdx.y][threadIdx.x] = b_val;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < MATMUL_TILE; ++k) {
            acc += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < Cout && col < Kdim && tile_start < tile_end) {
        atomic_add_dweight(&out[row * Kdim + col], acc);
    }
}

#define DWEIGHT_HALFFUSED_GEMM_SPLITK_LAUNCHER(TYPE, SUFFIX)                             \
    extern "C" __declspec(dllexport) int cf_dweight_halffused_gemm_splitk_##SUFFIX(     \
        const TYPE* dycolT, const TYPE* x, TYPE* out,                                   \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                         \
        int SH, int SW, int PH, int PW, int Hout, int Wout,                             \
        int num_k_splits, void* stream) {                                               \
        int Kdim = Cin * KH * KW;                                                       \
        cudaMemsetAsync(out, 0, static_cast<size_t>(Cout) * Kdim * sizeof(TYPE), (cudaStream_t)stream); \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                         \
        dim3 blocks(                                                                    \
            (Kdim + MATMUL_TILE - 1) / MATMUL_TILE, (Cout + MATMUL_TILE - 1) / MATMUL_TILE, \
            num_k_splits);                                                              \
        k_dweight_halffused_gemm_splitk<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>( \
            dycolT, x, out, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout,      \
            num_k_splits);                                                              \
        return static_cast<int>(cudaGetLastError());                                    \
    }

DWEIGHT_HALFFUSED_GEMM_SPLITK_LAUNCHER(float, f32)
DWEIGHT_HALFFUSED_GEMM_SPLITK_LAUNCHER(double, f64)

// -- MaxPool2d forward ----------------------------------------------------------
//
// Ties within a window break to the first maximum encountered in row-major
// (kh, then kw) scan order -- the strict `v > best` comparison below never
// replaces an already-found maximum with an equal later value. `CPUBackend.
// max_pool2d_backward` (forge/backend/cpu.py) applies `np.argmax` to a window
// flattened in that same order, so both backends agree on which element in a
// tied window receives the gradient.

template <typename T>
__global__ void k_maxpool2d_forward(
    const T* x, T* out,
    int N, int C, int H, int W,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(N) * C * Hout * Wout;
    if (idx >= total) return;

    int wo = static_cast<int>(idx % Wout);
    long long t1 = idx / Wout;
    int ho = static_cast<int>(t1 % Hout);
    long long t2 = t1 / Hout;
    int c = static_cast<int>(t2 % C);
    int n = static_cast<int>(t2 / C);

    T best = cf_neg_max<T>();
    for (int kh = 0; kh < KH; ++kh) {
        int hi = ho * SH - PH + kh;
        if (hi < 0 || hi >= H) continue;
        for (int kw_ = 0; kw_ < KW; ++kw_) {
            int wi = wo * SW - PW + kw_;
            if (wi < 0 || wi >= W) continue;
            long long x_idx = ((static_cast<long long>(n) * C + c) * H + hi) * W + wi;
            T v = x[x_idx];
            if (v > best) best = v;
        }
    }
    out[idx] = best;
}

#define MAXPOOL2D_FORWARD_LAUNCHER(TYPE, SUFFIX)                                          \
    extern "C" __declspec(dllexport) int cf_maxpool2d_forward_##SUFFIX(                   \
        const TYPE* x, TYPE* out,                                                         \
        int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW,       \
        int Hout, int Wout, void* stream) {                                               \
        long long total = static_cast<long long>(N) * C * Hout * Wout;                    \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        k_maxpool2d_forward<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(          \
            x, out, N, C, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout);                       \
        return static_cast<int>(cudaGetLastError());                                      \
    }

MAXPOOL2D_FORWARD_LAUNCHER(float, f32)
MAXPOOL2D_FORWARD_LAUNCHER(double, f64)

// -- MaxPool2d backward ----------------------------------------------------------
//
// One thread per *output* element (same indexing as forward), recomputing
// that window's argmax from the saved input `x` and scattering the upstream
// gradient there with `atomicAdd` -- overlapping windows (stride < kernel)
// can otherwise target the same input element from more than one thread, so
// a plain (non-atomic) write would race. `grad_x` is zeroed by the launcher
// (`cudaMemset`, mirroring `cf_sum_*`'s own launcher above) before any thread
// writes to it.

template <typename T>
__global__ void k_maxpool2d_backward(
    const T* x, const T* grad_out, T* grad_x,
    int N, int C, int H, int W,
    int KH, int KW, int SH, int SW, int PH, int PW,
    int Hout, int Wout)
{
    long long idx = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    long long total = static_cast<long long>(N) * C * Hout * Wout;
    if (idx >= total) return;

    int wo = static_cast<int>(idx % Wout);
    long long t1 = idx / Wout;
    int ho = static_cast<int>(t1 % Hout);
    long long t2 = t1 / Hout;
    int c = static_cast<int>(t2 % C);
    int n = static_cast<int>(t2 / C);

    T best = cf_neg_max<T>();
    long long best_idx = -1;
    for (int kh = 0; kh < KH; ++kh) {
        int hi = ho * SH - PH + kh;
        if (hi < 0 || hi >= H) continue;
        for (int kw_ = 0; kw_ < KW; ++kw_) {
            int wi = wo * SW - PW + kw_;
            if (wi < 0 || wi >= W) continue;
            long long x_idx = ((static_cast<long long>(n) * C + c) * H + hi) * W + wi;
            T v = x[x_idx];
            if (v > best) { best = v; best_idx = x_idx; }
        }
    }
    if (best_idx >= 0) {
        atomic_add_generic<T>(&grad_x[best_idx], grad_out[idx]);
    }
}

#define MAXPOOL2D_BACKWARD_LAUNCHER(TYPE, SUFFIX)                                         \
    extern "C" __declspec(dllexport) int cf_maxpool2d_backward_##SUFFIX(                  \
        const TYPE* x, const TYPE* grad_out, TYPE* grad_x,                                \
        int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PH, int PW,       \
        int Hout, int Wout, void* stream) {                                               \
        cudaStream_t s = (cudaStream_t)stream;                                            \
        cudaError_t memset_err = cudaMemsetAsync(                                         \
            grad_x, 0, sizeof(TYPE) * static_cast<size_t>(N) * C * H * W, s);             \
        if (memset_err != cudaSuccess) return static_cast<int>(memset_err);               \
        long long total = static_cast<long long>(N) * C * Hout * Wout;                    \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        k_maxpool2d_backward<TYPE><<<blocks, threads, 0, s>>>(                            \
            x, grad_out, grad_x, N, C, H, W, KH, KW, SH, SW, PH, PW, Hout, Wout);          \
        return static_cast<int>(cudaGetLastError());                                      \
    }

MAXPOOL2D_BACKWARD_LAUNCHER(float, f32)
MAXPOOL2D_BACKWARD_LAUNCHER(double, f64)

// -- Dropout mask (Milestone 16) ----------------------------------------------
//
// Every element's Bernoulli(1-p) draw is generated entirely on-device, one
// thread per element, from a stateless hash of (seed, index) -- no curand
// dependency, no per-element host round-trip, no NumPy involvement (the
// milestone brief explicitly rules out generating the mask via NumPy and
// forbids any CPU fallback for CUDA Dropout). `seed` is the one piece of
// host-side randomness: a single integer drawn once per forward call from
// `forge.random`'s own generator (`CUDABackend.dropout_mask`,
// `forge/backend/cuda/backend.py`) -- a cheap scalar draw, not the
// per-element array generation the brief forbids.
//
// `cf_splitmix64` is the standard SplitMix64 finalizer (Vigna, 2015): a
// correctness-first, statistically well-distributed stateless hash, not a
// cryptographic or sophisticated GPU RNG library -- exactly what the
// milestone brief asks for ("a simple correctness-first CUDA implementation
// is sufficient... do not attempt to implement a sophisticated GPU RNG
// library"). Combining `seed` with each element's flat index gives every
// element an independent-looking, reproducible-given-`seed` draw, computed
// directly from `mask`'s own thread index -- no shared state, no RNG object
// to allocate or free.

__device__ inline unsigned long long cf_splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x ^= (x >> 31);
    return x;
}

template <typename T>
__global__ void k_dropout_mask(T* mask, long long n, double p, unsigned long long seed) {
    long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (i >= n) return;

    unsigned long long h = cf_splitmix64(seed ^ cf_splitmix64(static_cast<unsigned long long>(i)));
    // Top 53 bits -> a uniform double in [0, 1), the same bit-width IEEE 754
    // doubles can represent exactly -- avoids the modulo bias a naive
    // `h % range` would introduce.
    double u = static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0);  // 2^53
    bool keep = u >= p;
    mask[i] = keep ? static_cast<T>(1.0 / (1.0 - p)) : static_cast<T>(0);
}

#define DROPOUT_MASK_LAUNCHER(TYPE, SUFFIX)                                              \
    extern "C" __declspec(dllexport) int cf_dropout_mask_##SUFFIX(                       \
        TYPE* mask, long long n, double p, unsigned long long seed, void* stream) {      \
        int blocks, threads;                                                             \
        launch_config(n, blocks, threads);                                               \
        k_dropout_mask<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(mask, n, p, seed); \
        return static_cast<int>(cudaGetLastError());                                     \
    }

DROPOUT_MASK_LAUNCHER(float, f32)
DROPOUT_MASK_LAUNCHER(double, f64)

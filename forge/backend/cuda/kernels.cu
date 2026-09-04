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

    T acc = static_cast<T>(0);
    for (int co = 0; co < Cout; ++co) {
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
                long long g_idx = ((static_cast<long long>(n) * Cout + co) * Hout + ho) * Wout + wo;
                long long w_idx = ((static_cast<long long>(co) * Cin + ci) * KH + kh) * KW + kw_;
                acc += grad_out[g_idx] * w[w_idx];
            }
        }
    }
    grad_x[idx] = acc;
}

#define CONV2D_BACKWARD_INPUT_LAUNCHER(TYPE, SUFFIX)                                      \
    extern "C" __declspec(dllexport) int cf_conv2d_backward_input_##SUFFIX(               \
        const TYPE* grad_out, const TYPE* w, TYPE* grad_x,                                \
        int N, int Cin, int H, int W, int Cout, int KH, int KW,                           \
        int SH, int SW, int PH, int PW, int Hout, int Wout, void* stream) {               \
        long long total = static_cast<long long>(N) * Cin * H * W;                        \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        k_conv2d_backward_input<TYPE><<<blocks, threads, 0, (cudaStream_t)stream>>>(      \
            grad_out, w, grad_x, N, Cin, H, W, Cout, KH, KW, SH, SW, PH, PW, Hout, Wout);  \
        return static_cast<int>(cudaGetLastError());                                      \
    }

CONV2D_BACKWARD_INPUT_LAUNCHER(float, f32)
CONV2D_BACKWARD_INPUT_LAUNCHER(double, f64)

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

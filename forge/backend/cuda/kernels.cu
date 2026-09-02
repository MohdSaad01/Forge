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
        const TYPE* a, const TYPE* b, TYPE* out, long long n) {                            \
        int blocks, threads;                                                               \
        launch_config(n, blocks, threads);                                                 \
        KERNEL<TYPE><<<blocks, threads>>>(a, b, out, n);                                   \
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
        long long rows, long long cols, int vec_is_left) {                                \
        long long total = rows * cols;                                                    \
        int blocks, threads;                                                              \
        launch_config(total, blocks, threads);                                            \
        KERNEL<TYPE><<<blocks, threads>>>(mat, vec, out, rows, cols, vec_is_left);         \
        return static_cast<int>(cudaGetLastError());                                      \
    }

BCAST_LAUNCHER(cf_add, k_add_bcast, float, f32)
BCAST_LAUNCHER(cf_add, k_add_bcast, double, f64)
BCAST_LAUNCHER(cf_sub, k_sub_bcast, float, f32)
BCAST_LAUNCHER(cf_sub, k_sub_bcast, double, f64)
BCAST_LAUNCHER(cf_mul, k_mul_bcast, float, f32)
BCAST_LAUNCHER(cf_mul, k_mul_bcast, double, f64)

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
        const TYPE* a, TYPE* out, long long n) {                                          \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        KERNEL<TYPE><<<blocks, threads>>>(a, out, n);                                     \
        return static_cast<int>(cudaGetLastError());                                      \
    }

UNARY_LAUNCHER(cf_relu, k_relu, float, f32)
UNARY_LAUNCHER(cf_relu, k_relu, double, f64)

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
        const TYPE* in, TYPE* out, long long rows, long long cols) {                      \
        dim3 threads(16, 16);                                                             \
        dim3 blocks(static_cast<unsigned int>((cols + 15) / 16),                          \
                    static_cast<unsigned int>((rows + 15) / 16));                         \
        k_transpose<TYPE><<<blocks, threads>>>(in, out, rows, cols);                      \
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
        const TYPE* mat, TYPE* out, long long rows, long long cols) {                     \
        int blocks, threads;                                                              \
        launch_config(cols, blocks, threads);                                             \
        k_reduce_rows<TYPE><<<blocks, threads>>>(mat, out, rows, cols);                   \
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
        TYPE* param, const TYPE* grad, double lr, long long n) {                          \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        k_sgd_step<TYPE><<<blocks, threads>>>(param, grad, lr, n);                        \
        return static_cast<int>(cudaGetLastError());                                      \
    }

SGD_STEP_LAUNCHER(float, f32)
SGD_STEP_LAUNCHER(double, f64)

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
        const TYPE* scalar, TYPE* out, long long n) {                                     \
        int blocks, threads;                                                              \
        launch_config(n, blocks, threads);                                                \
        k_broadcast_scalar<TYPE><<<blocks, threads>>>(scalar, out, n);                    \
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
        const TYPE* A, const TYPE* B, TYPE* C, int M, int K, int N) {                      \
        dim3 threads(MATMUL_TILE, MATMUL_TILE);                                            \
        dim3 blocks((N + MATMUL_TILE - 1) / MATMUL_TILE, (M + MATMUL_TILE - 1) / MATMUL_TILE); \
        k_matmul<TYPE><<<blocks, threads>>>(A, B, C, M, K, N);                              \
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
    extern "C" __declspec(dllexport) int cf_sum_##SUFFIX(const TYPE* a, TYPE* out, long long n) { \
        cudaError_t err = cudaMemset(out, 0, sizeof(TYPE));                                 \
        if (err != cudaSuccess) return static_cast<int>(err);                               \
        int threads = 256;                                                                  \
        long long b = (n + threads - 1) / threads;                                          \
        int blocks = b < 1 ? 1 : static_cast<int>(b);                                       \
        k_sum<TYPE><<<blocks, threads, threads * sizeof(TYPE)>>>(a, out, n);                \
        return static_cast<int>(cudaGetLastError());                                        \
    }

SUM_LAUNCHER(float, f32)
SUM_LAUNCHER(double, f64)

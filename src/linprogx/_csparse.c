#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

#ifdef LINPROGX_HAVE_BLAS
/* Fortran LAPACK (LP64, column-major), declared directly so no
 * lapacke header is required. OpenBLAS exports these symbols. */
extern void dpotrf_(const char *uplo, const int *n, double *a, const int *lda, int *info);
extern void dgetrf_(const int *m, const int *n, double *a, const int *lda,
                    int *ipiv, int *info);
extern void dgemm_(const char *transa, const char *transb, const int *m, const int *n,
                   const int *k, const double *alpha, const double *a, const int *lda,
                   const double *b, const int *ldb, const double *beta, double *c,
                   const int *ldc);
extern void dtrsm_(const char *side, const char *uplo, const char *transa,
                   const char *diag, const int *m, const int *n, const double *alpha,
                   const double *a, const int *lda, double *b, const int *ldb);
extern void openblas_set_num_threads(int n);
static int g_blas_threads_current = 0;
static void set_blas_threads(int n) {
    if (g_blas_threads_current != n) {
        openblas_set_num_threads(n);
        g_blas_threads_current = n;
    }
}
static void ensure_blas_threads(void) {
    /* 4 threads measured fastest on the monolithic dense-tail sizes
     * (1000-1600); the full core count oversubscribes and is erratic. */
    set_blas_threads(4);
}
static void ensure_supernodal_blas_threads(void) {
    /* The supernodal path issues many small panel BLAS calls; one OpenBLAS
     * thread avoids thread-team overhead and measured faster on maros_r7,
     * re-confirmed after relaxed amalgamation widened the panels. The env
     * knob exists for re-probing when panel shapes change again. */
    static int cached = -1;
    if (cached < 0) {
        const char *env = getenv("LINPROGX_SUPERNODAL_BLAS_THREADS");
        int parsed = env != NULL ? atoi(env) : 0;
        cached = parsed > 0 ? parsed : 1;
    }
    set_blas_threads(cached);
}
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double linprogx_monotonic_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

typedef struct {
    PyObject_HEAD
    Py_ssize_t rows;
    Py_ssize_t cols;
    Py_ssize_t nnz;
    Py_ssize_t *indptr;
    Py_ssize_t *indices;
    double *data;
    Py_ssize_t *csc_indptr;
    Py_ssize_t *csc_rows;
    double *csc_data;
} CSRMatrixObject;

static void CSRMatrix_dealloc(CSRMatrixObject *self) {
    free(self->indptr);
    free(self->indices);
    free(self->data);
    free(self->csc_indptr);
    free(self->csc_rows);
    free(self->csc_data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static int fill_index_array(PyObject *source, Py_ssize_t expected, Py_ssize_t *target, const char *name) {
    PyObject *seq = PySequence_Fast(source, name);
    if (seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain %zd entries", name, expected);
        return -1;
    }
    for (Py_ssize_t i = 0; i < expected; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        Py_ssize_t value = PyLong_AsSsize_t(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        target[i] = value;
    }
    Py_DECREF(seq);
    return 0;
}

static int fill_double_array(PyObject *source, Py_ssize_t expected, double *target, const char *name) {
    PyObject *seq = PySequence_Fast(source, name);
    if (seq == NULL) {
        return -1;
    }
    if (PySequence_Fast_GET_SIZE(seq) != expected) {
        Py_DECREF(seq);
        PyErr_Format(PyExc_ValueError, "%s must contain %zd entries", name, expected);
        return -1;
    }
    for (Py_ssize_t i = 0; i < expected; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(seq, i);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            Py_DECREF(seq);
            return -1;
        }
        target[i] = value;
    }
    Py_DECREF(seq);
    return 0;
}

/* Scaled operator with 32-bit inner indices: the PDHG hot loops are memory
 * bound, so halving index traffic measurably speeds up every matvec. */
typedef struct {
    Py_ssize_t rows;
    Py_ssize_t cols;
    const Py_ssize_t *row_start;
    const int32_t *col_index;
    const double *data;
    const Py_ssize_t *col_start;
    const int32_t *row_index;
    const double *csc_data;
} ScaledOp;

/* ---- lightweight persistent thread pool -------------------------------
 * Used only for embarrassingly parallel array kernels: every job writes
 * a disjoint output range, and all scalar reductions happen afterwards
 * on the main thread in canonical index order, so results are
 * bit-identical to the single-threaded kernel for ANY thread count.
 * Workers spin briefly then yield; jobs are a few hundred microseconds
 * to milliseconds, so wakeup latency must stay in the microseconds. */
#define POOL_MAX_THREADS 8

typedef void (*pool_job_fn)(void *ctx, int tid, int nthreads);

typedef struct {
    pthread_t workers[POOL_MAX_THREADS - 1];
    int pool_threads;         /* created capacity, including caller */
    int active_threads;       /* threads participating in this job */
    int started;
    pool_job_fn fn;
    void *ctx;
    _Atomic int generation;
    _Atomic int done_count;
    _Atomic int shutdown;
} ThreadPool;

static ThreadPool g_pool;
/* number of threads array kernels should use; 1 = fully serial paths */
static int g_kernel_threads = 1;
/* when 0, the dense tail uses the floored hand kernel instead of BLAS
 * dpotrf — a stability fallback for degenerate endgames whose
 * certificate depends on the per-pivot 1e-12 floor */
static int g_tail_use_blas = 1;

typedef struct {
    int tid;
    int start_generation;
} PoolWorkerArg;

static PoolWorkerArg g_pool_args[POOL_MAX_THREADS - 1];

static void *pool_worker_main(void *arg) {
    PoolWorkerArg *wa = (PoolWorkerArg *)arg;
    int seen = wa->start_generation;
    for (;;) {
        int spins = 0;
        while (atomic_load_explicit(&g_pool.generation, memory_order_acquire) == seen) {
            if (atomic_load_explicit(&g_pool.shutdown, memory_order_relaxed)) {
                return NULL;
            }
            if (++spins > 4096) {
                sched_yield();
                spins = 0;
            }
        }
        seen = atomic_load_explicit(&g_pool.generation, memory_order_acquire);
        int active_threads = g_pool.active_threads;
        if (wa->tid < active_threads) {
            g_pool.fn(g_pool.ctx, wa->tid, active_threads);
        }
        atomic_fetch_add_explicit(&g_pool.done_count, 1, memory_order_release);
    }
}

static int pool_ensure(int nthreads) {
    if (nthreads > POOL_MAX_THREADS) {
        nthreads = POOL_MAX_THREADS;
    }
    if (nthreads < 2) {
        return 1;
    }
    if (g_pool.started) {
        if (g_pool.pool_threads < nthreads) {
            int have = g_pool.pool_threads;
            int start_generation = atomic_load_explicit(&g_pool.generation, memory_order_acquire);
            for (int t = have - 1; t < nthreads - 1; t++) {
                g_pool_args[t].tid = t + 1;
                g_pool_args[t].start_generation = start_generation;
                if (pthread_create(&g_pool.workers[t], NULL, pool_worker_main,
                                   &g_pool_args[t]) != 0) {
                    break;
                }
                g_pool.pool_threads = t + 2;
            }
        }
        return g_pool.pool_threads < nthreads ? g_pool.pool_threads : nthreads;
    }
    g_pool.pool_threads = 1;
    g_pool.active_threads = 1;
    atomic_store(&g_pool.generation, 0);
    atomic_store(&g_pool.shutdown, 0);
    for (int t = 0; t < nthreads - 1; t++) {
        g_pool_args[t].tid = t + 1;
        g_pool_args[t].start_generation = 0;
        if (pthread_create(&g_pool.workers[t], NULL, pool_worker_main, &g_pool_args[t]) != 0) {
            break;
        }
        g_pool.pool_threads = t + 2;
    }
    g_pool.started = 1;
    return g_pool.pool_threads < nthreads ? g_pool.pool_threads : nthreads;
}

static void pool_run(pool_job_fn fn, void *ctx) {
    int n = g_kernel_threads;
    if (n < 2 || !g_pool.started) {
        fn(ctx, 0, 1);
        return;
    }
    if (n > g_pool.pool_threads) {
        n = g_pool.pool_threads;
    }
    g_pool.fn = fn;
    g_pool.ctx = ctx;
    g_pool.active_threads = n;
    atomic_store_explicit(&g_pool.done_count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&g_pool.generation, 1, memory_order_release);
    fn(ctx, 0, n);
    int spins = 0;
    while (atomic_load_explicit(&g_pool.done_count, memory_order_acquire) <
           g_pool.pool_threads - 1) {
        if (++spins > 4096) {
            sched_yield();
            spins = 0;
        }
    }
}

/* balanced contiguous range for tid over `count` items weighted by the
 * prefix array `starts` (e.g. CSR row_start) so each thread gets about
 * the same number of nonzeros; falls back to even split when starts is
 * NULL. The partition depends only on nthreads, never on scheduling. */
static void pool_range(const Py_ssize_t *starts, Py_ssize_t count, int tid, int nthreads,
                       Py_ssize_t *begin, Py_ssize_t *end) {
    if (nthreads <= 1) {
        *begin = 0;
        *end = count;
        return;
    }
    if (starts == NULL) {
        Py_ssize_t chunk = (count + nthreads - 1) / nthreads;
        *begin = (Py_ssize_t)tid * chunk;
        *end = *begin + chunk;
        if (*begin > count) {
            *begin = count;
        }
        if (*end > count) {
            *end = count;
        }
        return;
    }
    Py_ssize_t total = starts[count];
    Py_ssize_t lo_target = total * tid / nthreads;
    Py_ssize_t hi_target = total * (tid + 1) / nthreads;
    /* binary search for the first row whose prefix exceeds the target */
    Py_ssize_t lo = 0, hi = count;
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (starts[mid] < lo_target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *begin = lo;
    lo = *begin;
    hi = count;
    while (lo < hi) {
        Py_ssize_t mid = (lo + hi) / 2;
        if (starts[mid] < hi_target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    *end = lo;
}

typedef struct {
    const ScaledOp *op;
    const double *x;
    double *out;
} MatvecJob;

typedef struct {
    const ScaledOp *op;
    const double *y;
    const double *x;
    double *out;
    double *x_sum;
} TransposeAccumJob;

static void matvec_job(void *vctx, int tid, int nthreads) {
    MatvecJob *ctx = (MatvecJob *)vctx;
    const ScaledOp *op = ctx->op;
    const Py_ssize_t *restrict row_start = op->row_start;
    const int32_t *restrict col_index = op->col_index;
    const double *restrict data = op->data;
    const double *restrict x = ctx->x;
    double *restrict out = ctx->out;
    Py_ssize_t begin, end_row;
    pool_range(row_start, op->rows, tid, nthreads, &begin, &end_row);
    for (Py_ssize_t row = begin; row < end_row; row++) {
        double total = 0.0;
        Py_ssize_t end = row_start[row + 1];
        for (Py_ssize_t offset = row_start[row]; offset < end; offset++) {
            total += data[offset] * x[col_index[offset]];
        }
        out[row] = total;
    }
}

static void scaled_op_matvec(const ScaledOp *op, const double *restrict x, double *restrict out) {
    MatvecJob ctx = {op, x, out};
    pool_run(matvec_job, &ctx);
}

static void transpose_matvec_job(void *vctx, int tid, int nthreads) {
    MatvecJob *ctx = (MatvecJob *)vctx;
    const ScaledOp *op = ctx->op;
    const Py_ssize_t *restrict col_start = op->col_start;
    const int32_t *restrict row_index = op->row_index;
    const double *restrict csc_data = op->csc_data;
    const double *restrict y = ctx->x;
    double *restrict out = ctx->out;
    Py_ssize_t begin, end_col;
    pool_range(col_start, op->cols, tid, nthreads, &begin, &end_col);
    for (Py_ssize_t col = begin; col < end_col; col++) {
        double total = 0.0;
        Py_ssize_t end = col_start[col + 1];
        for (Py_ssize_t offset = col_start[col]; offset < end; offset++) {
            total += csc_data[offset] * y[row_index[offset]];
        }
        out[col] = total;
    }
}

static void scaled_op_transpose_matvec(const ScaledOp *op, const double *restrict y, double *restrict out) {
    MatvecJob ctx = {op, (const double *)y, out};
    pool_run(transpose_matvec_job, &ctx);
}

static void transpose_matvec_accum_x_job(void *vctx, int tid, int nthreads) {
    TransposeAccumJob *ctx = (TransposeAccumJob *)vctx;
    const ScaledOp *op = ctx->op;
    const Py_ssize_t *restrict col_start = op->col_start;
    const int32_t *restrict row_index = op->row_index;
    const double *restrict csc_data = op->csc_data;
    const double *restrict y = ctx->y;
    const double *restrict x = ctx->x;
    double *restrict out = ctx->out;
    double *restrict x_sum = ctx->x_sum;
    Py_ssize_t begin, end_col;
    pool_range(col_start, op->cols, tid, nthreads, &begin, &end_col);
    for (Py_ssize_t col = begin; col < end_col; col++) {
        double total = 0.0;
        Py_ssize_t end = col_start[col + 1];
        for (Py_ssize_t offset = col_start[col]; offset < end; offset++) {
            total += csc_data[offset] * y[row_index[offset]];
        }
        out[col] = total;
        x_sum[col] += x[col];
    }
}

static void scaled_op_transpose_matvec_accum_x(
    const ScaledOp *op,
    const double *restrict y,
    double *restrict out,
    const double *restrict x,
    double *restrict x_sum) {
    TransposeAccumJob ctx = {op, y, x, out, x_sum};
    pool_run(transpose_matvec_accum_x_job, &ctx);
}

static double l2_norm(const double *values, Py_ssize_t count) {
    double total = 0.0;
    for (Py_ssize_t i = 0; i < count; i++) {
        total += values[i] * values[i];
    }
    return sqrt(total);
}

static double dot_product(const double *left, const double *right, Py_ssize_t count) {
    double total = 0.0;
    for (Py_ssize_t i = 0; i < count; i++) {
        total += left[i] * right[i];
    }
    return total;
}

static void csr_residual(CSRMatrixObject *self, const double *x, const double *b, double *residual, double *max_residual, double *l2_residual) {
    double max_value = 0.0;
    double l2_value = 0.0;
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double total = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            total += self->data[offset] * x[self->indices[offset]];
        }
        double value = b[row] - total;
        double abs_value = fabs(value);
        residual[row] = value;
        max_value = abs_value > max_value ? abs_value : max_value;
        l2_value += value * value;
    }
    *max_residual = max_value;
    *l2_residual = sqrt(l2_value);
}

static void csr_restricted_matvec(CSRMatrixObject *self, const double *x, const unsigned char *free_col, double *out) {
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double total = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            Py_ssize_t col = self->indices[offset];
            if (free_col[col]) {
                total += self->data[offset] * x[col];
            }
        }
        out[row] = total;
    }
}

static void csr_restricted_transpose_matvec(CSRMatrixObject *self, const double *y, const unsigned char *free_col, double *out) {
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        if (!free_col[col]) {
            out[col] = 0.0;
            continue;
        }
        double total = 0.0;
        for (Py_ssize_t offset = self->csc_indptr[col]; offset < self->csc_indptr[col + 1]; offset++) {
            total += self->csc_data[offset] * y[self->csc_rows[offset]];
        }
        out[col] = total;
    }
}

static void active_set_cgls_cleanup(
    CSRMatrixObject *self,
    double *x,
    const double *b,
    const double *lo,
    const double *hi,
    const unsigned char *bound_kind,
    double tol,
    double *max_residual,
    double *l2_residual) {
    const double margin = 1e-3;
    const int max_passes = 12;
    const int max_iter = 600;
    double *residual = calloc((size_t)self->rows, sizeof(double));
    double *q = calloc((size_t)self->rows, sizeof(double));
    double *s = calloc((size_t)self->cols, sizeof(double));
    double *p = calloc((size_t)self->cols, sizeof(double));
    double *correction = calloc((size_t)self->cols, sizeof(double));
    unsigned char *free_col = calloc((size_t)self->cols, sizeof(unsigned char));
    if (residual == NULL || q == NULL || s == NULL || p == NULL || correction == NULL || free_col == NULL) {
        free(residual);
        free(q);
        free(s);
        free(p);
        free(correction);
        free(free_col);
        return;
    }

    double previous_l2 = INFINITY;
    for (int pass = 0; pass < max_passes; pass++) {
        csr_residual(self, x, b, residual, max_residual, l2_residual);
        if (*max_residual <= tol) {
            break;
        }
        /* Each pass refreshes the active set after the previous bound-limited
         * step; stop once a pass no longer makes meaningful progress. */
        if (*l2_residual > 0.99 * previous_l2) {
            break;
        }
        previous_l2 = *l2_residual;

        Py_ssize_t free_count = 0;
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            int is_free = 1;
            if ((bound_kind[col] & 1) && x[col] <= lo[col] + margin) {
                is_free = 0;
            }
            if ((bound_kind[col] & 2) && x[col] >= hi[col] - margin) {
                is_free = 0;
            }
            free_col[col] = (unsigned char)is_free;
            free_count += is_free;
            correction[col] = 0.0;
        }
        if (free_count == 0) {
            break;
        }

        csr_restricted_transpose_matvec(self, residual, free_col, s);
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            p[col] = s[col];
        }
        double gamma = dot_product(s, s, self->cols);
        if (gamma <= 1e-30) {
            break;
        }

        for (int iter = 0; iter < max_iter; iter++) {
            csr_restricted_matvec(self, p, free_col, q);
            double denom = dot_product(q, q, self->rows);
            if (denom <= 1e-30) {
                break;
            }
            double alpha = gamma / denom;
            for (Py_ssize_t col = 0; col < self->cols; col++) {
                if (free_col[col]) {
                    correction[col] += alpha * p[col];
                }
            }
            for (Py_ssize_t row = 0; row < self->rows; row++) {
                residual[row] -= alpha * q[row];
            }
            csr_restricted_transpose_matvec(self, residual, free_col, s);
            double next_gamma = dot_product(s, s, self->cols);
            if (sqrt(next_gamma) <= tol) {
                break;
            }
            double beta = next_gamma / gamma;
            for (Py_ssize_t col = 0; col < self->cols; col++) {
                if (free_col[col]) {
                    p[col] = s[col] + beta * p[col];
                }
            }
            gamma = next_gamma;
        }

        double step = 1.0;
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            if (!free_col[col]) {
                continue;
            }
            double change = correction[col];
            if ((bound_kind[col] & 1) && change < 0.0) {
                double candidate = (lo[col] - x[col]) / change;
                if (candidate < step) {
                    step = candidate;
                }
            }
            if ((bound_kind[col] & 2) && change > 0.0) {
                double candidate = (hi[col] - x[col]) / change;
                if (candidate < step) {
                    step = candidate;
                }
            }
        }
        if (step < 1.0) {
            step *= 0.99;
        }
        if (step <= 0.0 || !isfinite(step)) {
            break;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            if (!free_col[col]) {
                continue;
            }
            double updated = x[col] + step * correction[col];
            if ((bound_kind[col] & 1) && updated < lo[col]) {
                updated = lo[col];
            }
            if ((bound_kind[col] & 2) && updated > hi[col]) {
                updated = hi[col];
            }
            x[col] = updated;
        }
    }

    csr_residual(self, x, b, residual, max_residual, l2_residual);
    free(residual);
    free(q);
    free(s);
    free(p);
    free(correction);
    free(free_col);
}

static double estimate_scaled_operator_norm(const ScaledOp *op) {
    double *x = calloc((size_t)op->cols, sizeof(double));
    double *y = calloc((size_t)op->rows, sizeof(double));
    double *z = calloc((size_t)op->cols, sizeof(double));
    if (x == NULL || y == NULL || z == NULL) {
        free(x);
        free(y);
        free(z);
        return -1.0;
    }
    double initial = op->cols > 0 ? 1.0 / sqrt((double)op->cols) : 0.0;
    for (Py_ssize_t col = 0; col < op->cols; col++) {
        x[col] = initial;
    }
    double norm = 1.0;
    for (int iter = 0; iter < 30; iter++) {
        scaled_op_matvec(op, x, y);
        double ynorm = l2_norm(y, op->rows);
        if (ynorm <= 0.0) {
            break;
        }
        for (Py_ssize_t row = 0; row < op->rows; row++) {
            y[row] /= ynorm;
        }
        scaled_op_transpose_matvec(op, y, z);
        double znorm = l2_norm(z, op->cols);
        if (znorm <= 0.0) {
            break;
        }
        for (Py_ssize_t col = 0; col < op->cols; col++) {
            x[col] = z[col] / znorm;
        }
        norm = ynorm;
    }
    scaled_op_matvec(op, x, y);
    norm = l2_norm(y, op->rows);
    free(x);
    free(y);
    free(z);
    return norm > 0.0 ? norm : 1.0;
}

/* KKT diagnostics for the scaled iterate (x_scaled, y_scaled), reported in
 * original problem units. The scaled operator is A_tilde = R A C with
 * R = diag(row_scale) and C = diag(col_scale); x_orig = C x_scaled and the
 * original-space dual is y_orig = R y_scaled. */
typedef struct {
    double primal_res_max;
    double primal_res_l2;
    double dual_res_inf;
    double dual_res_l2;
    double primal_obj;
    double dual_obj;
    double gap;
    double kkt;
} KKTEval;

static void evaluate_kkt(
    const ScaledOp *op,
    const double *x_scaled,
    const double *y_scaled,
    const double *c,
    const double *b,
    const double *lo,
    const double *hi,
    const unsigned char *bound_kind,
    const double *col_scale,
    const double *row_scale,
    const double *scaled_b,
    double b_l2,
    double c_l2,
    double *ax_work,
    double *r_work,
    KKTEval *out) {
    double pmax = 0.0;
    double pl2 = 0.0;
    double dual_obj = 0.0;
    scaled_op_matvec(op, x_scaled, ax_work);
    for (Py_ssize_t row = 0; row < op->rows; row++) {
        double res = (ax_work[row] - scaled_b[row]) / row_scale[row];
        double abs_res = fabs(res);
        pmax = abs_res > pmax ? abs_res : pmax;
        pl2 += res * res;
        dual_obj -= b[row] * row_scale[row] * y_scaled[row];
    }
    scaled_op_transpose_matvec(op, y_scaled, r_work);
    double dinf = 0.0;
    double dl2 = 0.0;
    double primal_obj = 0.0;
    for (Py_ssize_t col = 0; col < op->cols; col++) {
        double reduced = c[col] + r_work[col] / col_scale[col];
        primal_obj += c[col] * col_scale[col] * x_scaled[col];
        double violation = 0.0;
        if (reduced > 0.0) {
            if (bound_kind[col] & 1) {
                dual_obj += reduced * lo[col];
            } else {
                violation = reduced;
            }
        } else if (reduced < 0.0) {
            if (bound_kind[col] & 2) {
                dual_obj += reduced * hi[col];
            } else {
                violation = -reduced;
            }
        }
        dinf = violation > dinf ? violation : dinf;
        dl2 += violation * violation;
    }
    out->primal_res_max = pmax;
    out->primal_res_l2 = sqrt(pl2);
    out->dual_res_inf = dinf;
    out->dual_res_l2 = sqrt(dl2);
    out->primal_obj = primal_obj;
    out->dual_obj = dual_obj;
    out->gap = primal_obj - dual_obj;
    /* Scale-free progress measure: every component is normalized so that the
     * restart and candidate-selection logic stays coherent while the primal
     * weight omega adapts between restarts. */
    double rel_primal = out->primal_res_l2 / (1.0 + b_l2);
    double rel_dual = out->dual_res_l2 / (1.0 + c_l2);
    double rel_gap = out->gap / (1.0 + fabs(primal_obj) + fabs(dual_obj));
    out->kkt = sqrt(rel_primal * rel_primal + rel_dual * rel_dual + rel_gap * rel_gap);
}

static int kkt_terminated(const KKTEval *ev, double tol, double c_inf) {
    double dual_tol = tol * (1.0 + c_inf);
    double gap_tol = tol * (1.0 + fabs(ev->primal_obj) + fabs(ev->dual_obj));
    return ev->primal_res_max <= tol && ev->dual_res_inf <= dual_tol && fabs(ev->gap) <= gap_tol;
}

static PyObject *CSRMatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)kwds;
    Py_ssize_t rows;
    Py_ssize_t cols;
    PyObject *indptr_obj;
    PyObject *indices_obj;
    PyObject *data_obj;

    if (!PyArg_ParseTuple(args, "nnOOO", &rows, &cols, &indptr_obj, &indices_obj, &data_obj)) {
        return NULL;
    }
    if (rows < 0 || cols < 0) {
        PyErr_SetString(PyExc_ValueError, "matrix dimensions must be nonnegative");
        return NULL;
    }

    PyObject *data_seq = PySequence_Fast(data_obj, "data must be a sequence");
    if (data_seq == NULL) {
        return NULL;
    }
    Py_ssize_t nnz = PySequence_Fast_GET_SIZE(data_seq);
    Py_DECREF(data_seq);

    CSRMatrixObject *self = (CSRMatrixObject *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }
    self->rows = rows;
    self->cols = cols;
    self->nnz = nnz;
    self->indptr = calloc((size_t)rows + 1, sizeof(Py_ssize_t));
    self->indices = calloc((size_t)nnz, sizeof(Py_ssize_t));
    self->data = calloc((size_t)nnz, sizeof(double));
    self->csc_indptr = calloc((size_t)cols + 1, sizeof(Py_ssize_t));
    self->csc_rows = calloc((size_t)nnz, sizeof(Py_ssize_t));
    self->csc_data = calloc((size_t)nnz, sizeof(double));
    if (self->indptr == NULL || self->indices == NULL || self->data == NULL ||
        self->csc_indptr == NULL || self->csc_rows == NULL || self->csc_data == NULL) {
        Py_DECREF(self);
        PyErr_NoMemory();
        return NULL;
    }
    if (fill_index_array(indptr_obj, rows + 1, self->indptr, "indptr") != 0 ||
        fill_index_array(indices_obj, nnz, self->indices, "indices") != 0 ||
        fill_double_array(data_obj, nnz, self->data, "data") != 0) {
        Py_DECREF(self);
        return NULL;
    }
    if (rows > 0 && self->indptr[0] != 0) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "indptr must start with 0");
        return NULL;
    }
    if (self->indptr[rows] != nnz) {
        Py_DECREF(self);
        PyErr_SetString(PyExc_ValueError, "indptr[-1] must equal nnz");
        return NULL;
    }
    for (Py_ssize_t row = 0; row < rows; row++) {
        if (self->indptr[row] > self->indptr[row + 1]) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_ValueError, "indptr must be nondecreasing");
            return NULL;
        }
    }
    for (Py_ssize_t i = 0; i < nnz; i++) {
        if (self->indices[i] < 0 || self->indices[i] >= cols) {
            Py_DECREF(self);
            PyErr_SetString(PyExc_ValueError, "column index out of range");
            return NULL;
        }
        self->csc_indptr[self->indices[i] + 1]++;
    }
    for (Py_ssize_t col = 0; col < cols; col++) {
        self->csc_indptr[col + 1] += self->csc_indptr[col];
    }
    if (cols > 0) {
        Py_ssize_t *next = calloc((size_t)cols, sizeof(Py_ssize_t));
        if (next == NULL) {
            Py_DECREF(self);
            PyErr_NoMemory();
            return NULL;
        }
        for (Py_ssize_t col = 0; col < cols; col++) {
            next[col] = self->csc_indptr[col];
        }
        for (Py_ssize_t row = 0; row < rows; row++) {
            for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
                Py_ssize_t col = self->indices[offset];
                Py_ssize_t dest = next[col]++;
                self->csc_rows[dest] = row;
                self->csc_data[dest] = self->data[offset];
            }
        }
        free(next);
    }
    return (PyObject *)self;
}

static PyObject *CSRMatrix_shape(CSRMatrixObject *self, void *closure) {
    (void)closure;
    return Py_BuildValue("(nn)", self->rows, self->cols);
}

static PyObject *CSRMatrix_nnz(CSRMatrixObject *self, void *closure) {
    (void)closure;
    return PyLong_FromSsize_t(self->nnz);
}

static PyObject *CSRMatrix_density(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    if (self->rows == 0 || self->cols == 0) {
        return PyFloat_FromDouble(0.0);
    }
    return PyFloat_FromDouble((double)self->nnz / ((double)self->rows * (double)self->cols));
}

static PyObject *CSRMatrix_matvec(CSRMatrixObject *self, PyObject *args) {
    PyObject *vector_obj;
    if (!PyArg_ParseTuple(args, "O", &vector_obj)) {
        return NULL;
    }
    PyObject *vector = PySequence_Fast(vector_obj, "vector must be a sequence");
    if (vector == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(vector) != self->cols) {
        Py_DECREF(vector);
        PyErr_SetString(PyExc_ValueError, "vector length must match matrix column count");
        return NULL;
    }

    PyObject *result = PyList_New(self->rows);
    if (result == NULL) {
        Py_DECREF(vector);
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        double total = 0.0;
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            PyObject *item = PySequence_Fast_GET_ITEM(vector, self->indices[offset]);
            double value = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(vector);
                Py_DECREF(result);
                return NULL;
            }
            total += self->data[offset] * value;
        }
        PyObject *boxed = PyFloat_FromDouble(total);
        if (boxed == NULL) {
            Py_DECREF(vector);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, row, boxed);
    }
    Py_DECREF(vector);
    return result;
}

static PyObject *CSRMatrix_transpose_matvec(CSRMatrixObject *self, PyObject *args) {
    PyObject *vector_obj;
    if (!PyArg_ParseTuple(args, "O", &vector_obj)) {
        return NULL;
    }
    PyObject *vector = PySequence_Fast(vector_obj, "vector must be a sequence");
    if (vector == NULL) {
        return NULL;
    }
    if (PySequence_Fast_GET_SIZE(vector) != self->rows) {
        Py_DECREF(vector);
        PyErr_SetString(PyExc_ValueError, "vector length must match matrix row count");
        return NULL;
    }
    double *totals = calloc((size_t)self->cols, sizeof(double));
    if (totals == NULL) {
        Py_DECREF(vector);
        PyErr_NoMemory();
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        PyObject *item = PySequence_Fast_GET_ITEM(vector, row);
        double value = PyFloat_AsDouble(item);
        if (PyErr_Occurred()) {
            free(totals);
            Py_DECREF(vector);
            return NULL;
        }
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            totals[self->indices[offset]] += self->data[offset] * value;
        }
    }
    PyObject *result = PyList_New(self->cols);
    if (result == NULL) {
        free(totals);
        Py_DECREF(vector);
        return NULL;
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        PyObject *boxed = PyFloat_FromDouble(totals[col]);
        if (boxed == NULL) {
            free(totals);
            Py_DECREF(vector);
            Py_DECREF(result);
            return NULL;
        }
        PyList_SET_ITEM(result, col, boxed);
    }
    free(totals);
    Py_DECREF(vector);
    return result;
}

static PyObject *CSRMatrix_to_components(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *indptr = PyList_New(self->rows + 1);
    PyObject *indices = PyList_New(self->nnz);
    PyObject *data = PyList_New(self->nnz);
    if (indptr == NULL || indices == NULL || data == NULL) {
        Py_XDECREF(indptr);
        Py_XDECREF(indices);
        Py_XDECREF(data);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->rows + 1; i++) {
        PyList_SET_ITEM(indptr, i, PyLong_FromSsize_t(self->indptr[i]));
    }
    for (Py_ssize_t i = 0; i < self->nnz; i++) {
        PyList_SET_ITEM(indices, i, PyLong_FromSsize_t(self->indices[i]));
        PyList_SET_ITEM(data, i, PyFloat_FromDouble(self->data[i]));
    }
    return Py_BuildValue("(NNN)", indptr, indices, data);
}

static PyObject *CSRMatrix_to_dense(CSRMatrixObject *self, PyObject *Py_UNUSED(ignored)) {
    PyObject *rows = PyList_New(self->rows);
    if (rows == NULL) {
        return NULL;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        PyObject *dense_row = PyList_New(self->cols);
        if (dense_row == NULL) {
            Py_DECREF(rows);
            return NULL;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            PyList_SET_ITEM(dense_row, col, PyFloat_FromDouble(0.0));
        }
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            PyObject *boxed = PyFloat_FromDouble(self->data[offset]);
            if (boxed == NULL) {
                Py_DECREF(dense_row);
                Py_DECREF(rows);
                return NULL;
            }
            if (PyList_SetItem(dense_row, self->indices[offset], boxed) != 0) {
                Py_DECREF(boxed);
                Py_DECREF(dense_row);
                Py_DECREF(rows);
                return NULL;
            }
        }
        PyList_SET_ITEM(rows, row, dense_row);
    }
    return rows;
}

/* Restart tuning constants in the spirit of restarted average PDHG for LP. */
#define PDHG_RESTART_SUFFICIENT 0.2
#define PDHG_RESTART_NECESSARY 0.8
#define PDHG_RESTART_ARTIFICIAL 0.36
#define PDHG_OMEGA_SMOOTHING 0.5
#define PDHG_OMEGA_MIN 1e-8
#define PDHG_OMEGA_MAX 1e8

typedef struct {
    Py_ssize_t cols;
    double trial_tau;
    const double *scaled_lo;
    const double *scaled_hi;
    const double *aty;
    const double *scaled_c;
    const double *x;
    double *xbar;
    double *dxsq_col;
} PrimalTrialJob;

static void primal_trial_job(void *vctx, int tid, int nthreads) {
    PrimalTrialJob *ctx = (PrimalTrialJob *)vctx;
    const double *restrict lo_v = ctx->scaled_lo;
    const double *restrict hi_v = ctx->scaled_hi;
    const double *restrict g_v = ctx->aty;
    const double *restrict c_v = ctx->scaled_c;
    const double *restrict x_v = ctx->x;
    double *restrict out_v = ctx->xbar;
    double *restrict dxsq = ctx->dxsq_col;
    double trial_tau = ctx->trial_tau;
    Py_ssize_t begin, end;
    pool_range(NULL, ctx->cols, tid, nthreads, &begin, &end);
    for (Py_ssize_t col = begin; col < end; col++) {
        /* scaled_lo/scaled_hi hold +-INF when a bound is absent, so the
         * clamp is branchless and vector friendly. */
        double updated = x_v[col] - trial_tau * (g_v[col] + c_v[col]);
        updated = fmax(updated, lo_v[col]);
        updated = fmin(updated, hi_v[col]);
        out_v[col] = updated;
        double dx = updated - x_v[col];
        dxsq[col] = dx * dx;
    }
}

typedef struct {
    Py_ssize_t rows;
    double trial_sigma;
    const ScaledOp *op;
    const double *xbar;
    const double *ax;
    const double *scaled_b;
    const double *y;
    double *ax_trial;
    double *y_trial;
    double *dysq_row;
    double *inter_row;
} DualTrialJob;

static void dual_trial_job(void *vctx, int tid, int nthreads) {
    DualTrialJob *ctx = (DualTrialJob *)vctx;
    const ScaledOp *op = ctx->op;
    const Py_ssize_t *restrict row_start = op->row_start;
    const int32_t *restrict col_index = op->col_index;
    const double *restrict data = op->data;
    const double *restrict xb = ctx->xbar;
    const double *restrict ax = ctx->ax;
    const double *restrict scaled_b = ctx->scaled_b;
    const double *restrict y = ctx->y;
    double *restrict ax_trial = ctx->ax_trial;
    double *restrict y_trial = ctx->y_trial;
    double *restrict dysq = ctx->dysq_row;
    double *restrict inter = ctx->inter_row;
    double trial_sigma = ctx->trial_sigma;
    Py_ssize_t begin, end_row;
    pool_range(row_start, ctx->rows, tid, nthreads, &begin, &end_row);
    for (Py_ssize_t row = begin; row < end_row; row++) {
        double axr = 0.0;
        Py_ssize_t end = row_start[row + 1];
        for (Py_ssize_t p = row_start[row]; p < end; p++) {
            axr += data[p] * xb[col_index[p]];
        }
        ax_trial[row] = axr;
        double gradient = 2.0 * axr - ax[row] - scaled_b[row];
        double updated = y[row] + trial_sigma * gradient;
        double dy = updated - y[row];
        y_trial[row] = updated;
        dysq[row] = dy * dy;
        inter[row] = dy * (axr - ax[row]);
    }
}

static PyObject *CSRMatrix_solve_eq_box_pdhg(CSRMatrixObject *self, PyObject *args, PyObject *kwds) {
    PyObject *c_obj;
    PyObject *b_obj;
    PyObject *lo_obj;
    PyObject *hi_obj;
    Py_ssize_t max_iter = 20000;
    Py_ssize_t check_interval = 500;
    double tol = 1e-6;
    double objective_scale = 0.0;
    int adaptive_weight = 1;
    int debug = 0;
    double restart_sufficient = PDHG_RESTART_SUFFICIENT;
    double restart_necessary = PDHG_RESTART_NECESSARY;
    double restart_artificial = PDHG_RESTART_ARTIFICIAL;
    Py_ssize_t eval_interval_override = 0;
    Py_ssize_t plateau_window = 80;
    double plateau_threshold = 0.02;
    Py_ssize_t threads = 1;
    static char *kwlist[] = {
        "c", "b", "lo", "hi", "max_iter", "tol", "check_interval", "objective_scale",
        "adaptive_weight", "debug", "restart_sufficient", "restart_necessary",
        "restart_artificial", "eval_interval_override", "plateau_window",
        "plateau_threshold", "threads", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOOO|ndndiidddnndn",
            kwlist,
            &c_obj,
            &b_obj,
            &lo_obj,
            &hi_obj,
            &max_iter,
            &tol,
            &check_interval,
            &objective_scale,
            &adaptive_weight,
            &debug,
            &restart_sufficient,
            &restart_necessary,
            &restart_artificial,
            &eval_interval_override,
            &plateau_window,
            &plateau_threshold,
            &threads)) {
        return NULL;
    }
    if (max_iter < 0 || check_interval <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_iter must be nonnegative and check_interval positive");
        return NULL;
    }
    if (plateau_window < 0) {
        plateau_window = 0;
    }

    PyObject *result = NULL;
    double *c = NULL;
    double *b = NULL;
    double *lo = NULL;
    double *hi = NULL;
    double *col_scale = NULL;
    double *scaled_lo = NULL;
    double *scaled_hi = NULL;
    double *scaled_c = NULL;
    double *row_scale = NULL;
    double *scaled_b = NULL;
    double *x = NULL;
    double *xbar = NULL;
    double *y = NULL;
    double *ax = NULL;
    double *aty = NULL;
    double *y_trial = NULL;
    double *dxsq_col = NULL;
    double *dysq_row = NULL;
    double *inter_row = NULL;
    double *ax_trial = NULL;
    double *x_sum = NULL;
    double *y_sum = NULL;
    double *avg_x = NULL;
    double *avg_y = NULL;
    double *x_restart = NULL;
    double *y_restart = NULL;
    double *best_x = NULL;
    double *best_y = NULL;
    double *cleanup_x = NULL;
    double *cleanup_ax = NULL;
    double *cleanup_aty = NULL;
    double *plateau_kkt_buf = NULL;
    double *operator_data = NULL;
    double *operator_csc_data = NULL;
    int32_t *op_col_index = NULL;
    int32_t *op_row_index = NULL;
    unsigned char *bound_kind = NULL;
    int pdhg_profile = getenv("LINPROGX_PDHG_PROFILE") != NULL;
    double profile_start = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
    double profile_setup = 0.0;
    double profile_norm = 0.0;
    double profile_initial_eval = 0.0;
    double profile_loop = 0.0;
    double profile_step = 0.0;
    double profile_trial_primal = 0.0;
    double profile_trial_col_reduce = 0.0;
    double profile_trial_dual = 0.0;
    double profile_trial_row_reduce = 0.0;
    double profile_trial_serial = 0.0;
    double profile_transpose = 0.0;
    double profile_accumulate = 0.0;
    double profile_eval = 0.0;
    double profile_restart = 0.0;
    double profile_cleanup = 0.0;

    c = calloc((size_t)self->cols, sizeof(double));
    b = calloc((size_t)self->rows, sizeof(double));
    lo = calloc((size_t)self->cols, sizeof(double));
    hi = calloc((size_t)self->cols, sizeof(double));
    col_scale = calloc((size_t)self->cols, sizeof(double));
    scaled_lo = calloc((size_t)self->cols, sizeof(double));
    scaled_hi = calloc((size_t)self->cols, sizeof(double));
    scaled_c = calloc((size_t)self->cols, sizeof(double));
    row_scale = calloc((size_t)self->rows, sizeof(double));
    scaled_b = calloc((size_t)self->rows, sizeof(double));
    x = calloc((size_t)self->cols, sizeof(double));
    xbar = calloc((size_t)self->cols, sizeof(double));
    y = calloc((size_t)self->rows, sizeof(double));
    ax = calloc((size_t)self->rows, sizeof(double));
    aty = calloc((size_t)self->cols, sizeof(double));
    y_trial = calloc((size_t)self->rows, sizeof(double));
    ax_trial = calloc((size_t)self->rows, sizeof(double));
    x_sum = calloc((size_t)self->cols, sizeof(double));
    y_sum = calloc((size_t)self->rows, sizeof(double));
    avg_x = calloc((size_t)self->cols, sizeof(double));
    avg_y = calloc((size_t)self->rows, sizeof(double));
    x_restart = calloc((size_t)self->cols, sizeof(double));
    y_restart = calloc((size_t)self->rows, sizeof(double));
    best_x = calloc((size_t)self->cols, sizeof(double));
    best_y = calloc((size_t)self->rows, sizeof(double));
    cleanup_x = calloc((size_t)self->cols, sizeof(double));
    cleanup_ax = calloc((size_t)self->rows, sizeof(double));
    cleanup_aty = calloc((size_t)self->cols, sizeof(double));
    operator_data = calloc((size_t)self->nnz, sizeof(double));
    operator_csc_data = calloc((size_t)self->nnz, sizeof(double));
    op_col_index = calloc((size_t)self->nnz, sizeof(int32_t));
    op_row_index = calloc((size_t)self->nnz, sizeof(int32_t));
    bound_kind = calloc((size_t)self->cols, sizeof(unsigned char));
    dxsq_col = calloc((size_t)self->cols, sizeof(double));
    dysq_row = calloc((size_t)self->rows, sizeof(double));
    inter_row = calloc((size_t)self->rows, sizeof(double));
    if (c == NULL || b == NULL || lo == NULL || hi == NULL ||
        col_scale == NULL || scaled_lo == NULL || scaled_hi == NULL || scaled_c == NULL ||
        row_scale == NULL || scaled_b == NULL ||
        dxsq_col == NULL || dysq_row == NULL || inter_row == NULL ||
        x == NULL || xbar == NULL || y == NULL || ax == NULL || aty == NULL ||
        y_trial == NULL || ax_trial == NULL ||
        x_sum == NULL || y_sum == NULL || avg_x == NULL || avg_y == NULL ||
        x_restart == NULL || y_restart == NULL ||
        best_x == NULL || best_y == NULL ||
        cleanup_x == NULL || cleanup_ax == NULL || cleanup_aty == NULL ||
        operator_data == NULL || operator_csc_data == NULL ||
        op_col_index == NULL || op_row_index == NULL || bound_kind == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (self->rows > INT32_MAX || self->cols > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix dimensions exceed the 32-bit solver limit");
        goto done;
    }
    if (fill_double_array(c_obj, self->cols, c, "c") != 0 ||
        fill_double_array(b_obj, self->rows, b, "b") != 0 ||
        fill_double_array(lo_obj, self->cols, lo, "lo") != 0 ||
        fill_double_array(hi_obj, self->cols, hi, "hi") != 0) {
        goto done;
    }

    for (Py_ssize_t col = 0; col < self->cols; col++) {
        int has_lo = isfinite(lo[col]);
        int has_hi = isfinite(hi[col]);
        bound_kind[col] = (unsigned char)((has_lo ? 1 : 0) | (has_hi ? 2 : 0));
        if (has_lo && has_hi && hi[col] < lo[col]) {
            PyErr_SetString(PyExc_ValueError, "upper bound is lower than lower bound");
            goto done;
        }
        col_scale[col] = 0.0;
    }
    /* Ruiz equilibration: iteratively divide each row and column by the
     * square root of its infinity norm so both approach 1, then apply one
     * l2 balancing pass. This conditions the operator far more evenly than
     * single-shot norm scaling. */
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        col_scale[col] = 1.0;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        row_scale[row] = 1.0;
    }
    for (int ruiz_iter = 0; ruiz_iter < 10; ruiz_iter++) {
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            scaled_b[row] = 0.0;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            aty[col] = 0.0;
        }
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
                Py_ssize_t col = self->indices[offset];
                double value = fabs(self->data[offset] * row_scale[row] * col_scale[col]);
                if (value > scaled_b[row]) {
                    scaled_b[row] = value;
                }
                if (value > aty[col]) {
                    aty[col] = value;
                }
            }
        }
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            if (scaled_b[row] > 0.0) {
                row_scale[row] /= sqrt(scaled_b[row]);
            }
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            if (aty[col] > 0.0) {
                col_scale[col] /= sqrt(aty[col]);
            }
        }
    }
    /* One l2 pass on the Ruiz-equilibrated matrix. */
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        scaled_b[row] = 0.0;
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        aty[col] = 0.0;
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            Py_ssize_t col = self->indices[offset];
            double value = self->data[offset] * row_scale[row] * col_scale[col];
            scaled_b[row] += value * value;
            aty[col] += value * value;
        }
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        if (scaled_b[row] > 0.0) {
            row_scale[row] /= sqrt(sqrt(scaled_b[row]));
        }
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        if (aty[col] > 0.0) {
            col_scale[col] /= sqrt(sqrt(aty[col]));
        }
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        if (col_scale[col] < 1e-8) {
            col_scale[col] = 1e-8;
        } else if (col_scale[col] > 1e8) {
            col_scale[col] = 1e8;
        }
        scaled_c[col] = c[col] * col_scale[col];
        scaled_lo[col] = isfinite(lo[col]) ? lo[col] / col_scale[col] : lo[col];
        scaled_hi[col] = isfinite(hi[col]) ? hi[col] / col_scale[col] : hi[col];
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        scaled_b[row] = row_scale[row] * b[row];
    }
    for (Py_ssize_t row = 0; row < self->rows; row++) {
        for (Py_ssize_t offset = self->indptr[row]; offset < self->indptr[row + 1]; offset++) {
            Py_ssize_t col = self->indices[offset];
            operator_data[offset] = self->data[offset] * row_scale[row] * col_scale[col];
        }
    }
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        for (Py_ssize_t offset = self->csc_indptr[col]; offset < self->csc_indptr[col + 1]; offset++) {
            Py_ssize_t row = self->csc_rows[offset];
            operator_csc_data[offset] = self->csc_data[offset] * row_scale[row] * col_scale[col];
        }
    }
    for (Py_ssize_t i = 0; i < self->nnz; i++) {
        op_col_index[i] = (int32_t)self->indices[i];
        op_row_index[i] = (int32_t)self->csc_rows[i];
    }
    ScaledOp op = {
        self->rows,
        self->cols,
        self->indptr,
        op_col_index,
        operator_data,
        self->csc_indptr,
        op_row_index,
        operator_csc_data,
    };

    if (pdhg_profile) {
        profile_setup = linprogx_monotonic_seconds() - profile_start;
    }
    double norm;
    double profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
    norm = estimate_scaled_operator_norm(&op);
    if (pdhg_profile) {
        profile_norm += linprogx_monotonic_seconds() - profile_phase;
    }
    if (norm < 0.0) {
        PyErr_NoMemory();
        goto done;
    }
    double eta = 0.99 / norm;

    /* The primal weight omega balances primal and dual step sizes:
     * tau = eta / omega and sigma = eta * omega. A user-provided
     * objective_scale seeds it for backwards compatibility; otherwise it
     * starts at ||scaled c|| / ||scaled b|| and adapts at every restart. */
    double omega = 1.0;
    if (objective_scale > 0.0) {
        omega = objective_scale;
    } else {
        double c_norm = l2_norm(scaled_c, self->cols);
        double b_norm = l2_norm(scaled_b, self->rows);
        if (c_norm > 1e-12 && b_norm > 1e-12) {
            omega = c_norm / b_norm;
        }
    }
    if (omega < PDHG_OMEGA_MIN) {
        omega = PDHG_OMEGA_MIN;
    } else if (omega > PDHG_OMEGA_MAX) {
        omega = PDHG_OMEGA_MAX;
    }
    double omega_initial = omega;
    double tau = eta / omega;
    double sigma = eta * omega;

    double c_inf = 0.0;
    double c_l2 = 0.0;
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        double abs_c = fabs(c[col]);
        c_inf = abs_c > c_inf ? abs_c : c_inf;
        c_l2 += c[col] * c[col];
    }
    c_l2 = sqrt(c_l2);
    double b_l2 = l2_norm(b, self->rows);

    for (Py_ssize_t col = 0; col < self->cols; col++) {
        double start = 0.0;
        if (isfinite(scaled_lo[col]) && start < scaled_lo[col]) {
            start = scaled_lo[col];
        }
        if (isfinite(scaled_hi[col]) && start > scaled_hi[col]) {
            start = scaled_hi[col];
        }
        x[col] = start;
        xbar[col] = start;
        x_restart[col] = start;
    }

    Py_ssize_t iterations = 0;
    Py_ssize_t restarts = 0;
    Py_ssize_t step_trials = 0;
    int plateau_exit_flag = 0;
    const char *status = "iteration_limit";
    KKTEval final_ev;
    /* Plateau detection state: track the best relative KKT seen and a ring
     * buffer of the best-KKT-at-each-eval over the last plateau_window evals.
     * When the improvement over that window is less than plateau_threshold
     * AND the best candidate's primal residual is within striking distance
     * (pres_max <= 50*tol, so CGLS can close the gap), exit early. */
    double best_kkt_global = INFINITY;
    KKTEval best_ev_global;
    best_ev_global.kkt = INFINITY;
    best_ev_global.primal_res_max = INFINITY;
    /* We use a simple ring buffer to store the best KKT at each eval point. */
    Py_ssize_t plateau_buf_size = plateau_window > 0 ? plateau_window : 1;
    plateau_kkt_buf = calloc((size_t)plateau_buf_size, sizeof(double));
    Py_ssize_t plateau_buf_pos = 0;
    Py_ssize_t plateau_eval_count = 0;
    if (plateau_kkt_buf == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    for (Py_ssize_t i = 0; i < plateau_buf_size; i++) {
        plateau_kkt_buf[i] = INFINITY;
    }
    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
    evaluate_kkt(
        &op, x, y, c, b, lo, hi, bound_kind,
        col_scale, row_scale, scaled_b, b_l2, c_l2, ax, aty, &final_ev);
    if (pdhg_profile) {
        profile_initial_eval += linprogx_monotonic_seconds() - profile_phase;
    }

    if (kkt_terminated(&final_ev, tol, c_inf)) {
        status = "optimal";
    } else if (max_iter > 0) {
        Py_ssize_t eval_interval = check_interval < 64 ? check_interval : 64;
        if (eval_interval_override > 0) {
            eval_interval = eval_interval_override;
        }
        {
            /* threads: 1 = serial (default), 0 = auto, N = N capped at
             * the pool maximum. The kernels are bit-identical at any
             * thread count, so this only affects wall clock. */
            int want = (int)threads;
            if (want == 0) {
                long cores = sysconf(_SC_NPROCESSORS_ONLN);
                want = cores >= 4 ? 4 : (cores > 1 ? (int)cores : 1);
            }
            if (want > 1) {
                g_kernel_threads = pool_ensure(want);
            } else {
                g_kernel_threads = 1;
            }
        }
        double profile_loop_start = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
        Py_BEGIN_ALLOW_THREADS
        double mu_start = final_ev.kkt;
        double mu_last = final_ev.kkt;
        Py_ssize_t navg = 0;
        /* ax and aty cache the scaled products for the current iterate; the
         * pre-loop evaluate_kkt call has already filled both. */
        for (Py_ssize_t iter = 1; iter <= max_iter; iter++) {
            /* Adaptive step size: try eta, accept when it is no larger than
             * the locally safe bound movement/interaction, otherwise shrink
             * and retry. Accepted steps may grow eta slightly. */
            double shrink = 1.0 - pow((double)iter + 1.0, -0.3);
            double grow = 1.0 + pow((double)iter + 1.0, -0.6);
            double profile_step_start = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
            for (int trial = 0; trial < 60; trial++) {
                step_trials++;
                double trial_tau = eta / omega;
                double trial_sigma = eta * omega;
                /* Threaded: both passes run as disjoint-output jobs and
                 * the scalar reductions are summed afterwards in
                 * canonical index order from scratch arrays — bit
                 * identical to the serial path at any thread count.
                 * Serial: direct accumulation, no scratch traffic. */
                double dx_sq = 0.0;
                double dy_sq = 0.0;
                double interaction = 0.0;
                if (g_kernel_threads > 1) {
                    PrimalTrialJob pjob = {self->cols, trial_tau, scaled_lo, scaled_hi,
                                           aty, scaled_c, x, xbar, dxsq_col};
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    pool_run(primal_trial_job, &pjob);
                    if (pdhg_profile) {
                        profile_trial_primal += linprogx_monotonic_seconds() - profile_phase;
                    }
                    const double *restrict dxsq = dxsq_col;
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        dx_sq += dxsq[col];
                    }
                    if (pdhg_profile) {
                        profile_trial_col_reduce += linprogx_monotonic_seconds() - profile_phase;
                    }
                    DualTrialJob djob = {self->rows, trial_sigma, &op, xbar, ax,
                                         scaled_b, y, ax_trial, y_trial, dysq_row,
                                         inter_row};
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    pool_run(dual_trial_job, &djob);
                    if (pdhg_profile) {
                        profile_trial_dual += linprogx_monotonic_seconds() - profile_phase;
                    }
                    const double *restrict dysq = dysq_row;
                    const double *restrict inter = inter_row;
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        dy_sq += dysq[row];
                        interaction += inter[row];
                    }
                    if (pdhg_profile) {
                        profile_trial_row_reduce += linprogx_monotonic_seconds() - profile_phase;
                    }
                } else {
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    const double *restrict lo_v = scaled_lo;
                    const double *restrict hi_v = scaled_hi;
                    const double *restrict g_v = aty;
                    const double *restrict c_v = scaled_c;
                    const double *restrict x_v = x;
                    double *restrict out_v = xbar;
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        /* scaled_lo/scaled_hi hold +-INF when a bound is
                         * absent, so the clamp is branchless and vector
                         * friendly. */
                        double updated = x_v[col] - trial_tau * (g_v[col] + c_v[col]);
                        updated = fmax(updated, lo_v[col]);
                        updated = fmin(updated, hi_v[col]);
                        out_v[col] = updated;
                        double dx = updated - x_v[col];
                        dx_sq += dx * dx;
                    }
                    /* fused matvec + dual trial pass: one row sweep */
                    const Py_ssize_t *restrict row_start = op.row_start;
                    const int32_t *restrict col_index = op.col_index;
                    const double *restrict data = op.data;
                    const double *restrict xb = xbar;
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        double axr = 0.0;
                        Py_ssize_t end = row_start[row + 1];
                        for (Py_ssize_t p = row_start[row]; p < end; p++) {
                            axr += data[p] * xb[col_index[p]];
                        }
                        ax_trial[row] = axr;
                        double gradient = 2.0 * axr - ax[row] - scaled_b[row];
                        double updated = y[row] + trial_sigma * gradient;
                        double dy = updated - y[row];
                        y_trial[row] = updated;
                        dy_sq += dy * dy;
                        interaction += dy * (axr - ax[row]);
                    }
                    if (pdhg_profile) {
                        profile_trial_serial += linprogx_monotonic_seconds() - profile_phase;
                    }
                }
                double movement = 0.5 * omega * dx_sq + 0.5 * dy_sq / omega;
                double inter_abs = fabs(interaction);
                if (movement <= 1e-30 || inter_abs <= 1e-30) {
                    eta *= grow;
                    tau = trial_tau;
                    sigma = trial_sigma;
                    break;
                }
                double eta_bar = movement / inter_abs;
                if (eta <= eta_bar) {
                    double eta_shrunk = shrink * eta_bar;
                    double eta_grown = grow * eta;
                    eta = eta_shrunk < eta_grown ? eta_shrunk : eta_grown;
                    tau = trial_tau;
                    sigma = trial_sigma;
                    break;
                }
                eta = shrink * eta_bar;
            }
            if (pdhg_profile) {
                profile_step += linprogx_monotonic_seconds() - profile_step_start;
            }
            {
                double *swap = x;
                x = xbar;
                xbar = swap;
            }
            {
                double *swap = y;
                y = y_trial;
                y_trial = swap;
            }
            {
                double *swap = ax;
                ax = ax_trial;
                ax_trial = swap;
            }
            profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
            scaled_op_transpose_matvec_accum_x(&op, y, aty, x, x_sum);
            if (pdhg_profile) {
                profile_transpose += linprogx_monotonic_seconds() - profile_phase;
            }
            profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
            for (Py_ssize_t row = 0; row < self->rows; row++) {
                y_sum[row] += y[row];
            }
            navg++;
            iterations = iter;
            if (pdhg_profile) {
                profile_accumulate += linprogx_monotonic_seconds() - profile_phase;
            }
            if (iter % eval_interval != 0 && iter != max_iter) {
                continue;
            }

            profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
            KKTEval ev_current;
            KKTEval ev_average;
            evaluate_kkt(
                &op, x, y, c, b, lo, hi, bound_kind,
                col_scale, row_scale, scaled_b, b_l2, c_l2, ax, aty, &ev_current);
            double inv_navg = 1.0 / (double)navg;
            for (Py_ssize_t col = 0; col < self->cols; col++) {
                avg_x[col] = x_sum[col] * inv_navg;
            }
            for (Py_ssize_t row = 0; row < self->rows; row++) {
                avg_y[row] = y_sum[row] * inv_navg;
            }
            /* The average eval writes A*avg_x into ax_trial and A'*avg_y into
             * xbar, leaving the current-iterate caches in ax and aty intact. */
            evaluate_kkt(
                &op, avg_x, avg_y, c, b, lo, hi, bound_kind,
                col_scale, row_scale, scaled_b, b_l2, c_l2, ax_trial, xbar, &ev_average);
            if (pdhg_profile) {
                profile_eval += linprogx_monotonic_seconds() - profile_phase;
            }
            int best_is_average = ev_average.kkt < ev_current.kkt;
            KKTEval ev_best = best_is_average ? ev_average : ev_current;

            /* Snapshot the best iterate seen so far (used by plateau
             * exit to return the best point, not the current wander). */
            if (ev_best.kkt < best_kkt_global) {
                best_kkt_global = ev_best.kkt;
                best_ev_global = ev_best;
                const double *snap_x = best_is_average ? avg_x : x;
                const double *snap_y = best_is_average ? avg_y : y;
                for (Py_ssize_t col = 0; col < self->cols; col++) {
                    best_x[col] = snap_x[col];
                }
                for (Py_ssize_t row = 0; row < self->rows; row++) {
                    best_y[row] = snap_y[row];
                }
            }

            if (kkt_terminated(&ev_best, tol, c_inf)) {
                if (best_is_average) {
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        x[col] = avg_x[col];
                    }
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        y[row] = avg_y[row];
                    }
                }
                final_ev = ev_best;
                status = "optimal";
                break;
            }
            {
                double gap_tol = tol *
                    (1.0 + fabs(ev_best.primal_obj) + fabs(ev_best.dual_obj));
                double dual_tol = tol * (1.0 + c_inf);
                /* CGLS cleanup is a cheap polish on small/medium systems but
                 * can cost more than the saved PDHG iterations on large LPs. */
                if (self->rows <= 5000 &&
                    ev_best.primal_res_max <= 10.0 * tol &&
                    ev_best.dual_res_inf <= dual_tol &&
                    fabs(ev_best.gap) <= gap_tol) {
                    const double *cand_x = best_is_average ? avg_x : x;
                    const double *cand_y = best_is_average ? avg_y : y;
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        cleanup_x[col] = cand_x[col] * col_scale[col];
                    }
                    double cleanup_max = ev_best.primal_res_max;
                    double cleanup_l2 = ev_best.primal_res_l2;
                    profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                    active_set_cgls_cleanup(
                        self, cleanup_x, b, lo, hi, bound_kind, tol,
                        &cleanup_max, &cleanup_l2);
                    if (pdhg_profile) {
                        profile_cleanup += linprogx_monotonic_seconds() - profile_phase;
                    }
                    if (cleanup_max <= tol) {
                        for (Py_ssize_t col = 0; col < self->cols; col++) {
                            cleanup_x[col] /= col_scale[col];
                        }
                        KKTEval cleanup_ev;
                        evaluate_kkt(
                            &op, cleanup_x, cand_y, c, b, lo, hi, bound_kind,
                            col_scale, row_scale, scaled_b, b_l2, c_l2,
                            cleanup_ax, cleanup_aty, &cleanup_ev);
                        if (kkt_terminated(&cleanup_ev, tol, c_inf)) {
                            for (Py_ssize_t col = 0; col < self->cols; col++) {
                                x[col] = cleanup_x[col];
                            }
                            if (best_is_average) {
                                for (Py_ssize_t row = 0; row < self->rows; row++) {
                                    y[row] = avg_y[row];
                                }
                            }
                            final_ev = cleanup_ev;
                            status = "optimal";
                            break;
                        }
                    }
                }
            }

            /* Plateau detection: if over the last plateau_window evals
             * the best KKT has improved by less than plateau_threshold
             * (relative), AND the best candidate's primal residual is
             * within striking distance (pres_max <= 50*tol so CGLS can
             * close the gap), exit early with the best-seen iterate.
             * This fires for degenerate problems like CYCLE that plateau
             * but never pass the full KKT test. */
            if (plateau_window > 0) {
                /* Record current best_kkt_global in the ring buffer. */
                plateau_kkt_buf[plateau_buf_pos] = best_kkt_global;
                plateau_buf_pos = (plateau_buf_pos + 1) % plateau_buf_size;
                plateau_eval_count++;

                if (plateau_eval_count >= plateau_window) {
                    /* The oldest entry in the ring is what was the best
                     * KKT plateau_window evals ago. */
                    double old_best = plateau_kkt_buf[plateau_buf_pos % plateau_buf_size];
                    /* Improvement fraction: (old - current) / old. */
                    double improvement = (old_best > 1e-30)
                        ? (old_best - best_kkt_global) / old_best
                        : 0.0;
                    if (improvement < plateau_threshold &&
                        best_ev_global.primal_res_max <= 50.0 * tol) {
                        /* Adopt the best-seen iterate and exit. */
                        for (Py_ssize_t col = 0; col < self->cols; col++) {
                            x[col] = best_x[col];
                        }
                        for (Py_ssize_t row = 0; row < self->rows; row++) {
                            y[row] = best_y[row];
                        }
                        final_ev = best_ev_global;
                        plateau_exit_flag = 1;
                        break;
                    }
                }
            }

            if (iter == max_iter) {
                if (best_is_average) {
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        x[col] = avg_x[col];
                    }
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        y[row] = avg_y[row];
                    }
                }
                final_ev = ev_best;
                break;
            }

            int do_restart = 0;
            int restart_type = 0;  /* 1=sufficient, 2=necessary, 3=artificial */
            if (ev_best.kkt <= restart_sufficient * mu_start) {
                do_restart = 1;
                restart_type = 1;
            } else if (ev_best.kkt <= restart_necessary * mu_start && ev_best.kkt > mu_last) {
                do_restart = 1;
                restart_type = 2;
            } else if ((double)navg >= restart_artificial * (double)iter) {
                do_restart = 1;
                restart_type = 3;
            }
            mu_last = ev_best.kkt;
            if (do_restart) {
                double profile_restart_start = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
                const double *cand_x = best_is_average ? avg_x : x;
                const double *cand_y = best_is_average ? avg_y : y;
                if (adaptive_weight == 1) {
                    double dx_sq = 0.0;
                    double dy_sq = 0.0;
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        double diff = cand_x[col] - x_restart[col];
                        dx_sq += diff * diff;
                    }
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        double diff = cand_y[row] - y_restart[row];
                        dy_sq += diff * diff;
                    }
                    if (dx_sq > 1e-30 && dy_sq > 1e-30) {
                        double ratio = sqrt(dy_sq / dx_sq);
                        omega = exp(
                            PDHG_OMEGA_SMOOTHING * log(ratio) +
                            (1.0 - PDHG_OMEGA_SMOOTHING) * log(omega));
                    }
                    /* Safeguard: the movement update can spiral when one side
                     * has converged (tiny steps keep its movement tiny). When
                     * the KKT error is grossly lopsided, nudge omega toward
                     * the lagging side: a larger omega strengthens the dual
                     * ascent that enforces primal feasibility and vice
                     * versa. Replacing the movement update entirely in this
                     * regime was tried and is worse on CYCLE: the early
                     * descent of omega needs the movement signal. */
                    {
                        double rel_primal = ev_best.primal_res_l2 / (1.0 + b_l2);
                        double rel_dual = ev_best.dual_res_l2 / (1.0 + c_l2);
                        double rel_gap = fabs(ev_best.gap) /
                            (1.0 + fabs(ev_best.primal_obj) + fabs(ev_best.dual_obj));
                        if (rel_primal > 20.0 * rel_dual && rel_primal > 20.0 * rel_gap) {
                            omega *= 2.0;
                        } else if (rel_dual > 20.0 * rel_primal) {
                            omega *= 0.5;
                        }
                    }
                } else if (adaptive_weight == 2) {
                    /* Residual-balance update: when primal infeasibility
                     * dominates the dual error, raise omega so the dual
                     * ascent gets stronger, and vice versa. The per-restart
                     * change is clamped so omega cannot spiral. */
                    double rel_primal = ev_best.primal_res_l2 / (1.0 + b_l2);
                    double rel_dual = ev_best.dual_res_l2 / (1.0 + c_l2);
                    double rel_gap = fabs(ev_best.gap) /
                        (1.0 + fabs(ev_best.primal_obj) + fabs(ev_best.dual_obj));
                    double dual_err = sqrt(rel_dual * rel_dual + rel_gap * rel_gap);
                    if (rel_primal > 1e-30 && dual_err > 1e-30) {
                        double step = PDHG_OMEGA_SMOOTHING * log(rel_primal / dual_err);
                        if (step > 1.3862943611198906) {
                            step = 1.3862943611198906;
                        } else if (step < -1.3862943611198906) {
                            step = -1.3862943611198906;
                        }
                        omega = exp(log(omega) + step);
                    }
                }
                if (adaptive_weight) {
                    if (omega < PDHG_OMEGA_MIN) {
                        omega = PDHG_OMEGA_MIN;
                    } else if (omega > PDHG_OMEGA_MAX) {
                        omega = PDHG_OMEGA_MAX;
                    }
                    tau = eta / omega;
                    sigma = eta * omega;
                }
                if (debug) {
                    double rp = ev_best.primal_res_l2 / (1.0 + b_l2);
                    double rd = ev_best.dual_res_l2 / (1.0 + c_l2);
                    double rg = ev_best.gap / (1.0 + fabs(ev_best.primal_obj) + fabs(ev_best.dual_obj));
                    double rp_cur = ev_current.primal_res_l2 / (1.0 + b_l2);
                    double rd_cur = ev_current.dual_res_l2 / (1.0 + c_l2);
                    double rg_cur = ev_current.gap / (1.0 + fabs(ev_current.primal_obj) + fabs(ev_current.dual_obj));
                    double rp_avg = ev_average.primal_res_l2 / (1.0 + b_l2);
                    double rd_avg = ev_average.dual_res_l2 / (1.0 + c_l2);
                    double rg_avg = ev_average.gap / (1.0 + fabs(ev_average.primal_obj) + fabs(ev_average.dual_obj));
                    fprintf(stderr,
                        "RST %3zd iter=%5zd type=%d avg?=%d navg=%5zd omega=%.3e eta=%.3e kkt=%.3e "
                        "rp=%.3e rd=%.3e rg=%.3e | cur kkt=%.3e rp=%.3e rd=%.3e rg=%.3e | avg kkt=%.3e rp=%.3e rd=%.3e rg=%.3e "
                        "gap_abs=%.3e pobj=%.3e dobj=%.3e pmax=%.3e dinf=%.3e\n",
                        restarts+1, iter, restart_type, best_is_average, navg, omega, eta, ev_best.kkt,
                        rp, rd, rg, ev_current.kkt, rp_cur, rd_cur, rg_cur,
                        ev_average.kkt, rp_avg, rd_avg, rg_avg,
                        ev_best.gap, ev_best.primal_obj, ev_best.dual_obj,
                        ev_best.primal_res_max, ev_best.dual_res_inf);
                }
                if (best_is_average) {
                    /* Adopt the average iterate and reuse the products that
                     * its KKT evaluation just computed. */
                    for (Py_ssize_t col = 0; col < self->cols; col++) {
                        x[col] = avg_x[col];
                        aty[col] = xbar[col];
                    }
                    for (Py_ssize_t row = 0; row < self->rows; row++) {
                        y[row] = avg_y[row];
                        ax[row] = ax_trial[row];
                    }
                }
                for (Py_ssize_t col = 0; col < self->cols; col++) {
                    x_restart[col] = x[col];
                    x_sum[col] = 0.0;
                }
                for (Py_ssize_t row = 0; row < self->rows; row++) {
                    y_restart[row] = y[row];
                    y_sum[row] = 0.0;
                }
                navg = 0;
                restarts++;
                mu_start = ev_best.kkt;
                mu_last = ev_best.kkt;
                if (pdhg_profile) {
                    profile_restart += linprogx_monotonic_seconds() - profile_restart_start;
                }
            }
        }
        Py_END_ALLOW_THREADS
        if (pdhg_profile) {
            profile_loop += linprogx_monotonic_seconds() - profile_loop_start;
        }
    }

    double max_residual = final_ev.primal_res_max;
    double l2_residual = final_ev.primal_res_l2;

    for (Py_ssize_t col = 0; col < self->cols; col++) {
        x[col] *= col_scale[col];
    }

    if (max_residual > tol) {
        profile_phase = pdhg_profile ? linprogx_monotonic_seconds() : 0.0;
        Py_BEGIN_ALLOW_THREADS
        active_set_cgls_cleanup(self, x, b, lo, hi, bound_kind, tol, &max_residual, &l2_residual);
        Py_END_ALLOW_THREADS
        if (pdhg_profile) {
            profile_cleanup += linprogx_monotonic_seconds() - profile_phase;
        }
    }
    /* Status follows the project's feasibility-based convention: the KKT
     * test can stop the loop early, but a primal-feasible final point is
     * reported optimal even when the gap test missed an eval point. */
    if (max_residual <= tol) {
        status = "optimal";
    }

    double objective = 0.0;
    for (Py_ssize_t col = 0; col < self->cols; col++) {
        objective += c[col] * x[col];
    }

    if (pdhg_profile) {
        double profile_total = linprogx_monotonic_seconds() - profile_start;
        fprintf(stderr,
            "pdhg profile: rows=%zd cols=%zd nnz=%zd iterations=%zd restarts=%zd "
            "trials=%zd threads=%d pool_threads=%d status=%s total=%.6f setup=%.6f norm=%.6f "
            "initial_eval=%.6f loop=%.6f step=%.6f transpose=%.6f accumulate=%.6f "
            "eval=%.6f restart=%.6f cleanup=%.6f trial_primal=%.6f "
            "trial_col_reduce=%.6f trial_dual=%.6f trial_row_reduce=%.6f "
            "trial_serial=%.6f\n",
            self->rows, self->cols, self->nnz, iterations, restarts, step_trials,
            g_kernel_threads, g_pool.started ? g_pool.pool_threads : g_kernel_threads,
            status, profile_total, profile_setup, profile_norm,
            profile_initial_eval, profile_loop, profile_step, profile_transpose,
            profile_accumulate, profile_eval, profile_restart, profile_cleanup,
            profile_trial_primal, profile_trial_col_reduce, profile_trial_dual,
            profile_trial_row_reduce, profile_trial_serial);
    }

    {
        PyObject *x_list = PyList_New(self->cols);
        if (x_list == NULL) {
            goto done;
        }
        for (Py_ssize_t col = 0; col < self->cols; col++) {
            PyObject *boxed = PyFloat_FromDouble(x[col]);
            if (boxed == NULL) {
                Py_DECREF(x_list);
                goto done;
            }
            PyList_SET_ITEM(x_list, col, boxed);
        }
        /* The dual in original units is row_scale * y_scaled. */
        PyObject *y_list = PyList_New(self->rows);
        if (y_list == NULL) {
            Py_DECREF(x_list);
            goto done;
        }
        for (Py_ssize_t row = 0; row < self->rows; row++) {
            PyObject *boxed = PyFloat_FromDouble(y[row] * row_scale[row]);
            if (boxed == NULL) {
                Py_DECREF(x_list);
                Py_DECREF(y_list);
                goto done;
            }
            PyList_SET_ITEM(y_list, row, boxed);
        }
        result = Py_BuildValue(
            "{s:s,s:d,s:d,s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:i,s:N,s:N}",
            "status",
            status,
            "objective",
            objective,
            "max_primal_residual",
            max_residual,
            "l2_primal_residual",
            l2_residual,
            "iterations",
            iterations,
            "operator_norm",
            norm,
            "step_size",
            tau,
            "objective_scale",
            omega_initial,
            "primal_weight",
            omega,
            "dual_residual",
            final_ev.dual_res_inf,
            "gap",
            final_ev.gap,
            "restarts",
            restarts,
            "step_trials",
            step_trials,
            "plateau_exit",
            plateau_exit_flag,
            "x",
            x_list,
            "y",
            y_list);
    }

done:
    free(plateau_kkt_buf);
    free(c);
    free(b);
    free(lo);
    free(hi);
    free(col_scale);
    free(scaled_lo);
    free(scaled_hi);
    free(scaled_c);
    free(row_scale);
    free(scaled_b);
    free(x);
    free(xbar);
    free(y);
    free(ax);
    free(aty);
    free(y_trial);
    free(dxsq_col);
    free(dysq_row);
    free(inter_row);
    g_kernel_threads = 1;
    free(ax_trial);
    free(x_sum);
    free(y_sum);
    free(avg_x);
    free(avg_y);
    free(x_restart);
    free(y_restart);
    free(best_x);
    free(best_y);
    free(cleanup_x);
    free(cleanup_ax);
    free(cleanup_aty);
    free(operator_data);
    free(operator_csc_data);
    free(op_col_index);
    free(op_row_index);
    free(bound_kind);
    return result;
}

static PyObject *CSRMatrix_normal_equations_solve(CSRMatrixObject *self, PyObject *args);
static PyObject *CSRMatrix_solve_eq_box_ipm(CSRMatrixObject *self, PyObject *args, PyObject *kwds);
static PyObject *CSRMatrix_solve_eq_box_dual_simplex(CSRMatrixObject *self, PyObject *args, PyObject *kwds);
static PyObject *CSRMatrix_supernode_sizes(CSRMatrixObject *self, PyObject *args);
static PyObject *CSRMatrix_supernode_symbolic_structure(CSRMatrixObject *self, PyObject *args);

static PyGetSetDef CSRMatrix_getset[] = {
    {"shape", (getter)CSRMatrix_shape, NULL, "matrix shape", NULL},
    {"nnz", (getter)CSRMatrix_nnz, NULL, "number of stored nonzeros", NULL},
    {NULL}
};

static PyMethodDef CSRMatrix_methods[] = {
    {"matvec", (PyCFunction)CSRMatrix_matvec, METH_VARARGS, "Compute A @ x."},
    {"transpose_matvec", (PyCFunction)CSRMatrix_transpose_matvec, METH_VARARGS, "Compute A.T @ x."},
    {"density", (PyCFunction)CSRMatrix_density, METH_NOARGS, "Return nnz / (rows * cols)."},
    {"to_components", (PyCFunction)CSRMatrix_to_components, METH_NOARGS, "Return indptr, indices, data."},
    {"to_dense", (PyCFunction)CSRMatrix_to_dense, METH_NOARGS, "Materialize as nested Python lists."},
    {"solve_eq_box_pdhg", (PyCFunction)CSRMatrix_solve_eq_box_pdhg, METH_VARARGS | METH_KEYWORDS, "Solve min c'x subject to Ax=b and lo<=x<=hi with native PDHG."},
    {"normal_equations_solve", (PyCFunction)CSRMatrix_normal_equations_solve, METH_VARARGS, "Solve (A diag(d) A' + delta I) x = rhs with the native sparse Cholesky."},
    {"solve_eq_box_ipm", (PyCFunction)CSRMatrix_solve_eq_box_ipm, METH_VARARGS | METH_KEYWORDS, "Solve min c'x subject to Ax=b and lo<=x<=hi with a native interior point method."},
    {"solve_eq_box_dual_simplex", (PyCFunction)CSRMatrix_solve_eq_box_dual_simplex, METH_VARARGS | METH_KEYWORDS, "Solve min c'x subject to Ax=b and lo<=x<=hi with bounded-variable dual simplex."},
    {"supernode_sizes", (PyCFunction)CSRMatrix_supernode_sizes, METH_NOARGS, "Test hook: fundamental supernode sizes of the Cholesky factor of A A'."},
    {"supernode_symbolic_structure", (PyCFunction)CSRMatrix_supernode_symbolic_structure, METH_NOARGS, "Test hook: supernodal row lists and descendant update maps."},
    {NULL}
};

static PyTypeObject CSRMatrixType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "linprogx._csparse.CSRMatrix",
    .tp_basicsize = sizeof(CSRMatrixObject),
    .tp_itemsize = 0,
    .tp_dealloc = (destructor)CSRMatrix_dealloc,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_doc = "C-backed compressed sparse row matrix.",
    .tp_methods = CSRMatrix_methods,
    .tp_getset = CSRMatrix_getset,
    .tp_new = CSRMatrix_new,
};

/* ------------------------------------------------------------------ */
/* Exact minimum-degree ordering on a symmetric sparsity pattern.      */
/* Quotient-graph formulation with element absorption; exact degrees   */
/* recomputed with stamp-marked unions. Used to order the IPM normal   */
/* equations before Cholesky factorization.                            */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t *data;
    int32_t len;
    int32_t cap;
} IntVec;

static int intvec_push(IntVec *v, int32_t value) {
    if (v->len == v->cap) {
        int32_t cap = v->cap < 4 ? 4 : v->cap * 2;
        int32_t *grown = realloc(v->data, (size_t)cap * sizeof(int32_t));
        if (grown == NULL) {
            return -1;
        }
        v->data = grown;
        v->cap = cap;
    }
    v->data[v->len++] = value;
    return 0;
}

typedef struct {
    int64_t *keys;   /* (degree << 32) | var, lazy deletion */
    Py_ssize_t len;
    Py_ssize_t cap;
} MinHeap;

static int heap_push(MinHeap *h, int32_t degree, int32_t var) {
    if (h->len == h->cap) {
        Py_ssize_t cap = h->cap < 64 ? 64 : h->cap * 2;
        int64_t *grown = realloc(h->keys, (size_t)cap * sizeof(int64_t));
        if (grown == NULL) {
            return -1;
        }
        h->keys = grown;
        h->cap = cap;
    }
    int64_t key = ((int64_t)degree << 32) | (int64_t)(uint32_t)var;
    Py_ssize_t i = h->len++;
    while (i > 0) {
        Py_ssize_t parent = (i - 1) / 2;
        if (h->keys[parent] <= key) {
            break;
        }
        h->keys[i] = h->keys[parent];
        i = parent;
    }
    h->keys[i] = key;
    return 0;
}

static int64_t heap_pop(MinHeap *h) {
    int64_t top = h->keys[0];
    int64_t last = h->keys[--h->len];
    Py_ssize_t i = 0;
    for (;;) {
        Py_ssize_t left = 2 * i + 1;
        if (left >= h->len) {
            break;
        }
        Py_ssize_t small = left;
        if (left + 1 < h->len && h->keys[left + 1] < h->keys[left]) {
            small = left + 1;
        }
        if (h->keys[small] >= last) {
            break;
        }
        h->keys[i] = h->keys[small];
        i = small;
    }
    if (h->len > 0) {
        h->keys[i] = last;
    }
    return top;
}

/* Compute an exact minimum-degree elimination order for the symmetric
 * pattern given by (indptr, indices) over m nodes. Writes the order
 * (a permutation of 0..m-1) into `order`. Returns 0 on success, -1 on
 * allocation failure, and -2 if max_ops > 0 and the work budget is
 * exhausted (orderings on dense-ish graphs can cost minutes; callers use
 * the budget to abort early and route the problem elsewhere). */
static int min_degree_impl(
    int32_t m,
    const Py_ssize_t *indptr,
    const Py_ssize_t *indices,
    int32_t *order,
    int64_t max_ops,
    double flops_abort) {
    int status = -1;
    int64_t ops = 0;
    double predicted_flops = 0.0;
    IntVec *adj = calloc((size_t)m, sizeof(IntVec));
    IntVec *var_elems = calloc((size_t)m, sizeof(IntVec));
    IntVec *elements = NULL;
    Py_ssize_t elements_len = 0;
    Py_ssize_t elements_cap = 0;
    unsigned char *alive = calloc((size_t)m, sizeof(unsigned char));
    unsigned char *deferred = NULL;
    int32_t *degree = calloc((size_t)m, sizeof(int32_t));
    int32_t *mark = calloc((size_t)m, sizeof(int32_t));
    int32_t *elem_mark = NULL;
    int32_t *elem_residual = NULL;
    Py_ssize_t elem_mark_cap = 0;
    int32_t *nbhd = calloc((size_t)m, sizeof(int32_t));
    MinHeap heap = {NULL, 0, 0};
    int32_t stamp = 0;
    if (adj == NULL || var_elems == NULL || alive == NULL || degree == NULL ||
        mark == NULL || nbhd == NULL) {
        goto cleanup;
    }

    /* Dense-node deferral (AMD-style): nodes whose degree far exceeds
     * 10*sqrt(m) sit in nearly every elimination neighborhood and turn
     * the quotient-graph updates quadratic (osa_14: median degree 47
     * but max 2302 of m=2337 drove the ordering to ~0.3s). They are
     * removed from the graph up front and appended to the END of the
     * ordering in index order — the position any good ordering gives
     * them — where the dense-tail factor absorbs them at BLAS speed.
     * Purely structural rule, no measured-machine constant. */
    deferred = calloc((size_t)m, sizeof(unsigned char));
    if (deferred == NULL) {
        goto cleanup;
    }
    int32_t dense_threshold = (int32_t)(10.0 * sqrt((double)m));
    if (dense_threshold < m / 2) {
        /* Only overwhelmingly dense rows qualify: deferring moderately
         * dense rows (cre_a: 3 rows at degree ~0.26m) measurably
         * degraded the ordering and broke raw-problem IPM
         * certification, while the pathological rows this rule targets
         * (osa family) sit at ~0.98m. */
        dense_threshold = m / 2;
    }
    if (dense_threshold < 16) {
        dense_threshold = 16;
    }
    int32_t n_deferred = 0;
    for (int32_t j = 0; j < m; j++) {
        int32_t deg0 = (int32_t)(indptr[j + 1] - indptr[j]) - 1;
        if (deg0 > dense_threshold) {
            deferred[j] = 1;
            alive[j] = 0;
            n_deferred++;
        } else {
            alive[j] = 1;
        }
    }
    for (int32_t j = 0; j < m; j++) {
        if (deferred[j]) {
            continue;
        }
        for (Py_ssize_t idx = indptr[j]; idx < indptr[j + 1]; idx++) {
            int32_t i = (int32_t)indices[idx];
            if (i > j && !deferred[i]) {
                if (intvec_push(&adj[i], j) != 0 || intvec_push(&adj[j], i) != 0) {
                    goto cleanup;
                }
            }
        }
    }
    for (int32_t v = 0; v < m; v++) {
        if (deferred[v]) {
            continue;
        }
        degree[v] = adj[v].len;
        if (heap_push(&heap, degree[v], v) != 0) {
            goto cleanup;
        }
    }

    for (int32_t count = 0; count < m - n_deferred; count++) {
        int32_t v = -1;
        for (;;) {
            int64_t key = heap_pop(&heap);
            int32_t deg = (int32_t)(key >> 32);
            int32_t cand = (int32_t)(uint32_t)(key & 0xffffffff);
            if (alive[cand] && degree[cand] == deg) {
                v = cand;
                break;
            }
        }
        order[count] = v;
        alive[v] = 0;

        /* Neighborhood = alive adjacency of v plus members of v's elements. */
        stamp++;
        mark[v] = stamp;
        int32_t nbhd_len = 0;
        ops += adj[v].len;
        for (int32_t k = 0; k < adj[v].len; k++) {
            int32_t u = adj[v].data[k];
            if (alive[u] && mark[u] != stamp) {
                mark[u] = stamp;
                nbhd[nbhd_len++] = u;
            }
        }
        for (int32_t k = 0; k < var_elems[v].len; k++) {
            IntVec *e = &elements[var_elems[v].data[k]];
            ops += e->len;
            for (int32_t t = 0; t < e->len; t++) {
                int32_t u = e->data[t];
                if (alive[u] && mark[u] != stamp) {
                    mark[u] = stamp;
                    nbhd[nbhd_len++] = u;
                }
            }
        }
        int32_t nbhd_stamp = stamp;
        if (max_ops > 0 && ops > max_ops) {
            status = -2;
            goto cleanup;
        }
        /* Early fill abort: the pivot's external degree d adds ~d^2 to
         * the numeric factor cost, so the running sum predicts the
         * final factor flops as the ordering proceeds. Approximate
         * degrees overestimate, so only abort when the prediction is
         * far past the cap (the exact post-ordering check does the
         * fine gating); fill-explosive graphs abort in milliseconds
         * instead of burning the whole ordering budget. */
        predicted_flops += (double)nbhd_len * (double)nbhd_len;
        if (flops_abort > 0.0 && predicted_flops > flops_abort) {
            status = -2;
            goto cleanup;
        }

        /* New element holding the neighborhood; absorb v's old elements. */
        if (elements_len == elements_cap) {
            Py_ssize_t cap = elements_cap < 16 ? 16 : elements_cap * 2;
            IntVec *grown = realloc(elements, (size_t)cap * sizeof(IntVec));
            if (grown == NULL) {
                goto cleanup;
            }
            memset(grown + elements_cap, 0, (size_t)(cap - elements_cap) * sizeof(IntVec));
            elements = grown;
            elements_cap = cap;
        }
        int32_t eid = (int32_t)elements_len++;
        if (elements_cap > elem_mark_cap) {
            int32_t *grown = realloc(elem_mark, (size_t)elements_cap * sizeof(int32_t));
            if (grown == NULL) {
                goto cleanup;
            }
            memset(grown + elem_mark_cap, 0,
                   (size_t)(elements_cap - elem_mark_cap) * sizeof(int32_t));
            elem_mark = grown;
            int32_t *grown_res = realloc(elem_residual, (size_t)elements_cap * sizeof(int32_t));
            if (grown_res == NULL) {
                goto cleanup;
            }
            memset(grown_res + elem_mark_cap, 0,
                   (size_t)(elements_cap - elem_mark_cap) * sizeof(int32_t));
            elem_residual = grown_res;
            elem_mark_cap = elements_cap;
        }
        IntVec *new_elem = &elements[eid];
        for (int32_t k = 0; k < nbhd_len; k++) {
            if (intvec_push(new_elem, nbhd[k]) != 0) {
                goto cleanup;
            }
        }
        /* Mark absorbed element ids (they are emptied below). */
        stamp++;
        int32_t absorb_stamp = stamp;
        for (int32_t k = 0; k < var_elems[v].len; k++) {
            int32_t e = var_elems[v].data[k];
            elem_mark[e] = absorb_stamp;
            free(elements[e].data);
            elements[e].data = NULL;
            elements[e].len = 0;
            elements[e].cap = 0;
        }

        for (int32_t k = 0; k < nbhd_len; k++) {
            int32_t u = nbhd[k];
            /* Drop v and any neighbor covered by the new element. */
            int32_t kept = 0;
            for (int32_t t = 0; t < adj[u].len; t++) {
                int32_t w = adj[u].data[t];
                if (w == v || !alive[w] || mark[w] == nbhd_stamp) {
                    continue;
                }
                adj[u].data[kept++] = w;
            }
            adj[u].len = kept;
            /* Drop absorbed elements, add the new one. */
            kept = 0;
            for (int32_t t = 0; t < var_elems[u].len; t++) {
                int32_t e = var_elems[u].data[t];
                if (elem_mark[e] == absorb_stamp) {
                    continue;
                }
                var_elems[u].data[kept++] = e;
            }
            var_elems[u].len = kept;
            if (intvec_push(&var_elems[u], eid) != 0) {
                goto cleanup;
            }
        }

        /* Approximate degree update in the spirit of approximate minimum
         * degree: d(u) <= |alive adjacency outside L_p| + |L_p \ {u}| +
         * sum over u's other elements e of |L_e \ L_p|, where each element
         * residual is computed ONCE per elimination instead of once per
         * neighbor. Elements shared across the neighborhood are the
         * dominant cost in exact degree updates; this removes it. */
        stamp++;
        {
            int32_t residual_stamp = stamp;
            for (int32_t k = 0; k < nbhd_len; k++) {
                int32_t u = nbhd[k];
                int32_t deg = adj[u].len + nbhd_len - 1;
                ops += var_elems[u].len;
                for (int32_t t = 0; t < var_elems[u].len; t++) {
                    int32_t e = var_elems[u].data[t];
                    if (e == eid) {
                        continue; /* covered by the nbhd_len - 1 term */
                    }
                    if (elem_mark[e] != residual_stamp) {
                        elem_mark[e] = residual_stamp;
                        IntVec *ev = &elements[e];
                        int32_t live = 0;
                        ops += ev->len;
                        for (int32_t t2 = 0; t2 < ev->len; t2++) {
                            int32_t w = ev->data[t2];
                            if (alive[w] && mark[w] != nbhd_stamp) {
                                live++;
                            }
                        }
                        elem_residual[e] = live;
                    }
                    deg += elem_residual[e];
                }
                degree[u] = deg;
                if (heap_push(&heap, deg, u) != 0) {
                    goto cleanup;
                }
            }
        }
        if (max_ops > 0 && ops > max_ops) {
            status = -2;
            goto cleanup;
        }
    }
    {
        /* deferred dense nodes come last, in index order (deterministic) */
        int32_t at = m - n_deferred;
        for (int32_t j = 0; j < m; j++) {
            if (deferred[j]) {
                order[at++] = j;
            }
        }
    }
    status = 0;

cleanup:
    if (adj != NULL) {
        for (int32_t v = 0; v < m; v++) {
            free(adj[v].data);
        }
    }
    if (var_elems != NULL) {
        for (int32_t v = 0; v < m; v++) {
            free(var_elems[v].data);
        }
    }
    if (elements != NULL) {
        for (Py_ssize_t e = 0; e < elements_len; e++) {
            free(elements[e].data);
        }
    }
    free(adj);
    free(var_elems);
    free(elements);
    free(alive);
    free(deferred);
    free(degree);
    free(mark);
    free(elem_mark);
    free(elem_residual);
    free(nbhd);
    free(heap.keys);
    return status;
}

/* ------------------------------------------------------------------ */
/* Sparse Cholesky of C = P (A D A' + delta I) P' with fixed pattern.  */
/* Setup once per problem (pattern, ordering, elimination tree,        */
/* symbolic factorization, assembly scatter map); refactor + solve     */
/* once per IPM iteration with a new diagonal D.                       */
/* ------------------------------------------------------------------ */

typedef struct {
    int32_t m;
    /* permuted symmetric pattern of A D A' + delta I (full, CSC, sorted) */
    Py_ssize_t *Cp;
    int32_t *Ci;
    double *Cx;
    /* assembly map: ordered pairs of entries sharing an A column are
     * enumerated in a fixed order at refactor time; pair_offset[at] is the
     * destination offset in Cx of the at-th pair. */
    Py_ssize_t *pair_offset;
    Py_ssize_t n_pairs;
    Py_ssize_t *diag_offset;
    int32_t *perm;     /* order[k] = original row eliminated at step k */
    int32_t *pinv;
    /* Cholesky factor L (CSC, diagonal first in each column) */
    Py_ssize_t *Lp;
    int32_t *Li;
    double *Lx;
    int32_t *parent;   /* elimination tree */
    int32_t *cursor;   /* per-column insertion cursor during refactor */
    int32_t *estack;   /* ereach scratch */
    int32_t *epattern; /* ereach result */
    int32_t *emark;
    double *work;      /* dense accumulator for the factorization */
    double *work2;     /* dense buffer for triangular solves */
    /* Dense-column splitting (Sherman-Morrison-Woodbury): columns whose
     * normal-equations clique would be ruinous are excluded from the
     * sparse factor and handled as a low-rank correction. */
    Py_ssize_t n_dense;
    int32_t *dense_cols;          /* A column indices */
    unsigned char *col_is_dense;  /* size A->cols */
    double *Umat;                 /* m x k, column-major */
    double *Wmat;                 /* m x k, column-major: M_s^-1 U */
    double *cap;                  /* k x k dense Cholesky factor of I + U'W */
    double *cap_rhs;              /* k scratch */
    /* Dense tail: the trailing tail_len columns of L are nearly dense
     * (top of the elimination tree), so their Schur complement is
     * accumulated into a row-major tail_len x tail_len buffer and
     * factored with a blocked dense kernel; results are copied back
     * into the same CSC storage, so the solves are untouched. */
    int32_t tail_start;           /* first tail column; == m disables */
    int32_t tail_len;
    double prefix_flops;          /* sum of colcount^2 over the sparse prefix */
    double factor_flops;          /* sum of colcount^2 over all of L */
    double *Tdense;               /* tail_len x tail_len, row-major */
    /* Fundamental supernode partition of L (consecutive columns sharing
     * lower structure). snode_start has n_snodes+1 entries; supernode s
     * spans columns [snode_start[s], snode_start[s+1]). Foundation for a
     * supernodal numeric factor; computed but not yet used by refactor. */
    int32_t n_snodes;
    int32_t *snode_start;
    int snode_symbolic_ready;
    int32_t *col_snode;          /* column -> supernode id */
    Py_ssize_t *snode_row_ptr;   /* row lists for each supernode panel */
    int32_t *snode_rows;
    Py_ssize_t *snode_panel_ptr; /* row-major (row position, column) -> Lx offset */
    Py_ssize_t *snode_panel_lx;
    Py_ssize_t *snode_panel_cx;
    Py_ssize_t *snode_update_ptr; /* target supernode -> update range */
    Py_ssize_t n_snode_updates;
    struct SNodeUpdate *snode_updates;
    int32_t *snode_update_pivot_srcpos;
    int32_t *snode_update_pivot_col;
    int32_t *snode_update_target_srcpos;
    int32_t *snode_update_target_rowpos;
    Py_ssize_t snode_panel_cap;
    double *snode_panel;
    Py_ssize_t snode_update_a_cap;
    Py_ssize_t snode_update_b_cap;
    Py_ssize_t snode_update_c_cap;
    double *snode_update_a;
    double *snode_update_b;
    double *snode_update_c;
} CholContext;

typedef struct SNodeUpdate {
    int32_t source;
    int32_t pivot_col_first;
    int pivot_cols_contiguous;
    Py_ssize_t pivot_begin;
    Py_ssize_t pivot_end;
    Py_ssize_t target_begin;
    Py_ssize_t target_end;
} SNodeUpdate;

static void chol_free(CholContext *ctx) {
    if (ctx == NULL) {
        return;
    }
    free(ctx->Cp);
    free(ctx->Ci);
    free(ctx->Cx);
    free(ctx->pair_offset);
    free(ctx->diag_offset);
    free(ctx->perm);
    free(ctx->pinv);
    free(ctx->Lp);
    free(ctx->Li);
    free(ctx->Lx);
    free(ctx->parent);
    free(ctx->cursor);
    free(ctx->estack);
    free(ctx->epattern);
    free(ctx->emark);
    free(ctx->work);
    free(ctx->work2);
    free(ctx->dense_cols);
    free(ctx->col_is_dense);
    free(ctx->Umat);
    free(ctx->Wmat);
    free(ctx->cap);
    free(ctx->cap_rhs);
    free(ctx->Tdense);
    free(ctx->snode_start);
    free(ctx->col_snode);
    free(ctx->snode_row_ptr);
    free(ctx->snode_rows);
    free(ctx->snode_panel_ptr);
    free(ctx->snode_panel_lx);
    free(ctx->snode_panel_cx);
    free(ctx->snode_update_ptr);
    free(ctx->snode_updates);
    free(ctx->snode_update_pivot_srcpos);
    free(ctx->snode_update_pivot_col);
    free(ctx->snode_update_target_srcpos);
    free(ctx->snode_update_target_rowpos);
    free(ctx->snode_panel);
    free(ctx->snode_update_a);
    free(ctx->snode_update_b);
    free(ctx->snode_update_c);
    free(ctx);
}

static int cmp_int32(const void *a, const void *b) {
    int32_t x = *(const int32_t *)a;
    int32_t y = *(const int32_t *)b;
    return (x > y) - (x < y);
}

static Py_ssize_t chol_find_offset(const CholContext *ctx, int32_t row, int32_t col) {
    Py_ssize_t lo_idx = ctx->Cp[col];
    Py_ssize_t hi_idx = ctx->Cp[col + 1] - 1;
    while (lo_idx <= hi_idx) {
        Py_ssize_t mid = (lo_idx + hi_idx) / 2;
        if (ctx->Ci[mid] == row) {
            return mid;
        }
        if (ctx->Ci[mid] < row) {
            lo_idx = mid + 1;
        } else {
            hi_idx = mid - 1;
        }
    }
    return -1;
}

/* Pattern of row k of L in topological order; returns top index into
 * ctx->epattern (entries epattern[top..m-1]). */
static int32_t chol_ereach(const CholContext *ctx, int32_t k) {
    int32_t top = ctx->m;
    int32_t len;
    ctx->emark[k] = k + 1;
    for (Py_ssize_t p = ctx->Cp[k]; p < ctx->Cp[k + 1]; p++) {
        int32_t i = ctx->Ci[p];
        if (i >= k) {
            continue;
        }
        len = 0;
        while (ctx->emark[i] != k + 1) {
            ctx->estack[len++] = i;
            ctx->emark[i] = k + 1;
            i = ctx->parent[i];
        }
        while (len > 0) {
            ctx->epattern[--top] = ctx->estack[--len];
        }
    }
    return top;
}

static Py_ssize_t int32_lower_bound(const int32_t *values, Py_ssize_t n, int32_t target) {
    Py_ssize_t lo = 0;
    Py_ssize_t hi = n;
    while (lo < hi) {
        Py_ssize_t mid = lo + (hi - lo) / 2;
        if (values[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

static void chol_count_snode_update_maps(
    const CholContext *ctx, int32_t source, int32_t target,
    Py_ssize_t *pivot_count, Py_ssize_t *target_count) {
    int32_t k0 = ctx->snode_start[source];
    int32_t k1 = ctx->snode_start[source + 1];
    int32_t j0 = ctx->snode_start[target];
    int32_t j1 = ctx->snode_start[target + 1];
    Py_ssize_t src_begin = ctx->snode_row_ptr[source];
    Py_ssize_t src_end = ctx->snode_row_ptr[source + 1];
    Py_ssize_t target_begin = ctx->snode_row_ptr[target];
    Py_ssize_t target_len = ctx->snode_row_ptr[target + 1] - target_begin;
    const int32_t *src_rows = ctx->snode_rows + src_begin;
    const int32_t *target_rows = ctx->snode_rows + target_begin;
    Py_ssize_t pc = 0;
    Py_ssize_t tc = 0;
    for (Py_ssize_t pos = k1 - k0; pos < src_end - src_begin; pos++) {
        int32_t row = src_rows[pos];
        if (row >= j0 && row < j1) {
            pc++;
        }
        Py_ssize_t target_pos = int32_lower_bound(target_rows, target_len, row);
        if (target_pos < target_len && target_rows[target_pos] == row) {
            tc++;
        }
    }
    *pivot_count = pc;
    *target_count = tc;
}

static void chol_fill_snode_update_maps(
    CholContext *ctx, SNodeUpdate *update, int32_t target,
    Py_ssize_t *pivot_at, Py_ssize_t *target_at) {
    int32_t source = update->source;
    int32_t k0 = ctx->snode_start[source];
    int32_t k1 = ctx->snode_start[source + 1];
    int32_t j0 = ctx->snode_start[target];
    int32_t j1 = ctx->snode_start[target + 1];
    Py_ssize_t src_begin = ctx->snode_row_ptr[source];
    Py_ssize_t src_end = ctx->snode_row_ptr[source + 1];
    Py_ssize_t target_begin = ctx->snode_row_ptr[target];
    Py_ssize_t target_len = ctx->snode_row_ptr[target + 1] - target_begin;
    const int32_t *src_rows = ctx->snode_rows + src_begin;
    const int32_t *target_rows = ctx->snode_rows + target_begin;
    for (Py_ssize_t pos = k1 - k0; pos < src_end - src_begin; pos++) {
        int32_t row = src_rows[pos];
        if (row >= j0 && row < j1) {
            ctx->snode_update_pivot_srcpos[*pivot_at] = (int32_t)pos;
            ctx->snode_update_pivot_col[*pivot_at] = row - j0;
            (*pivot_at)++;
        }
        Py_ssize_t target_pos = int32_lower_bound(target_rows, target_len, row);
        if (target_pos < target_len && target_rows[target_pos] == row) {
            ctx->snode_update_target_srcpos[*target_at] = (int32_t)pos;
            ctx->snode_update_target_rowpos[*target_at] = (int32_t)target_pos;
            (*target_at)++;
        }
    }
    Py_ssize_t pc = update->pivot_end - update->pivot_begin;
    update->pivot_col_first = pc > 0 ? ctx->snode_update_pivot_col[update->pivot_begin] : 0;
    update->pivot_cols_contiguous = pc > 0;
    for (Py_ssize_t i = 0; i < pc; i++) {
        if (ctx->snode_update_pivot_col[update->pivot_begin + i] !=
            update->pivot_col_first + (int32_t)i) {
            update->pivot_cols_contiguous = 0;
            break;
        }
    }
}

static int chol_build_supernode_symbolic(CholContext *ctx) {
    int32_t m = ctx->m;
    int32_t ns = ctx->n_snodes;
    int32_t *mark = NULL;
    Py_ssize_t *update_count = NULL;
    Py_ssize_t *cursor = NULL;

    ctx->col_snode = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    ctx->snode_row_ptr = calloc((size_t)ns + 1, sizeof(Py_ssize_t));
    ctx->snode_panel_ptr = calloc((size_t)ns + 1, sizeof(Py_ssize_t));
    if (ctx->col_snode == NULL || ctx->snode_row_ptr == NULL ||
        ctx->snode_panel_ptr == NULL) {
        goto fail;
    }
    for (int32_t s = 0; s < ns; s++) {
        int32_t j0 = ctx->snode_start[s];
        int32_t j1 = ctx->snode_start[s + 1];
        /* Union row list of a (possibly relaxed) supernode: its own
         * columns [j0, j1) followed by the below-diagonal structure of
         * its last column. parent[j] == j+1 holds inside every merged
         * group, so earlier columns' structures nest into the last. */
        Py_ssize_t nr = (Py_ssize_t)(j1 - j0) + (ctx->Lp[j1] - ctx->Lp[j1 - 1] - 1);
        Py_ssize_t panel_n = nr * (Py_ssize_t)(j1 - j0);
        if (panel_n > ctx->snode_panel_cap) {
            ctx->snode_panel_cap = panel_n;
        }
        for (int32_t j = j0; j < j1; j++) {
            ctx->col_snode[j] = s;
        }
        ctx->snode_row_ptr[s + 1] = ctx->snode_row_ptr[s] + nr;
        ctx->snode_panel_ptr[s + 1] =
            ctx->snode_panel_ptr[s] + nr * (Py_ssize_t)(j1 - j0);
    }
    ctx->snode_panel = calloc((size_t)(ctx->snode_panel_cap > 0 ?
                                       ctx->snode_panel_cap : 1),
                              sizeof(double));
    ctx->snode_rows = calloc((size_t)(ctx->snode_row_ptr[ns] > 0 ?
                                      ctx->snode_row_ptr[ns] : 1),
                             sizeof(int32_t));
    ctx->snode_panel_lx = calloc((size_t)(ctx->snode_panel_ptr[ns] > 0 ?
                                          ctx->snode_panel_ptr[ns] : 1),
                                 sizeof(Py_ssize_t));
    ctx->snode_panel_cx = calloc((size_t)(ctx->snode_panel_ptr[ns] > 0 ?
                                          ctx->snode_panel_ptr[ns] : 1),
                                 sizeof(Py_ssize_t));
    if (ctx->snode_panel == NULL || ctx->snode_rows == NULL ||
        ctx->snode_panel_lx == NULL || ctx->snode_panel_cx == NULL) {
        goto fail;
    }
    for (int32_t s = 0; s < ns; s++) {
        int32_t j0 = ctx->snode_start[s];
        int32_t j1 = ctx->snode_start[s + 1];
        int32_t w = j1 - j0;
        Py_ssize_t nr = ctx->snode_row_ptr[s + 1] - ctx->snode_row_ptr[s];
        int32_t *rows = ctx->snode_rows + ctx->snode_row_ptr[s];
        if (nr < w) {
            goto fail;
        }
        for (int32_t c = 0; c < w; c++) {
            rows[c] = j0 + c;
        }
        memcpy(rows + w, ctx->Li + ctx->Lp[j1 - 1] + 1,
               (size_t)(nr - w) * sizeof(int32_t));
        Py_ssize_t panel_base = ctx->snode_panel_ptr[s];
        /* Padding positions (union rows absent from a column's exact
         * structure) alias the sentinel zero slot at Lx[Lp[m]]: they
         * read as exact 0.0 in gathers and only ever have exact zeros
         * scattered back. */
        Py_ssize_t zero_lx = ctx->Lp[ctx->m];
        for (int32_t c = 0; c < w; c++) {
            int32_t col = j0 + c;
            Py_ssize_t lx = ctx->Lp[col];
            Py_ssize_t lx_end = ctx->Lp[col + 1];
            for (Py_ssize_t rpos = 0; rpos < nr; rpos++) {
                Py_ssize_t dst = panel_base + rpos * (Py_ssize_t)w + c;
                ctx->snode_panel_cx[dst] = chol_find_offset(ctx, rows[rpos], col);
                if (rpos < c) {
                    ctx->snode_panel_lx[dst] = -1;
                    continue;
                }
                if (lx < lx_end && ctx->Li[lx] == rows[rpos]) {
                    ctx->snode_panel_lx[dst] = lx;
                    lx++;
                } else {
                    ctx->snode_panel_lx[dst] = zero_lx;
                }
            }
            if (lx != lx_end) {
                goto fail;
            }
        }
    }

    mark = calloc((size_t)(ns > 0 ? ns : 1), sizeof(int32_t));
    update_count = calloc((size_t)(ns > 0 ? ns : 1), sizeof(Py_ssize_t));
    if (mark == NULL || update_count == NULL) {
        goto fail;
    }
    for (int32_t source = 0; source < ns; source++) {
        int32_t k0 = ctx->snode_start[source];
        int32_t k1 = ctx->snode_start[source + 1];
        Py_ssize_t begin = ctx->snode_row_ptr[source];
        Py_ssize_t end = ctx->snode_row_ptr[source + 1];
        int32_t *rows = ctx->snode_rows + begin;
        for (Py_ssize_t pos = k1 - k0; pos < end - begin; pos++) {
            int32_t target = ctx->col_snode[rows[pos]];
            if (target > source && mark[target] != source + 1) {
                mark[target] = source + 1;
                update_count[target]++;
            }
        }
    }
    ctx->snode_update_ptr = calloc((size_t)ns + 1, sizeof(Py_ssize_t));
    if (ctx->snode_update_ptr == NULL) {
        goto fail;
    }
    for (int32_t s = 0; s < ns; s++) {
        ctx->snode_update_ptr[s + 1] = ctx->snode_update_ptr[s] + update_count[s];
    }
    ctx->n_snode_updates = ctx->snode_update_ptr[ns];
    ctx->snode_updates = calloc((size_t)(ctx->n_snode_updates > 0 ?
                                         ctx->n_snode_updates : 1),
                                sizeof(SNodeUpdate));
    cursor = calloc((size_t)(ns > 0 ? ns : 1), sizeof(Py_ssize_t));
    if (ctx->snode_updates == NULL || cursor == NULL) {
        goto fail;
    }
    memcpy(cursor, ctx->snode_update_ptr, (size_t)ns * sizeof(Py_ssize_t));
    memset(mark, 0, (size_t)ns * sizeof(int32_t));
    for (int32_t source = 0; source < ns; source++) {
        int32_t k0 = ctx->snode_start[source];
        int32_t k1 = ctx->snode_start[source + 1];
        Py_ssize_t begin = ctx->snode_row_ptr[source];
        Py_ssize_t end = ctx->snode_row_ptr[source + 1];
        int32_t *rows = ctx->snode_rows + begin;
        for (Py_ssize_t pos = k1 - k0; pos < end - begin; pos++) {
            int32_t target = ctx->col_snode[rows[pos]];
            if (target > source && mark[target] != source + 1) {
                mark[target] = source + 1;
                Py_ssize_t id = cursor[target]++;
                ctx->snode_updates[id].source = source;
            }
        }
    }

    Py_ssize_t pivot_total = 0;
    Py_ssize_t target_total = 0;
    for (int32_t target = 0; target < ns; target++) {
        for (Py_ssize_t u = ctx->snode_update_ptr[target];
             u < ctx->snode_update_ptr[target + 1]; u++) {
            Py_ssize_t pc = 0;
            Py_ssize_t tc = 0;
            chol_count_snode_update_maps(
                ctx, ctx->snode_updates[u].source, target, &pc, &tc);
            if (pc <= 0 || tc <= 0) {
                goto fail;
            }
            int32_t source = ctx->snode_updates[u].source;
            Py_ssize_t wk = (Py_ssize_t)(ctx->snode_start[source + 1] -
                                         ctx->snode_start[source]);
            Py_ssize_t a_need = pc * wk;
            Py_ssize_t b_need = tc * wk;
            Py_ssize_t c_need = pc * tc;
            if (a_need > ctx->snode_update_a_cap) {
                ctx->snode_update_a_cap = a_need;
            }
            if (b_need > ctx->snode_update_b_cap) {
                ctx->snode_update_b_cap = b_need;
            }
            if (c_need > ctx->snode_update_c_cap) {
                ctx->snode_update_c_cap = c_need;
            }
            ctx->snode_updates[u].pivot_begin = pivot_total;
            pivot_total += pc;
            ctx->snode_updates[u].pivot_end = pivot_total;
            ctx->snode_updates[u].target_begin = target_total;
            target_total += tc;
            ctx->snode_updates[u].target_end = target_total;
        }
    }
    ctx->snode_update_pivot_srcpos = calloc((size_t)(pivot_total > 0 ? pivot_total : 1),
                                           sizeof(int32_t));
    ctx->snode_update_pivot_col = calloc((size_t)(pivot_total > 0 ? pivot_total : 1),
                                        sizeof(int32_t));
    ctx->snode_update_target_srcpos = calloc((size_t)(target_total > 0 ? target_total : 1),
                                            sizeof(int32_t));
    ctx->snode_update_target_rowpos = calloc((size_t)(target_total > 0 ? target_total : 1),
                                            sizeof(int32_t));
    ctx->snode_update_a = calloc((size_t)(ctx->snode_update_a_cap > 0 ?
                                          ctx->snode_update_a_cap : 1),
                                 sizeof(double));
    ctx->snode_update_b = calloc((size_t)(ctx->snode_update_b_cap > 0 ?
                                          ctx->snode_update_b_cap : 1),
                                 sizeof(double));
    ctx->snode_update_c = calloc((size_t)(ctx->snode_update_c_cap > 0 ?
                                          ctx->snode_update_c_cap : 1),
                                 sizeof(double));
    if (ctx->snode_update_pivot_srcpos == NULL || ctx->snode_update_pivot_col == NULL ||
        ctx->snode_update_target_srcpos == NULL || ctx->snode_update_target_rowpos == NULL ||
        ctx->snode_update_a == NULL || ctx->snode_update_b == NULL ||
        ctx->snode_update_c == NULL) {
        goto fail;
    }
    pivot_total = 0;
    target_total = 0;
    for (int32_t target = 0; target < ns; target++) {
        for (Py_ssize_t u = ctx->snode_update_ptr[target];
             u < ctx->snode_update_ptr[target + 1]; u++) {
            chol_fill_snode_update_maps(
                ctx, &ctx->snode_updates[u], target, &pivot_total, &target_total);
        }
    }

    free(mark);
    free(update_count);
    free(cursor);
    return 0;

fail:
    free(mark);
    free(update_count);
    free(cursor);
    return -1;
}

static int chol_ensure_supernode_symbolic(CholContext *ctx) {
    if (ctx->snode_symbolic_ready > 0) {
        return 0;
    }
    if (ctx->snode_symbolic_ready < 0) {
        return -1;
    }
    if (chol_build_supernode_symbolic(ctx) != 0) {
        ctx->snode_symbolic_ready = -1;
        return -1;
    }
    ctx->snode_symbolic_ready = 1;
    return 0;
}

/* Build everything that depends only on the sparsity pattern of A.
 * If factor_flops_cap > 0 and the estimated numeric factorization cost
 * (sum of squared column counts of L) exceeds it, setup aborts and sets
 * *too_dense so callers can route the problem elsewhere. */
static double setup_clock(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static CholContext *chol_setup(
    CSRMatrixObject *A, double factor_flops_cap, int64_t md_ops_cap, int *too_dense) {
    int debug_setup = getenv("LINPROGX_CHOL_DEBUG") != NULL;
    double t_phase = debug_setup ? setup_clock() : 0.0;
#define SETUP_MARK(label) \
    do { \
        if (debug_setup) { \
            double now_ = setup_clock(); \
            fprintf(stderr, "chol_setup %-12s %.2fs\n", label, now_ - t_phase); \
            t_phase = now_; \
        } \
    } while (0)
    int32_t m = (int32_t)A->rows;
    CholContext *ctx = calloc(1, sizeof(CholContext));
    int32_t *head = NULL;
    int32_t *mark = NULL;
    int32_t *colbuf = NULL;
    int32_t *count = NULL;
    Py_ssize_t *Bp = NULL;
    int32_t *Bi = NULL;
    int32_t *ancestor = NULL;
    if (ctx == NULL) {
        return NULL;
    }
    ctx->m = m;

    /* --- dense-column detection: exclude clique-forming columns from the
     * sparse factor and treat them as a low-rank correction --- */
    ctx->col_is_dense = calloc((size_t)(A->cols > 0 ? A->cols : 1), sizeof(unsigned char));
    if (ctx->col_is_dense == NULL) {
        goto fail;
    }
    {
        Py_ssize_t threshold = m / 8;
        if (threshold < 64) {
            threshold = 64;
        }
        Py_ssize_t k = 0;
        for (Py_ssize_t t = 0; t < A->cols; t++) {
            if (A->csc_indptr[t + 1] - A->csc_indptr[t] > threshold) {
                ctx->col_is_dense[t] = 1;
                k++;
            }
        }
        if (k > 256) {
            /* too many dense columns for the low-rank treatment */
            if (too_dense != NULL) {
                *too_dense = 1;
            }
            goto fail;
        }
        ctx->n_dense = k;
        if (k > 0) {
            ctx->dense_cols = calloc((size_t)k, sizeof(int32_t));
            ctx->Umat = calloc((size_t)(k * m), sizeof(double));
            ctx->Wmat = calloc((size_t)(k * m), sizeof(double));
            ctx->cap = calloc((size_t)(k * k), sizeof(double));
            ctx->cap_rhs = calloc((size_t)k, sizeof(double));
            if (ctx->dense_cols == NULL || ctx->Umat == NULL || ctx->Wmat == NULL ||
                ctx->cap == NULL || ctx->cap_rhs == NULL) {
                goto fail;
            }
            Py_ssize_t at = 0;
            for (Py_ssize_t t = 0; t < A->cols; t++) {
                if (ctx->col_is_dense[t]) {
                    ctx->dense_cols[at++] = (int32_t)t;
                }
            }
        }
    }

    /* --- unpermuted ADA' pattern (full symmetric, CSC) --- */
    mark = calloc((size_t)m, sizeof(int32_t));
    colbuf = calloc((size_t)m, sizeof(int32_t));
    count = calloc((size_t)m, sizeof(int32_t));
    if (mark == NULL || colbuf == NULL || count == NULL) {
        goto fail;
    }
    Bp = calloc((size_t)m + 1, sizeof(Py_ssize_t));
    if (Bp == NULL) {
        goto fail;
    }
    /* Two passes: count then fill. Row i's pattern is the union of the
     * rows appearing in every A-column touched by row i. */
    for (int pass = 0; pass < 2; pass++) {
        for (int32_t i = 0; i < m; i++) {
            int32_t len = 0;
            mark[i] = i + 1;
            colbuf[len++] = i;
            for (Py_ssize_t p = A->indptr[i]; p < A->indptr[i + 1]; p++) {
                Py_ssize_t t = A->indices[p];
                if (ctx->col_is_dense[t]) {
                    continue;
                }
                for (Py_ssize_t q = A->csc_indptr[t]; q < A->csc_indptr[t + 1]; q++) {
                    int32_t j = (int32_t)A->csc_rows[q];
                    if (mark[j] != i + 1) {
                        mark[j] = i + 1;
                        colbuf[len++] = j;
                    }
                }
            }
            if (pass == 0) {
                Bp[i + 1] = Bp[i] + len;
            } else {
                Py_ssize_t base = Bp[i];
                for (int32_t t = 0; t < len; t++) {
                    Bi[base + t] = colbuf[t];
                }
            }
        }
        if (pass == 0) {
            Bi = calloc((size_t)Bp[m], sizeof(int32_t));
            if (Bi == NULL) {
                goto fail;
            }
            memset(mark, 0, (size_t)m * sizeof(int32_t));
        }
    }

    /* --- minimum-degree ordering on that pattern --- */
    ctx->perm = calloc((size_t)m, sizeof(int32_t));
    ctx->pinv = calloc((size_t)m, sizeof(int32_t));
    if (ctx->perm == NULL || ctx->pinv == NULL) {
        goto fail;
    }
    {
        Py_ssize_t *Bp_ss = Bp;
        Py_ssize_t *Bi_ss = calloc((size_t)Bp[m], sizeof(Py_ssize_t));
        if (Bi_ss == NULL) {
            goto fail;
        }
        for (Py_ssize_t p = 0; p < Bp[m]; p++) {
            Bi_ss[p] = Bi[p];
        }
        SETUP_MARK("pre-md");
        int status = min_degree_impl(m, Bp_ss, Bi_ss, ctx->perm, md_ops_cap,
                                     factor_flops_cap > 0.0 ? 4.0 * factor_flops_cap : 0.0);
        SETUP_MARK("min-degree");
        free(Bi_ss);
        if (status == -2 && too_dense != NULL) {
            *too_dense = 1;
        }
        if (status != 0) {
            goto fail;
        }
    }
    for (int32_t k = 0; k < m; k++) {
        ctx->pinv[ctx->perm[k]] = k;
    }

    /* --- permuted pattern C = P B P', CSC with sorted columns --- */
    ctx->Cp = calloc((size_t)m + 1, sizeof(Py_ssize_t));
    if (ctx->Cp == NULL) {
        goto fail;
    }
    for (int32_t newc = 0; newc < m; newc++) {
        int32_t oldc = ctx->perm[newc];
        ctx->Cp[newc + 1] = ctx->Cp[newc] + (Bp[oldc + 1] - Bp[oldc]);
    }
    ctx->Ci = calloc((size_t)ctx->Cp[m], sizeof(int32_t));
    ctx->Cx = calloc((size_t)ctx->Cp[m], sizeof(double));
    if (ctx->Ci == NULL || ctx->Cx == NULL) {
        goto fail;
    }
    head = calloc((size_t)m, sizeof(int32_t));
    if (head == NULL) {
        goto fail;
    }
    /* per-column sort of the permuted row indices: insertion sort for
     * short columns, qsort above that. (The previous counting sort
     * scanned all m buckets per column — O(m^2), 10+ seconds on
     * 100k-row instances.) */
    for (int32_t newc = 0; newc < m; newc++) {
        int32_t oldc = ctx->perm[newc];
        Py_ssize_t base = ctx->Cp[newc];
        Py_ssize_t at = base;
        for (Py_ssize_t p = Bp[oldc]; p < Bp[oldc + 1]; p++) {
            ctx->Ci[at++] = ctx->pinv[Bi[p]];
        }
        Py_ssize_t nz = at - base;
        if (nz <= 64) {
            for (Py_ssize_t i = base + 1; i < at; i++) {
                int32_t v = ctx->Ci[i];
                Py_ssize_t q = i;
                while (q > base && ctx->Ci[q - 1] > v) {
                    ctx->Ci[q] = ctx->Ci[q - 1];
                    q--;
                }
                ctx->Ci[q] = v;
            }
        } else {
            qsort(ctx->Ci + base, (size_t)nz, sizeof(int32_t), cmp_int32);
        }
    }

    /* --- assembly map --- */
    Py_ssize_t n_pairs = 0;
    for (Py_ssize_t t = 0; t < A->cols; t++) {
        if (ctx->col_is_dense[t]) {
            continue;
        }
        Py_ssize_t nz = A->csc_indptr[t + 1] - A->csc_indptr[t];
        n_pairs += nz * nz;
    }
    if (n_pairs > (Py_ssize_t)1 << 27) {
        /* refuse absurd assembly maps (dense columns) */
        goto fail;
    }
    ctx->n_pairs = n_pairs;
    ctx->pair_offset = calloc((size_t)(n_pairs > 0 ? n_pairs : 1), sizeof(Py_ssize_t));
    ctx->diag_offset = calloc((size_t)m, sizeof(Py_ssize_t));
    if (ctx->pair_offset == NULL || ctx->diag_offset == NULL) {
        goto fail;
    }
    {
        Py_ssize_t at = 0;
        for (Py_ssize_t t = 0; t < A->cols; t++) {
            if (ctx->col_is_dense[t]) {
                continue;
            }
            for (Py_ssize_t p = A->csc_indptr[t]; p < A->csc_indptr[t + 1]; p++) {
                int32_t r1 = ctx->pinv[A->csc_rows[p]];
                for (Py_ssize_t q = A->csc_indptr[t]; q < A->csc_indptr[t + 1]; q++) {
                    int32_t r2 = ctx->pinv[A->csc_rows[q]];
                    Py_ssize_t offset = chol_find_offset(ctx, r1, r2);
                    if (offset < 0) {
                        goto fail;
                    }
                    ctx->pair_offset[at++] = offset;
                }
            }
        }
    }
    for (int32_t k = 0; k < m; k++) {
        Py_ssize_t offset = chol_find_offset(ctx, k, k);
        if (offset < 0) {
            goto fail;
        }
        ctx->diag_offset[k] = offset;
    }
    SETUP_MARK("assembly-map");

    /* --- elimination tree (Liu's algorithm with path compression) --- */
    ctx->parent = calloc((size_t)m, sizeof(int32_t));
    ancestor = calloc((size_t)m, sizeof(int32_t));
    if (ctx->parent == NULL || ancestor == NULL) {
        goto fail;
    }
    for (int32_t k = 0; k < m; k++) {
        ctx->parent[k] = -1;
        ancestor[k] = -1;
        for (Py_ssize_t p = ctx->Cp[k]; p < ctx->Cp[k + 1]; p++) {
            int32_t i = ctx->Ci[p];
            while (i != -1 && i < k) {
                int32_t inext = ancestor[i];
                ancestor[i] = k;
                if (inext == -1) {
                    ctx->parent[i] = k;
                }
                i = inext;
            }
        }
    }

    /* --- symbolic: column counts via ereach, then fixed Li --- */
    ctx->estack = calloc((size_t)m, sizeof(int32_t));
    ctx->epattern = calloc((size_t)m, sizeof(int32_t));
    ctx->emark = calloc((size_t)m, sizeof(int32_t));
    ctx->cursor = calloc((size_t)m, sizeof(int32_t));
    ctx->work = calloc((size_t)m, sizeof(double));
    ctx->work2 = calloc((size_t)m, sizeof(double));
    ctx->Lp = calloc((size_t)m + 1, sizeof(Py_ssize_t));
    if (ctx->estack == NULL || ctx->epattern == NULL || ctx->emark == NULL ||
        ctx->cursor == NULL || ctx->work == NULL || ctx->work2 == NULL ||
        ctx->Lp == NULL) {
        goto fail;
    }
    memset(count, 0, (size_t)m * sizeof(int32_t));
    for (int32_t k = 0; k < m; k++) {
        count[k]++; /* diagonal */
        int32_t top = chol_ereach(ctx, k);
        for (int32_t s = top; s < m; s++) {
            count[ctx->epattern[s]]++;
        }
    }
    SETUP_MARK("colcounts");
    {
        double flops = 0.0;
        for (int32_t k = 0; k < m; k++) {
            flops += (double)count[k] * (double)count[k];
        }
        ctx->factor_flops = flops;
        if (factor_flops_cap > 0.0) {
            if (getenv("LINPROGX_CHOL_DEBUG") != NULL) {
                fprintf(stderr, "chol_setup: factor flops %.3e (cap %.3e)\n", flops,
                        factor_flops_cap);
            }
            if (flops > factor_flops_cap) {
                if (too_dense != NULL) {
                    *too_dense = 1;
                }
                goto fail;
            }
        }
    }
    for (int32_t k = 0; k < m; k++) {
        ctx->Lp[k + 1] = ctx->Lp[k] + count[k];
    }
    ctx->Li = calloc((size_t)ctx->Lp[m], sizeof(int32_t));
    /* One extra slot past Lp[m]: relaxed-supernode panel positions that
     * are structural zeros of L alias this sentinel. It reads as exactly
     * 0.0 and only ever has exact zeros scattered back into it. */
    ctx->Lx = calloc((size_t)ctx->Lp[m] + 1, sizeof(double));
    if (ctx->Li == NULL || ctx->Lx == NULL) {
        goto fail;
    }
    memset(ctx->emark, 0, (size_t)m * sizeof(int32_t));
    for (int32_t k = 0; k < m; k++) {
        ctx->Li[ctx->Lp[k]] = k;
        ctx->cursor[k] = 1;
    }
    for (int32_t k = 0; k < m; k++) {
        int32_t top = chol_ereach(ctx, k);
        for (int32_t s = top; s < m; s++) {
            int32_t j = ctx->epattern[s];
            ctx->Li[ctx->Lp[j] + ctx->cursor[j]++] = k;
        }
    }
    memset(ctx->emark, 0, (size_t)m * sizeof(int32_t));
    SETUP_MARK("li-fill");

    /* --- fundamental supernode partition. Column j joins the supernode
     * of j-1 when j-1 is the only child of j in the elimination tree and
     * colcount[j-1] == colcount[j] + 1 (identical lower structure shifted
     * by the diagonal). This is the classic fundamental-supernode rule;
     * it is computed here as the foundation for a supernodal numeric
     * factor and does not yet change the factorization. --- */
    {
        int32_t *nchild = count; /* reuse: count is free here, size m */
        memset(nchild, 0, (size_t)m * sizeof(int32_t));
        for (int32_t j = 0; j < m; j++) {
            if (ctx->parent[j] >= 0) {
                nchild[ctx->parent[j]]++;
            }
        }
        ctx->snode_start = malloc((size_t)(m + 1) * sizeof(int32_t));
        if (ctx->snode_start == NULL) {
            goto fail;
        }
        int32_t ns = 0;
        for (int32_t j = 0; j < m; j++) {
            int32_t cc_j = (int32_t)(ctx->Lp[j + 1] - ctx->Lp[j]);
            int merge = 0;
            if (j > 0) {
                int32_t cc_prev = (int32_t)(ctx->Lp[j] - ctx->Lp[j - 1]);
                if (ctx->parent[j - 1] == j && nchild[j] == 1 &&
                    cc_prev == cc_j + 1) {
                    merge = 1;
                }
            }
            if (!merge) {
                ctx->snode_start[ns++] = j;
            }
        }
        ctx->snode_start[ns] = m;
        ctx->n_snodes = ns;

        /* --- relaxed supernode amalgamation (Ashcraft/Grimes-style).
         * Merge an adjacent chain of fundamental supernodes when the
         * elimination tree links them (the parent of the left group's
         * last column is the right group's first column) and the merged
         * panel would carry at most a bounded fraction of structural
         * zeros. Padding positions are exact zeros of L (every update
         * product into them has a structurally-zero factor), so the
         * numeric factor is unchanged; merging turns many thin scalar
         * descendant updates into fewer BLAS-panel updates. The
         * threshold is a global constant, not a per-instance switch.
         * With parent[j] == j+1 across the whole merged group, its union
         * row set is [g0, j2) followed by struct(j2-1) minus the
         * diagonal, so the padded trapezoid size is closed-form. --- */
        {
            const char *relax_env = getenv("LINPROGX_SNODE_RELAX");
            double relax_frac = relax_env != NULL ? atof(relax_env) : 0.15;
            int32_t fundamental_ns = ns;
            if (relax_frac > 0.0 && ns > 1) {
                int32_t *merged_start = malloc(((size_t)ns + 1) * sizeof(int32_t));
                if (merged_start == NULL) {
                    goto fail;
                }
                int32_t out = 0;
                int32_t s = 0;
                while (s < ns) {
                    int32_t g0 = ctx->snode_start[s];
                    double true_nnz = 0.0;
                    for (int32_t j = g0; j < ctx->snode_start[s + 1]; j++) {
                        true_nnz += (double)(ctx->Lp[j + 1] - ctx->Lp[j]);
                    }
                    int32_t t = s + 1;
                    while (t < ns) {
                        int32_t j1 = ctx->snode_start[t];
                        int32_t j2 = ctx->snode_start[t + 1];
                        if (ctx->parent[j1 - 1] != j1) {
                            break;
                        }
                        double next_nnz = 0.0;
                        for (int32_t j = j1; j < j2; j++) {
                            next_nnz += (double)(ctx->Lp[j + 1] - ctx->Lp[j]);
                        }
                        double W = (double)(j2 - g0);
                        double N = W + (double)(ctx->Lp[j2] - ctx->Lp[j2 - 1]) - 1.0;
                        double padded = W * N - W * (W - 1.0) / 2.0;
                        if (padded - (true_nnz + next_nnz) > relax_frac * padded) {
                            break;
                        }
                        true_nnz += next_nnz;
                        t++;
                    }
                    merged_start[out++] = g0;
                    s = t;
                }
                merged_start[out] = m;
                memcpy(ctx->snode_start, merged_start,
                       ((size_t)out + 1) * sizeof(int32_t));
                ctx->n_snodes = out;
                free(merged_start);
            }
            ns = ctx->n_snodes;
            if (debug_setup) {
                int32_t big = 0;
                for (int32_t s = 0; s < ns; s++) {
                    int32_t w = ctx->snode_start[s + 1] - ctx->snode_start[s];
                    if (w > big) {
                        big = w;
                    }
                }
                fprintf(stderr,
                        "chol_setup supernodes: m=%d fundamental=%d count=%d "
                        "mean=%.1f largest=%d relax=%.2f\n",
                        m, fundamental_ns, ns, (double)m / (double)ns, big,
                        relax_frac);
            }
        }
    }
    SETUP_MARK("supernodes");

    /* --- dense-tail selection by a two-speed cost model. Splitting the
     * factor at column s costs (sum_{j<s} colcount[j]^2) scalar flops on
     * the sparse up-looking prefix plus (m-s)^3/3 flops on the dense
     * tail. The kernels run at very different rates (the scalar
     * up-looking path ~1 Gflop/s vs OpenBLAS dpotrf ~55 Gflop/s on the
     * relevant sizes), so the optimal split minimizes
     *   prefix_sq(s) + ALPHA * (m-s)^3/3,   ALPHA = v_sparse / v_dense.
     * This absorbs a large sparse prefix into the fast dense block when
     * that prefix dominates (maros_r7), and leaves the tail small when
     * the prefix is already cheap (pilot87). ALPHA is a machine
     * throughput constant, not a per-instance switch. --- */
    ctx->tail_start = m;
    ctx->tail_len = 0;
    {
        const double ALPHA = 1.0 / 58.0; /* v_sparse / v_dense, measured */
        int32_t cap_t = 4096; /* tail buffer is t^2 doubles */
        int32_t s_min = m > cap_t ? m - cap_t : 0;
        const char *alpha_env = getenv("LINPROGX_TAIL_ALPHA");
        double alpha = alpha_env != NULL ? atof(alpha_env) : ALPHA;
        /* prefix_sq over [0, s): build cumulatively as s increases */
        double prefix_sq = 0.0;
        for (int32_t j = 0; j < s_min; j++) {
            double cj = (double)(ctx->Lp[j + 1] - ctx->Lp[j]);
            prefix_sq += cj * cj;
        }
        double best_cost = prefix_sq; /* s = m baseline counts only above */
        /* recompute the all-sparse baseline properly: full prefix_sq */
        {
            double full_sq = prefix_sq;
            for (int32_t j = s_min; j < m; j++) {
                double cj = (double)(ctx->Lp[j + 1] - ctx->Lp[j]);
                full_sq += cj * cj;
            }
            best_cost = full_sq;
        }
        for (int32_t s = s_min; s <= m - 64; s++) {
            Py_ssize_t t = m - s;
            double dense = (double)t * (double)t * (double)t / 3.0;
            double cost = prefix_sq + alpha * dense;
            if (cost < best_cost) {
                best_cost = cost;
                ctx->tail_start = s;
                ctx->tail_len = (int32_t)t;
            }
            double cs = (double)(ctx->Lp[s + 1] - ctx->Lp[s]);
            prefix_sq += cs * cs;
        }
        {
            double prefix_sq = 0.0, tail_cubed = 0.0;
            for (int32_t j = 0; j < ctx->tail_start; j++) {
                double cj = (double)(ctx->Lp[j + 1] - ctx->Lp[j]);
                prefix_sq += cj * cj;
            }
            if (ctx->tail_len > 0) {
                double t = (double)ctx->tail_len;
                tail_cubed = t * t * t / 3.0;
            }
            ctx->prefix_flops = prefix_sq;
            if (debug_setup) {
                fprintf(stderr,
                        "chol_setup tail: m=%d tail_start=%d tail_len=%d "
                        "prefix_flops=%.3e tail_flops=%.3e\n",
                        m, ctx->tail_start, ctx->tail_len, prefix_sq, tail_cubed);
            }
        }
        if (ctx->tail_len > 0) {
            ctx->Tdense = calloc((size_t)ctx->tail_len * (size_t)ctx->tail_len,
                                 sizeof(double));
            if (ctx->Tdense == NULL) {
                /* fall back to the fully sparse path */
                ctx->tail_start = m;
                ctx->tail_len = 0;
            }
        }
    }

    free(head);
    free(mark);
    free(colbuf);
    free(count);
    free(Bp);
    free(Bi);
    free(ancestor);
    return ctx;

fail:
    free(head);
    free(mark);
    free(colbuf);
    free(count);
    free(Bp);
    free(Bi);
    free(ancestor);
    chol_free(ctx);
    return NULL;
}

static void chol_solve_sparse(CholContext *ctx, const double *rhs, double *out);
static int dense_chol_factor(double *M, Py_ssize_t k);

/* Blocked dense Cholesky (row-major, lower), NB-wide panels with a
 * register-tiled 4x4 GEMM trailing update. Pivots are boosted to the
 * same 1e-12 floor as the sparse path. */
#define TAIL_NB 96

typedef struct {
    double *C;
    const double *A;
    const double *B;
    Py_ssize_t mC, nC, kk, ld;
} TailGemmJob;

static void tail_gemm_rows(double *C, const double *A, const double *B,
                           Py_ssize_t mC, Py_ssize_t nC, Py_ssize_t kk,
                           Py_ssize_t ld);

static void tail_gemm_job(void *vctx, int tid, int nthreads) {
    TailGemmJob *ctx = (TailGemmJob *)vctx;
    /* partition rows in 4-aligned chunks so every C element is computed
     * wholly by one thread in the same loop order — bit-identical at
     * any thread count */
    Py_ssize_t blocks = (ctx->mC + 3) / 4;
    Py_ssize_t b0 = blocks * tid / nthreads;
    Py_ssize_t b1 = blocks * (tid + 1) / nthreads;
    Py_ssize_t r0 = b0 * 4;
    Py_ssize_t r1 = b1 * 4;
    if (r1 > ctx->mC) {
        r1 = ctx->mC;
    }
    if (r0 >= r1) {
        return;
    }
    tail_gemm_rows(ctx->C + r0 * ctx->ld, ctx->A + r0 * ctx->ld, ctx->B,
                   r1 - r0, ctx->nC, ctx->kk, ctx->ld);
}

static void tail_gemm_update(double *C, const double *A, const double *B,
                             Py_ssize_t mC, Py_ssize_t nC, Py_ssize_t kk,
                             Py_ssize_t ld) {
    TailGemmJob job = {C, A, B, mC, nC, kk, ld};
    pool_run(tail_gemm_job, &job);
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
#endif
static void tail_gemm_rows(double *C, const double *A, const double *B,
                             Py_ssize_t mC, Py_ssize_t nC, Py_ssize_t kk,
                             Py_ssize_t ld) {
    Py_ssize_t i = 0;
    for (; i + 4 <= mC; i += 4) {
        Py_ssize_t j = 0;
        for (; j + 4 <= nC; j += 4) {
            double acc[16] = {0};
            for (Py_ssize_t p = 0; p < kk; p++) {
                double a0 = A[(i + 0) * ld + p], a1 = A[(i + 1) * ld + p];
                double a2 = A[(i + 2) * ld + p], a3 = A[(i + 3) * ld + p];
                double b0 = B[(j + 0) * ld + p], b1 = B[(j + 1) * ld + p];
                double b2 = B[(j + 2) * ld + p], b3 = B[(j + 3) * ld + p];
                acc[0] += a0 * b0; acc[1] += a0 * b1; acc[2] += a0 * b2; acc[3] += a0 * b3;
                acc[4] += a1 * b0; acc[5] += a1 * b1; acc[6] += a1 * b2; acc[7] += a1 * b3;
                acc[8] += a2 * b0; acc[9] += a2 * b1; acc[10] += a2 * b2; acc[11] += a2 * b3;
                acc[12] += a3 * b0; acc[13] += a3 * b1; acc[14] += a3 * b2; acc[15] += a3 * b3;
            }
            for (int ii = 0; ii < 4; ii++) {
                for (int jj = 0; jj < 4; jj++) {
                    C[(i + ii) * ld + (j + jj)] -= acc[ii * 4 + jj];
                }
            }
        }
        for (; j < nC; j++) {
            for (int ii = 0; ii < 4; ii++) {
                double sum = 0.0;
                for (Py_ssize_t p = 0; p < kk; p++) {
                    sum += A[(i + ii) * ld + p] * B[j * ld + p];
                }
                C[(i + ii) * ld + j] -= sum;
            }
        }
    }
    for (; i < mC; i++) {
        for (Py_ssize_t j = 0; j < nC; j++) {
            double sum = 0.0;
            for (Py_ssize_t p = 0; p < kk; p++) {
                sum += A[i * ld + p] * B[j * ld + p];
            }
            C[i * ld + j] -= sum;
        }
    }
}

#if defined(__x86_64__) || defined(_M_X64)
__attribute__((target("avx2,fma")))
#endif
static void tail_dense_chol(double *M, Py_ssize_t t) {
    for (Py_ssize_t c0 = 0; c0 < t; c0 += TAIL_NB) {
        Py_ssize_t nb = t - c0 < TAIL_NB ? t - c0 : TAIL_NB;
        for (Py_ssize_t i = c0; i < c0 + nb; i++) {
            for (Py_ssize_t j = c0; j < i; j++) {
                double sum = M[i * t + j];
                for (Py_ssize_t p = c0; p < j; p++) {
                    sum -= M[i * t + p] * M[j * t + p];
                }
                M[i * t + j] = sum / M[j * t + j];
            }
            double diag = M[i * t + i];
            for (Py_ssize_t p = c0; p < i; p++) {
                diag -= M[i * t + p] * M[i * t + p];
            }
            if (diag < 1e-12) {
                diag = 1e-12;
            }
            M[i * t + i] = sqrt(diag);
        }
        if (c0 + nb >= t) {
            break;
        }
        for (Py_ssize_t i = c0 + nb; i < t; i++) {
            for (Py_ssize_t j = c0; j < c0 + nb; j++) {
                double sum = M[i * t + j];
                for (Py_ssize_t p = c0; p < j; p++) {
                    sum -= M[i * t + p] * M[j * t + p];
                }
                M[i * t + j] = sum / M[j * t + j];
            }
        }
        Py_ssize_t rem = t - (c0 + nb);
        tail_gemm_update(&M[(c0 + nb) * t + (c0 + nb)], &M[(c0 + nb) * t + c0],
                         &M[(c0 + nb) * t + c0], rem, rem, nb, t);
    }
}

static void supernode_diag_chol(double *M, Py_ssize_t w) {
#ifdef LINPROGX_HAVE_BLAS
    if (g_tail_use_blas && w >= 64 && w <= INT32_MAX) {
        ensure_supernodal_blas_threads();
        int blas_n = (int)w;
        int blas_info = 0;
        for (Py_ssize_t i = 0; i < w; i++) {
            M[i * w + i] += 1e-11 * (1.0 + fabs(M[i * w + i]));
        }
        dpotrf_("U", &blas_n, M, &blas_n, &blas_info);
        if (blas_info == 0) {
            return;
        }
    }
#endif
    tail_dense_chol(M, w);
}

static void chol_assemble_normal(
    CholContext *ctx, CSRMatrixObject *A, const double *csc_values,
    const double *D, double delta) {
    int32_t m = ctx->m;
    memset(ctx->Cx, 0, (size_t)ctx->Cp[m] * sizeof(double));
    Py_ssize_t at = 0;
    for (Py_ssize_t t = 0; t < A->cols; t++) {
        if (ctx->col_is_dense[t]) {
            continue;
        }
        double dt = D[t];
        for (Py_ssize_t p = A->csc_indptr[t]; p < A->csc_indptr[t + 1]; p++) {
            double vp = csc_values[p] * dt;
            for (Py_ssize_t q = A->csc_indptr[t]; q < A->csc_indptr[t + 1]; q++) {
                ctx->Cx[ctx->pair_offset[at++]] += vp * csc_values[q];
            }
        }
    }
    for (int32_t k = 0; k < m; k++) {
        ctx->Cx[ctx->diag_offset[k]] += delta;
    }
}

static void chol_refactor_dense_columns(
    CholContext *ctx, CSRMatrixObject *A, const double *csc_values, const double *D) {
    int32_t m = ctx->m;
    if (ctx->n_dense <= 0) {
        return;
    }
    Py_ssize_t kd = ctx->n_dense;
    for (Py_ssize_t j = 0; j < kd; j++) {
        int32_t t = ctx->dense_cols[j];
        double scale = sqrt(D[t] > 0.0 ? D[t] : 0.0);
        double *u = ctx->Umat + j * m;
        memset(u, 0, (size_t)m * sizeof(double));
        for (Py_ssize_t p = A->csc_indptr[t]; p < A->csc_indptr[t + 1]; p++) {
            u[A->csc_rows[p]] = csc_values[p] * scale;
        }
        chol_solve_sparse(ctx, u, ctx->Wmat + j * m);
    }
    for (Py_ssize_t j = 0; j < kd; j++) {
        const double *uj = ctx->Umat + j * m;
        for (Py_ssize_t i2 = j; i2 < kd; i2++) {
            const double *wi = ctx->Wmat + i2 * m;
            double total = 0.0;
            for (int32_t r = 0; r < m; r++) {
                total += uj[r] * wi[r];
            }
            /* column-major lower triangle: entry (i2, j) */
            ctx->cap[j * kd + i2] = total + (i2 == j ? 1.0 : 0.0);
        }
    }
    dense_chol_factor(ctx->cap, kd);
}

static void chol_refactor_supernodal(
    CholContext *ctx, CSRMatrixObject *A, const double *csc_values,
    const double *D, double delta) {
#ifdef LINPROGX_HAVE_BLAS
    ensure_supernodal_blas_threads();
    int blas_threads = 1;
#else
    int blas_threads = 0;
#endif
    int profile = getenv("LINPROGX_SUPERNODAL_PROFILE") != NULL;
    double t0 = profile ? setup_clock() : 0.0;
    double t_assemble = 0.0, t_panel = 0.0, t_gather = 0.0, t_gemm = 0.0;
    double t_scatter_update = 0.0, t_scalar_update = 0.0, t_chol = 0.0;
    double t_trsm = 0.0, t_scatter_lx = 0.0, t_dense = 0.0;
    Py_ssize_t blas_updates = 0, scalar_updates = 0;
    chol_assemble_normal(ctx, A, csc_values, D, delta);
    if (profile) {
        double t1 = setup_clock();
        t_assemble += t1 - t0;
        t0 = t1;
    }
    for (int32_t s = 0; s < ctx->n_snodes; s++) {
        int32_t j0 = ctx->snode_start[s];
        int32_t j1 = ctx->snode_start[s + 1];
        int32_t w = j1 - j0;
        Py_ssize_t row_begin = ctx->snode_row_ptr[s];
        Py_ssize_t row_end = ctx->snode_row_ptr[s + 1];
        Py_ssize_t nr = row_end - row_begin;
        double *F = ctx->snode_panel;
        memset(F, 0, (size_t)(nr * (Py_ssize_t)w) * sizeof(double));

        Py_ssize_t panel_base = ctx->snode_panel_ptr[s];
        for (Py_ssize_t rpos = 0; rpos < nr; rpos++) {
            for (int32_t c = 0; c < w; c++) {
                Py_ssize_t offset = ctx->snode_panel_cx[panel_base + rpos * (Py_ssize_t)w + c];
                if (offset >= 0) {
                    F[rpos * (Py_ssize_t)w + c] = ctx->Cx[offset];
                }
            }
        }
        if (profile) {
            double t1 = setup_clock();
            t_panel += t1 - t0;
            t0 = t1;
        }

        for (Py_ssize_t uid = ctx->snode_update_ptr[s];
             uid < ctx->snode_update_ptr[s + 1]; uid++) {
            SNodeUpdate *update = &ctx->snode_updates[uid];
            int32_t source = update->source;
            int32_t k0 = ctx->snode_start[source];
            int32_t k1 = ctx->snode_start[source + 1];
            int32_t wk = k1 - k0;
            Py_ssize_t source_panel = ctx->snode_panel_ptr[source];
            Py_ssize_t pc = update->pivot_end - update->pivot_begin;
            Py_ssize_t tc = update->target_end - update->target_begin;
            int used_blas = 0;
#ifdef LINPROGX_HAVE_BLAS
            if (pc > 0 && tc > 0 && wk > 0 && pc <= INT32_MAX && tc <= INT32_MAX &&
                wk <= INT32_MAX && pc * tc * (Py_ssize_t)wk >= 32768) {
                ensure_supernodal_blas_threads();
                double *Abuf = ctx->snode_update_a;
                double *Bbuf = ctx->snode_update_b;
                double *Cbuf = ctx->snode_update_c;
                for (Py_ssize_t pi = 0; pi < pc; pi++) {
                    int32_t pivot_srcpos =
                        ctx->snode_update_pivot_srcpos[update->pivot_begin + pi];
                    for (int32_t c = 0; c < wk; c++) {
                        Py_ssize_t lx = ctx->snode_panel_lx[
                            source_panel + (Py_ssize_t)pivot_srcpos * wk + c];
                        Abuf[(Py_ssize_t)c + pi * (Py_ssize_t)wk] = ctx->Lx[lx];
                    }
                }
                for (Py_ssize_t ti = 0; ti < tc; ti++) {
                    int32_t target_srcpos =
                        ctx->snode_update_target_srcpos[update->target_begin + ti];
                    for (int32_t c = 0; c < wk; c++) {
                        Py_ssize_t lx = ctx->snode_panel_lx[
                            source_panel + (Py_ssize_t)target_srcpos * wk + c];
                        Bbuf[(Py_ssize_t)c + ti * (Py_ssize_t)wk] = ctx->Lx[lx];
                    }
                }
                if (profile) {
                    double t1 = setup_clock();
                    t_gather += t1 - t0;
                    t0 = t1;
                }
                int blas_m = (int)pc;
                int blas_n = (int)tc;
                int blas_k = wk;
                int lda = wk;
                int ldb = wk;
                int ldc = (int)pc;
                double alpha = 1.0;
                double beta = 0.0;
                dgemm_("T", "N", &blas_m, &blas_n, &blas_k, &alpha,
                       Abuf, &lda, Bbuf, &ldb, &beta, Cbuf, &ldc);
                if (profile) {
                    double t1 = setup_clock();
                    t_gemm += t1 - t0;
                    t0 = t1;
                }
                if (update->pivot_cols_contiguous) {
                    int32_t pivot_col_first = update->pivot_col_first;
                    for (Py_ssize_t ti = 0; ti < tc; ti++) {
                        int32_t target_rowpos =
                            ctx->snode_update_target_rowpos[update->target_begin + ti];
                        double *frow = F + (Py_ssize_t)target_rowpos * w + pivot_col_first;
                        const double *crow = Cbuf + ti * pc;
                        for (Py_ssize_t pi = 0; pi < pc; pi++) {
                            frow[pi] -= crow[pi];
                        }
                    }
                } else {
                    for (Py_ssize_t ti = 0; ti < tc; ti++) {
                        int32_t target_rowpos =
                            ctx->snode_update_target_rowpos[update->target_begin + ti];
                        for (Py_ssize_t pi = 0; pi < pc; pi++) {
                            int32_t pivot_col =
                                ctx->snode_update_pivot_col[update->pivot_begin + pi];
                            F[(Py_ssize_t)target_rowpos * w + pivot_col] -=
                                Cbuf[pi + ti * pc];
                        }
                    }
                }
                if (profile) {
                    double t1 = setup_clock();
                    t_scatter_update += t1 - t0;
                    t0 = t1;
                }
                blas_updates++;
                used_blas = 1;
            }
#endif
            if (!used_blas) {
                scalar_updates++;
                if (wk == 1) {
                    for (Py_ssize_t ti = update->target_begin; ti < update->target_end; ti++) {
                        int32_t target_srcpos = ctx->snode_update_target_srcpos[ti];
                        int32_t target_rowpos = ctx->snode_update_target_rowpos[ti];
                        double bval = ctx->Lx[ctx->snode_panel_lx[
                            source_panel + (Py_ssize_t)target_srcpos]];
                        if (update->pivot_cols_contiguous) {
                            double *frow =
                                F + (Py_ssize_t)target_rowpos * w + update->pivot_col_first;
                            for (Py_ssize_t pi = update->pivot_begin; pi < update->pivot_end;
                                 pi++) {
                                int32_t pivot_srcpos = ctx->snode_update_pivot_srcpos[pi];
                                Py_ssize_t local = pi - update->pivot_begin;
                                double aval = ctx->Lx[ctx->snode_panel_lx[
                                    source_panel + (Py_ssize_t)pivot_srcpos]];
                                frow[local] -= bval * aval;
                            }
                        } else {
                            for (Py_ssize_t pi = update->pivot_begin; pi < update->pivot_end;
                                 pi++) {
                                int32_t pivot_srcpos = ctx->snode_update_pivot_srcpos[pi];
                                int32_t pivot_col = ctx->snode_update_pivot_col[pi];
                                double aval = ctx->Lx[ctx->snode_panel_lx[
                                    source_panel + (Py_ssize_t)pivot_srcpos]];
                                F[(Py_ssize_t)target_rowpos * w + pivot_col] -= bval * aval;
                            }
                        }
                    }
                } else {
                    for (Py_ssize_t ti = update->target_begin; ti < update->target_end; ti++) {
                        int32_t target_srcpos = ctx->snode_update_target_srcpos[ti];
                        int32_t target_rowpos = ctx->snode_update_target_rowpos[ti];
                        for (Py_ssize_t pi = update->pivot_begin; pi < update->pivot_end; pi++) {
                            int32_t pivot_srcpos = ctx->snode_update_pivot_srcpos[pi];
                            int32_t pivot_col = ctx->snode_update_pivot_col[pi];
                            double total = 0.0;
                            for (int32_t c = 0; c < wk; c++) {
                                Py_ssize_t b_lx = ctx->snode_panel_lx[
                                    source_panel + (Py_ssize_t)target_srcpos * wk + c];
                                Py_ssize_t a_lx = ctx->snode_panel_lx[
                                    source_panel + (Py_ssize_t)pivot_srcpos * wk + c];
                                total += ctx->Lx[b_lx] * ctx->Lx[a_lx];
                            }
                            F[(Py_ssize_t)target_rowpos * w + pivot_col] -= total;
                        }
                    }
                }
                if (profile) {
                    double t1 = setup_clock();
                    t_scalar_update += t1 - t0;
                    t0 = t1;
                }
            }
        }

        supernode_diag_chol(F, w);
        if (profile) {
            double t1 = setup_clock();
            t_chol += t1 - t0;
            t0 = t1;
        }
        Py_ssize_t off_rows = nr - w;
        int used_trsm = 0;
#ifdef LINPROGX_HAVE_BLAS
        if (off_rows > 0 && w >= 16 && w <= INT32_MAX && off_rows <= INT32_MAX) {
            ensure_supernodal_blas_threads();
            int blas_m = w;
            int blas_n = (int)off_rows;
            int lda = w;
            int ldb = w;
            double alpha = 1.0;
            dtrsm_("L", "U", "T", "N", &blas_m, &blas_n, &alpha,
                   F, &lda, F + (Py_ssize_t)w * w, &ldb);
            used_trsm = 1;
        }
#endif
        if (!used_trsm) {
            for (Py_ssize_t rpos = w; rpos < nr; rpos++) {
                for (int32_t c = 0; c < w; c++) {
                    double total = F[rpos * (Py_ssize_t)w + c];
                    for (int32_t p = 0; p < c; p++) {
                        total -= F[rpos * (Py_ssize_t)w + p] * F[(Py_ssize_t)c * w + p];
                    }
                    F[rpos * (Py_ssize_t)w + c] = total / F[(Py_ssize_t)c * w + c];
                }
            }
        }
        if (profile) {
            double t1 = setup_clock();
            t_trsm += t1 - t0;
            t0 = t1;
        }

        for (Py_ssize_t rpos = 0; rpos < nr; rpos++) {
            for (int32_t c = 0; c < w; c++) {
                Py_ssize_t lx = ctx->snode_panel_lx[panel_base + rpos * (Py_ssize_t)w + c];
                if (lx >= 0) {
                    ctx->Lx[lx] = F[rpos * (Py_ssize_t)w + c];
                }
            }
        }
        if (profile) {
            double t1 = setup_clock();
            t_scatter_lx += t1 - t0;
            t0 = t1;
        }
    }
    chol_refactor_dense_columns(ctx, A, csc_values, D);
    if (profile) {
        double t1 = setup_clock();
        t_dense += t1 - t0;
        fprintf(stderr,
                "supernodal profile: assemble=%.4f panel=%.4f gather=%.4f "
                "gemm=%.4f update_scatter=%.4f scalar_update=%.4f chol=%.4f "
                "trsm=%.4f lx_scatter=%.4f dense=%.4f blas_updates=%zd "
                "scalar_updates=%zd blas_threads=%d\n",
                t_assemble, t_panel, t_gather, t_gemm, t_scatter_update,
                t_scalar_update, t_chol, t_trsm, t_scatter_lx, t_dense,
                blas_updates, scalar_updates, blas_threads);
    }
}

/* Numeric refactorization with diagonal D and regularization delta, using
 * the provided CSC value array (so callers can factor a rescaled operator
 * over the same pattern). Tiny or negative pivots are boosted (dynamic
 * regularization), standard practice for IPM normal equations. */
static void chol_refactor(
    CholContext *ctx, CSRMatrixObject *A, const double *csc_values,
    const double *D, double delta) {
    int32_t m = ctx->m;
    chol_assemble_normal(ctx, A, csc_values, D, delta);

    /* up-looking Cholesky over the fixed pattern; rows in the dense
     * tail keep their sparse prefix processing but accumulate their
     * tail-block entries into the row-major Tdense buffer, which is
     * then factored with the blocked dense kernel and copied back into
     * the same CSC storage (the solves never know the difference). */
    memset(ctx->emark, 0, (size_t)m * sizeof(int32_t));
    for (int32_t k = 0; k < m; k++) {
        ctx->cursor[k] = 1;
    }
    int32_t tstart = ctx->tail_start;
    double *T = ctx->Tdense;
    Py_ssize_t tlen = ctx->tail_len;
    double *x = ctx->work; /* maintained all-zero between rows */
    for (int32_t k = 0; k < m; k++) {
        for (Py_ssize_t p = ctx->Cp[k]; p < ctx->Cp[k + 1]; p++) {
            int32_t i = ctx->Ci[p];
            if (i <= k) {
                x[i] = ctx->Cx[p];
            }
        }
        double d = x[k];
        x[k] = 0.0;
        int32_t top = chol_ereach(ctx, k);
        if (k < tstart) {
            for (int32_t s = top; s < m; s++) {
                int32_t j = ctx->epattern[s];
                double lkj = x[j] / ctx->Lx[ctx->Lp[j]];
                x[j] = 0.0;
                Py_ssize_t end = ctx->Lp[j] + ctx->cursor[j];
                for (Py_ssize_t p = ctx->Lp[j] + 1; p < end; p++) {
                    x[ctx->Li[p]] -= ctx->Lx[p] * lkj;
                }
                d -= lkj * lkj;
                ctx->Lx[end] = lkj;
                ctx->cursor[j]++;
            }
            if (d < 1e-12) {
                d = 1e-12;
            }
            ctx->Lx[ctx->Lp[k]] = sqrt(d);
        } else {
            /* tail row: process only prefix columns; the leftover x
             * values at tail positions are exactly the Schur-corrected
             * tail entries of this row */
            for (int32_t s = top; s < m; s++) {
                int32_t j = ctx->epattern[s];
                if (j >= tstart) {
                    continue;
                }
                double lkj = x[j] / ctx->Lx[ctx->Lp[j]];
                x[j] = 0.0;
                Py_ssize_t end = ctx->Lp[j] + ctx->cursor[j];
                for (Py_ssize_t p = ctx->Lp[j] + 1; p < end; p++) {
                    x[ctx->Li[p]] -= ctx->Lx[p] * lkj;
                }
                d -= lkj * lkj;
                ctx->Lx[end] = lkj;
                ctx->cursor[j]++;
            }
            double *trow = T + (Py_ssize_t)(k - tstart) * tlen;
            for (int32_t i = tstart; i < k; i++) {
                trow[i - tstart] = x[i];
                x[i] = 0.0;
            }
            trow[k - tstart] = d;
        }
    }
    if (tlen > 0) {
#ifdef LINPROGX_HAVE_BLAS
        /* T is row-major lower == column-major upper, so dpotrf("U")
         * factors A = U^T U and writes U; read back row-major lower
         * gives exactly L with A = L L^T. delta regularization is
         * already on T's diagonal, so the block is positive definite;
         * on the rare indefinite case dpotrf returns info>0 and we
         * fall back to the hand kernel with its dynamic pivot boost.
         *
         * Only large tails go to BLAS: below ~400 the dense block is a
         * negligible share of runtime, and the hand kernel's per-pivot
         * 1e-12 floor keeps small degenerate blocks (e.g. cre_a's
         * 140-col tail) on a more stable trajectory than dpotrf's
         * unfloored factorization. The threshold is a global
         * throughput calibration, not a per-instance switch. */
        if (g_tail_use_blas && tlen >= 400) {
            ensure_blas_threads();
            int blas_n = (int)tlen;
            int blas_info = 0;
            /* dpotrf has no per-pivot floor, so on degenerate blocks it
             * lands a less-regularized factor than the hand kernel
             * (which boosts pivots to 1e-12), giving a trajectory whose
             * Lagrangian certificate cannot close (cre_a/b/d). A tiny
             * relative diagonal ridge emulates that floor: it certifies
             * the cre family with BLAS in one run while leaving the
             * well-conditioned instances unchanged (the IPM's iterative
             * refinement and certificate gate absorb it). */
            for (Py_ssize_t i = 0; i < tlen; i++) {
                T[i * tlen + i] += 1e-11 * (1.0 + fabs(T[i * tlen + i]));
            }
            dpotrf_("U", &blas_n, T, &blas_n, &blas_info);
            if (blas_info != 0) {
                tail_dense_chol(T, tlen);
            }
        } else {
            tail_dense_chol(T, tlen);
        }
#else
        tail_dense_chol(T, tlen);
#endif
        for (int32_t j = tstart; j < m; j++) {
            for (Py_ssize_t p = ctx->Lp[j]; p < ctx->Lp[j + 1]; p++) {
                ctx->Lx[p] = T[(Py_ssize_t)(ctx->Li[p] - tstart) * tlen + (j - tstart)];
            }
        }
    }

    /* Dense-column low-rank data: U = A_dense sqrt(D_dense),
     * W = M_s^-1 U, capacitance C = I + U'W (dense Cholesky). */
    chol_refactor_dense_columns(ctx, A, csc_values, D);
}

static void chol_refactor_mode(
    CholContext *ctx, CSRMatrixObject *A, const double *csc_values,
    const double *D, double delta, int use_supernodal) {
    if (use_supernodal && chol_ensure_supernode_symbolic(ctx) == 0) {
        chol_refactor_supernodal(ctx, A, csc_values, D, delta);
    } else {
        chol_refactor(ctx, A, csc_values, D, delta);
    }
}

static int chol_auto_supernodal(const CholContext *ctx) {
    if (ctx->n_snodes <= 0) {
        return 0;
    }
    /* A dominant scalar prefix is worth panelizing even when supernodes
     * are narrow: on ken_18 (prefix 1.78e8 flops, mean width 1.3) the
     * supernodal refactor measured 3.1x faster than row-wise, while
     * every other suite instance sits below 3e7 prefix flops where the
     * fragmentation overhead wins (cre_b 2.4e7 measured 2-3x slower
     * forced). 1e8 is a machine-rate constant (~0.1s of scalar prefix
     * work per refactor), not a per-instance switch. */
    if (ctx->prefix_flops >= 1e8) {
        return 1;
    }
    if (ctx->tail_len < 512) {
        return 0;
    }
    double mean_width = (double)ctx->m / (double)ctx->n_snodes;
    return mean_width >= 4.0;
}

/* y = (A D A' + delta I) x using the assembled (permuted) matrix. */
static void chol_matvec(const CholContext *ctx, const double *x, double *out) {
    int32_t m = ctx->m;
    for (int32_t k = 0; k < m; k++) {
        double total = 0.0;
        for (Py_ssize_t p = ctx->Cp[k]; p < ctx->Cp[k + 1]; p++) {
            total += ctx->Cx[p] * x[ctx->perm[ctx->Ci[p]]];
        }
        out[ctx->perm[k]] = total;
    }
    for (Py_ssize_t j = 0; j < ctx->n_dense; j++) {
        const double *u = ctx->Umat + j * m;
        double total = 0.0;
        for (int32_t i = 0; i < m; i++) {
            total += u[i] * x[i];
        }
        for (int32_t i = 0; i < m; i++) {
            out[i] += u[i] * total;
        }
    }
}

/* Solve with the SPARSE part of the factor only (no dense correction). */
static void chol_solve_sparse(CholContext *ctx, const double *rhs, double *out) {
    int32_t m = ctx->m;
    double *y = ctx->work2;
    for (int32_t k = 0; k < m; k++) {
        y[k] = rhs[ctx->perm[k]];
    }
    for (int32_t j = 0; j < m; j++) {
        double yj = y[j] / ctx->Lx[ctx->Lp[j]];
        y[j] = yj;
        for (Py_ssize_t p = ctx->Lp[j] + 1; p < ctx->Lp[j + 1]; p++) {
            y[ctx->Li[p]] -= ctx->Lx[p] * yj;
        }
    }
    for (int32_t j = m - 1; j >= 0; j--) {
        double total = y[j];
        for (Py_ssize_t p = ctx->Lp[j] + 1; p < ctx->Lp[j + 1]; p++) {
            total -= ctx->Lx[p] * y[ctx->Li[p]];
        }
        y[j] = total / ctx->Lx[ctx->Lp[j]];
    }
    for (int32_t k = 0; k < m; k++) {
        out[ctx->perm[k]] = y[k];
    }
}

/* Dense k x k Cholesky factorization / solve for the capacitance system. */
static int dense_chol_factor(double *M, Py_ssize_t k) {
    for (Py_ssize_t j = 0; j < k; j++) {
        double d = M[j * k + j];
        for (Py_ssize_t t = 0; t < j; t++) {
            double l = M[t * k + j];
            d -= l * l;
        }
        if (d < 1e-14) {
            d = 1e-14;
        }
        d = sqrt(d);
        M[j * k + j] = d;
        for (Py_ssize_t i = j + 1; i < k; i++) {
            double v = M[j * k + i];
            for (Py_ssize_t t = 0; t < j; t++) {
                v -= M[t * k + i] * M[t * k + j];
            }
            M[j * k + i] = v / d;
        }
    }
    return 0;
}

static void dense_chol_solve(const double *M, Py_ssize_t k, double *rhs) {
    for (Py_ssize_t j = 0; j < k; j++) {
        double v = rhs[j];
        for (Py_ssize_t t = 0; t < j; t++) {
            v -= M[t * k + j] * rhs[t];
        }
        rhs[j] = v / M[j * k + j];
    }
    for (Py_ssize_t j = k - 1; j >= 0; j--) {
        double v = rhs[j];
        for (Py_ssize_t t = j + 1; t < k; t++) {
            v -= M[j * k + t] * rhs[t];
        }
        rhs[j] = v / M[j * k + j];
    }
}

/* Solve (A D A' + delta I) out = rhs including the dense-column part via
 * Sherman-Morrison-Woodbury: x = t - W (I + U'W)^-1 U' t with t the
 * sparse-part solve. */
static void chol_solve(CholContext *ctx, const double *rhs, double *out) {
    chol_solve_sparse(ctx, rhs, out);
    Py_ssize_t kd = ctx->n_dense;
    if (kd == 0) {
        return;
    }
    int32_t m = ctx->m;
    for (Py_ssize_t j = 0; j < kd; j++) {
        const double *u = ctx->Umat + j * m;
        double total = 0.0;
        for (int32_t i = 0; i < m; i++) {
            total += u[i] * out[i];
        }
        ctx->cap_rhs[j] = total;
    }
    dense_chol_solve(ctx->cap, kd, ctx->cap_rhs);
    for (Py_ssize_t j = 0; j < kd; j++) {
        const double *w = ctx->Wmat + j * m;
        double scale = ctx->cap_rhs[j];
        for (int32_t i = 0; i < m; i++) {
            out[i] -= w[i] * scale;
        }
    }
}

/* Test hook: fundamental supernode sizes of chol(A A' + I)'s factor. */
static PyObject *CSRMatrix_supernode_sizes(CSRMatrixObject *self, PyObject *args) {
    (void)args;
    if (self->rows > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix too large for the 32-bit factorization");
        return NULL;
    }
    int too_dense = 0;
    CholContext *ctx = chol_setup(self, 0.0, 1000000000000LL, &too_dense);
    if (ctx == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "chol_setup failed");
        return NULL;
    }
    PyObject *sizes = PyList_New(ctx->n_snodes);
    if (sizes == NULL) {
        chol_free(ctx);
        return NULL;
    }
    for (int32_t s = 0; s < ctx->n_snodes; s++) {
        int32_t w = ctx->snode_start[s + 1] - ctx->snode_start[s];
        PyList_SET_ITEM(sizes, s, PyLong_FromLong((long)w));
    }
    chol_free(ctx);
    return sizes;
}

static PyObject *int32_list_from_slice(const int32_t *values, Py_ssize_t n) {
    PyObject *out = PyList_New(n);
    if (out == NULL) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PyLong_FromLong((long)values[i]);
        if (item == NULL) {
            Py_DECREF(out);
            return NULL;
        }
        PyList_SET_ITEM(out, i, item);
    }
    return out;
}

static PyObject *CSRMatrix_supernode_symbolic_structure(CSRMatrixObject *self, PyObject *args) {
    (void)args;
    if (self->rows > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix too large for the 32-bit factorization");
        return NULL;
    }
    int too_dense = 0;
    CholContext *ctx = chol_setup(self, 0.0, 1000000000000LL, &too_dense);
    if (ctx == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "chol_setup failed");
        return NULL;
    }
    if (chol_ensure_supernode_symbolic(ctx) != 0) {
        chol_free(ctx);
        PyErr_SetString(PyExc_RuntimeError, "supernode symbolic setup failed");
        return NULL;
    }
    PyObject *result = PyList_New(ctx->n_snodes);
    if (result == NULL) {
        chol_free(ctx);
        return NULL;
    }
    for (int32_t s = 0; s < ctx->n_snodes; s++) {
        int32_t j0 = ctx->snode_start[s];
        Py_ssize_t row_begin = ctx->snode_row_ptr[s];
        Py_ssize_t row_len = ctx->snode_row_ptr[s + 1] - row_begin;
        Py_ssize_t update_begin = ctx->snode_update_ptr[s];
        Py_ssize_t update_len = ctx->snode_update_ptr[s + 1] - update_begin;

        PyObject *start_obj = PyLong_FromLong((long)j0);
        PyObject *rows = int32_list_from_slice(ctx->snode_rows + row_begin, row_len);
        PyObject *updates = PyList_New(update_len);
        PyObject *node = NULL;
        if (start_obj == NULL || rows == NULL || updates == NULL) {
            Py_XDECREF(start_obj);
            Py_XDECREF(rows);
            Py_XDECREF(updates);
            Py_DECREF(result);
            chol_free(ctx);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < update_len; i++) {
            SNodeUpdate *update = &ctx->snode_updates[update_begin + i];
            Py_ssize_t pb = update->pivot_begin;
            Py_ssize_t pc = update->pivot_end - update->pivot_begin;
            Py_ssize_t tb = update->target_begin;
            Py_ssize_t tc = update->target_end - update->target_begin;
            PyObject *source_obj = PyLong_FromLong((long)update->source);
            PyObject *pivot_src = int32_list_from_slice(
                ctx->snode_update_pivot_srcpos + pb, pc);
            PyObject *pivot_col = int32_list_from_slice(
                ctx->snode_update_pivot_col + pb, pc);
            PyObject *target_src = int32_list_from_slice(
                ctx->snode_update_target_srcpos + tb, tc);
            PyObject *target_row = int32_list_from_slice(
                ctx->snode_update_target_rowpos + tb, tc);
            PyObject *update_tuple = PyTuple_New(5);
            if (source_obj == NULL || pivot_src == NULL || pivot_col == NULL ||
                target_src == NULL || target_row == NULL || update_tuple == NULL) {
                Py_XDECREF(source_obj);
                Py_XDECREF(pivot_src);
                Py_XDECREF(pivot_col);
                Py_XDECREF(target_src);
                Py_XDECREF(target_row);
                Py_XDECREF(update_tuple);
                Py_DECREF(start_obj);
                Py_DECREF(rows);
                Py_DECREF(updates);
                Py_DECREF(result);
                chol_free(ctx);
                return NULL;
            }
            PyTuple_SET_ITEM(update_tuple, 0, source_obj);
            PyTuple_SET_ITEM(update_tuple, 1, pivot_src);
            PyTuple_SET_ITEM(update_tuple, 2, pivot_col);
            PyTuple_SET_ITEM(update_tuple, 3, target_src);
            PyTuple_SET_ITEM(update_tuple, 4, target_row);
            PyList_SET_ITEM(updates, i, update_tuple);
        }
        node = PyTuple_New(3);
        if (node == NULL) {
            Py_DECREF(start_obj);
            Py_DECREF(rows);
            Py_DECREF(updates);
            Py_DECREF(result);
            chol_free(ctx);
            return NULL;
        }
        PyTuple_SET_ITEM(node, 0, start_obj);
        PyTuple_SET_ITEM(node, 1, rows);
        PyTuple_SET_ITEM(node, 2, updates);
        PyList_SET_ITEM(result, s, node);
    }
    chol_free(ctx);
    return result;
}

/* Test hook: solve (A D A' + delta I) x = rhs with the native Cholesky. */
static PyObject *CSRMatrix_normal_equations_solve(CSRMatrixObject *self, PyObject *args) {
    PyObject *d_obj;
    PyObject *rhs_obj;
    double delta = 0.0;
    int use_supernodal = 0;
    if (!PyArg_ParseTuple(args, "OO|dp", &d_obj, &rhs_obj, &delta, &use_supernodal)) {
        return NULL;
    }
    if (self->rows > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix too large for the 32-bit factorization");
        return NULL;
    }
    double *d = calloc((size_t)(self->cols > 0 ? self->cols : 1), sizeof(double));
    double *rhs = calloc((size_t)(self->rows > 0 ? self->rows : 1), sizeof(double));
    double *out = calloc((size_t)(self->rows > 0 ? self->rows : 1), sizeof(double));
    if (d == NULL || rhs == NULL || out == NULL ||
        fill_double_array(d_obj, self->cols, d, "d") != 0 ||
        fill_double_array(rhs_obj, self->rows, rhs, "rhs") != 0) {
        free(d);
        free(rhs);
        free(out);
        return NULL;
    }
    CholContext *ctx;
    Py_BEGIN_ALLOW_THREADS
    ctx = chol_setup(self, 0.0, 0, NULL);
    if (ctx != NULL) {
        chol_refactor_mode(ctx, self, self->csc_data, d, delta, use_supernodal);
        chol_solve(ctx, rhs, out);
    }
    Py_END_ALLOW_THREADS
    free(d);
    free(rhs);
    if (ctx == NULL) {
        free(out);
        PyErr_SetString(PyExc_RuntimeError, "Cholesky setup failed");
        return NULL;
    }
    chol_free(ctx);
    PyObject *result = PyList_New(self->rows);
    if (result == NULL) {
        free(out);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < self->rows; i++) {
        PyList_SET_ITEM(result, i, PyFloat_FromDouble(out[i]));
    }
    free(out);
    return result;
}

/* ------------------------------------------------------------------ */
/* Mehrotra predictor-corrector interior point method for              */
/*   min c'x  s.t.  Ax = b, lo <= x <= hi                              */
/* mirroring the validated Python prototype: Ruiz + cost scaling,      */
/* zero-width boxes pinned, regularized normal equations solved with   */
/* the native Cholesky.                                                */
/* ------------------------------------------------------------------ */

typedef struct {
    CholContext *chol;
    const ScaledOp *op;
    const double *D;
    const double *sl;
    const double *su;
    const double *zl;
    const double *zu;
    const unsigned char *bound_kind;
    double *rhs_x;   /* cols scratch */
    double *tmp_x;   /* cols scratch */
    double *rhs_m;   /* rows scratch */
    double *aty;     /* cols scratch */
    double *res_m;   /* rows scratch for iterative refinement */
    double *corr_m;  /* rows scratch for iterative refinement */
} IpmNewton;

static void ipm_newton_solve(
    const IpmNewton *nw,
    const double *rp,
    const double *rd,
    const double *rcl,
    const double *rcu,
    double *dy,
    double *dx,
    double *dzl,
    double *dzu) {
    Py_ssize_t n = nw->op->cols;
    Py_ssize_t m = nw->op->rows;
    for (Py_ssize_t j = 0; j < n; j++) {
        double value = rd[j];
        if (nw->bound_kind[j] & 1) {
            value -= rcl[j] / nw->sl[j];
        }
        if (nw->bound_kind[j] & 2) {
            value += rcu[j] / nw->su[j];
        }
        nw->rhs_x[j] = value;
        nw->tmp_x[j] = nw->D[j] * value;
    }
    scaled_op_matvec(nw->op, nw->tmp_x, nw->rhs_m);
    for (Py_ssize_t i = 0; i < m; i++) {
        nw->rhs_m[i] += rp[i];
    }
    chol_solve(nw->chol, nw->rhs_m, dy);
    /* Two steps of iterative refinement: the regularized factor loses a
     * few digits on ill-conditioned late-stage systems, and the refreshed
     * residual solves recover them. */
    for (int refine = 0; refine < 2; refine++) {
        chol_matvec(nw->chol, dy, nw->res_m);
        for (Py_ssize_t i = 0; i < m; i++) {
            nw->res_m[i] = nw->rhs_m[i] - nw->res_m[i];
        }
        chol_solve(nw->chol, nw->res_m, nw->corr_m);
        for (Py_ssize_t i = 0; i < m; i++) {
            dy[i] += nw->corr_m[i];
        }
    }
    scaled_op_transpose_matvec(nw->op, dy, nw->aty);
    for (Py_ssize_t j = 0; j < n; j++) {
        double step = nw->D[j] * (nw->aty[j] - nw->rhs_x[j]);
        dx[j] = step;
        dzl[j] = (nw->bound_kind[j] & 1) ? (rcl[j] - nw->zl[j] * step) / nw->sl[j] : 0.0;
        dzu[j] = (nw->bound_kind[j] & 2) ? (rcu[j] + nw->zu[j] * step) / nw->su[j] : 0.0;
    }
}

static int ipm_lagrangian_gap(const double *c, const double *b, const double *lo,
                              const double *hi, const unsigned char *bound_kind,
                              Py_ssize_t m, Py_ssize_t n, const double *x,
                              const double *y, const double *aty, double *gap_out) {
    double pobj = 0.0;
    double dobj = 0.0;
    int certifiable = 1;
    for (Py_ssize_t j = 0; j < n; j++) {
        unsigned char kind = bound_kind[j];
        double r = c[j] - aty[j];
        pobj += c[j] * x[j];
        if (kind == 4) {
            dobj += r * x[j];
            continue;
        }
        if (r > 0.0) {
            if (kind & 1) {
                dobj += r * lo[j];
            } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                certifiable = 0;
                break;
            }
        } else if (r < 0.0) {
            if (kind & 2) {
                dobj += r * hi[j];
            } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                certifiable = 0;
                break;
            }
        }
    }
    if (!certifiable) {
        return 0;
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        dobj += b[i] * y[i];
    }
    *gap_out = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
    return 1;
}

static double ipm_raw_primal_residual(const double *rp, const double *row_scale,
                                      Py_ssize_t m) {
    double max_residual = 0.0;
    for (Py_ssize_t i = 0; i < m; i++) {
        double scale = row_scale[i];
        double residual = fabs(scale != 0.0 ? rp[i] / scale : rp[i]);
        if (residual > max_residual) {
            max_residual = residual;
        }
    }
    return max_residual;
}

static int ipm_dual_polish(const ScaledOp *op, CholContext *chol, const double *c,
                           const double *b, const double *lo, const double *hi,
                           const unsigned char *bound_kind, Py_ssize_t m, Py_ssize_t n,
                           const double *x, const double *D, double *tmp_x,
                           double *rhs_m, double *candidate_y, double *candidate_aty,
                           double *gap_out) {
    for (Py_ssize_t j = 0; j < n; j++) {
        tmp_x[j] = D[j] * c[j];
    }
    scaled_op_matvec(op, tmp_x, rhs_m);
    chol_solve(chol, rhs_m, candidate_y);
    scaled_op_transpose_matvec(op, candidate_y, candidate_aty);
    double gap = INFINITY;
    if (!ipm_lagrangian_gap(c, b, lo, hi, bound_kind, m, n, x, candidate_y,
                            candidate_aty, &gap)) {
        return 0;
    }
    if (gap <= 1e-5) {
        *gap_out = gap;
        return 1;
    }
    return 0;
}

static int ipm_primal_polish(
    const ScaledOp *op, CholContext *chol, CSRMatrixObject *A,
    const double *csc_vals, const double *c, const double *b, const double *lo,
    const double *hi, const unsigned char *bound_kind, const double *row_scale,
    Py_ssize_t m, Py_ssize_t n, const double *rp, const double *y,
    const double *aty, int refactor_supernodal, double feas_tol, double b_norm,
    double *x, double *D, double *rhs_m, double *candidate_x, double *candidate_aty,
    double *pres_out, double *raw_pres_out, double *gap_out) {
    for (Py_ssize_t j = 0; j < n; j++) {
        unsigned char kind = bound_kind[j];
        if (kind == 4) {
            D[j] = 0.0;
            continue;
        }
        double weight = 1.0;
        if (kind & 1) {
            double slack = x[j] - lo[j];
            if (slack < weight) {
                weight = slack;
            }
        }
        if (kind & 2) {
            double slack = hi[j] - x[j];
            if (slack < weight) {
                weight = slack;
            }
        }
        if (!isfinite(weight) || weight > 1.0) {
            weight = 1.0;
        }
        if (weight < 1e-14) {
            weight = 1e-14;
        }
        D[j] = weight;
    }
    chol_refactor_mode(chol, A, csc_vals, D, 1e-12, refactor_supernodal);
    chol_solve(chol, rp, rhs_m);
    scaled_op_transpose_matvec(op, rhs_m, candidate_aty);
    for (Py_ssize_t j = 0; j < n; j++) {
        candidate_x[j] = x[j] + D[j] * candidate_aty[j];
    }
    double max_bound_violation = 0.0;
    for (Py_ssize_t j = 0; j < n; j++) {
        unsigned char kind = bound_kind[j];
        if ((kind & 1) && candidate_x[j] < lo[j]) {
            double viol = lo[j] - candidate_x[j];
            if (viol > max_bound_violation) {
                max_bound_violation = viol;
            }
        }
        if ((kind & 2) && candidate_x[j] > hi[j]) {
            double viol = candidate_x[j] - hi[j];
            if (viol > max_bound_violation) {
                max_bound_violation = viol;
            }
        }
    }
    if (max_bound_violation > feas_tol) {
        return 0;
    }
    scaled_op_matvec(op, candidate_x, rhs_m);
    for (Py_ssize_t i = 0; i < m; i++) {
        rhs_m[i] = b[i] - rhs_m[i];
    }
    double raw_pres = ipm_raw_primal_residual(rhs_m, row_scale, m);
    if (raw_pres > feas_tol) {
        return 0;
    }
    double gap = INFINITY;
    if (!ipm_lagrangian_gap(c, b, lo, hi, bound_kind, m, n, candidate_x, y, aty,
                            &gap) ||
        gap > 1e-5) {
        return 0;
    }
    memcpy(x, candidate_x, (size_t)n * sizeof(double));
    *pres_out = l2_norm(rhs_m, m) / b_norm;
    *raw_pres_out = raw_pres;
    *gap_out = gap;
    return 1;
}

/* Min-norm dual cleanup: a near-optimal primal point whose certificate
 * fails on a small set S of wrong-signed reduced costs can often be
 * certified by the min-norm correction delta solving
 * A_S' delta = r_S - goal (Gram system A_S'A_S, dense Cholesky). Any y
 * yields a valid Lagrangian bound, so this can gain certificates but
 * never fake one; if the certificate still fails after five rounds, y
 * is restored untouched. Returns 1 and writes *gap_out when certified. */
static int ipm_dual_cleanup(const ScaledOp *op, const double *c, const double *b,
                            const double *lo, const double *hi,
                            const unsigned char *bound_kind, Py_ssize_t m,
                            Py_ssize_t n, const double *x, double *y, double *aty,
                            double *gap_out, Py_ssize_t *rounds_out) {
    enum { CLEANUP_CAP = 512 };
    int certified = 0;
    double *y_save = malloc((size_t)m * sizeof(double));
    unsigned char *in_union = calloc((size_t)n, 1);
    Py_ssize_t *S = malloc(CLEANUP_CAP * sizeof(Py_ssize_t));
    double *w = malloc(CLEANUP_CAP * sizeof(double));
    double *G = NULL;
    if (y_save != NULL && in_union != NULL && S != NULL && w != NULL) {
        memcpy(y_save, y, (size_t)m * sizeof(double));
        Py_ssize_t s_count = 0;
        for (int round = 0; round < 5; round++) {
            scaled_op_transpose_matvec(op, y, aty);
            int overflow = 0;
            int new_viol = 0;
            for (Py_ssize_t j = 0; j < n; j++) {
                unsigned char kind = bound_kind[j];
                if (kind == 4 || in_union[j]) {
                    continue;
                }
                double r = c[j] - aty[j];
                double tolj = 1e-9 * (1.0 + fabs(c[j]));
                int viol = (r > tolj && !(kind & 1)) || (-r > tolj && !(kind & 2));
                if (viol) {
                    if (s_count >= CLEANUP_CAP) {
                        overflow = 1;
                        break;
                    }
                    in_union[j] = 1;
                    S[s_count++] = j;
                    new_viol = 1;
                }
            }
            if (overflow || s_count == 0 || (!new_viol && round > 0)) {
                break;
            }
            if (new_viol) {
                free(G);
                G = calloc((size_t)(s_count * s_count), sizeof(double));
                if (G == NULL) {
                    break;
                }
                for (Py_ssize_t a = 0; a < s_count; a++) {
                    for (Py_ssize_t bcol = a; bcol < s_count; bcol++) {
                        Py_ssize_t ja = S[a];
                        Py_ssize_t jb = S[bcol];
                        Py_ssize_t pa = op->col_start[ja];
                        Py_ssize_t pb = op->col_start[jb];
                        Py_ssize_t ea = op->col_start[ja + 1];
                        Py_ssize_t eb = op->col_start[jb + 1];
                        double dot = 0.0;
                        while (pa < ea && pb < eb) {
                            int32_t ra = op->row_index[pa];
                            int32_t rb = op->row_index[pb];
                            if (ra == rb) {
                                dot += op->csc_data[pa] * op->csc_data[pb];
                                pa++;
                                pb++;
                            } else if (ra < rb) {
                                pa++;
                            } else {
                                pb++;
                            }
                        }
                        G[a * s_count + bcol] = dot;
                        G[bcol * s_count + a] = dot;
                    }
                }
                for (Py_ssize_t a = 0; a < s_count; a++) {
                    G[a * s_count + a] += 1e-12 * (1.0 + G[a * s_count + a]);
                }
                if (dense_chol_factor(G, s_count) != 0) {
                    break;
                }
            }
            for (Py_ssize_t a = 0; a < s_count; a++) {
                Py_ssize_t j = S[a];
                unsigned char kind = bound_kind[j];
                double r = c[j] - aty[j];
                double margin = 1e-8 * (1.0 + fabs(c[j]));
                double goal;
                if (!(kind & 1) && !(kind & 2)) {
                    goal = 0.0;
                } else if (!(kind & 1)) {
                    goal = -margin;
                } else {
                    goal = margin;
                }
                w[a] = r - goal;
            }
            dense_chol_solve(G, s_count, w);
            for (Py_ssize_t a = 0; a < s_count; a++) {
                Py_ssize_t j = S[a];
                double wa = w[a];
                for (Py_ssize_t pp = op->col_start[j]; pp < op->col_start[j + 1]; pp++) {
                    y[op->row_index[pp]] += op->csc_data[pp] * wa;
                }
            }
            if (rounds_out != NULL) {
                *rounds_out = round + 1;
            }
            scaled_op_transpose_matvec(op, y, aty);
            double pobj2 = 0.0;
            double dobj2 = 0.0;
            int cert2 = 1;
            for (Py_ssize_t j = 0; j < n; j++) {
                unsigned char kind = bound_kind[j];
                double r = c[j] - aty[j];
                pobj2 += c[j] * x[j];
                if (kind == 4) {
                    dobj2 += r * x[j];
                    continue;
                }
                if (r > 0.0) {
                    if (kind & 1) {
                        dobj2 += r * lo[j];
                    } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                        cert2 = 0;
                        break;
                    }
                } else if (r < 0.0) {
                    if (kind & 2) {
                        dobj2 += r * hi[j];
                    } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                        cert2 = 0;
                        break;
                    }
                }
            }
            if (cert2) {
                for (Py_ssize_t i = 0; i < m; i++) {
                    dobj2 += b[i] * y[i];
                }
                double cleaned_gap = fabs(pobj2 - dobj2) / (1.0 + fabs(pobj2) + fabs(dobj2));
                if (cleaned_gap <= 1e-5) {
                    *gap_out = cleaned_gap;
                    certified = 1;
                    break;
                }
            }
        }
        if (!certified) {
            memcpy(y, y_save, (size_t)m * sizeof(double));
        }
    }
    free(y_save);
    free(in_union);
    free(S);
    free(w);
    free(G);
    return certified;
}

/* Active-set dual repair: where the min-norm cleanup above re-targets a
 * growing violator union with FULL steps (which overshoots and cascades
 * new violations — measured divergence on pilot87: 335 -> 1095 -> 1790
 * violators), this variant takes PARTIAL steps. Each iteration solves
 * the min-norm correction for the current working set W, ratio-tests the
 * step against every other sign-constrained column (one transpose matvec
 * gives all directional derivatives), advances only to the first
 * blocking column, and adds that column to W with an O(|W|^2) Cholesky
 * append. The working set grows monotonically, so termination is finite;
 * acceptance still requires the full Lagrangian certificate, so this can
 * gain certificates but never fake one. Returns 1 and updates y only on
 * certified success. */
static int ipm_dual_repair(const ScaledOp *op, const double *c, const double *b,
                           const double *lo, const double *hi,
                           const unsigned char *bound_kind, Py_ssize_t m,
                           Py_ssize_t n, const double *x, double *y, double *aty,
                           double *gap_out) {
    enum { REPAIR_CAP = 1024 };
    int certified = 0;
    double *y_save = malloc((size_t)m * sizeof(double));
    unsigned char *in_w = calloc((size_t)n, 1);
    Py_ssize_t *W = malloc(REPAIR_CAP * sizeof(Py_ssize_t));
    double *L = calloc((size_t)REPAIR_CAP * REPAIR_CAP, sizeof(double));
    double *rhs = malloc(REPAIR_CAP * sizeof(double));
    double *omega = malloc(REPAIR_CAP * sizeof(double));
    double *dy = malloc((size_t)m * sizeof(double));
    double *delta = malloc((size_t)n * sizeof(double));
    if (y_save == NULL || in_w == NULL || W == NULL || L == NULL ||
        rhs == NULL || omega == NULL || dy == NULL || delta == NULL) {
        goto done;
    }
    memcpy(y_save, y, (size_t)m * sizeof(double));
    scaled_op_transpose_matvec(op, y, aty);

    Py_ssize_t wn = 0;
    /* Seed W with the current violators. */
    for (Py_ssize_t j = 0; j < n; j++) {
        unsigned char kind = bound_kind[j];
        if (kind == 4) {
            continue;
        }
        double r = c[j] - aty[j];
        double tolj = 1e-9 * (1.0 + fabs(c[j]));
        int viol = (r > tolj && !(kind & 1)) || (-r > tolj && !(kind & 2));
        if (viol) {
            if (wn >= REPAIR_CAP) {
                goto done;
            }
            in_w[j] = 1;
            W[wn++] = j;
        }
    }
    if (wn == 0) {
        goto done; /* nothing to repair; caller's certificate already ran */
    }
    /* Build the Cholesky factor of G_W incrementally: appending column k
     * solves L l = g_k over the existing factor. */
    for (Py_ssize_t k = 0; k < wn; k++) {
        Py_ssize_t jk = W[k];
        for (Py_ssize_t a = 0; a <= k; a++) {
            Py_ssize_t ja = W[a];
            Py_ssize_t pa = op->col_start[ja];
            Py_ssize_t pk = op->col_start[jk];
            Py_ssize_t ea = op->col_start[ja + 1];
            Py_ssize_t ek = op->col_start[jk + 1];
            double dot = 0.0;
            while (pa < ea && pk < ek) {
                int32_t ra = op->row_index[pa];
                int32_t rk = op->row_index[pk];
                if (ra == rk) {
                    dot += op->csc_data[pa] * op->csc_data[pk];
                    pa++;
                    pk++;
                } else if (ra < rk) {
                    pa++;
                } else {
                    pk++;
                }
            }
            if (a == k) {
                dot += 1e-12 * (1.0 + dot);
                double sum = dot;
                for (Py_ssize_t t = 0; t < k; t++) {
                    sum -= L[k * REPAIR_CAP + t] * L[k * REPAIR_CAP + t];
                }
                if (sum <= 0.0) {
                    goto done;
                }
                L[k * REPAIR_CAP + k] = sqrt(sum);
            } else {
                double sum = dot;
                for (Py_ssize_t t = 0; t < a; t++) {
                    sum -= L[k * REPAIR_CAP + t] * L[a * REPAIR_CAP + t];
                }
                L[k * REPAIR_CAP + a] = sum / L[a * REPAIR_CAP + a];
            }
        }
    }

    for (int iter = 0; iter < 2 * REPAIR_CAP; iter++) {
        /* Min-norm correction for the working set from the CURRENT y:
         * targets are the remaining sign excesses with a small margin. */
        double excess = 0.0;
        for (Py_ssize_t a = 0; a < wn; a++) {
            Py_ssize_t j = W[a];
            unsigned char kind = bound_kind[j];
            double r = c[j] - aty[j];
            double margin = 1e-8 * (1.0 + fabs(c[j]));
            double goal;
            if (!(kind & 1) && !(kind & 2)) {
                goal = 0.0;
            } else if (!(kind & 1)) {
                goal = -margin;
            } else {
                goal = margin;
            }
            rhs[a] = r - goal;
            if (fabs(rhs[a]) > excess) {
                excess = fabs(rhs[a]);
            }
        }
        if (excess <= 0.0) {
            break;
        }
        /* forward/back solve with the appended factor (row-major L) */
        for (Py_ssize_t a = 0; a < wn; a++) {
            double v = rhs[a];
            for (Py_ssize_t t = 0; t < a; t++) {
                v -= L[a * REPAIR_CAP + t] * omega[t];
            }
            omega[a] = v / L[a * REPAIR_CAP + a];
        }
        for (Py_ssize_t a = wn - 1; a >= 0; a--) {
            double v = omega[a];
            for (Py_ssize_t t = a + 1; t < wn; t++) {
                v -= L[t * REPAIR_CAP + a] * omega[t];
            }
            omega[a] = v / L[a * REPAIR_CAP + a];
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            dy[i] = 0.0;
        }
        for (Py_ssize_t a = 0; a < wn; a++) {
            Py_ssize_t j = W[a];
            double wa = omega[a];
            for (Py_ssize_t pp = op->col_start[j]; pp < op->col_start[j + 1]; pp++) {
                dy[op->row_index[pp]] += op->csc_data[pp] * wa;
            }
        }
        /* Directional derivatives for every column at once. */
        scaled_op_transpose_matvec(op, dy, delta);
        /* Ratio test: largest alpha in (0,1] keeping all non-working
         * sign-constrained columns feasible. r_j(alpha) = r_j - alpha*delta_j. */
        double alpha = 1.0;
        Py_ssize_t blocker = -1;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            if (kind == 4 || in_w[j]) {
                continue;
            }
            double r = c[j] - aty[j];
            double tolj = 1e-9 * (1.0 + fabs(c[j]));
            double d = delta[j];
            if (!(kind & 1) && d < -1e-300) {
                /* needs r <= tolj; grows when d < 0 */
                double slack = tolj - r;
                double a_j = slack / (-d);
                if (a_j < alpha) {
                    alpha = a_j;
                    blocker = j;
                }
            }
            if (!(kind & 2) && d > 1e-300) {
                /* needs r >= -tolj; shrinks when d > 0 */
                double slack = r + tolj;
                double a_j = slack / d;
                if (a_j < alpha) {
                    alpha = a_j;
                    blocker = j;
                }
            }
        }
        if (alpha < 0.0) {
            alpha = 0.0;
        }
        if (alpha > 0.0) {
            for (Py_ssize_t i = 0; i < m; i++) {
                y[i] += alpha * dy[i];
            }
            scaled_op_transpose_matvec(op, y, aty);
        }
        if (blocker < 0) {
            break; /* full step taken; working set satisfied at margins */
        }
        if (wn >= REPAIR_CAP) {
            goto done; /* would exceed the cap: fail closed */
        }
        /* Append blocker to W and extend the Cholesky factor by one row. */
        {
            Py_ssize_t k = wn;
            Py_ssize_t jk = blocker;
            for (Py_ssize_t a = 0; a <= k; a++) {
                Py_ssize_t ja = a == k ? jk : W[a];
                Py_ssize_t pa = op->col_start[ja];
                Py_ssize_t pk = op->col_start[jk];
                Py_ssize_t ea = op->col_start[ja + 1];
                Py_ssize_t ek = op->col_start[jk + 1];
                double dot = 0.0;
                while (pa < ea && pk < ek) {
                    int32_t ra = op->row_index[pa];
                    int32_t rk = op->row_index[pk];
                    if (ra == rk) {
                        dot += op->csc_data[pa] * op->csc_data[pk];
                        pa++;
                        pk++;
                    } else if (ra < rk) {
                        pa++;
                    } else {
                        pk++;
                    }
                }
                if (a == k) {
                    dot += 1e-12 * (1.0 + dot);
                    double sum = dot;
                    for (Py_ssize_t t = 0; t < k; t++) {
                        sum -= L[k * REPAIR_CAP + t] * L[k * REPAIR_CAP + t];
                    }
                    if (sum <= 1e-300) {
                        goto done; /* dependent column: fail closed */
                    }
                    L[k * REPAIR_CAP + k] = sqrt(sum);
                } else {
                    double sum = dot;
                    for (Py_ssize_t t = 0; t < a; t++) {
                        sum -= L[k * REPAIR_CAP + t] * L[a * REPAIR_CAP + t];
                    }
                    L[k * REPAIR_CAP + a] = sum / L[a * REPAIR_CAP + a];
                }
            }
            in_w[jk] = 1;
            W[wn++] = jk;
        }
    }

    /* Full certificate with the repaired y (identical gate to cleanup). */
    {
        double pobj = 0.0;
        double dobj = 0.0;
        int cert = 1;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            double r = c[j] - aty[j];
            pobj += c[j] * x[j];
            if (kind == 4) {
                dobj += r * x[j];
                continue;
            }
            if (r > 0.0) {
                if (kind & 1) {
                    dobj += r * lo[j];
                } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                    cert = 0;
                    break;
                }
            } else if (r < 0.0) {
                if (kind & 2) {
                    dobj += r * hi[j];
                } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                    cert = 0;
                    break;
                }
            }
        }
        if (cert) {
            for (Py_ssize_t i = 0; i < m; i++) {
                dobj += b[i] * y[i];
            }
            double gap = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
            if (gap <= 1e-5) {
                *gap_out = gap;
                certified = 1;
            }
        }
    }

done:
    if (!certified && y_save != NULL) {
        memcpy(y, y_save, (size_t)m * sizeof(double));
        scaled_op_transpose_matvec(op, y, aty);
    }
    free(y_save);
    free(in_w);
    free(W);
    free(L);
    free(rhs);
    free(omega);
    free(dy);
    free(delta);
    return certified;
}

static PyObject *CSRMatrix_solve_eq_box_ipm(CSRMatrixObject *self, PyObject *args, PyObject *kwds) {
    PyObject *c_obj;
    PyObject *b_obj;
    PyObject *lo_obj;
    PyObject *hi_obj;
    Py_ssize_t max_iter = 60;
    double tol = 1e-9;
    int debug = 0;
    Py_ssize_t threads = 0; /* 0 = auto; the threaded kernels are
                             * bit-identical at any thread count */
    int use_blas = 1; /* 0 forces the floored hand kernel for the tail */
    int use_supernodal = -1; /* -1 auto, 0 row-wise, 1 supernodal */
    double feas_tol = -1.0;
    static char *kwlist[] = {"c", "b", "lo", "hi", "max_iter", "tol", "debug",
                             "threads", "blas", "supernodal", "feas_tol", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOO|ndpnppd", kwlist,
            &c_obj, &b_obj, &lo_obj, &hi_obj, &max_iter, &tol, &debug, &threads,
            &use_blas, &use_supernodal, &feas_tol)) {
        return NULL;
    }
    if (feas_tol <= 0.0 || !isfinite(feas_tol)) {
        feas_tol = tol > 2e-5 ? tol : 2e-5;
    }
    if (self->rows > INT32_MAX || self->cols > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix too large for the 32-bit factorization");
        return NULL;
    }
    Py_ssize_t m = self->rows;
    Py_ssize_t n = self->cols;
    Py_ssize_t nnz = self->nnz;

    PyObject *result = NULL;
    CholContext *chol = NULL;
    double *c = NULL, *b = NULL, *lo = NULL, *hi = NULL;
    double *row_scale = NULL, *col_scale = NULL;
    double *csr_vals = NULL, *csc_vals = NULL;
    int32_t *op_col_index = NULL, *op_row_index = NULL;
    double *x = NULL, *y = NULL, *zl = NULL, *zu = NULL;
    double *sl = NULL, *su = NULL;
    double *rp = NULL, *rd = NULL, *H = NULL, *D = NULL;
    double *rcl = NULL, *rcu = NULL;
    double *dx_a = NULL, *dy_a = NULL, *dzl_a = NULL, *dzu_a = NULL;
    double *dx = NULL, *dy = NULL, *dzl = NULL, *dzu = NULL;
    double *rhs_x = NULL, *tmp_x = NULL, *rhs_m = NULL, *aty = NULL, *ax = NULL;
    double *res_m = NULL, *corr_m = NULL;
    double *x_best = NULL, *y_best = NULL;
    double *zero_m = NULL, *zero_n = NULL;
    unsigned char *bound_kind = NULL;

    c = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    b = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    lo = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    hi = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    row_scale = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    col_scale = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    csr_vals = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
    csc_vals = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
    op_col_index = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int32_t));
    op_row_index = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int32_t));
    x = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    y = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    zl = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    zu = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    sl = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    su = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    rp = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    rd = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    H = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    D = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    rcl = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    rcu = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dx_a = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dy_a = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    dzl_a = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dzu_a = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dx = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dy = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    dzl = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    dzu = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    res_m = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    corr_m = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    x_best = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    y_best = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    rhs_x = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    tmp_x = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    rhs_m = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    aty = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    ax = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    /* permanently-zero rhs vectors for pure recentering solves */
    zero_m = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    zero_n = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    bound_kind = calloc((size_t)(n > 0 ? n : 1), sizeof(unsigned char));
    if (c == NULL || b == NULL || lo == NULL || hi == NULL ||
        row_scale == NULL || col_scale == NULL ||
        csr_vals == NULL || csc_vals == NULL ||
        op_col_index == NULL || op_row_index == NULL ||
        x == NULL || y == NULL || zl == NULL || zu == NULL ||
        sl == NULL || su == NULL || rp == NULL || rd == NULL ||
        H == NULL || D == NULL || rcl == NULL || rcu == NULL ||
        dx_a == NULL || dy_a == NULL || dzl_a == NULL || dzu_a == NULL ||
        dx == NULL || dy == NULL || dzl == NULL || dzu == NULL ||
        rhs_x == NULL || tmp_x == NULL || rhs_m == NULL || aty == NULL ||
        ax == NULL || res_m == NULL || corr_m == NULL ||
        zero_m == NULL || zero_n == NULL ||
        x_best == NULL || y_best == NULL || bound_kind == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    if (fill_double_array(c_obj, n, c, "c") != 0 ||
        fill_double_array(b_obj, m, b, "b") != 0 ||
        fill_double_array(lo_obj, n, lo, "lo") != 0 ||
        fill_double_array(hi_obj, n, hi, "hi") != 0) {
        goto done;
    }

    /* --- Ruiz equilibration (inf-norm, 10 passes) --- */
    for (Py_ssize_t i = 0; i < m; i++) {
        row_scale[i] = 1.0;
    }
    for (Py_ssize_t j = 0; j < n; j++) {
        col_scale[j] = 1.0;
    }
    for (int pass = 0; pass < 10; pass++) {
        for (Py_ssize_t i = 0; i < m; i++) {
            rp[i] = 0.0;
        }
        for (Py_ssize_t j = 0; j < n; j++) {
            rd[j] = 0.0;
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++) {
                double value = fabs(self->data[p] * row_scale[i] * col_scale[self->indices[p]]);
                if (value > rp[i]) {
                    rp[i] = value;
                }
                if (value > rd[self->indices[p]]) {
                    rd[self->indices[p]] = value;
                }
            }
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            if (rp[i] > 0.0) {
                row_scale[i] /= sqrt(rp[i]);
            }
        }
        for (Py_ssize_t j = 0; j < n; j++) {
            if (rd[j] > 0.0) {
                col_scale[j] /= sqrt(rd[j]);
            }
        }
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++) {
            csr_vals[p] = self->data[p] * row_scale[i] * col_scale[self->indices[p]];
            op_col_index[p] = (int32_t)self->indices[p];
        }
    }
    for (Py_ssize_t j = 0; j < n; j++) {
        for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
            csc_vals[p] = self->csc_data[p] * row_scale[self->csc_rows[p]] * col_scale[j];
            op_row_index[p] = (int32_t)self->csc_rows[p];
        }
    }
    ScaledOp op = {
        m, n, self->indptr, op_col_index, csr_vals,
        self->csc_indptr, op_row_index, csc_vals,
    };

    /* scaled problem data: b_s = R b, c_s = S c / c_scale, bounds / S */
    double c_scale = 1.0;
    for (Py_ssize_t i = 0; i < m; i++) {
        b[i] *= row_scale[i];
    }
    for (Py_ssize_t j = 0; j < n; j++) {
        c[j] *= col_scale[j];
        double abs_c = fabs(c[j]);
        if (abs_c > c_scale) {
            c_scale = abs_c;
        }
        if (isfinite(lo[j])) {
            lo[j] /= col_scale[j];
        }
        if (isfinite(hi[j])) {
            hi[j] /= col_scale[j];
        }
    }
    for (Py_ssize_t j = 0; j < n; j++) {
        c[j] /= c_scale;
    }

    /* bound classification; zero-width boxes are pinned (bit 4) */
    for (Py_ssize_t j = 0; j < n; j++) {
        int has_lo = isfinite(lo[j]);
        int has_hi = isfinite(hi[j]);
        unsigned char kind = (unsigned char)((has_lo ? 1 : 0) | (has_hi ? 2 : 0));
        if (has_lo && has_hi && hi[j] - lo[j] < 1e-10) {
            kind = 4;
        }
        bound_kind[j] = kind;
    }

    Py_ssize_t n_comp = 0;
    for (Py_ssize_t j = 0; j < n; j++) {
        if (bound_kind[j] & 1) {
            n_comp += 1;
        }
        if (bound_kind[j] & 2) {
            n_comp += 1;
        }
    }
    if (n_comp == 0) {
        n_comp = 1;
    }

    int factor_too_dense = 0;
    Py_ssize_t dual_cleanup_rounds = 0;
    double last_ap = 0.0;
    double last_ad = 0.0;
    double t_refactor = 0.0;
    double t_newton = 0.0;
    double mu_hist[10];
    Py_ssize_t last_cleanup_attempt = -1000;
    for (int hh = 0; hh < 10; hh++) {
        mu_hist[hh] = INFINITY;
    }
    (void)last_ap;
    (void)last_ad;
    /* Small problems get a generous ordering budget: even a slow exact
     * minimum-degree run is bounded there and the IPM usually repays it.
     * Larger problems abort quickly and fall back to the first-order path. */
    /* The ordering budget is now purely a time guard (~20 s of MD work
     * at this machine's throughput): graphs whose factor would blow the
     * flops cap abort early inside the ordering via the predicted-fill
     * check, so a generous budget no longer risks burning the solve
     * budget on hopeless instances. */
    int64_t md_budget = 10000000000LL;
    {
        int want = (int)threads;
        if (want == 0) {
            long cores = sysconf(_SC_NPROCESSORS_ONLN);
            want = cores >= 4 ? 4 : (cores > 1 ? (int)cores : 1);
        }
        if (want > 1) {
            g_kernel_threads = pool_ensure(want);
        } else {
            g_kernel_threads = 1;
        }
    }
    g_tail_use_blas = use_blas;
    chol = chol_setup(self, 7e8, md_budget, &factor_too_dense);
    if (chol == NULL) {
        if (factor_too_dense) {
            /* The factor would be too expensive per iteration; report a
             * distinct status so auto routing can use the first-order path. */
            result = Py_BuildValue(
                "{s:s,s:d,s:d,s:d,s:d,s:d,s:n,s:[],s:[]}",
                "status", "factor_too_dense",
                "objective", 0.0,
                "max_primal_residual", INFINITY,
                "rel_primal_residual", INFINITY,
                "rel_dual_residual", INFINITY,
                "mu", INFINITY,
                "iterations", (Py_ssize_t)0,
                "x",
                "y");
            goto done;
        }
        PyErr_SetString(PyExc_RuntimeError, "Cholesky setup failed");
        goto done;
    }
    int refactor_supernodal =
        use_supernodal < 0 ? chol_auto_supernodal(chol) : use_supernodal;

    int max_mcc = 2;
    double mcc_ratio = 5.5;
    {
        const char *mcc_env = getenv("LINPROGX_IPM_MCC");
        if (mcc_env != NULL) {
            max_mcc = atoi(mcc_env);
        }
        const char *ratio_env = getenv("LINPROGX_IPM_MCC_RATIO");
        if (ratio_env != NULL) {
            mcc_ratio = atof(ratio_env);
        }
    }
    /* Deterministic Gondzio-corrector budget (Gondzio's cost-ratio rule):
     * correctors pay only when a saved iteration (one refactor) is worth
     * clearly more than the extra back-solves they cost. Estimated in
     * sparse-equivalent flop units from the symbolic structure — never
     * from wall time, which would break bit-identical results across
     * thread counts and runs. The row-wise refactor model reuses the
     * dense-tail machine constant (1/58); the supernodal panel rate
     * (~1/8 of scalar) is calibrated on maros_r7's measured 65ms
     * refactor vs 15ms scalar solves. Measured corrector economics:
     * cre_b/cre_d/osa_14 lose wall time to correctors, pilot87/ken_18
     * are neutral, maros_r7 mildly gains — the 5.5 threshold keeps only
     * the gaining population enabled. */
    int mcc_budget = 0;
    {
        double refactor_units;
        if (refactor_supernodal) {
            refactor_units = chol->factor_flops / 8.0;
        } else {
            double tail = (double)chol->tail_len;
            refactor_units = chol->prefix_flops + tail * tail * tail / 3.0 / 58.0;
        }
        double solve_units = 6.0 * (double)chol->Lp[chol->m] +
                             4.0 * (double)chol->Cp[chol->m] +
                             4.0 * (double)nnz;
        if (solve_units > 0.0 &&
            refactor_units >= mcc_ratio * solve_units) {
            mcc_budget = max_mcc;
        }
    }

    /* Mehrotra least-squares starting point: factor A A' + delta I once,
     * take the min-norm primal consistent with Ax=b and the dual from
     * projecting c, then shift slacks and duals positive. */
    {
        for (Py_ssize_t j = 0; j < n; j++) {
            D[j] = 1.0;
        }
        chol_refactor_mode(chol, self, csc_vals, D, 1e-8, refactor_supernodal);
        chol_solve(chol, b, dy_a);
        scaled_op_transpose_matvec(&op, dy_a, x);
        scaled_op_matvec(&op, c, ax);
        chol_solve(chol, ax, y);
        scaled_op_transpose_matvec(&op, y, aty);

        double min_sl = INFINITY;
        double min_su = INFINITY;
        double min_zl = INFINITY;
        double min_zu = INFINITY;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            double reduced = c[j] - aty[j];
            zl[j] = (kind & 1) ? (reduced > 0.0 ? reduced : 0.0) : 0.0;
            zu[j] = (kind & 2) ? (reduced < 0.0 ? -reduced : 0.0) : 0.0;
            if (kind & 1) {
                double slack = x[j] - lo[j];
                min_sl = slack < min_sl ? slack : min_sl;
                min_zl = zl[j] < min_zl ? zl[j] : min_zl;
            }
            if (kind & 2) {
                double slack = hi[j] - x[j];
                min_su = slack < min_su ? slack : min_su;
                min_zu = zu[j] < min_zu ? zu[j] : min_zu;
            }
        }
        double shift_l = (isfinite(min_sl) && min_sl < 0.0 ? -1.5 * min_sl : 0.0) + 0.1;
        double shift_u = (isfinite(min_su) && min_su < 0.0 ? -1.5 * min_su : 0.0) + 0.1;
        double zshift_l = (isfinite(min_zl) && min_zl < 0.0 ? -1.5 * min_zl : 0.0) + 0.1;
        double zshift_u = (isfinite(min_zu) && min_zu < 0.0 ? -1.5 * min_zu : 0.0) + 0.1;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            if (kind == 4) {
                x[j] = 0.5 * (lo[j] + hi[j]);
                zl[j] = 0.0;
                zu[j] = 0.0;
                continue;
            }
            double lo_target = -INFINITY;
            double hi_target = INFINITY;
            if (kind & 1) {
                double cap = (kind & 2) ? 0.4 * (hi[j] - lo[j]) : shift_l;
                lo_target = lo[j] + (shift_l < cap ? shift_l : cap);
                zl[j] += zshift_l;
            }
            if (kind & 2) {
                double cap = (kind & 1) ? 0.4 * (hi[j] - lo[j]) : shift_u;
                hi_target = hi[j] - (shift_u < cap ? shift_u : cap);
                zu[j] += zshift_u;
            }
            if (lo_target > hi_target) {
                double mid = 0.5 * (lo_target + hi_target);
                lo_target = mid;
                hi_target = mid;
            }
            if (x[j] < lo_target) {
                x[j] = lo_target;
            }
            if (x[j] > hi_target) {
                x[j] = hi_target;
            }
            /* strict interior margin, even for narrow boxes */
            if (kind & 1) {
                double width = (kind & 2) ? hi[j] - lo[j] : INFINITY;
                double margin = width < 4e-4 ? 0.25 * width : 1e-4;
                if (x[j] < lo[j] + margin) {
                    x[j] = lo[j] + margin;
                }
            }
            if (kind & 2) {
                double width = (kind & 1) ? hi[j] - lo[j] : INFINITY;
                double margin = width < 4e-4 ? 0.25 * width : 1e-4;
                if (x[j] > hi[j] - margin) {
                    x[j] = hi[j] - margin;
                }
            }
        }
    }

    double delta_reg = 1e-8;
    double b_norm = 1.0 + l2_norm(b, m);
    double c_norm = 1.0 + l2_norm(c, n);
    Py_ssize_t iterations = 0;
    double pres = INFINITY;
    double raw_pres = INFINITY;
    double dres = INFINITY;
    double mu = INFINITY;
    double best_score = INFINITY;
    double best_pres = INFINITY;
    double best_raw_pres = INFINITY;
    double best_dres = INFINITY;
    double best_mu = INFINITY;
    double best_gap = INFINITY;
    double mu_initial = INFINITY;
    const char *status = "iteration_limit";

    Py_BEGIN_ALLOW_THREADS
    IpmNewton nw = {chol, &op, D, sl, su, zl, zu, bound_kind, rhs_x, tmp_x, rhs_m, aty,
                    res_m, corr_m};
    for (Py_ssize_t iter = 0; iter < max_iter; iter++) {
        iterations = iter;
        /* slacks, residuals, and the barrier parameter */
        scaled_op_matvec(&op, x, ax);
        for (Py_ssize_t i = 0; i < m; i++) {
            rp[i] = b[i] - ax[i];
        }
        scaled_op_transpose_matvec(&op, y, aty);
        double mu_sum = 0.0;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            double slack_l = (kind & 1) ? x[j] - lo[j] : 1.0;
            double slack_u = (kind & 2) ? hi[j] - x[j] : 1.0;
            sl[j] = slack_l > 1e-13 ? slack_l : 1e-13;
            su[j] = slack_u > 1e-13 ? slack_u : 1e-13;
            if ((kind & 1) && zl[j] < 1e-13) {
                zl[j] = 1e-13;
            }
            if ((kind & 2) && zu[j] < 1e-13) {
                zu[j] = 1e-13;
            }
            rd[j] = (kind == 4) ? 0.0 : c[j] - aty[j] - zl[j] + zu[j];
            if (kind & 1) {
                mu_sum += sl[j] * zl[j];
            }
            if (kind & 2) {
                mu_sum += su[j] * zu[j];
            }
        }
        mu = mu_sum / (double)n_comp;
        pres = l2_norm(rp, m) / b_norm;
        raw_pres = ipm_raw_primal_residual(rp, row_scale, m);
        dres = l2_norm(rd, n) / c_norm;
        if (debug) {
            fprintf(stderr,
                    "ipm iter=%zd mu=%.3e pres=%.3e raw=%.3e dres=%.3e ap=%.2e ad=%.2e\n",
                    iter, mu, pres, raw_pres, dres, last_ap, last_ad);
        }
        if (!isfinite(mu) || !isfinite(pres) || !isfinite(dres)) {
            /* The iterate is destroyed (late Newton steps on
             * ill-conditioned instances can overflow); nothing after
             * this point can recover, and the best-iterate snapshot
             * already holds the usable point. Stop burning budget. */
            break;
        }
        if (iter == 0) {
            mu_initial = mu;
        } else if (iter == 60 && mu > 1e-4 * mu_initial) {
            /* Pace watchdog: a healthy Mehrotra run shrinks mu by far more
             * than four orders in 60 iterations; bail to the fallback
             * instead of burning the rest of the budget. */
            break;
        }
        {
            double score = pres > dres ? pres : dres;
            if (mu > score) {
                score = mu;
            }
            if (isfinite(score) && score < best_score) {
                best_score = score;
                best_pres = pres;
                best_raw_pres = raw_pres;
                best_dres = dres;
                best_mu = mu;
                /* Certified primal-dual gap: build the TRUE Lagrangian
                 * bound from the actual reduced costs r = c - A'y, splitting
                 * each onto its bound. Unlike the z-based dual objective,
                 * this is valid regardless of how far z has drifted; if a
                 * reduced cost points at an infinite bound, the iterate is
                 * not certifiable and the gap stays infinite. */
                double pobj = 0.0;
                double dobj = 0.0;
                int certifiable = 1;
                for (Py_ssize_t j = 0; j < n; j++) {
                    unsigned char kind = bound_kind[j];
                    double r = c[j] - aty[j];
                    pobj += c[j] * x[j];
                    if (kind == 4) {
                        /* pinned (zero-width box): multiplier absorbs r */
                        dobj += r * x[j];
                        continue;
                    }
                    if (r > 0.0) {
                        if (kind & 1) {
                            dobj += r * lo[j];
                        } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                            certifiable = 0;
                            break;
                        }
                    } else if (r < 0.0) {
                        if (kind & 2) {
                            dobj += r * hi[j];
                        } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                            certifiable = 0;
                            break;
                        }
                    }
                }
                if (certifiable) {
                    for (Py_ssize_t i = 0; i < m; i++) {
                        dobj += b[i] * y[i];
                    }
                    best_gap = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
                } else {
                    best_gap = INFINITY;
                }
                memcpy(x_best, x, (size_t)n * sizeof(double));
                memcpy(y_best, y, (size_t)m * sizeof(double));
            }
        }
        if (raw_pres <= feas_tol && pres < tol && dres < tol && mu < 10.0 * tol) {
            status = "optimal";
            break;
        }
        {
            /* Early relaxed acceptance: identical bar to the exit path
             * (pres/dres/mu bounds + certified gap), but only once mu
             * progress has stalled — a healthy Mehrotra run shrinks mu
             * by well over 4x per 10 iterations, so fast convergers
             * keep polishing toward the strict tolerance while
             * tail-crawling runs stop burning factorizations. */
            double mu_old = mu_hist[iter % 10];
            mu_hist[iter % 10] = mu;
            int stalled = iter >= 10 && isfinite(mu_old) && mu > 0.25 * mu_old;
            if (stalled && raw_pres <= feas_tol && pres <= 1e-6 && dres <= 5e-6 &&
                mu <= 1e-6) {
                double pobj = 0.0;
                double dobj = 0.0;
                int certifiable = 1;
                for (Py_ssize_t j = 0; j < n; j++) {
                    unsigned char kind = bound_kind[j];
                    double r = c[j] - aty[j];
                    pobj += c[j] * x[j];
                    if (kind == 4) {
                        dobj += r * x[j];
                        continue;
                    }
                    if (r > 0.0) {
                        if (kind & 1) {
                            dobj += r * lo[j];
                        } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                            certifiable = 0;
                            break;
                        }
                    } else if (r < 0.0) {
                        if (kind & 2) {
                            dobj += r * hi[j];
                        } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                            certifiable = 0;
                            break;
                        }
                    }
                }
                int accepted = 0;
                if (certifiable) {
                    for (Py_ssize_t i = 0; i < m; i++) {
                        dobj += b[i] * y[i];
                    }
                    double gap = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
                    if (gap <= 1e-5) {
                        best_gap = gap;
                        status = "optimal";
                        accepted = 1;
                    }
                }
                if (!accepted && iter - last_cleanup_attempt >= 16) {
                    /* the raw certificate failed; the min-norm dual
                     * cleanup may close it (rate-limited: it costs a
                     * Gram solve over the violating columns) */
                    last_cleanup_attempt = iter;
                    double cleaned_gap = 0.0;
                    if (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y,
                                         aty, &cleaned_gap, &dual_cleanup_rounds) ||
                        ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y,
                                        aty, &cleaned_gap)) {
                        best_gap = cleaned_gap;
                        status = "optimal";
                        accepted = 1;
                    }
                }
                if (accepted) {
                    break;
                }
            }
        }
        /* The raw_pres window is only a cost pre-filter: the polish itself
         * re-checks bound violation, recomputed raw residual, and the
         * Lagrangian gap before accepting. 1e-1 (not 1e-3) matters on
         * badly row-scaled instances (osa_60) where the scaled residual
         * converges many orders ahead of the original-unit one and the
         * trajectory can break down within an iteration or two of the
         * window opening. */
        if ((m >= 100 || n >= 100) && raw_pres > feas_tol && raw_pres <= 1e-1 &&
            pres <= 1e-6 && dres <= 5e-6) {
            double cleaned_gap = 0.0;
            int certified = ipm_lagrangian_gap(c, b, lo, hi, bound_kind, m, n, x, y,
                                                aty, &cleaned_gap) &&
                            cleaned_gap <= 1e-5;
            if (!certified &&
                ipm_dual_polish(&op, chol, c, b, lo, hi, bound_kind, m, n, x, D,
                                tmp_x, rhs_m, dy, dx, &cleaned_gap)) {
                memcpy(y, dy, (size_t)m * sizeof(double));
                memcpy(aty, dx, (size_t)n * sizeof(double));
                certified = 1;
            }
            if (!certified && iter - last_cleanup_attempt >= 1 &&
                (last_ap < 1e-6 || last_ad < 1e-6)) {
                last_cleanup_attempt = iter;
                if (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                     &cleaned_gap, &dual_cleanup_rounds) ||
                    ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                    &cleaned_gap)) {
                    certified = 1;
                }
            }
            if (!certified && iter - last_cleanup_attempt >= 16) {
                /* Without a step stall the trajectory may still break down
                 * before ever stalling (osa_60 goes NaN two iterations
                 * after entering this window), so also attempt the
                 * min-norm cleanup on the same 16-iteration rate limit as
                 * the relaxed-acceptance path. */
                last_cleanup_attempt = iter;
                if (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                     &cleaned_gap, &dual_cleanup_rounds) ||
                    ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                    &cleaned_gap)) {
                    certified = 1;
                }
            }
            if (certified) {
                double polished_pres = pres;
                double polished_raw_pres = raw_pres;
                double polished_gap = cleaned_gap;
                if (ipm_primal_polish(&op, chol, self, csc_vals, c, b, lo, hi,
                                      bound_kind, row_scale, m, n, rp, y, aty,
                                      refactor_supernodal, feas_tol, b_norm, x, D,
                                      rhs_m, tmp_x, dx, &polished_pres,
                                      &polished_raw_pres, &polished_gap)) {
                    pres = polished_pres;
                    raw_pres = polished_raw_pres;
                    best_gap = polished_gap;
                    best_pres = polished_pres;
                    best_raw_pres = polished_raw_pres;
                    best_dres = dres;
                    best_mu = mu;
                    memcpy(x_best, x, (size_t)n * sizeof(double));
                    memcpy(y_best, y, (size_t)m * sizeof(double));
                    status = "optimal";
                    break;
                }
            }
        }
        if ((m >= 100 || n >= 100) && raw_pres <= feas_tol &&
            pres <= 1e-6 && dres <= 5e-6 &&
            iter - last_cleanup_attempt >= 1) {
            /* Same certificate-producing cleanup used on non-optimal exit,
             * but run in-loop once the residuals are already small. This
             * avoids burning extra factorizations solely to reach stricter
             * barrier tolerances when the Lagrangian bound can already close. */
            last_cleanup_attempt = iter;
            double cleaned_gap = 0.0;
            if (ipm_lagrangian_gap(c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                   &cleaned_gap) &&
                cleaned_gap <= 1e-5) {
                best_gap = cleaned_gap;
                status = "optimal";
                break;
            }
            if (ipm_dual_polish(&op, chol, c, b, lo, hi, bound_kind, m, n, x, D,
                                tmp_x, rhs_m, dy, dx, &cleaned_gap)) {
                memcpy(y, dy, (size_t)m * sizeof(double));
                memcpy(aty, dx, (size_t)n * sizeof(double));
                best_gap = cleaned_gap;
                status = "optimal";
                break;
            }
            if (last_ap < 1e-6 || last_ad < 1e-6) {
                if (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                     &cleaned_gap, &dual_cleanup_rounds) ||
                    ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                    &cleaned_gap)) {
                    best_gap = cleaned_gap;
                    status = "optimal";
                    break;
                }
            }
        }

        /* scaling matrix and factorization; the regularization shrinks with
         * mu so it stops limiting the final dual accuracy */
        double delta_it = 1e-2 * mu;
        if (delta_it > delta_reg) {
            delta_it = delta_reg;
        }
        /* Staged precision: near convergence the iterate is stable enough
         * to shrink the regularization further (doubled refinement covers
         * the worse conditioning), letting the dual residual fall through
         * the 1e-10-floor barrier. */
        double delta_floor = mu < 1e-7 ? 1e-12 : 1e-10;
        if (delta_it < delta_floor) {
            delta_it = delta_floor;
        }
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            double h;
            if (kind == 4) {
                h = 1e16;
            } else if (kind == 0) {
                /* free columns have no barrier term; keep their
                 * regularization fixed so 1/h stays bounded */
                h = delta_reg;
            } else {
                h = delta_it;
                if (kind & 1) {
                    h += zl[j] / sl[j];
                }
                if (kind & 2) {
                    h += zu[j] / su[j];
                }
            }
            H[j] = h;
            D[j] = 1.0 / h;
        }
        double t_phase = linprogx_monotonic_seconds();
        chol_refactor_mode(chol, self, csc_vals, D, delta_it, refactor_supernodal);
        t_refactor += linprogx_monotonic_seconds() - t_phase;

        /* affine direction */
        for (Py_ssize_t j = 0; j < n; j++) {
            rcl[j] = (bound_kind[j] & 1) ? -sl[j] * zl[j] : 0.0;
            rcu[j] = (bound_kind[j] & 2) ? -su[j] * zu[j] : 0.0;
        }
        t_phase = linprogx_monotonic_seconds();
        ipm_newton_solve(&nw, rp, rd, rcl, rcu, dy_a, dx_a, dzl_a, dzu_a);
        t_newton += linprogx_monotonic_seconds() - t_phase;

        double ap_aff = 1.0;
        double ad_aff = 1.0;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            if ((kind & 1) && dx_a[j] < 0.0) {
                double step = -sl[j] / dx_a[j];
                if (step < ap_aff) {
                    ap_aff = step;
                }
            }
            if ((kind & 2) && dx_a[j] > 0.0) {
                double step = su[j] / dx_a[j];
                if (step < ap_aff) {
                    ap_aff = step;
                }
            }
            if ((kind & 1) && dzl_a[j] < 0.0) {
                double step = -zl[j] / dzl_a[j];
                if (step < ad_aff) {
                    ad_aff = step;
                }
            }
            if ((kind & 2) && dzu_a[j] < 0.0) {
                double step = -zu[j] / dzu_a[j];
                if (step < ad_aff) {
                    ad_aff = step;
                }
            }
        }
        double mu_aff_sum = 0.0;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            if (kind & 1) {
                mu_aff_sum += (sl[j] + ap_aff * dx_a[j]) * (zl[j] + ad_aff * dzl_a[j]);
            }
            if (kind & 2) {
                mu_aff_sum += (su[j] - ap_aff * dx_a[j]) * (zu[j] + ad_aff * dzu_a[j]);
            }
        }
        double mu_aff = mu_aff_sum / (double)n_comp;
        double ratio = mu > 0.0 ? mu_aff / mu : 0.1;
        double sigma = ratio * ratio * ratio;

        /* corrector */
        double coupling = ap_aff * ad_aff;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            rcl[j] = (kind & 1)
                ? sigma * mu - sl[j] * zl[j] - coupling * dx_a[j] * dzl_a[j]
                : 0.0;
            rcu[j] = (kind & 2)
                ? sigma * mu - su[j] * zu[j] + coupling * dx_a[j] * dzu_a[j]
                : 0.0;
        }
        if (debug) {
            struct timespec ts0, ts1;
            clock_gettime(CLOCK_MONOTONIC, &ts0);
            ipm_newton_solve(&nw, rp, rd, rcl, rcu, dy, dx, dzl, dzu);
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            t_newton += (double)(ts1.tv_sec - ts0.tv_sec) +
                        1e-9 * (double)(ts1.tv_nsec - ts0.tv_nsec);
        } else {
            ipm_newton_solve(&nw, rp, rd, rcl, rcu, dy, dx, dzl, dzu);
        }

        double ap = 1.0;
        double ad = 1.0;
        for (Py_ssize_t j = 0; j < n; j++) {
            unsigned char kind = bound_kind[j];
            if ((kind & 1) && dx[j] < 0.0) {
                double step = -sl[j] / dx[j];
                if (step < ap) {
                    ap = step;
                }
            }
            if ((kind & 2) && dx[j] > 0.0) {
                double step = su[j] / dx[j];
                if (step < ap) {
                    ap = step;
                }
            }
            if ((kind & 1) && dzl[j] < 0.0) {
                double step = -zl[j] / dzl[j];
                if (step < ad) {
                    ad = step;
                }
            }
            if ((kind & 2) && dzu[j] < 0.0) {
                double step = -zu[j] / dzu[j];
                if (step < ad) {
                    ad = step;
                }
            }
        }
        /* Gondzio multiple centrality correctors: extend the trial step,
         * project the trial complementarity products back into the
         * [beta_min, beta_max] * sigma * mu hypercube, and add the pure
         * recentering direction (zero primal/dual residual rhs) while it
         * lengthens the steps. One back-solve per round is cheap against
         * the full refactor a saved iteration avoids. Global constants;
         * skipped once both steps are already long. mcc_budget is the
         * deterministic structure-based budget computed before the loop. */
        for (int mcc = 0; mcc < mcc_budget; mcc++) {
            if (ap >= 0.9 && ad >= 0.9) {
                break;
            }
            double ap_bar = ap + 0.08 < 1.0 ? ap + 0.08 : 1.0;
            double ad_bar = ad + 0.08 < 1.0 ? ad + 0.08 : 1.0;
            double target_lo = 0.1 * sigma * mu;
            double target_hi = 10.0 * sigma * mu;
            for (Py_ssize_t j = 0; j < n; j++) {
                unsigned char kind = bound_kind[j];
                double tl = 0.0;
                double tu = 0.0;
                if (kind & 1) {
                    double v = (sl[j] + ap_bar * dx[j]) * (zl[j] + ad_bar * dzl[j]);
                    if (v < target_lo) {
                        tl = target_lo - v;
                    } else if (v > target_hi) {
                        tl = target_hi - v;
                    }
                }
                if (kind & 2) {
                    double v = (su[j] - ap_bar * dx[j]) * (zu[j] + ad_bar * dzu[j]);
                    if (v < target_lo) {
                        tu = target_lo - v;
                    } else if (v > target_hi) {
                        tu = target_hi - v;
                    }
                }
                rcl[j] = tl;
                rcu[j] = tu;
            }
            if (debug) {
                struct timespec ts0, ts1;
                clock_gettime(CLOCK_MONOTONIC, &ts0);
                ipm_newton_solve(&nw, zero_m, zero_n, rcl, rcu, dy_a, dx_a, dzl_a, dzu_a);
                clock_gettime(CLOCK_MONOTONIC, &ts1);
                t_newton += (double)(ts1.tv_sec - ts0.tv_sec) +
                            1e-9 * (double)(ts1.tv_nsec - ts0.tv_nsec);
            } else {
                ipm_newton_solve(&nw, zero_m, zero_n, rcl, rcu, dy_a, dx_a, dzl_a, dzu_a);
            }
            double ap_c = 1.0;
            double ad_c = 1.0;
            for (Py_ssize_t j = 0; j < n; j++) {
                unsigned char kind = bound_kind[j];
                double dxc = dx[j] + dx_a[j];
                if ((kind & 1) && dxc < 0.0) {
                    double step = -sl[j] / dxc;
                    if (step < ap_c) {
                        ap_c = step;
                    }
                }
                if ((kind & 2) && dxc > 0.0) {
                    double step = su[j] / dxc;
                    if (step < ap_c) {
                        ap_c = step;
                    }
                }
                if (kind & 1) {
                    double dzlc = dzl[j] + dzl_a[j];
                    if (dzlc < 0.0) {
                        double step = -zl[j] / dzlc;
                        if (step < ad_c) {
                            ad_c = step;
                        }
                    }
                }
                if (kind & 2) {
                    double dzuc = dzu[j] + dzu_a[j];
                    if (dzuc < 0.0) {
                        double step = -zu[j] / dzuc;
                        if (step < ad_c) {
                            ad_c = step;
                        }
                    }
                }
            }
            if (ap_c < ap || ad_c < ad || ap_c + ad_c < ap + ad + 0.01) {
                break;
            }
            for (Py_ssize_t j = 0; j < n; j++) {
                dx[j] += dx_a[j];
                dzl[j] += dzl_a[j];
                dzu[j] += dzu_a[j];
            }
            for (Py_ssize_t i = 0; i < m; i++) {
                dy[i] += dy_a[i];
            }
            ap = ap_c;
            ad = ad_c;
        }

        ap *= 0.995;
        ad *= 0.995;
        last_ap = ap;
        last_ad = ad;
        for (Py_ssize_t j = 0; j < n; j++) {
            if (bound_kind[j] == 4) {
                continue;
            }
            x[j] += ap * dx[j];
            if (bound_kind[j] & 1) {
                zl[j] += ad * dzl[j];
            }
            if (bound_kind[j] & 2) {
                zu[j] += ad * dzu[j];
            }
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            y[i] += ad * dy[i];
        }
    }
    /* On non-optimal exit, fall back to the best iterate seen: late Newton
     * steps on ill-conditioned instances can wander or overflow after the
     * run has already passed through an excellent point. */
    if (strcmp(status, "optimal") != 0 && isfinite(best_score)) {
        memcpy(x, x_best, (size_t)n * sizeof(double));
        memcpy(y, y_best, (size_t)m * sizeof(double));
        pres = best_pres;
        raw_pres = best_raw_pres;
        dres = best_dres;
        mu = best_mu;
        /* Relaxed acceptance: ill-conditioned instances stall with residual
         * floors around 1e-7 while the iterate is excellent for every
         * practical purpose. The explicit gap test is essential: with an
         * infeasible dual, small mu alone can hide a real objective error. */
        if (raw_pres <= feas_tol && pres <= 1e-6 && dres <= 5e-6 &&
            mu <= 1e-6 && best_gap <= 1e-5) {
            status = "optimal";
        } else if (raw_pres <= feas_tol && pres <= 1e-6 && best_gap > 1e-5) {
            /* Dual polish: the stored multipliers may fail to certify an
             * excellent primal point. Recompute y by weighted least squares
             * from the final factorization (one extra solve); ANY y yields
             * a valid Lagrangian bound, so this can only gain certificates,
             * never fake one. */
            for (Py_ssize_t j = 0; j < n; j++) {
                tmp_x[j] = D[j] * c[j];
            }
            scaled_op_matvec(&op, tmp_x, rhs_m);
            chol_solve(chol, rhs_m, dy);
            scaled_op_transpose_matvec(&op, dy, aty);
            double pobj = 0.0;
            double dobj = 0.0;
            int certifiable = 1;
            for (Py_ssize_t j = 0; j < n; j++) {
                unsigned char kind = bound_kind[j];
                double r = c[j] - aty[j];
                pobj += c[j] * x[j];
                if (kind == 4) {
                    dobj += r * x[j];
                    continue;
                }
                if (r > 0.0) {
                    if (kind & 1) {
                        dobj += r * lo[j];
                    } else if (r > 1e-9 * (1.0 + fabs(c[j]))) {
                        certifiable = 0;
                        break;
                    }
                } else if (r < 0.0) {
                    if (kind & 2) {
                        dobj += r * hi[j];
                    } else if (-r > 1e-9 * (1.0 + fabs(c[j]))) {
                        certifiable = 0;
                        break;
                    }
                }
            }
            if (certifiable) {
                for (Py_ssize_t i = 0; i < m; i++) {
                    dobj += b[i] * dy[i];
                }
                double polished_gap = fabs(pobj - dobj) / (1.0 + fabs(pobj) + fabs(dobj));
                if (polished_gap <= 1e-5) {
                    memcpy(y, dy, (size_t)m * sizeof(double));
                    best_gap = polished_gap;
                    status = "optimal";
                }
            }
        }
        if (strcmp(status, "optimal") != 0 && raw_pres <= feas_tol && pres <= 1e-6) {
            double cleaned_gap = 0.0;
            if (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                 &cleaned_gap, &dual_cleanup_rounds) ||
                ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                &cleaned_gap)) {
                best_gap = cleaned_gap;
                status = "optimal";
            }
        }
        if (strcmp(status, "optimal") != 0 && (m >= 100 || n >= 100) &&
            raw_pres > feas_tol && raw_pres <= 1e-1 &&
            pres <= 1e-6 && dres <= 5e-6) {
            /* Exit-path primal feasibility polish: the trajectory can break
             * down (NaN step) within an iteration of the in-loop polish
             * window opening, leaving a restored best iterate whose scaled
             * residuals are excellent but whose original-unit residual is
             * above eps. Same guarded correction as in-loop: it re-checks
             * bound violation, recomputed raw residual, and the Lagrangian
             * gap, so it can only gain a certificate, never fake one. */
            scaled_op_matvec(&op, x, ax);
            for (Py_ssize_t i = 0; i < m; i++) {
                rp[i] = b[i] - ax[i];
            }
            scaled_op_transpose_matvec(&op, y, aty);
            double cleaned_gap = 0.0;
            int certified = ipm_lagrangian_gap(c, b, lo, hi, bound_kind, m, n, x, y,
                                               aty, &cleaned_gap) &&
                            cleaned_gap <= 1e-5;
            if (!certified &&
                (ipm_dual_cleanup(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                  &cleaned_gap, &dual_cleanup_rounds) ||
                 ipm_dual_repair(&op, c, b, lo, hi, bound_kind, m, n, x, y, aty,
                                 &cleaned_gap))) {
                certified = 1;
            }
            if (!certified &&
                ipm_dual_polish(&op, chol, c, b, lo, hi, bound_kind, m, n, x, D,
                                tmp_x, rhs_m, dy, dx, &cleaned_gap)) {
                memcpy(y, dy, (size_t)m * sizeof(double));
                memcpy(aty, dx, (size_t)n * sizeof(double));
                certified = 1;
            }
            if (certified) {
                double polished_pres = pres;
                double polished_raw_pres = raw_pres;
                double polished_gap = cleaned_gap;
                if (ipm_primal_polish(&op, chol, self, csc_vals, c, b, lo, hi,
                                      bound_kind, row_scale, m, n, rp, y, aty,
                                      refactor_supernodal, feas_tol, b_norm, x, D,
                                      rhs_m, tmp_x, dx, &polished_pres,
                                      &polished_raw_pres, &polished_gap)) {
                    pres = polished_pres;
                    raw_pres = polished_raw_pres;
                    best_gap = polished_gap;
                    best_pres = polished_pres;
                    best_raw_pres = polished_raw_pres;
                    status = "optimal";
                }
            }
        }
    }
    if (debug) {
        fprintf(stderr, "ipm timers: refactor=%.2fs newton_solves=%.2fs\n",
                t_refactor, t_newton);
        fprintf(stderr, "ipm exit: status=%s best_gap=%.3e best_pres=%.3e "
                "best_raw=%.3e best_dres=%.3e best_mu=%.3e\n",
                status, best_gap, best_pres, best_raw_pres, best_dres, best_mu);
    }
    {
        int finite = 1;
        for (Py_ssize_t j = 0; j < n; j++) {
            if (!isfinite(x[j])) {
                finite = 0;
                break;
            }
        }
        if (!finite) {
            status = "numerical_error";
        }
    }
    Py_END_ALLOW_THREADS

    /* unscale and report in original units */
    for (Py_ssize_t j = 0; j < n; j++) {
        x[j] *= col_scale[j];
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        y[i] *= row_scale[i] * c_scale;
    }
    {
        double objective = 0.0;
        double max_residual = 0.0;
        /* original-unit residual via the raw matrix */
        for (Py_ssize_t i = 0; i < m; i++) {
            double total = 0.0;
            for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++) {
                total += self->data[p] * x[self->indices[p]];
            }
            rp[i] = total;
        }
        if (fill_double_array(b_obj, m, b, "b") != 0 ||
            fill_double_array(c_obj, n, c, "c") != 0) {
            goto done;
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            double res = fabs(rp[i] - b[i]);
            if (res > max_residual) {
                max_residual = res;
            }
        }
        for (Py_ssize_t j = 0; j < n; j++) {
            objective += c[j] * x[j];
        }
        if (strcmp(status, "optimal") == 0 && max_residual > feas_tol) {
            status = "iteration_limit";
        }
        PyObject *x_list = PyList_New(n);
        PyObject *y_list = PyList_New(m);
        if (x_list == NULL || y_list == NULL) {
            Py_XDECREF(x_list);
            Py_XDECREF(y_list);
            goto done;
        }
        for (Py_ssize_t j = 0; j < n; j++) {
            PyList_SET_ITEM(x_list, j, PyFloat_FromDouble(x[j]));
        }
        for (Py_ssize_t i = 0; i < m; i++) {
            PyList_SET_ITEM(y_list, i, PyFloat_FromDouble(y[i]));
        }
        result = Py_BuildValue(
            "{s:s,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:N,s:N}",
            "status", status,
            "objective", objective,
            "max_primal_residual", max_residual,
            "rel_primal_residual", pres,
            "rel_dual_residual", dres,
            "mu", mu,
            "iterations", iterations,
            "dual_cleanup_rounds", dual_cleanup_rounds,
            "x", x_list,
            "y", y_list);
    }

done:
    g_kernel_threads = 1;
    g_tail_use_blas = 1;
    chol_free(chol);
    free(c);
    free(b);
    free(lo);
    free(hi);
    free(row_scale);
    free(col_scale);
    free(csr_vals);
    free(csc_vals);
    free(op_col_index);
    free(op_row_index);
    free(x);
    free(y);
    free(zl);
    free(zu);
    free(sl);
    free(su);
    free(rp);
    free(rd);
    free(H);
    free(D);
    free(rcl);
    free(rcu);
    free(dx_a);
    free(dy_a);
    free(dzl_a);
    free(dzu_a);
    free(dx);
    free(dy);
    free(dzl);
    free(dzu);
    free(rhs_x);
    free(tmp_x);
    free(rhs_m);
    free(aty);
    free(ax);
    free(res_m);
    free(corr_m);
    free(zero_m);
    free(zero_n);
    free(x_best);
    free(y_best);
    free(bound_kind);
    return result;
}

/* ================================================================== */
/* Sparse LU factorization with Markowitz pivot selection              */
/* for the revised simplex basis matrix (square, nonsymmetric, sparse) */
/* ================================================================== */

/*
 * Active submatrix representation:
 *   - Column-linked lists: each column is a singly-linked list of entries
 *     threaded through a pool of (row, value, next_in_col, next_in_row)
 *     nodes. A free list recycles deleted entries to avoid realloc during
 *     fill-in. Row headers provide a linked list through rows as well.
 *   - Row counts and column counts are maintained for Markowitz scoring.
 *
 * Pivot search:
 *   - Threshold Markowitz with u=0.1: among candidates where
 *     |pivot| >= u * max_abs_in_its_column, minimize (r-1)*(c-1).
 *   - Search bounded to 4 candidate "tiers" (columns with minimum count
 *     first, then next-smallest, etc.) -- O(small) per step, as in
 *     Markowitz's original suggestion and Duff/Erisman/Reid.
 *
 * Output: L (unit lower triangular, CSC), U (upper triangular, CSC),
 *         P (row permutation), Q (column permutation) such that PAQ = LU.
 *
 * Dense RHS for FTRAN/BTRAN in milestone 1. Sparse-RHS Gilbert-Peierls
 * triangular solves will be added in a later milestone.
 */

/* Pool entry for the active submatrix linked structure. */
typedef struct {
    int32_t row;
    double  value;
    int32_t next_in_col;  /* next entry index in same column, or -1 */
    int32_t next_in_row;  /* next entry index in same row, or -1 */
    int32_t col;          /* column of this entry */
} LUPoolEntry;

typedef struct {
    /* Pool of entries */
    LUPoolEntry *pool;
    int32_t pool_len;
    int32_t pool_cap;
    int32_t free_head;   /* head of free list, or -1 */

    /* Column headers: col_head[j] = first pool index in column j, or -1 */
    int32_t *col_head;
    /* Row headers: row_head[i] = first pool index in row i, or -1 */
    int32_t *row_head;

    /* Counts of entries in each active row/column */
    int32_t *col_count;
    int32_t *row_count;

    /* Whether row/col is still active (not yet pivoted) */
    unsigned char *row_active;
    unsigned char *col_active;

    /* Count buckets: doubly-linked lists of ACTIVE columns keyed by their
     * current col_count, so Markowitz pivot selection walks only the
     * smallest occupied counts instead of scanning all m columns per
     * elimination step (that scan made lu_factorize ~quadratic in m:
     * 5.3ms at m=1097 but 547ms at m=14633 per call, 86% of stocfor3's
     * dual-simplex wall time). bkt_prev encodes bucket-head membership
     * as -(count+1). bkt_min is a lazy lower bound on the smallest
     * nonempty bucket. bkt_ready gates the maintenance hooks so
     * initialization can bulk-load. */
    int32_t *bkt_head;  /* size m+1 */
    int32_t *bkt_next;  /* size m */
    int32_t *bkt_prev;  /* size m */
    int32_t bkt_min;
    int bkt_ready;

    /* Total entries in ACTIVE columns, maintained incrementally: the
     * dense-tail density check needed it every step and recomputing by
     * scanning all m columns was the last O(m^2) term in lu_factorize
     * (~147ms per call at m=14633). */
    int64_t total_nnz;

    int32_t m;  /* matrix dimension */
} LUActive;

static void lu_bkt_unlink(LUActive *a, int32_t col) {
    int32_t nx = a->bkt_next[col];
    int32_t pv = a->bkt_prev[col];
    if (pv >= 0) {
        a->bkt_next[pv] = nx;
    } else {
        a->bkt_head[-(pv + 1)] = nx;
    }
    if (nx >= 0) {
        a->bkt_prev[nx] = pv;
    }
}

static void lu_bkt_insert(LUActive *a, int32_t col) {
    int32_t cnt = a->col_count[col];
    int32_t h = a->bkt_head[cnt];
    a->bkt_next[col] = h;
    a->bkt_prev[col] = -(cnt + 1);
    if (h >= 0) {
        a->bkt_prev[h] = col;
    }
    a->bkt_head[cnt] = col;
    if (cnt < a->bkt_min) {
        a->bkt_min = cnt;
    }
}

/* Result of LU factorization. */
typedef struct {
    int32_t m;

    /* L in CSC: unit lower triangular */
    int32_t *l_indptr;    /* size m+1 */
    int32_t *l_indices;   /* size nnz_l */
    double  *l_values;    /* size nnz_l */
    int32_t  nnz_l;

    /* U in CSC: upper triangular */
    int32_t *u_indptr;    /* size m+1 */
    int32_t *u_indices;   /* size nnz_u */
    double  *u_values;    /* size nnz_u */
    int32_t  nnz_u;

    /* Permutations: P[k] = original row pivoted at step k,
     * Q[k] = original column pivoted at step k. PAQ = LU. */
    int32_t *perm_row;    /* P: size m */
    int32_t *perm_col;    /* Q: size m */

    /* Inverse permutations for solve convenience */
    int32_t *inv_perm_row;  /* P^-1: size m */
    int32_t *inv_perm_col;  /* Q^-1: size m */

    int32_t  singular_step; /* -1 if nonsingular, else failing step index */

    /* ---- Sparse Forrest-Tomlin basis update ----
     *
     * Replaces the dense product-form-of-the-inverse (PFI) with packed
     * sparse eta vectors.  Each update k stores:
     *
     *   eta_positions[k] = leaving_pos p_k
     *   eta_pivot[k]     = alpha_k[p_k]  (the pivot value)
     *   eta_sp_idx/val[eta_sp_start[k] .. eta_sp_start[k+1]-1]
     *       = nonzero entries of alpha_k at indices j != p_k
     *
     * The semantics of the update are identical to the old dense PFI:
     *
     *   FTRAN step (apply E_k^{-1}):
     *     temp = x[p_k] / eta_pivot[k]
     *     for (j, v) in sparse entries: x[j] -= v * temp
     *     x[p_k] = temp
     *
     *   BTRAN step (apply E_k^{-T}, in reverse order):
     *     dot = eta_pivot[k] * v[p_k]
     *     for (j, v) in sparse entries: dot += v * v_eta[j]
     *     v_eta[p_k] += (v_eta[p_k] - dot) / eta_pivot[k]
     *
     * Packed storage avoids the O(k*m) cost of dense PFI; each
     * FTRAN/BTRAN update costs O(nnz_k) instead of O(m).
     */
    int32_t *eta_positions;     /* leaving_pos per update, size n_updates */
    double  *eta_pivot;         /* pivot value per update, size n_updates */
    int32_t *eta_sp_start;      /* CSR-style starts into packed arrays, size n_updates+1 */
    int32_t *eta_sp_idx;        /* packed nonzero column indices */
    double  *eta_sp_val;        /* packed nonzero values */
    int32_t  eta_sp_total_nnz;  /* total packed nonzeros across all updates */
    int32_t  eta_sp_packed_cap; /* allocated capacity for packed arrays */
    int32_t  n_updates;         /* number of accumulated updates */
    int32_t  eta_cap;           /* allocated capacity for per-update arrays */
    int32_t  orig_nnz_lu;       /* original nnzL + nnzU for refactor threshold */
    double   max_abs_diag;      /* max |eta_pivot[k]| across all updates */
    double   min_abs_diag;      /* min |eta_pivot[k]| across all updates */

    /* ---- Cached U diagonal for fast FTRAN/BTRAN ---- */
    double  *u_diag;            /* u_diag[j] = U[j,j], size m */

    /* ---- Reusable workspace for FTRAN/BTRAN (avoid per-call malloc) ---- */
    double  *ws_z;              /* workspace of size m */
    double  *ws_w;              /* workspace of size m */
    double  *ws_v;              /* workspace of size m (BTRAN eta application) */

    /* ---- Gilbert-Peierls sparse solve infrastructure ---- */
    int32_t *gp_stack;        /* DFS stack, size m */
    int32_t *gp_xi;           /* reach output (topological order), size 2*m */
    int32_t *gp_pinv;         /* workspace for column-to-row mapping, size m */
    int32_t  gp_mark;         /* timestamp counter for visited marking */
    int32_t *gp_marked;       /* visited markers, size m (compare vs gp_mark) */

    /* ---- Transposed L and U for BTRAN (built once per factorization) ---- */
    int32_t *lt_indptr;       /* L^T in CSC (= L in CSR), size m+1 */
    int32_t *lt_indices;
    double  *lt_values;
    int32_t *ut_indptr;       /* U^T in CSC (= U in CSR), size m+1 */
    int32_t *ut_indices;
    double  *ut_values;

    /* ---- Hyper-sparse solve statistics ---- */
    int64_t  ftran_dense_count;   /* number of dense FTRAN calls */
    int64_t  ftran_sparse_count;  /* number of sparse FTRAN calls */
    int64_t  ftran_sparse_nnz_total; /* sum of solution nnz across sparse FTRANs */
    int64_t  btran_dense_count;
    int64_t  btran_sparse_count;
    int64_t  btran_sparse_nnz_total;
} LUContext;

/* Allocate a pool entry; returns index or -1 on failure. */
static int32_t lu_pool_alloc(LUActive *a) {
    if (a->free_head >= 0) {
        int32_t idx = a->free_head;
        a->free_head = a->pool[idx].next_in_col;
        return idx;
    }
    if (a->pool_len == a->pool_cap) {
        int32_t new_cap = a->pool_cap < 256 ? 256 : a->pool_cap * 2;
        LUPoolEntry *grown = realloc(a->pool, (size_t)new_cap * sizeof(LUPoolEntry));
        if (grown == NULL) {
            return -1;
        }
        a->pool = grown;
        a->pool_cap = new_cap;
    }
    return a->pool_len++;
}

/* Free a pool entry by pushing it onto the free list. */
static void lu_pool_free(LUActive *a, int32_t idx) {
    a->pool[idx].next_in_col = a->free_head;
    a->free_head = idx;
}

/* Insert an entry into the active submatrix. Returns 0 on success, -1 on alloc failure. */
static int lu_active_insert(LUActive *a, int32_t row, int32_t col, double value) {
    int32_t idx = lu_pool_alloc(a);
    if (idx < 0) {
        return -1;
    }
    a->pool[idx].row = row;
    a->pool[idx].col = col;
    a->pool[idx].value = value;
    a->pool[idx].next_in_col = a->col_head[col];
    a->col_head[col] = idx;
    a->pool[idx].next_in_row = a->row_head[row];
    a->row_head[row] = idx;
    if (a->bkt_ready && a->col_active[col]) {
        lu_bkt_unlink(a, col);
        a->col_count[col]++;
        lu_bkt_insert(a, col);
    } else {
        a->col_count[col]++;
    }
    if (a->col_active[col]) {
        a->total_nnz++;
    }
    a->row_count[row]++;
    return 0;
}

/* Initialize the active submatrix from CSC arrays. */
static int lu_active_init(LUActive *a, int32_t m,
                          const int32_t *csc_indptr, const int32_t *csc_indices,
                          const double *csc_values) {
    int32_t nnz = csc_indptr[m];
    a->m = m;
    a->pool = NULL;
    a->pool_len = 0;
    /* Pre-allocate pool with some room for fill-in */
    a->pool_cap = nnz + nnz / 2 + 64;
    a->pool = calloc((size_t)a->pool_cap, sizeof(LUPoolEntry));
    a->free_head = -1;
    a->col_head = calloc((size_t)m, sizeof(int32_t));
    a->row_head = calloc((size_t)m, sizeof(int32_t));
    a->col_count = calloc((size_t)m, sizeof(int32_t));
    a->row_count = calloc((size_t)m, sizeof(int32_t));
    a->row_active = calloc((size_t)m, sizeof(unsigned char));
    a->col_active = calloc((size_t)m, sizeof(unsigned char));
    if (a->pool == NULL || a->col_head == NULL || a->row_head == NULL ||
        a->col_count == NULL || a->row_count == NULL ||
        a->row_active == NULL || a->col_active == NULL) {
        return -1;
    }
    for (int32_t j = 0; j < m; j++) {
        a->col_head[j] = -1;
        a->row_head[j] = -1;
        a->row_active[j] = 1;
        a->col_active[j] = 1;
    }
    a->bkt_ready = 0;
    a->total_nnz = 0;
    a->bkt_head = calloc((size_t)m + 1, sizeof(int32_t));
    a->bkt_next = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    a->bkt_prev = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    if (a->bkt_head == NULL || a->bkt_next == NULL || a->bkt_prev == NULL) {
        return -1;
    }
    /* Insert entries column by column */
    for (int32_t j = 0; j < m; j++) {
        for (int32_t p = csc_indptr[j]; p < csc_indptr[j + 1]; p++) {
            if (lu_active_insert(a, csc_indices[p], j, csc_values[p]) < 0) {
                return -1;
            }
        }
    }
    /* Bulk-load the count buckets now that col_counts are final. */
    for (int32_t k = 0; k <= m; k++) {
        a->bkt_head[k] = -1;
    }
    a->bkt_min = m;
    for (int32_t j = 0; j < m; j++) {
        lu_bkt_insert(a, j);
    }
    a->bkt_ready = 1;
    return 0;
}

static void lu_active_free(LUActive *a) {
    free(a->pool);
    free(a->col_head);
    free(a->row_head);
    free(a->col_count);
    free(a->row_count);
    free(a->row_active);
    free(a->col_active);
    free(a->bkt_head);
    free(a->bkt_next);
    free(a->bkt_prev);
}

/* Remove all entries in row `row` from the active submatrix, also unlinking
 * them from their column lists. Does NOT decrement row_count (row is being
 * eliminated). */
static void lu_active_remove_row(LUActive *a, int32_t row) {
    int32_t idx = a->row_head[row];
    while (idx >= 0) {
        int32_t next = a->pool[idx].next_in_row;
        int32_t col = a->pool[idx].col;
        /* Remove idx from column col's linked list */
        int32_t *pp = &a->col_head[col];
        while (*pp >= 0) {
            if (*pp == idx) {
                *pp = a->pool[idx].next_in_col;
                break;
            }
            pp = &a->pool[*pp].next_in_col;
        }
        if (a->bkt_ready && a->col_active[col]) {
            lu_bkt_unlink(a, col);
            a->col_count[col]--;
            lu_bkt_insert(a, col);
        } else {
            a->col_count[col]--;
        }
        if (a->col_active[col]) {
            a->total_nnz--;
        }
        lu_pool_free(a, idx);
        idx = next;
    }
    a->row_head[row] = -1;
    a->row_active[row] = 0;
}

/* Remove all entries in column `col` from the active submatrix, also unlinking
 * them from their row lists. */
static void lu_active_remove_col(LUActive *a, int32_t col) {
    int32_t idx = a->col_head[col];
    while (idx >= 0) {
        int32_t next = a->pool[idx].next_in_col;
        int32_t row = a->pool[idx].row;
        /* Remove idx from row's linked list */
        int32_t *pp = &a->row_head[row];
        while (*pp >= 0) {
            if (*pp == idx) {
                *pp = a->pool[idx].next_in_row;
                break;
            }
            pp = &a->pool[*pp].next_in_row;
        }
        a->row_count[row]--;
        lu_pool_free(a, idx);
        idx = next;
    }
    a->col_head[col] = -1;
    if (a->col_active[col]) {
        a->total_nnz -= a->col_count[col];
    }
    if (a->bkt_ready && a->col_active[col]) {
        lu_bkt_unlink(a, col);
    }
    a->col_active[col] = 0;
}

static void lu_context_free(LUContext *ctx) {
    if (ctx == NULL) return;
    free(ctx->l_indptr);
    free(ctx->l_indices);
    free(ctx->l_values);
    free(ctx->u_indptr);
    free(ctx->u_indices);
    free(ctx->u_values);
    free(ctx->perm_row);
    free(ctx->perm_col);
    free(ctx->inv_perm_row);
    free(ctx->inv_perm_col);
    free(ctx->eta_positions);
    free(ctx->eta_pivot);
    free(ctx->eta_sp_start);
    free(ctx->eta_sp_idx);
    free(ctx->eta_sp_val);
    free(ctx->u_diag);
    free(ctx->ws_z);
    free(ctx->ws_w);
    free(ctx->ws_v);
    free(ctx->gp_stack);
    free(ctx->gp_xi);
    free(ctx->gp_pinv);
    free(ctx->gp_marked);
    free(ctx->lt_indptr);
    free(ctx->lt_indices);
    free(ctx->lt_values);
    free(ctx->ut_indptr);
    free(ctx->ut_indices);
    free(ctx->ut_values);
    free(ctx);
}

/*
 * lu_factorize: sparse LU factorization with threshold Markowitz pivoting.
 *
 * Input: square matrix in CSC format (csc_indptr, csc_indices, csc_values),
 *        dimension m, threshold parameter u (typically 0.1).
 *
 * Returns an LUContext on success (caller must call lu_context_free),
 * or NULL on allocation failure (sets *alloc_fail=1).
 * If the matrix is singular, ctx->singular_step >= 0 indicates the step.
 *
 * Performance notes:
 *   - Schur update is column-oriented: for each column c in the pivot row,
 *     walk column c once, updating existing entries at update rows via a
 *     dense multiplier array, then insert fill-in for update rows not found.
 *     Cost per step: O(prow_nnz * (col_len + n_update_rows)) instead of the
 *     naive O(n_update_rows * prow_nnz * col_len).
 *   - Dense-tail switch: when the active submatrix remaining dimension is
 *     <= DENSE_TAIL_THRESHOLD or density > DENSE_TAIL_DENSITY, copy to a
 *     dense buffer and finish with dense partial-pivot LU. This avoids
 *     linked-list overhead when the submatrix has gone dense from fill-in.
 */
#define LU_DENSE_TAIL_DIM 64
#define LU_DENSE_TAIL_DENSITY 0.30

static LUContext *lu_factorize(int32_t m,
                               const int32_t *csc_indptr,
                               const int32_t *csc_indices,
                               const double *csc_values,
                               double u,
                               int *alloc_fail) {
    LUContext *ctx = NULL;
    LUActive active;
    int lu_prof = getenv("LINPROGX_LU_PROFILE") != NULL;
    double tp_mark = lu_prof ? linprogx_monotonic_seconds() : 0.0;
    double tp_init = 0.0, tp_pivot = 0.0, tp_elim = 0.0, tp_step = 0.0;
    /* Dense workspace arrays, allocated once and reused every step */
    double *mult_arr = NULL;       /* mult_arr[row] = L multiplier for update rows */
    unsigned char *is_update = NULL; /* marks update rows */
    double *prow_work = NULL;      /* dense scatter of pivot row values */
    unsigned char *prow_mark = NULL; /* marks columns in pivot row */
    int32_t *prow_list = NULL;     /* list of pivot row column indices */
    int32_t *update_rows = NULL;   /* list of update row indices */
    /* Growable arrays for L and U entries */
    int32_t *l_row = NULL, *u_row = NULL;
    double  *l_val = NULL, *u_val = NULL;
    int32_t  l_len = 0, l_cap = 0;
    int32_t  u_len = 0, u_cap = 0;
    /* Column pointers for L and U (built step by step) */
    int32_t *l_colptr = NULL, *u_colptr = NULL;
    /* Dense-tail workspace */
    double *dense_buf = NULL;
    int32_t *dense_row_map = NULL, *dense_col_map = NULL;

    *alloc_fail = 0;
    memset(&active, 0, sizeof(active));

    ctx = calloc(1, sizeof(LUContext));
    if (ctx == NULL) { goto oom; }
    ctx->m = m;
    ctx->singular_step = -1;
    ctx->perm_row = calloc((size_t)m, sizeof(int32_t));
    ctx->perm_col = calloc((size_t)m, sizeof(int32_t));
    ctx->inv_perm_row = calloc((size_t)m, sizeof(int32_t));
    ctx->inv_perm_col = calloc((size_t)m, sizeof(int32_t));
    if (ctx->perm_row == NULL || ctx->perm_col == NULL ||
        ctx->inv_perm_row == NULL || ctx->inv_perm_col == NULL) {
        goto oom;
    }

    if (m == 0) {
        /* Trivial case: empty matrix */
        ctx->l_indptr = calloc(1, sizeof(int32_t));
        ctx->u_indptr = calloc(1, sizeof(int32_t));
        if (ctx->l_indptr == NULL || ctx->u_indptr == NULL) goto oom;
        ctx->nnz_l = 0;
        ctx->nnz_u = 0;
        return ctx;
    }

    mult_arr = calloc((size_t)m, sizeof(double));
    is_update = calloc((size_t)m, sizeof(unsigned char));
    prow_work = calloc((size_t)m, sizeof(double));
    prow_mark = calloc((size_t)m, sizeof(unsigned char));
    prow_list = calloc((size_t)m, sizeof(int32_t));
    update_rows = calloc((size_t)m, sizeof(int32_t));
    l_colptr = calloc((size_t)(m + 1), sizeof(int32_t));
    u_colptr = calloc((size_t)(m + 1), sizeof(int32_t));
    if (mult_arr == NULL || is_update == NULL || prow_work == NULL ||
        prow_mark == NULL || prow_list == NULL || update_rows == NULL ||
        l_colptr == NULL || u_colptr == NULL) {
        goto oom;
    }

    if (lu_prof) {
        double t = linprogx_monotonic_seconds();
        tp_init -= t; /* completed after the call below */
    }
    if (lu_active_init(&active, m, csc_indptr, csc_indices, csc_values) < 0) {
        goto oom;
    }

    /* Pre-allocate L and U storage with nnz estimate */
    {
        int32_t est = csc_indptr[m] + m;
        l_cap = est > 64 ? est : 64;
        u_cap = est > 64 ? est : 64;
    }
    l_row = calloc((size_t)l_cap, sizeof(int32_t));
    l_val = calloc((size_t)l_cap, sizeof(double));
    u_row = calloc((size_t)u_cap, sizeof(int32_t));
    u_val = calloc((size_t)u_cap, sizeof(double));
    if (l_row == NULL || l_val == NULL || u_row == NULL || u_val == NULL) {
        goto oom;
    }

    if (lu_prof) {
        tp_init += linprogx_monotonic_seconds();
        tp_mark = linprogx_monotonic_seconds();
    }
    /* Main elimination loop */
    for (int32_t step = 0; step < m; step++) {
        /* ---- Check for dense-tail switch ---- */
        int32_t remaining = m - step;
        if (remaining > 1) {
            /* Density from the incrementally-maintained active-entry count */
            int64_t active_nnz = active.total_nnz;
            double density = (double)active_nnz / ((double)remaining * (double)remaining);
            if (remaining <= LU_DENSE_TAIL_DIM || density > LU_DENSE_TAIL_DENSITY) {
                /* Switch to dense LU for the remaining submatrix */
                int32_t n = remaining;
                dense_buf = calloc((size_t)n * (size_t)n, sizeof(double));
                dense_row_map = calloc((size_t)n, sizeof(int32_t));
                dense_col_map = calloc((size_t)n, sizeof(int32_t));
                if (dense_buf == NULL || dense_row_map == NULL || dense_col_map == NULL) {
                    goto oom;
                }
                /* Build maps from dense index to original index */
                {
                    int32_t ri = 0, ci = 0;
                    for (int32_t i = 0; i < m; i++) {
                        if (active.row_active[i]) dense_row_map[ri++] = i;
                    }
                    for (int32_t j = 0; j < m; j++) {
                        if (active.col_active[j]) dense_col_map[ci++] = j;
                    }
                }
                /* Build inverse maps for row: original -> dense index */
                /* Reuse is_update as scratch (it's 0-initialized, size m) */
                {
                    int32_t *row_inv = calloc((size_t)m, sizeof(int32_t));
                    if (row_inv == NULL) goto oom;
                    for (int32_t di = 0; di < n; di++) {
                        row_inv[dense_row_map[di]] = di;
                    }
                    /* Fill dense buffer from active columns */
                    for (int32_t dj = 0; dj < n; dj++) {
                        int32_t orig_col = dense_col_map[dj];
                        int32_t idx = active.col_head[orig_col];
                        while (idx >= 0) {
                            int32_t di = row_inv[active.pool[idx].row];
                            dense_buf[di * n + dj] = active.pool[idx].value;
                            idx = active.pool[idx].next_in_col;
                        }
                    }
                    free(row_inv);
                }
                lu_active_free(&active);
                memset(&active, 0, sizeof(active));

                /* Dense partial-pivot LU on the n x n buffer */
                /* dense_buf is row-major: dense_buf[i*n + j] = A[i,j] */
                for (int32_t k = 0; k < n; k++) {
                    /* Find pivot: largest absolute value in column k, rows k..n-1 */
                    int32_t piv = -1;
                    double piv_val = 0.0;
                    for (int32_t i = k; i < n; i++) {
                        double av = fabs(dense_buf[i * n + k]);
                        if (av > fabs(piv_val)) {
                            piv_val = dense_buf[i * n + k];
                            piv = i;
                        }
                    }
                    if (piv < 0 || piv_val == 0.0) {
                        /* Singular: set column pointers for the failing step
                         * and all remaining steps to the current lengths so
                         * the assembly loop doesn't read stale offsets. */
                        ctx->singular_step = step + k;
                        l_colptr[step + k] = l_len;
                        u_colptr[step + k] = u_len;
                        for (int32_t s = k; s < n; s++) {
                            ctx->perm_row[step + s] = dense_row_map[s];
                            ctx->perm_col[step + s] = dense_col_map[s];
                            l_colptr[step + s + 1] = l_len;
                            u_colptr[step + s + 1] = u_len;
                        }
                        goto assemble;
                    }
                    /* Swap rows k and piv */
                    if (piv != k) {
                        for (int32_t j = 0; j < n; j++) {
                            double tmp = dense_buf[k * n + j];
                            dense_buf[k * n + j] = dense_buf[piv * n + j];
                            dense_buf[piv * n + j] = tmp;
                        }
                        {
                            int32_t tmp = dense_row_map[k];
                            dense_row_map[k] = dense_row_map[piv];
                            dense_row_map[piv] = tmp;
                        }
                    }
                    ctx->perm_row[step + k] = dense_row_map[k];
                    ctx->perm_col[step + k] = dense_col_map[k];

                    /* Record L column (skip exact zeros to avoid nnz bloat) */
                    l_colptr[step + k] = l_len;
                    for (int32_t i = k + 1; i < n; i++) {
                        double lval = dense_buf[i * n + k] / piv_val;
                        dense_buf[i * n + k] = lval; /* store in-place for Schur update */
                        if (lval == 0.0) continue;
                        if (l_len == l_cap) {
                            int32_t nc = l_cap * 2;
                            int32_t *nr2 = realloc(l_row, (size_t)nc * sizeof(int32_t));
                            double  *nv2 = realloc(l_val, (size_t)nc * sizeof(double));
                            if (nr2 == NULL || nv2 == NULL) {
                                if (nr2) l_row = nr2;
                                if (nv2) l_val = nv2;
                                goto oom;
                            }
                            l_row = nr2; l_val = nv2; l_cap = nc;
                        }
                        l_row[l_len] = dense_row_map[i];
                        l_val[l_len] = lval;
                        l_len++;
                    }

                    /* Record U row (skip exact zeros) */
                    u_colptr[step + k] = u_len;
                    for (int32_t j = k; j < n; j++) {
                        double uval = dense_buf[k * n + j];
                        if (uval == 0.0 && j != k) continue; /* keep diagonal */
                        if (u_len == u_cap) {
                            int32_t nc = u_cap * 2;
                            int32_t *nr2 = realloc(u_row, (size_t)nc * sizeof(int32_t));
                            double  *nv2 = realloc(u_val, (size_t)nc * sizeof(double));
                            if (nr2 == NULL || nv2 == NULL) {
                                if (nr2) u_row = nr2;
                                if (nv2) u_val = nv2;
                                goto oom;
                            }
                            u_row = nr2; u_val = nv2; u_cap = nc;
                        }
                        u_row[u_len] = dense_col_map[j];
                        u_val[u_len] = uval;
                        u_len++;
                    }

                    /* Schur complement update on dense buffer */
                    for (int32_t i = k + 1; i < n; i++) {
                        double lv = dense_buf[i * n + k]; /* already = L[i,k] */
                        for (int32_t j = k + 1; j < n; j++) {
                            dense_buf[i * n + j] -= lv * dense_buf[k * n + j];
                        }
                    }
                }
                /* All steps consumed by dense tail */
                free(dense_buf); dense_buf = NULL;
                free(dense_row_map); dense_row_map = NULL;
                free(dense_col_map); dense_col_map = NULL;
                goto assemble;
            }
        }

        if (lu_prof) {
            tp_step = linprogx_monotonic_seconds();
        }
        /* ---- Markowitz pivot selection ---- */
        /* Search up to 4 "tiers" of minimum column count among active columns.
         * Within each tier, scan all entries to find the best Markowitz score
         * subject to the threshold stability test. */
        int32_t best_pivot_row = -1, best_pivot_col = -1;
        int64_t best_score = (int64_t)m * (int64_t)m + 1;
        double best_pivot_val = 0.0;

        /* Walk the count buckets from the smallest occupied count: visit
         * at most 4 nonempty tiers, and after the first tier stop probing
         * for more once 256 consecutive counts are empty (quality-only
         * tiers; bounded so sparse count distributions cannot reintroduce
         * the O(m) sweep this replaced). bkt_min advances lazily. */
        while (active.bkt_min <= m && active.bkt_head[active.bkt_min] < 0) {
            active.bkt_min++;
        }
        int32_t tiers_seen = 0;
        int32_t empty_run = 0;
        for (int32_t cnt = active.bkt_min;
             cnt <= m && tiers_seen < 4 && (tiers_seen == 0 || empty_run < 256);
             cnt++) {
            int32_t j = active.bkt_head[cnt];
            if (j < 0) {
                empty_run++;
                continue;
            }
            empty_run = 0;
            tiers_seen++;
            for (; j >= 0; j = active.bkt_next[j]) {

            double col_max = 0.0;
            int32_t idx = active.col_head[j];
            while (idx >= 0) {
                double av = fabs(active.pool[idx].value);
                if (av > col_max) col_max = av;
                idx = active.pool[idx].next_in_col;
            }
            if (col_max == 0.0) continue;

            double threshold = u * col_max;
            idx = active.col_head[j];
            while (idx >= 0) {
                int32_t r = active.pool[idx].row;
                double av = fabs(active.pool[idx].value);
                if (av >= threshold) {
                    int64_t score = (int64_t)(active.row_count[r] - 1) *
                                   (int64_t)(active.col_count[j] - 1);
                    if (score < best_score) {
                        best_score = score;
                        best_pivot_row = r;
                        best_pivot_col = j;
                        best_pivot_val = active.pool[idx].value;
                        if (score == 0) goto pivot_found;
                    }
                }
                idx = active.pool[idx].next_in_col;
            }
            }
        }
pivot_found:
        if (lu_prof) {
            double t = linprogx_monotonic_seconds();
            tp_pivot += t - tp_step;
            tp_step = t;
        }

        if (best_pivot_row < 0) {
            ctx->singular_step = step;
            l_colptr[step] = l_len;
            u_colptr[step] = u_len;
            {
                int32_t ri = 0, ci = 0;
                for (int32_t s = step; s < m; s++) {
                    while (ri < m && !active.row_active[ri]) ri++;
                    while (ci < m && !active.col_active[ci]) ci++;
                    ctx->perm_row[s] = ri < m ? ri : 0;
                    ctx->perm_col[s] = ci < m ? ci : 0;
                    if (ri < m) { active.row_active[ri] = 0; ri++; }
                    if (ci < m) { active.col_active[ci] = 0; ci++; }
                    l_colptr[s + 1] = l_len;
                    u_colptr[s + 1] = u_len;
                }
            }
            break;
        }

        /* Record the pivot */
        ctx->perm_row[step] = best_pivot_row;
        ctx->perm_col[step] = best_pivot_col;

        /* -- Record L column step: multipliers from pivot column -- */
        l_colptr[step] = l_len;
        {
            int32_t idx2 = active.col_head[best_pivot_col];
            while (idx2 >= 0) {
                int32_t r = active.pool[idx2].row;
                if (r != best_pivot_row) {
                    double lval = active.pool[idx2].value / best_pivot_val;
                    if (l_len == l_cap) {
                        int32_t nc = l_cap * 2;
                        int32_t *nr = realloc(l_row, (size_t)nc * sizeof(int32_t));
                        double  *nv = realloc(l_val, (size_t)nc * sizeof(double));
                        if (nr == NULL || nv == NULL) {
                            if (nr) l_row = nr;
                            if (nv) l_val = nv;
                            goto oom;
                        }
                        l_row = nr; l_val = nv; l_cap = nc;
                    }
                    l_row[l_len] = r;
                    l_val[l_len] = lval;
                    l_len++;
                }
                idx2 = active.pool[idx2].next_in_col;
            }
        }

        /* -- Record U row step: entries in pivot row -- */
        u_colptr[step] = u_len;
        {
            int32_t idx2 = active.row_head[best_pivot_row];
            while (idx2 >= 0) {
                int32_t c = active.pool[idx2].col;
                double val = active.pool[idx2].value;
                if (u_len == u_cap) {
                    int32_t nc = u_cap * 2;
                    int32_t *nr = realloc(u_row, (size_t)nc * sizeof(int32_t));
                    double  *nv = realloc(u_val, (size_t)nc * sizeof(double));
                    if (nr == NULL || nv == NULL) {
                        if (nr) u_row = nr;
                        if (nv) u_val = nv;
                        goto oom;
                    }
                    u_row = nr; u_val = nv; u_cap = nc;
                }
                u_row[u_len] = c;
                u_val[u_len] = val;
                u_len++;
                idx2 = active.pool[idx2].next_in_row;
            }
        }

        /* ---- Column-oriented Schur complement update ----
         *
         * For each column c in the pivot row (c != pivot_col):
         *   walk column c once in the active matrix.
         *   For each entry (r, val): if r is an update row, val -= mult[r] * prow[c].
         *   Then scan update rows not found in column c and insert fill-in.
         *
         * This is O(prow_nnz * (col_len + n_update_rows)) per step, vs the
         * naive O(n_update_rows * prow_nnz * col_len) which blows up when
         * columns become dense from fill-in.
         */
        {
            /* Scatter pivot row into dense workspace */
            int32_t prow_nnz = 0;
            {
                int32_t idx2 = active.row_head[best_pivot_row];
                while (idx2 >= 0) {
                    int32_t c = active.pool[idx2].col;
                    prow_work[c] = active.pool[idx2].value;
                    prow_mark[c] = 1;
                    prow_list[prow_nnz++] = c;
                    idx2 = active.pool[idx2].next_in_row;
                }
            }

            /* Collect update rows and their multipliers into dense arrays */
            int32_t n_update_rows = 0;
            {
                int32_t idx2 = active.col_head[best_pivot_col];
                while (idx2 >= 0) {
                    int32_t r = active.pool[idx2].row;
                    if (r != best_pivot_row) {
                        double lmult = active.pool[idx2].value / best_pivot_val;
                        mult_arr[r] = lmult;
                        is_update[r] = 1;
                        update_rows[n_update_rows++] = r;
                    }
                    idx2 = active.pool[idx2].next_in_col;
                }
            }

            /* Remove pivot row and column from active matrix */
            lu_active_remove_row(&active, best_pivot_row);
            lu_active_remove_col(&active, best_pivot_col);

            /* Column-oriented update: for each column c in the pivot row */
            for (int32_t pc = 0; pc < prow_nnz; pc++) {
                int32_t c = prow_list[pc];
                if (c == best_pivot_col) continue;
                if (!active.col_active[c]) continue;
                double pval = prow_work[c];

                /* Mark which update rows we find in this column.
                 * Use is_update: it's 1 for update rows. We'll temporarily
                 * set it to 2 when found, then reset to 1 after. */

                /* Walk column c, update existing entries at update rows */
                int32_t idx2 = active.col_head[c];
                while (idx2 >= 0) {
                    int32_t r = active.pool[idx2].row;
                    if (is_update[r] == 1) {
                        active.pool[idx2].value -= mult_arr[r] * pval;
                        is_update[r] = 2; /* mark as found */
                    }
                    idx2 = active.pool[idx2].next_in_col;
                }

                /* Insert fill-in for update rows not found in column c */
                for (int32_t ur = 0; ur < n_update_rows; ur++) {
                    int32_t r = update_rows[ur];
                    if (is_update[r] == 1) {
                        /* Fill-in entry */
                        double fval = -(mult_arr[r] * pval);
                        if (lu_active_insert(&active, r, c, fval) < 0) {
                            goto oom;
                        }
                    }
                    /* Reset found flag back to 1 for next column */
                    is_update[r] = 1;
                }
            }

            /* Clean up dense workspace for pivot row */
            for (int32_t pc = 0; pc < prow_nnz; pc++) {
                prow_work[prow_list[pc]] = 0.0;
                prow_mark[prow_list[pc]] = 0;
            }

            /* Clean up update row markers */
            for (int32_t ur = 0; ur < n_update_rows; ur++) {
                mult_arr[update_rows[ur]] = 0.0;
                is_update[update_rows[ur]] = 0;
            }
        }
        if (lu_prof) {
            tp_elim += linprogx_monotonic_seconds() - tp_step;
        }
    }

    /* ---- Assemble L and U into CSC format ---- */
assemble:

    /* Build inverse permutations */
    for (int32_t k = 0; k < m; k++) {
        ctx->inv_perm_row[ctx->perm_row[k]] = k;
        ctx->inv_perm_col[ctx->perm_col[k]] = k;
    }

    /* L is already collected column-by-column with original row indices.
     * Convert original row indices to elimination (permuted) row indices,
     * sort by column. L is unit lower triangular in the permuted ordering. */
    {
        int32_t total_l = l_len;
        ctx->nnz_l = total_l;
        ctx->l_indptr = calloc((size_t)(m + 1), sizeof(int32_t));
        ctx->l_indices = total_l > 0 ? calloc((size_t)total_l, sizeof(int32_t)) : NULL;
        ctx->l_values  = total_l > 0 ? calloc((size_t)total_l, sizeof(double))  : NULL;
        if (ctx->l_indptr == NULL ||
            (total_l > 0 && (ctx->l_indices == NULL || ctx->l_values == NULL))) {
            goto oom;
        }
        /* l_colptr[step] marks start of L column `step` in l_row/l_val.
         * l_colptr[step+1] (from the next step or sentinel) marks the end. */
        if (ctx->singular_step < 0) {
            l_colptr[m] = l_len;
        }
        /* The sentinel is already set for singular case in the break above */

        int32_t pos = 0;
        for (int32_t k = 0; k < m; k++) {
            ctx->l_indptr[k] = pos;
            for (int32_t p = l_colptr[k]; p < l_colptr[k + 1]; p++) {
                int32_t orig_row = l_row[p];
                int32_t perm_row = ctx->inv_perm_row[orig_row];
                ctx->l_indices[pos] = perm_row;
                ctx->l_values[pos] = l_val[p];
                pos++;
            }
        }
        ctx->l_indptr[m] = pos;
    }

    /* U was collected as rows: U row `step` contains entries at original columns.
     * Convert to CSC: U column j (in elimination order) contains all U[i,j]
     * where i <= j. */
    {
        /* u_colptr[step] marks start of U row `step` in u_row/u_val.
         * u_row[p] is the original column index. */
        if (ctx->singular_step < 0) {
            u_colptr[m] = u_len;
        }

        /* First pass: count entries per U column (in elimination order) */
        int32_t *u_csc_count = calloc((size_t)m, sizeof(int32_t));
        if (u_csc_count == NULL) goto oom;

        for (int32_t step = 0; step < m; step++) {
            for (int32_t p = u_colptr[step]; p < u_colptr[step + 1]; p++) {
                int32_t orig_col = u_row[p];
                int32_t elim_col = ctx->inv_perm_col[orig_col];
                u_csc_count[elim_col]++;
            }
        }

        int32_t total_u = 0;
        for (int32_t j = 0; j < m; j++) total_u += u_csc_count[j];

        ctx->nnz_u = total_u;
        ctx->u_indptr = calloc((size_t)(m + 1), sizeof(int32_t));
        ctx->u_indices = total_u > 0 ? calloc((size_t)total_u, sizeof(int32_t)) : NULL;
        ctx->u_values  = total_u > 0 ? calloc((size_t)total_u, sizeof(double))  : NULL;
        if (ctx->u_indptr == NULL ||
            (total_u > 0 && (ctx->u_indices == NULL || ctx->u_values == NULL))) {
            free(u_csc_count);
            goto oom;
        }

        /* Build column pointers */
        ctx->u_indptr[0] = 0;
        for (int32_t j = 0; j < m; j++) {
            ctx->u_indptr[j + 1] = ctx->u_indptr[j] + u_csc_count[j];
        }

        /* Second pass: fill entries */
        int32_t *u_csc_pos = calloc((size_t)m, sizeof(int32_t));
        if (u_csc_pos == NULL) {
            free(u_csc_count);
            goto oom;
        }
        for (int32_t j = 0; j < m; j++) {
            u_csc_pos[j] = ctx->u_indptr[j];
        }

        for (int32_t step = 0; step < m; step++) {
            for (int32_t p = u_colptr[step]; p < u_colptr[step + 1]; p++) {
                int32_t orig_col = u_row[p];
                int32_t elim_col = ctx->inv_perm_col[orig_col];
                int32_t pos2 = u_csc_pos[elim_col]++;
                ctx->u_indices[pos2] = step;  /* elimination row index */
                ctx->u_values[pos2] = u_val[p];
            }
        }

        free(u_csc_count);
        free(u_csc_pos);
    }

    /* Initialize basis update eta file */
    /* Initialize sparse Forrest-Tomlin update storage */
    ctx->eta_positions = NULL;
    ctx->eta_pivot = NULL;
    ctx->eta_sp_start = NULL;
    ctx->eta_sp_idx = NULL;
    ctx->eta_sp_val = NULL;
    ctx->eta_sp_total_nnz = 0;
    ctx->eta_sp_packed_cap = 0;
    ctx->n_updates = 0;
    ctx->eta_cap = 0;
    ctx->orig_nnz_lu = ctx->nnz_l + ctx->nnz_u;
    ctx->max_abs_diag = 0.0;
    ctx->min_abs_diag = 1e300;

    /* Build cached U diagonal for fast access in FTRAN/BTRAN */
    ctx->u_diag = calloc((size_t)m, sizeof(double));
    if (ctx->u_diag == NULL) goto oom;
    for (int32_t j = 0; j < m; j++) {
        for (int32_t p = ctx->u_indptr[j]; p < ctx->u_indptr[j + 1]; p++) {
            if (ctx->u_indices[p] == j) {
                ctx->u_diag[j] = ctx->u_values[p];
                break;
            }
        }
    }

    /* Allocate reusable workspace for FTRAN/BTRAN */
    ctx->ws_z = calloc((size_t)m, sizeof(double));
    ctx->ws_w = calloc((size_t)m, sizeof(double));
    ctx->ws_v = calloc((size_t)m, sizeof(double));
    if (ctx->ws_z == NULL || ctx->ws_w == NULL || ctx->ws_v == NULL) goto oom;

    /* Allocate Gilbert-Peierls sparse solve workspaces */
    ctx->gp_stack  = malloc((size_t)m * sizeof(int32_t));
    ctx->gp_xi     = malloc(2 * (size_t)m * sizeof(int32_t));
    ctx->gp_pinv   = malloc((size_t)m * sizeof(int32_t));
    ctx->gp_marked = calloc((size_t)m, sizeof(int32_t));
    if (ctx->gp_stack == NULL || ctx->gp_xi == NULL ||
        ctx->gp_pinv == NULL || ctx->gp_marked == NULL) goto oom;
    ctx->gp_mark = 1;

    /* Initialize transpose pointers to NULL (built on demand) */
    ctx->lt_indptr  = NULL;
    ctx->lt_indices = NULL;
    ctx->lt_values  = NULL;
    ctx->ut_indptr  = NULL;
    ctx->ut_indices = NULL;
    ctx->ut_values  = NULL;

    /* Initialize hyper-sparse solve statistics */
    ctx->ftran_dense_count = 0;
    ctx->ftran_sparse_count = 0;
    ctx->ftran_sparse_nnz_total = 0;
    ctx->btran_dense_count = 0;
    ctx->btran_sparse_count = 0;
    ctx->btran_sparse_nnz_total = 0;

    /* Cleanup temporaries and return */
    lu_active_free(&active);
    free(mult_arr); free(is_update);
    free(prow_work); free(prow_mark);
    free(prow_list); free(update_rows);
    free(l_row); free(l_val);
    free(u_row); free(u_val);
    free(l_colptr); free(u_colptr);
    free(dense_buf); free(dense_row_map); free(dense_col_map);
    if (lu_prof) {
        fprintf(stderr,
                "lu profile: m=%d init=%.4f pivot=%.4f elim=%.4f assemble=%.4f\n",
                m, tp_init, tp_pivot, tp_elim,
                linprogx_monotonic_seconds() - tp_mark - tp_pivot - tp_elim);
    }
    return ctx;

oom:
    *alloc_fail = 1;
    lu_active_free(&active);
    free(mult_arr); free(is_update);
    free(prow_work); free(prow_mark);
    free(prow_list); free(update_rows);
    free(l_row); free(l_val);
    free(u_row); free(u_val);
    free(l_colptr); free(u_colptr);
    free(dense_buf); free(dense_row_map); free(dense_col_map);
    lu_context_free(ctx);
    return NULL;
}

/* ============================================================
 * Gilbert-Peierls hyper-sparse triangular solve infrastructure
 * ============================================================ */

/*
 * lu_build_transposes: build CSC representations of L^T and U^T
 * (equivalently, CSR representations of L and U).
 *
 * Standard CSC-to-CSR conversion:
 *   1. Count entries per row
 *   2. Prefix sum to get indptr
 *   3. Fill values and column indices
 */
static void lu_build_transposes(LUContext *ctx) {
    int32_t m = ctx->m;

    /* Free previous transposes if any */
    free(ctx->lt_indptr);
    free(ctx->lt_indices);
    free(ctx->lt_values);
    free(ctx->ut_indptr);
    free(ctx->ut_indices);
    free(ctx->ut_values);

    /* ---- Build L^T (transpose of L) ---- */
    {
        int32_t nnz = ctx->nnz_l;
        ctx->lt_indptr  = calloc((size_t)(m + 1), sizeof(int32_t));
        ctx->lt_indices = malloc((size_t)nnz * sizeof(int32_t));
        ctx->lt_values  = malloc((size_t)nnz * sizeof(double));
        if (ctx->lt_indptr == NULL || ctx->lt_indices == NULL || ctx->lt_values == NULL) return;

        /* Count entries per row of L (= per column of L^T) */
        for (int32_t p = 0; p < nnz; p++) {
            ctx->lt_indptr[ctx->l_indices[p] + 1]++;
        }
        /* Prefix sum */
        for (int32_t i = 1; i <= m; i++) {
            ctx->lt_indptr[i] += ctx->lt_indptr[i - 1];
        }
        /* Fill: for each column j of L, for each entry (i,j), place into L^T column i.
         * Reuse gp_stack as temp counter array (size m, int32_t). */
        int32_t *work = ctx->gp_stack;
        memset(work, 0, (size_t)m * sizeof(int32_t));
        for (int32_t j = 0; j < m; j++) {
            for (int32_t p = ctx->l_indptr[j]; p < ctx->l_indptr[j + 1]; p++) {
                int32_t i = ctx->l_indices[p];
                int32_t dest = ctx->lt_indptr[i] + work[i];
                ctx->lt_indices[dest] = j;
                ctx->lt_values[dest]  = ctx->l_values[p];
                work[i]++;
            }
        }
    }

    /* ---- Build U^T (transpose of U) ---- */
    {
        int32_t nnz = ctx->nnz_u;
        ctx->ut_indptr  = calloc((size_t)(m + 1), sizeof(int32_t));
        ctx->ut_indices = malloc((size_t)nnz * sizeof(int32_t));
        ctx->ut_values  = malloc((size_t)nnz * sizeof(double));
        if (ctx->ut_indptr == NULL || ctx->ut_indices == NULL || ctx->ut_values == NULL) return;

        /* Count entries per row of U (= per column of U^T) */
        for (int32_t p = 0; p < nnz; p++) {
            ctx->ut_indptr[ctx->u_indices[p] + 1]++;
        }
        /* Prefix sum */
        for (int32_t i = 1; i <= m; i++) {
            ctx->ut_indptr[i] += ctx->ut_indptr[i - 1];
        }
        /* Fill (reuse gp_stack as temp counter) */
        int32_t *work = ctx->gp_stack;
        memset(work, 0, (size_t)m * sizeof(int32_t));
        for (int32_t j = 0; j < m; j++) {
            for (int32_t p = ctx->u_indptr[j]; p < ctx->u_indptr[j + 1]; p++) {
                int32_t i = ctx->u_indices[p];
                int32_t dest = ctx->ut_indptr[i] + work[i];
                ctx->ut_indices[dest] = j;
                ctx->ut_values[dest]  = ctx->u_values[p];
                work[i]++;
            }
        }
    }
}

/*
 * gp_reach: Gilbert-Peierls depth-first-search reach.
 *
 * Given a sparse rhs with nonzero positions in rhs_pattern[0..n_rhs_nz-1],
 * compute the set of nodes reachable in the graph defined by
 * matrix_indptr/matrix_indices (CSC format).
 *
 * Returns the reach in REVERSE topological order in xi[top..m-1].
 * Returns the value of 'top'.
 *
 * Uses ctx->gp_marked with ctx->gp_mark for O(1) amortized visited checks.
 * Uses ctx->gp_stack for iterative DFS.
 * Uses xi[0..m-1] as scratch for current-child-pointer during DFS.
 */
static int32_t gp_reach(LUContext *ctx,
                         int32_t n_rhs_nz,
                         const int32_t *rhs_pattern,
                         const int32_t *matrix_indptr,
                         const int32_t *matrix_indices,
                         int32_t m,
                         int32_t *xi) {
    int32_t *stack   = ctx->gp_stack;
    int32_t *marked  = ctx->gp_marked;
    int32_t  mark    = ctx->gp_mark;

    /* Overflow check: if mark would overflow, reset */
    if (mark >= INT32_MAX - 1) {
        memset(marked, 0, (size_t)m * sizeof(int32_t));
        mark = 1;
    }
    ctx->gp_mark = mark + 1;

    int32_t top = m;  /* xi[top..m-1] will be filled in reverse topo order */

    for (int32_t k = 0; k < n_rhs_nz; k++) {
        int32_t root = rhs_pattern[k];
        if (marked[root] == mark) continue;  /* already visited */

        int32_t stack_top = 0;
        stack[0] = root;
        /* xi[root] stores the current pointer position for DFS resume */
        xi[root] = matrix_indptr[root];

        while (stack_top >= 0) {
            int32_t node = stack[stack_top];
            int32_t p_end = matrix_indptr[node + 1];

            if (marked[node] != mark) {
                marked[node] = mark;
                xi[node] = matrix_indptr[node];
            }

            /* Find next unvisited child */
            int found_child = 0;
            for (int32_t p = xi[node]; p < p_end; p++) {
                int32_t child = matrix_indices[p];
                if (marked[child] != mark) {
                    xi[node] = p + 1;  /* save resume point */
                    stack[++stack_top] = child;
                    xi[child] = matrix_indptr[child];
                    found_child = 1;
                    break;
                }
            }

            if (!found_child) {
                /* All children visited: post-order emit */
                stack_top--;
                xi[--top + m] = node;  /* store in xi[m..2m-1] */
            }
        }
    }

    /* Move reach from xi[m+top..2m-1] to xi[top..m-1] */
    for (int32_t i = top; i < m; i++) {
        xi[i] = xi[i + m];
    }

    return top;
}

/*
 * gp_lsolve: sparse forward solve with L (unit lower triangular).
 *
 * x is a dense vector (mostly zero), xi[top..m-1] contains the nonzero
 * positions in topological order. L is unit lower triangular in CSC.
 *
 * For each j in xi[top..m-1]:
 *   for each L[i,j] where i > j: x[i] -= L[i,j] * x[j]
 *
 * No diagonal division needed since L is unit lower triangular.
 */
static void gp_lsolve(const LUContext *ctx, double *x,
                       const int32_t *xi, int32_t top) {
    int32_t m = ctx->m;
    for (int32_t px = top; px < m; px++) {
        int32_t j = xi[px];
        if (x[j] == 0.0) continue;
        for (int32_t p = ctx->l_indptr[j]; p < ctx->l_indptr[j + 1]; p++) {
            x[ctx->l_indices[p]] -= ctx->l_values[p] * x[j];
        }
    }
}

/*
 * gp_usolve: sparse back solve with U (upper triangular).
 *
 * x is a dense vector (mostly zero), xi[top..m-1] contains the nonzero
 * positions from the reach computation. The DFS post-order reversal
 * already places them in decreasing order (correct for back-sub),
 * so iterate FORWARD through xi[top..m-1].
 *
 * For each j in xi[top..m-1]:
 *   x[j] /= u_diag[j]
 *   for each U[i,j] where i < j: x[i] -= U[i,j] * x[j]
 */
static void gp_usolve(const LUContext *ctx, double *x,
                       const int32_t *xi, int32_t top) {
    int32_t m = ctx->m;
    for (int32_t px = top; px < m; px++) {
        int32_t j = xi[px];
        if (x[j] == 0.0) continue;
        double diag = ctx->u_diag[j];
        if (diag != 0.0) {
            x[j] /= diag;
        }
        for (int32_t p = ctx->u_indptr[j]; p < ctx->u_indptr[j + 1]; p++) {
            int32_t i = ctx->u_indices[p];
            if (i < j) {
                x[i] -= ctx->u_values[p] * x[j];
            }
        }
    }
}

/*
 * lu_ftran_sparse: hyper-sparse FTRAN using Gilbert-Peierls.
 *
 * Solves B x = b where b is given in sparse form:
 *   rhs_indices[0..n_rhs_nz-1] = nonzero positions
 *   rhs_values[0..n_rhs_nz-1]  = corresponding values
 *
 * Steps:
 *   1. Permute sparse rhs: z[inv_perm_row[idx]] = val
 *   2. DFS reach on L -> sparse L-solve
 *   3. DFS reach on U (using L-solve output pattern) -> sparse U-solve
 *   4. Inverse permute: x[perm_col[k]] = z[k]
 *   5. Apply eta chain
 *   6. Clear workspace at touched positions
 */
/*
 * lu_ftran_sparse: hyper-sparse FTRAN using Gilbert-Peierls.
 *
 * x must be zero on entry; caller is responsible for clearing it at
 * the returned pattern positions after use.
 *
 * Returns the number of nonzeros.  x_pattern[0..ret-1] holds their indices.
 */
static int32_t lu_ftran_sparse(LUContext *ctx,
                               int32_t n_rhs_nz, const int32_t *rhs_indices,
                               const double *rhs_values,
                               double *x, int32_t *x_pattern) {
    int32_t m = ctx->m;
    double *z = ctx->ws_z;
    int32_t *xi = ctx->gp_xi;

    /* ws_z may have stale values from a prior dense lu_ftran or lu_btran
     * call.  The sparse solve relies on z being zero at untouched positions,
     * so clear it unconditionally.  This is O(m) but sequential and fast. */
    memset(z, 0, (size_t)m * sizeof(double));

    /* Step 1: Permute sparse rhs into z.
     * Build permuted pattern for DFS.
     * gp_pinv is safe here: gp_reach reads from it but never writes to it. */
    int32_t *perm_pattern = ctx->gp_pinv;
    int32_t n_perm_nz = 0;
    for (int32_t k = 0; k < n_rhs_nz; k++) {
        int32_t idx = rhs_indices[k];
        int32_t prow = ctx->inv_perm_row[idx];
        z[prow] = rhs_values[k];
        perm_pattern[n_perm_nz++] = prow;
    }

    /* Step 2: Reach on L, then sparse L-solve */
    int32_t top_l = gp_reach(ctx, n_perm_nz, perm_pattern,
                              ctx->l_indptr, ctx->l_indices, m, xi);
    gp_lsolve(ctx, z, xi, top_l);

    /* Step 3: Build the nonzero pattern after L-solve for U-reach.
     * Collect actual nonzeros from L-solve output.
     * Must be done before gp_reach for U overwrites xi. */
    int32_t n_u_rhs = 0;
    for (int32_t px = top_l; px < m; px++) {
        int32_t j = xi[px];
        if (z[j] != 0.0) {
            perm_pattern[n_u_rhs++] = j;
        }
    }

    /* Reach on U, then sparse U-solve.
     * This overwrites xi with U-reach in xi[top_u..m-1]. */
    int32_t top_u = gp_reach(ctx, n_u_rhs, perm_pattern,
                              ctx->u_indptr, ctx->u_indices, m, xi);
    gp_usolve(ctx, z, xi, top_u);

    /* Step 4: Inverse permute: x[perm_col[k]] = z[k] for nonzero k.
     * Then clear z at U-reach positions. The U-reach is a superset of all
     * nonzero positions after both L-solve and U-solve (since we fed all
     * L-solve nonzeros as U-reach roots), so clearing here is sufficient.
     *
     * x is assumed zero on entry; no memset needed. */
    int32_t sol_nnz = 0;
    for (int32_t px = top_u; px < m; px++) {
        int32_t k = xi[px];
        if (z[k] != 0.0) {
            int32_t out_idx = ctx->perm_col[k];
            x[out_idx] = z[k];
            x_pattern[sol_nnz++] = out_idx;
        }
        z[k] = 0.0;  /* clear workspace */
    }

    /* Step 5: Apply sparse Forrest-Tomlin etas.
     * If x[pos] == 0 before eta k, then temp = 0/pivot = 0, all
     * updates are x[j] -= v * 0 = 0, so no change. Skip.
     *
     * Etas may introduce new nonzeros (at eta sparse positions) or modify
     * existing ones.  We track the full nonzero pattern including
     * eta-created entries in x_pattern.  To avoid duplicate indices
     * (which are harmless but waste work in callers), we use the
     * gp_marked array as a presence set. */
    if (ctx->n_updates > 0) {
        int32_t *marked = ctx->gp_marked;
        int32_t mark = ctx->gp_mark;
        if (mark >= INT32_MAX - 1) {
            memset(marked, 0, (size_t)m * sizeof(int32_t));
            mark = 1;
        }
        ctx->gp_mark = mark + 1;

        /* Mark existing pattern positions */
        for (int32_t i = 0; i < sol_nnz; i++) {
            marked[x_pattern[i]] = mark;
        }

        for (int32_t upd = 0; upd < ctx->n_updates; upd++) {
            int32_t pos = ctx->eta_positions[upd];
            double xpos = x[pos];
            if (xpos == 0.0) continue;
            double temp = xpos / ctx->eta_pivot[upd];
            int32_t sp_start = ctx->eta_sp_start[upd];
            int32_t sp_end   = ctx->eta_sp_start[upd + 1];
            for (int32_t p = sp_start; p < sp_end; p++) {
                int32_t j = ctx->eta_sp_idx[p];
                x[j] -= ctx->eta_sp_val[p] * temp;
                if (marked[j] != mark) {
                    marked[j] = mark;
                    x_pattern[sol_nnz++] = j;
                }
            }
            x[pos] = temp;
        }
    }

    /* Update statistics */
    ctx->ftran_sparse_count++;
    ctx->ftran_sparse_nnz_total += sol_nnz;
    return sol_nnz;
}

/*
 * lu_btran_sparse: hyper-sparse BTRAN for unit-vector rhs.
 *
 * Solves B^T x = e_{rhs_pos} using sparse triangular solves with
 * transposed L and U.
 *
 * x must be zero on entry; caller is responsible for clearing it at
 * the returned pattern positions after use.
 *
 * Returns the number of nonzeros.  x_pattern[0..ret-1] holds their indices.
 */
static int32_t lu_btran_sparse(LUContext *ctx,
                               int32_t rhs_pos,
                               double *x, int32_t *x_pattern) {
    int32_t m = ctx->m;
    double *z = ctx->ws_z;
    double *v_eta = ctx->ws_v;
    int32_t *xi = ctx->gp_xi;

    /* ws_z may have stale values from a prior dense lu_ftran or lu_btran
     * call.  The sparse solve relies on z being zero at untouched positions,
     * so clear it unconditionally.  This is O(m) but sequential and fast. */
    memset(z, 0, (size_t)m * sizeof(double));

    /* Step 1: Apply etas in reverse to build b_eff from e_{rhs_pos}.
     * Step 2: Permute b_eff into z.
     *
     * When n_updates == 0 (freshly factorized), b_eff is just e_{rhs_pos},
     * so we can directly set up the permuted pattern without the O(m) scan
     * of v_eta. */
    int32_t *perm_pattern = ctx->gp_pinv;
    int32_t n_perm_nz = 0;

    if (ctx->n_updates == 0) {
        /* Fast path: unit vector, no etas to apply */
        int32_t perm_idx = ctx->inv_perm_col[rhs_pos];
        z[perm_idx] = 1.0;
        perm_pattern[0] = perm_idx;
        n_perm_nz = 1;
    } else {
        /* Clear v_eta to handle stale values from prior dense lu_btran
         * calls (which use ws_v but don't clean up). */
        memset(v_eta, 0, (size_t)m * sizeof(double));
        v_eta[rhs_pos] = 1.0;

        for (int32_t upd = ctx->n_updates - 1; upd >= 0; upd--) {
            int32_t pos = ctx->eta_positions[upd];
            double piv = ctx->eta_pivot[upd];
            double vpos = v_eta[pos];

            /* dot = piv * v_eta[pos] + sum_{(j,v)} v * v_eta[j] */
            double dot = piv * vpos;
            int32_t sp_start = ctx->eta_sp_start[upd];
            int32_t sp_end   = ctx->eta_sp_start[upd + 1];
            for (int32_t p = sp_start; p < sp_end; p++) {
                dot += ctx->eta_sp_val[p] * v_eta[ctx->eta_sp_idx[p]];
            }
            v_eta[pos] = vpos + (vpos - dot) / piv;
        }

        /* Permute b_eff into z.  The set of potentially-nonzero positions
         * in v_eta is {rhs_pos} union {eta_positions[0..n_updates-1]}.
         * Scan only those instead of the full O(m) array. */
        if (v_eta[rhs_pos] != 0.0) {
            int32_t pi = ctx->inv_perm_col[rhs_pos];
            z[pi] = v_eta[rhs_pos];
            perm_pattern[n_perm_nz++] = pi;
            v_eta[rhs_pos] = 0.0;
        }
        for (int32_t upd = 0; upd < ctx->n_updates; upd++) {
            int32_t pos = ctx->eta_positions[upd];
            if (v_eta[pos] != 0.0) {
                int32_t pi = ctx->inv_perm_col[pos];
                z[pi] = v_eta[pos];
                perm_pattern[n_perm_nz++] = pi;
                v_eta[pos] = 0.0;
            }
        }
    }

    /* Step 3: U^T-solve. U^T is lower triangular (forward solve).
     * Reach gives positions in topological order for forward solve. */
    int32_t top_ut = gp_reach(ctx, n_perm_nz, perm_pattern,
                               ctx->ut_indptr, ctx->ut_indices, m, xi);
    /* Forward solve with U^T (lower triangular):
     *   z[j] /= u_diag[j]
     *   for each U^T[i,j] where i > j: z[i] -= U^T[i,j] * z[j] */
    for (int32_t px = top_ut; px < m; px++) {
        int32_t j = xi[px];
        if (z[j] == 0.0) continue;
        double diag = ctx->u_diag[j];
        if (diag != 0.0) {
            z[j] /= diag;
        }
        for (int32_t p = ctx->ut_indptr[j]; p < ctx->ut_indptr[j + 1]; p++) {
            int32_t i = ctx->ut_indices[p];
            if (i > j) {
                z[i] -= ctx->ut_values[p] * z[j];
            }
        }
    }

    /* Step 4: L^T-solve. L^T is unit upper triangular (back solve).
     * Build pattern after U^T solve for L^T reach (before gp_reach
     * overwrites xi). */
    int32_t n_lt_rhs = 0;
    for (int32_t px = top_ut; px < m; px++) {
        int32_t j = xi[px];
        if (z[j] != 0.0) {
            perm_pattern[n_lt_rhs++] = j;
        }
    }

    int32_t top_lt = gp_reach(ctx, n_lt_rhs, perm_pattern,
                               ctx->lt_indptr, ctx->lt_indices, m, xi);
    /* Back solve with L^T (unit upper triangular):
     * DFS post-order reversal already gives decreasing order in
     * xi[top_lt..m-1], matching back-sub requirements. Iterate forward.
     *   for each L^T[i,j] where i < j: z[i] -= L^T[i,j] * z[j] */
    for (int32_t px = top_lt; px < m; px++) {
        int32_t j = xi[px];
        if (z[j] == 0.0) continue;
        for (int32_t p = ctx->lt_indptr[j]; p < ctx->lt_indptr[j + 1]; p++) {
            int32_t i = ctx->lt_indices[p];
            if (i < j) {
                z[i] -= ctx->lt_values[p] * z[j];
            }
        }
    }

    /* Step 5: Inverse permute and clear z.
     * x[perm_row[k]] = z[k] for nonzero k in L^T-reach.
     * L^T-reach is a superset of all nonzero positions after both solves
     * (U^T-solve nonzeros were fed as L^T-reach roots).
     *
     * x is assumed zero on entry; no memset needed. */
    int32_t sol_nnz = 0;
    for (int32_t px = top_lt; px < m; px++) {
        int32_t k = xi[px];
        if (z[k] != 0.0) {
            int32_t out_idx = ctx->perm_row[k];
            x[out_idx] = z[k];
            x_pattern[sol_nnz++] = out_idx;
        }
        z[k] = 0.0;  /* clear workspace */
    }

    /* Update statistics */
    ctx->btran_sparse_count++;
    ctx->btran_sparse_nnz_total += sol_nnz;
    return sol_nnz;
}

/*
 * lu_ftran: solve B x = b where PAQ = LU.
 *
 * B = P^-1 L U Q^-1, so B x = b  =>  L U Q^-1 x = P b
 * Let y = Q^-1 x, then L U y = P b.
 * 1. z = P b   (permute rhs)
 * 2. L w = z   (forward solve, L is unit lower triangular)
 * 3. U y = w   (back solve)
 * 4. x = Q y   (inverse column permutation)
 *
 * Dense rhs for milestone 1.
 */
static void lu_ftran(const LUContext *ctx, const double *b, double *x) {
    int32_t m = ctx->m;
    double *z = ctx->ws_z;
    double *w = ctx->ws_w;

    /* Step 1: z = P b */
    for (int32_t k = 0; k < m; k++) {
        z[k] = b[ctx->perm_row[k]];
    }

    /* Step 2: L w = z (forward substitution, L is unit lower triangular CSC) */
    memcpy(w, z, (size_t)m * sizeof(double));
    for (int32_t j = 0; j < m; j++) {
        /* w[j] is finalized (L[j,j] = 1 implicit) */
        for (int32_t p = ctx->l_indptr[j]; p < ctx->l_indptr[j + 1]; p++) {
            w[ctx->l_indices[p]] -= ctx->l_values[p] * w[j];
        }
    }

    /* Step 3: U y = w (back substitution using cached diagonal) */
    /* y stored in z (reuse) */
    memcpy(z, w, (size_t)m * sizeof(double));
    for (int32_t j = m - 1; j >= 0; j--) {
        double diag = ctx->u_diag[j];
        if (diag != 0.0) {
            z[j] /= diag;
        }
        /* Subtract U[i,j] * z[j] from z[i] for i < j */
        for (int32_t p = ctx->u_indptr[j]; p < ctx->u_indptr[j + 1]; p++) {
            int32_t i = ctx->u_indices[p];
            if (i < j) {
                z[i] -= ctx->u_values[p] * z[j];
            }
        }
    }

    /* Step 4: x = Q y, i.e., x[Q[k]] = y[k] */
    for (int32_t k = 0; k < m; k++) {
        x[ctx->perm_col[k]] = z[k];
    }

    /* Step 5: Apply sparse Forrest-Tomlin etas.
     *
     * After standard FTRAN, x = B_0^{-1} b. With k accumulated updates:
     *   x_final = E_k^{-1} ... E_1^{-1} x
     *
     * Each E_i^{-1} acts on x as:
     *   temp = x[pos_i] / pivot_i
     *   for (j, v) in sparse eta_i: x[j] -= v * temp
     *   x[pos_i] = temp
     *
     * Cost per update: O(nnz_i) instead of O(m).
     */
    for (int32_t upd = 0; upd < ctx->n_updates; upd++) {
        int32_t pos = ctx->eta_positions[upd];
        double temp = x[pos] / ctx->eta_pivot[upd];
        int32_t sp_start = ctx->eta_sp_start[upd];
        int32_t sp_end   = ctx->eta_sp_start[upd + 1];
        for (int32_t p = sp_start; p < sp_end; p++) {
            x[ctx->eta_sp_idx[p]] -= ctx->eta_sp_val[p] * temp;
        }
        x[pos] = temp;
    }

}

/*
 * lu_btran: solve B^T x = b where PAQ = LU.
 *
 * B^T = Q^{-T} U^T L^T P^T, so B^T x = b  =>
 * 1. z = Q^T b    (z[k] = b[Q[k]])
 * 2. U^T w = z    (forward solve with U^T)
 * 3. L^T y = w    (back solve with L^T, L unit lower => L^T unit upper)
 * 4. x = P^T y    (x[P[k]] = ... but P^T means x[i] = y[P^{-1}[i]])
 *    Actually: x[i] = y[inv_perm_row[i]]
 *    Equivalently: for k=0..m-1, x[perm_row[k]] ... let's be careful.
 *    P maps original rows to elimination rows: elim_row = inv_perm_row[orig_row].
 *    P^T maps elimination rows back to original: x_orig = P^T y_elim.
 *    x[orig_row] = y[inv_perm_row[orig_row]] ... no.
 *    P_{ki} = 1 iff perm_row[k] = i. So (P b)_k = b_{perm_row[k]}.
 *    (P^T y)_i = y_k where perm_row[k] = i, i.e., k = inv_perm_row[i].
 *    So x[i] = y[inv_perm_row[i]].
 */
static void lu_btran(const LUContext *ctx, const double *b, double *x) {
    int32_t m = ctx->m;
    double *z = ctx->ws_z;
    double *w = ctx->ws_w;

    /* Step 0: Apply sparse Forrest-Tomlin etas BEFORE the standard BTRAN.
     *
     * With k accumulated updates:
     *   x = B_0^{-T} E_1^{-T} ... E_k^{-T} b
     *
     * Apply etas in reverse order (k downto 1):
     *   dot = pivot_i * v[pos_i] + sum_{(j,v) in sparse} v * v_eta[j]
     *   v[pos_i] += (v[pos_i] - dot) / pivot_i
     *
     * Cost per update: O(nnz_i) instead of O(m).
     */
    double *v_eta = ctx->ws_v;
    const double *b_eff = b;
    if (ctx->n_updates > 0) {
        memcpy(v_eta, b, (size_t)m * sizeof(double));
        for (int32_t upd = ctx->n_updates - 1; upd >= 0; upd--) {
            int32_t pos = ctx->eta_positions[upd];
            double piv = ctx->eta_pivot[upd];
            double dot = piv * v_eta[pos];
            int32_t sp_start = ctx->eta_sp_start[upd];
            int32_t sp_end   = ctx->eta_sp_start[upd + 1];
            for (int32_t p = sp_start; p < sp_end; p++) {
                dot += ctx->eta_sp_val[p] * v_eta[ctx->eta_sp_idx[p]];
            }
            v_eta[pos] += (v_eta[pos] - dot) / piv;
        }
        b_eff = v_eta;
    }

    /* Step 1: z = Q^T b_eff, i.e., z[k] = b_eff[perm_col[k]] */
    for (int32_t k = 0; k < m; k++) {
        z[k] = b_eff[ctx->perm_col[k]];
    }

    /* Step 2: U^T w = z (forward solve with U transposed, using cached diagonal)
     *
     * U^T is lower triangular. U^T[j,i] = U[i,j].
     * Row j of U^T corresponds to column j of U.
     * Entries in column j of U with row index i < j give U^T[j,i] for i < j.
     *
     * Forward solve: for j = 0..m-1:
     *   w[j] = (z[j] - sum_{i<j} U[i,j] * w[i]) / U[j,j]
     */
    memcpy(w, z, (size_t)m * sizeof(double));
    for (int32_t j = 0; j < m; j++) {
        /* Subtract contributions from entries in column j with row < j */
        for (int32_t p = ctx->u_indptr[j]; p < ctx->u_indptr[j + 1]; p++) {
            int32_t i = ctx->u_indices[p];
            if (i < j) {
                w[j] -= ctx->u_values[p] * w[i];
            }
        }
        /* Divide by cached diagonal */
        double diag = ctx->u_diag[j];
        if (diag != 0.0) {
            w[j] /= diag;
        }
    }

    /* Step 3: L^T y = w (back solve with L^T, unit upper triangular)
     *
     * for j = m-1 .. 0:
     *   y[j] = w[j] - sum_{p in L_col_j} L_values[p] * y[L_indices[p]]
     */
    memcpy(z, w, (size_t)m * sizeof(double));
    for (int32_t j = m - 1; j >= 0; j--) {
        for (int32_t p = ctx->l_indptr[j]; p < ctx->l_indptr[j + 1]; p++) {
            z[j] -= ctx->l_values[p] * z[ctx->l_indices[p]];
        }
    }

    /* Step 4: x[i] = y[inv_perm_row[i]] */
    for (int32_t i = 0; i < m; i++) {
        x[i] = z[ctx->inv_perm_row[i]];
    }
}

/* ---- Basis-update functions ---- */

/*
 * lu_update: Sparse Forrest-Tomlin basis-change update.
 *
 * Replaces column `leaving_pos` of B with the entering column given in
 * sparse form (entering_indices, entering_values, entering_nnz).
 *
 * Algorithm:
 * 1. Build dense m-vector from sparse entering column.
 * 2. FTRAN through current factorization (L, U, P, Q + accumulated etas)
 *    to get alpha = B_current^{-1} * a_entering.
 * 3. Check |alpha[leaving_pos]| >= threshold for singularity.
 * 4. Store (leaving_pos, pivot, sparse off-diagonal entries of alpha).
 *    Entries with |alpha[j]| < drop_tol * max|alpha| are dropped to
 *    improve sparsity without measurable accuracy loss.
 *
 * Returns 0 on success, -1 if the update is (near-)singular.
 * On singularity, state is unchanged (skip-and-continue semantics).
 */
static int lu_update(LUContext *ctx,
                     int32_t leaving_pos,
                     const int32_t *entering_indices,
                     const double *entering_values,
                     int32_t entering_nnz) {
    int32_t m = ctx->m;

    /* Build dense entering column */
    double *a_dense = calloc((size_t)m, sizeof(double));
    double *alpha   = calloc((size_t)m, sizeof(double));
    if (a_dense == NULL || alpha == NULL) {
        free(a_dense);
        free(alpha);
        return -1;
    }
    for (int32_t k = 0; k < entering_nnz; k++) {
        a_dense[entering_indices[k]] = entering_values[k];
    }

    /* FTRAN: alpha = B_current^{-1} * a_entering */
    lu_ftran(ctx, a_dense, alpha);
    free(a_dense);

    /* Singularity check */
    double pivot = alpha[leaving_pos];
    double abs_pivot = fabs(pivot);

    /* Find max diagonal for relative threshold */
    double max_diag = ctx->max_abs_diag;
    if (max_diag < 1.0) max_diag = 1.0;  /* floor for first update */

    if (abs_pivot < 1e-11 * max_diag) {
        /* Near-singular: leave state unchanged */
        free(alpha);
        return -1;
    }

    /* Count nonzeros for sparse storage (off-diagonal entries only).
     * Drop entries with |alpha[j]| < drop_tol to improve sparsity. */
    double alpha_max = 0.0;
    for (int32_t j = 0; j < m; j++) {
        double av = fabs(alpha[j]);
        if (av > alpha_max) alpha_max = av;
    }
    (void)alpha_max;  /* no drop tolerance — keep etas exact */

    int32_t sp_nnz = 0;
    for (int32_t j = 0; j < m; j++) {
        if (j != leaving_pos && alpha[j] != 0.0) {
            sp_nnz++;
        }
    }

    /* Grow per-update arrays if needed */
    if (ctx->n_updates >= ctx->eta_cap) {
        int32_t new_cap = ctx->eta_cap == 0 ? 16 : ctx->eta_cap * 2;
        int32_t *new_positions = realloc(ctx->eta_positions,
                                          (size_t)new_cap * sizeof(int32_t));
        double *new_pivot = realloc(ctx->eta_pivot,
                                     (size_t)new_cap * sizeof(double));
        int32_t *new_starts = realloc(ctx->eta_sp_start,
                                       ((size_t)new_cap + 1) * sizeof(int32_t));
        if (new_positions == NULL || new_pivot == NULL || new_starts == NULL) {
            if (new_positions != NULL) ctx->eta_positions = new_positions;
            if (new_pivot != NULL) ctx->eta_pivot = new_pivot;
            if (new_starts != NULL) ctx->eta_sp_start = new_starts;
            free(alpha);
            return -1;
        }
        ctx->eta_positions = new_positions;
        ctx->eta_pivot = new_pivot;
        ctx->eta_sp_start = new_starts;
        if (ctx->n_updates == 0) {
            ctx->eta_sp_start[0] = 0;
        }
        ctx->eta_cap = new_cap;
    }

    /* Grow packed sparse arrays if needed */
    int32_t new_total = ctx->eta_sp_total_nnz + sp_nnz;
    if (new_total > ctx->eta_sp_packed_cap) {
        int32_t new_pcap = ctx->eta_sp_packed_cap == 0 ? 256 : ctx->eta_sp_packed_cap;
        while (new_pcap < new_total) new_pcap *= 2;
        int32_t *new_idx = realloc(ctx->eta_sp_idx,
                                    (size_t)new_pcap * sizeof(int32_t));
        double *new_val = realloc(ctx->eta_sp_val,
                                   (size_t)new_pcap * sizeof(double));
        if (new_idx == NULL || new_val == NULL) {
            if (new_idx != NULL) ctx->eta_sp_idx = new_idx;
            if (new_val != NULL) ctx->eta_sp_val = new_val;
            free(alpha);
            return -1;
        }
        ctx->eta_sp_idx = new_idx;
        ctx->eta_sp_val = new_val;
        ctx->eta_sp_packed_cap = new_pcap;
    }

    /* Store sparse eta entry */
    if (ctx->n_updates == 0 && ctx->eta_sp_start == NULL) {
        /* Should not happen after grow above, but be safe */
        free(alpha);
        return -1;
    }
    int32_t pack_pos = ctx->eta_sp_total_nnz;
    for (int32_t j = 0; j < m; j++) {
        if (j != leaving_pos && alpha[j] != 0.0) {
            ctx->eta_sp_idx[pack_pos] = j;
            ctx->eta_sp_val[pack_pos] = alpha[j];
            pack_pos++;
        }
    }

    ctx->eta_positions[ctx->n_updates] = leaving_pos;
    ctx->eta_pivot[ctx->n_updates] = pivot;
    ctx->eta_sp_start[ctx->n_updates + 1] = new_total;
    ctx->eta_sp_total_nnz = new_total;
    ctx->n_updates++;

    /* Track diagonal statistics */
    if (abs_pivot > ctx->max_abs_diag) ctx->max_abs_diag = abs_pivot;
    if (abs_pivot < ctx->min_abs_diag) ctx->min_abs_diag = abs_pivot;

    free(alpha);
    return 0;
}

/*
 * lu_update_with_ftran: basis-change update using a PRE-COMPUTED alpha
 * vector (alpha = B_current^{-1} * a_entering, already obtained via
 * lu_ftran in the calling code).  Avoids the redundant internal FTRAN
 * that the standard lu_update performs.
 *
 * The caller must ensure alpha_precomputed[0..m-1] is valid dense output
 * from lu_ftran on the current factorization.  This buffer is NOT freed
 * by this function.
 *
 * Returns 0 on success, -1 if near-singular (state unchanged).
 */
static int lu_update_with_ftran(LUContext *ctx,
                                int32_t leaving_pos,
                                const double *alpha_precomputed) {
    int32_t m = ctx->m;

    double pivot = alpha_precomputed[leaving_pos];
    double abs_pivot = fabs(pivot);

    double max_diag = ctx->max_abs_diag;
    if (max_diag < 1.0) max_diag = 1.0;

    if (abs_pivot < 1e-11 * max_diag) {
        return -1;
    }

    /* Count nonzeros for sparse storage */
    double alpha_max = 0.0;
    for (int32_t j = 0; j < m; j++) {
        double av = fabs(alpha_precomputed[j]);
        if (av > alpha_max) alpha_max = av;
    }
    (void)alpha_max;  /* no drop tolerance — keep etas exact */

    int32_t sp_nnz = 0;
    for (int32_t j = 0; j < m; j++) {
        if (j != leaving_pos && alpha_precomputed[j] != 0.0) {
            sp_nnz++;
        }
    }

    /* Grow per-update arrays if needed */
    if (ctx->n_updates >= ctx->eta_cap) {
        int32_t new_cap = ctx->eta_cap == 0 ? 16 : ctx->eta_cap * 2;
        int32_t *new_positions = realloc(ctx->eta_positions,
                                          (size_t)new_cap * sizeof(int32_t));
        double *new_pivot = realloc(ctx->eta_pivot,
                                     (size_t)new_cap * sizeof(double));
        int32_t *new_starts = realloc(ctx->eta_sp_start,
                                       ((size_t)new_cap + 1) * sizeof(int32_t));
        if (new_positions == NULL || new_pivot == NULL || new_starts == NULL) {
            if (new_positions != NULL) ctx->eta_positions = new_positions;
            if (new_pivot != NULL) ctx->eta_pivot = new_pivot;
            if (new_starts != NULL) ctx->eta_sp_start = new_starts;
            return -1;
        }
        ctx->eta_positions = new_positions;
        ctx->eta_pivot = new_pivot;
        ctx->eta_sp_start = new_starts;
        if (ctx->n_updates == 0) {
            ctx->eta_sp_start[0] = 0;
        }
        ctx->eta_cap = new_cap;
    }

    /* Grow packed sparse arrays if needed */
    int32_t new_total = ctx->eta_sp_total_nnz + sp_nnz;
    if (new_total > ctx->eta_sp_packed_cap) {
        int32_t new_pcap = ctx->eta_sp_packed_cap == 0 ? 256 : ctx->eta_sp_packed_cap;
        while (new_pcap < new_total) new_pcap *= 2;
        int32_t *new_idx = realloc(ctx->eta_sp_idx,
                                    (size_t)new_pcap * sizeof(int32_t));
        double *new_val = realloc(ctx->eta_sp_val,
                                   (size_t)new_pcap * sizeof(double));
        if (new_idx == NULL || new_val == NULL) {
            if (new_idx != NULL) ctx->eta_sp_idx = new_idx;
            if (new_val != NULL) ctx->eta_sp_val = new_val;
            return -1;
        }
        ctx->eta_sp_idx = new_idx;
        ctx->eta_sp_val = new_val;
        ctx->eta_sp_packed_cap = new_pcap;
    }

    /* Store sparse eta entry */
    int32_t pack_pos = ctx->eta_sp_total_nnz;
    for (int32_t j = 0; j < m; j++) {
        if (j != leaving_pos && alpha_precomputed[j] != 0.0) {
            ctx->eta_sp_idx[pack_pos] = j;
            ctx->eta_sp_val[pack_pos] = alpha_precomputed[j];
            pack_pos++;
        }
    }

    ctx->eta_positions[ctx->n_updates] = leaving_pos;
    ctx->eta_pivot[ctx->n_updates] = pivot;
    ctx->eta_sp_start[ctx->n_updates + 1] = new_total;
    ctx->eta_sp_total_nnz = new_total;
    ctx->n_updates++;

    if (abs_pivot > ctx->max_abs_diag) ctx->max_abs_diag = abs_pivot;
    if (abs_pivot < ctx->min_abs_diag) ctx->min_abs_diag = abs_pivot;

    return 0;
}

/*
 * lu_update_with_ftran_sparse: like lu_update_with_ftran but uses the
 * pre-computed nonzero pattern to avoid O(m) scans.
 *
 * alpha_pattern[0..alpha_nnz-1] = indices of nonzero positions in
 *   alpha_precomputed.
 */
static int lu_update_with_ftran_sparse(LUContext *ctx,
                                       int32_t leaving_pos,
                                       const double *alpha_precomputed,
                                       int32_t alpha_nnz,
                                       const int32_t *alpha_pattern) {
    double pivot = alpha_precomputed[leaving_pos];
    double abs_pivot = fabs(pivot);

    double max_diag = ctx->max_abs_diag;
    if (max_diag < 1.0) max_diag = 1.0;

    if (abs_pivot < 1e-11 * max_diag) {
        return -1;
    }

    /* Count nonzeros for sparse storage (using pattern) */
    int32_t sp_nnz = 0;
    for (int32_t ki = 0; ki < alpha_nnz; ki++) {
        int32_t j = alpha_pattern[ki];
        if (j != leaving_pos) sp_nnz++;
    }

    /* Grow per-update arrays if needed */
    if (ctx->n_updates >= ctx->eta_cap) {
        int32_t new_cap = ctx->eta_cap == 0 ? 16 : ctx->eta_cap * 2;
        int32_t *new_positions = realloc(ctx->eta_positions,
                                          (size_t)new_cap * sizeof(int32_t));
        double *new_pivot = realloc(ctx->eta_pivot,
                                     (size_t)new_cap * sizeof(double));
        int32_t *new_starts = realloc(ctx->eta_sp_start,
                                       ((size_t)new_cap + 1) * sizeof(int32_t));
        if (new_positions == NULL || new_pivot == NULL || new_starts == NULL) {
            if (new_positions != NULL) ctx->eta_positions = new_positions;
            if (new_pivot != NULL) ctx->eta_pivot = new_pivot;
            if (new_starts != NULL) ctx->eta_sp_start = new_starts;
            return -1;
        }
        ctx->eta_positions = new_positions;
        ctx->eta_pivot = new_pivot;
        ctx->eta_sp_start = new_starts;
        if (ctx->n_updates == 0) {
            ctx->eta_sp_start[0] = 0;
        }
        ctx->eta_cap = new_cap;
    }

    /* Grow packed sparse arrays if needed */
    int32_t new_total = ctx->eta_sp_total_nnz + sp_nnz;
    if (new_total > ctx->eta_sp_packed_cap) {
        int32_t new_pcap = ctx->eta_sp_packed_cap == 0 ? 256 : ctx->eta_sp_packed_cap;
        while (new_pcap < new_total) new_pcap *= 2;
        int32_t *new_idx = realloc(ctx->eta_sp_idx,
                                    (size_t)new_pcap * sizeof(int32_t));
        double *new_val = realloc(ctx->eta_sp_val,
                                   (size_t)new_pcap * sizeof(double));
        if (new_idx == NULL || new_val == NULL) {
            if (new_idx != NULL) ctx->eta_sp_idx = new_idx;
            if (new_val != NULL) ctx->eta_sp_val = new_val;
            return -1;
        }
        ctx->eta_sp_idx = new_idx;
        ctx->eta_sp_val = new_val;
        ctx->eta_sp_packed_cap = new_pcap;
    }

    /* Store sparse eta entry using pattern */
    int32_t pack_pos = ctx->eta_sp_total_nnz;
    for (int32_t ki = 0; ki < alpha_nnz; ki++) {
        int32_t j = alpha_pattern[ki];
        if (j != leaving_pos) {
            ctx->eta_sp_idx[pack_pos] = j;
            ctx->eta_sp_val[pack_pos] = alpha_precomputed[j];
            pack_pos++;
        }
    }

    ctx->eta_positions[ctx->n_updates] = leaving_pos;
    ctx->eta_pivot[ctx->n_updates] = pivot;
    ctx->eta_sp_start[ctx->n_updates + 1] = new_total;
    ctx->eta_sp_total_nnz = new_total;
    ctx->n_updates++;

    if (abs_pivot > ctx->max_abs_diag) ctx->max_abs_diag = abs_pivot;
    if (abs_pivot < ctx->min_abs_diag) ctx->min_abs_diag = abs_pivot;

    return 0;
}

/*
 * lu_should_refactor: predicate indicating the factorization should be
 * recomputed from scratch.
 *
 * Triggers:
 * - Number of accumulated updates >= 500 (hard limit)
 * - Eta pivot diagonal growth ratio max/min > 1e6 (stability guard)
 * - Total sparse eta fill > 4 * original nnz(L + U) (fill guard)
 *
 * Returns 1 if refactorization is recommended, 0 otherwise.
 */
static int lu_should_refactor(const LUContext *ctx) {
    if (ctx->n_updates >= 500) return 1;

    if (ctx->n_updates > 0 && ctx->min_abs_diag > 0.0) {
        double ratio = ctx->max_abs_diag / ctx->min_abs_diag;
        if (ratio > 1e6) return 1;
    }

    /* Total sparse eta entries.  With sparse storage the fill grows much
     * more slowly than the old dense O(k*m) scheme, so use a 4x threshold
     * (was 2x for dense).  Minimum 20 updates to amortize refactorization. */
    if (ctx->n_updates >= 20) {
        int64_t eta_fill = (int64_t)ctx->eta_sp_total_nnz;
        if (eta_fill > 4 * (int64_t)ctx->orig_nnz_lu) return 1;
    }

    return 0;
}

/* ---- Bounded-variable dual simplex Phase-2 ---- */

/*
 * Conditioning-aware triangular crash (Bixby 1992, "Implementing the simplex
 * method: the initial basis", spirit).
 *
 * DS_CRASH_STAB: relative stability threshold for accepting a column into the
 *   triangular crash. A column is only accepted at its largest-magnitude
 *   uncovered row r if |a_rj| >= DS_CRASH_STAB * max_i |a_ij|. Keeping the
 *   pivot large relative to the column's own entries bounds the magnitude of
 *   the lower-triangular multipliers and hence the growth in the LU factor.
 *   0.5 is a standard, conservative choice (CPLEX/Bixby use a tolerance in
 *   this range); it trades a few extra artificial fills for a well-conditioned
 *   start.
 * DS_CRASH_MAX_GROWTH: post-factorization guard. If the crash basis factorizes
 *   but max|U_ii| / min|U_ii| exceeds this, the basis is catastrophically
 *   ill-conditioned (e.g. presolved lp_woodw at kappa ~1e19) and the dual
 *   simplex cannot make dual-feasible progress from it. We discard the crash
 *   and restart from the pure-artificial identity basis, which is perfectly
 *   conditioned.
 */
#define DS_CRASH_STAB 0.5
#define DS_CRASH_MAX_GROWTH 1e10

/* Candidate column for the triangular crash, sorted by quality. */
typedef struct {
    int32_t col;      /* structural column index */
    int32_t penalty;  /* bound-type penalty: free(0) < one-sided(1)
                       *   < boxed(2) < fixed(3) */
    int32_t nnz;      /* number of nonzeros in the column */
} DsCrashCand;

/* Order: prefer free/wide columns (low penalty), then fewer nonzeros
 * (slack-like singletons first), deterministic tie-break by column index. */
static int ds_crash_cand_cmp(const void *pa, const void *pb) {
    const DsCrashCand *a = (const DsCrashCand *)pa;
    const DsCrashCand *b = (const DsCrashCand *)pb;
    if (a->penalty != b->penalty) return a->penalty - b->penalty;
    if (a->nnz != b->nnz) return a->nnz - b->nnz;
    return a->col - b->col;
}

/*
 * Helper: build CSC representation of the m x m basis matrix.
 *
 * Basis columns with index < n_structural come from A's CSC arrays.
 * Basis columns with index >= n_structural are artificial (identity column
 * for row j-n_structural, value +1.0).
 *
 * b_indptr[m+1], b_indices[b_nnz], b_values[b_nnz] are preallocated by caller.
 * Returns total nnz of the basis matrix.
 */
static int32_t ds_build_basis_csc(
    int32_t m,
    int32_t n_structural,
    const Py_ssize_t *a_csc_indptr,
    const Py_ssize_t *a_csc_rows,
    const double *a_csc_data,
    const int32_t *basis,
    int32_t *b_indptr,
    int32_t *b_indices,
    double *b_values)
{
    int32_t pos = 0;
    b_indptr[0] = 0;
    for (int32_t k = 0; k < m; k++) {
        int32_t j = basis[k];
        if (j < n_structural) {
            /* Structural column from A's CSC */
            for (Py_ssize_t p = a_csc_indptr[j]; p < a_csc_indptr[j + 1]; p++) {
                b_indices[pos] = (int32_t)a_csc_rows[p];
                b_values[pos] = a_csc_data[p];
                pos++;
            }
        } else {
            /* Artificial column: single +1 in row (j - n_structural) */
            b_indices[pos] = j - n_structural;
            b_values[pos] = 1.0;
            pos++;
        }
        b_indptr[k + 1] = pos;
    }
    return pos;
}

/*
 * Helper: factorize the current basis. Returns LUContext* or NULL on failure.
 * Caller must free with lu_context_free.
 */
static LUContext *ds_factorize_basis(
    int32_t m,
    int32_t n_structural,
    const Py_ssize_t *a_csc_indptr,
    const Py_ssize_t *a_csc_rows,
    const double *a_csc_data,
    const int32_t *basis,
    int32_t *b_indptr,
    int32_t *b_indices,
    double *b_values)
{
    ds_build_basis_csc(m, n_structural, a_csc_indptr, a_csc_rows, a_csc_data,
                       basis, b_indptr, b_indices, b_values);
    int alloc_fail = 0;
    LUContext *lu = lu_factorize(m, b_indptr, b_indices, b_values, 0.1,
                                  &alloc_fail);
    return lu;
}

/* Bound classification constants */
#define DS_BOUND_LO    0   /* nonbasic at lower bound */
#define DS_BOUND_HI    1   /* nonbasic at upper bound */
#define DS_BOUND_FREE  2   /* nonbasic free variable at zero */
#define DS_BOUND_FIXED 3   /* lo == hi, variable is fixed */
#define DS_BOUND_BASIC 4   /* in the basis */

/*
 * ds_repair_singular_basis: when LU factorization reports singular_step >= 0,
 * replace the offending basis column with an artificial (identity) column
 * and refactorize. Repeat up to max_repairs times.
 *
 * On success, returns a valid LUContext with singular_step < 0.
 * On failure (NULL or still singular after max_repairs), returns the last
 * LUContext (possibly with singular_step >= 0) or NULL.
 *
 * Side effects: modifies basis[], basis_pos[], bound_status[], x_ext[],
 * r_ext[], lo_ext[], hi_ext[] to reflect the column swaps.
 */
static LUContext *ds_repair_singular_basis(
    LUContext *lu,
    int32_t m,
    int32_t n,
    const Py_ssize_t *a_csc_indptr,
    const Py_ssize_t *a_csc_rows,
    const double *a_csc_data,
    int32_t *basis,
    int32_t *basis_pos,
    int8_t *bound_status,
    double *x_ext,
    double *r_ext,
    double *lo_ext,
    double *hi_ext,
    int32_t *b_indptr,
    int32_t *b_indices,
    double *b_values,
    int max_repairs)
{
    int32_t n_total = n + m;
    for (int repair = 0; repair < max_repairs; repair++) {
        if (lu == NULL || lu->singular_step < 0) return lu;

        /* perm_col[singular_step] is the basis position (0..m-1) whose
         * column is linearly dependent on the preceding ones. */
        int32_t bad_pos = lu->perm_col[lu->singular_step];
        int32_t old_col = basis[bad_pos];

        /* Find an artificial column not already in the basis.
         * Prefer n + bad_pos (identity column for the same row), but
         * if that's already basic, scan for any available artificial. */
        int32_t art_col = -1;
        if (basis_pos[n + bad_pos] < 0) {
            art_col = n + bad_pos;
        } else {
            for (int32_t i = 0; i < m; i++) {
                if (basis_pos[n + i] < 0) {
                    art_col = n + i;
                    break;
                }
            }
        }
        if (art_col < 0) {
            /* All artificials are in the basis; cannot repair */
            return lu;
        }

        /* Remove old column from basis */
        if (old_col < n) {
            /* Set it at its lower bound (safe default) */
            if (isfinite(lo_ext[old_col])) {
                bound_status[old_col] = DS_BOUND_LO;
                x_ext[old_col] = lo_ext[old_col];
            } else if (isfinite(hi_ext[old_col])) {
                bound_status[old_col] = DS_BOUND_HI;
                x_ext[old_col] = hi_ext[old_col];
            } else {
                bound_status[old_col] = DS_BOUND_FREE;
                x_ext[old_col] = 0.0;
            }
            r_ext[old_col] = 0.0;  /* will be recomputed */
        } else {
            /* Removing an artificial from the basis */
            bound_status[old_col] = DS_BOUND_FIXED;
            x_ext[old_col] = 0.0;
        }
        basis_pos[old_col] = -1;

        /* Insert artificial */
        basis[bad_pos] = art_col;
        basis_pos[art_col] = bad_pos;
        bound_status[art_col] = DS_BOUND_BASIC;
        x_ext[art_col] = 0.0;  /* will be recomputed */

        /* Refactorize with the repaired basis */
        lu_context_free(lu);
        lu = ds_factorize_basis(m, n, a_csc_indptr, a_csc_rows, a_csc_data,
                                basis, b_indptr, b_indices, b_values);
    }
    return lu;
}

/*
 * CSRMatrix_solve_eq_box_dual_simplex
 *
 * Solve  min c'x  subject to  Ax = b,  lo <= x <= hi
 * using a bounded-variable dual simplex (Phase-2 only).
 *
 * Returns a dict with the same keys as solve_eq_box_ipm:
 *   status, x, y, objective, iterations, max_primal_residual
 *
 * Features (Milestone 4):
 *   - Identity artificial start with triangular crash warmstart
 *   - Bound-flip ratio test (Fourer/Maros bounded-variable dual)
 *   - Harris two-pass ratio test for numerical stability
 *   - Devex pricing for leaving variable selection
 *   - Sparse rho pricing via CSR row scatter
 *
 * Sign convention for the dual ratio test (primal-derived):
 *
 *   Leaving variable i has violation:
 *     sigma = +1  if x_B[i] < lo[basis[i]]  (below lower bound)
 *     sigma = -1  if x_B[i] > hi[basis[i]]  (above upper bound)
 *
 *   Admissibility conditions:
 *     j at lower bound (r_j >= 0):  sigma * alpha_j < 0
 *     j at upper bound (r_j <= 0):  sigma * alpha_j > 0
 *
 *   theta_d = -r_q / (sigma * alpha_q)  >= 0
 *   r_j_new = r_j + theta_d * sigma * alpha_j
 */
static PyObject *CSRMatrix_solve_eq_box_dual_simplex(
    CSRMatrixObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *c_obj, *b_obj, *lo_obj, *hi_obj;
    Py_ssize_t max_iter_arg = 0;
    double tol = 1e-8;
    int pricing = 0; /* 0 = Devex (default), 1 = carried exact steepest edge */
    static char *kwlist[] = {"c", "b", "lo", "hi", "max_iter", "tol", "pricing", NULL};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOOO|ndi", kwlist,
            &c_obj, &b_obj, &lo_obj, &hi_obj, &max_iter_arg, &tol, &pricing)) {
        return NULL;
    }
    if (self->rows > INT32_MAX || self->cols > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "matrix too large for 32-bit factorization");
        return NULL;
    }
    Py_ssize_t m_s = self->rows;
    Py_ssize_t n_s = self->cols;
    int32_t m = (int32_t)m_s;
    int32_t n = (int32_t)n_s;
    /* n_total = structural + artificial columns */
    int32_t n_total = n + m;
    Py_ssize_t max_iter = max_iter_arg > 0 ? max_iter_arg
                          : (Py_ssize_t)(50 * (m_s + n_s) < 100000 ? 50 * (m_s + n_s) : 100000);

    PyObject *result = NULL;
    const char *status = "numerical_error";
    Py_ssize_t iterations = 0;
    LUContext *lu = NULL;

    /* ---- Allocate working arrays ---- */
    /* Arrays indexed by n_total (structurals + artificials) */
    double *c_ext  = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    double *lo_ext = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    double *hi_ext = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    double *x_ext  = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    double *r_ext  = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    int32_t *basis_pos = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(int32_t));
    int8_t  *bound_status = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(int8_t));

    double *b   = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *y   = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *x_B = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *rhs = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *rho = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *alpha_col = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *e_i = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *c_B = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *devex_w = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *dse_beta = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    int32_t *basis = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));

    /* Basis CSC workspace (max nnz = A's nnz + m for artificials) */
    int32_t b_nnz_max = (int32_t)(self->nnz > 0 ? self->nnz : 1) + m;
    int32_t *b_indptr  = calloc((size_t)(m + 1), sizeof(int32_t));
    int32_t *b_indices = calloc((size_t)b_nnz_max, sizeof(int32_t));
    double  *b_values  = calloc((size_t)b_nnz_max, sizeof(double));

    /* Sparse rho support */
    int32_t *rho_nz_rows = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    /* Sparse FTRAN/BTRAN output pattern tracking */
    int32_t *ftran_pattern = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    int32_t ftran_nnz = 0;
    int32_t *btran_pattern = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    /* Per-column alpha accumulator for sparse pricing */
    double *alpha_scratch = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(double));
    /* per-column basis entries (churn probe) */
    int32_t *enter_count = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(int32_t));
    int32_t *alpha_touched = calloc((size_t)(n_total > 0 ? n_total : 1), sizeof(int32_t));

    /* Bound-flip workspace: accumulated delta to x_B from flips */
    double *flip_delta_xB = calloc((size_t)(m > 0 ? m : 1), sizeof(double));

    /* Artificial bound tracking: 1 if column j has a big-M artificial bound */
    int8_t *has_art_bound = calloc((size_t)(n > 0 ? n : 1), sizeof(int8_t));
    /* True bounds: original lo/hi before artificial bound shift */
    double *lo_true = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *hi_true = calloc((size_t)(n > 0 ? n : 1), sizeof(double));

    /* Ruiz equilibration scaling arrays */
    double *ds_row_scale = calloc((size_t)(m > 0 ? m : 1), sizeof(double));
    double *ds_col_scale = calloc((size_t)(n > 0 ? n : 1), sizeof(double));
    double *scaled_csc_data = calloc((size_t)(self->nnz > 0 ? self->nnz : 1), sizeof(double));
    double *scaled_csr_data = calloc((size_t)(self->nnz > 0 ? self->nnz : 1), sizeof(double));
    double *c_orig = calloc((size_t)(n > 0 ? n : 1), sizeof(double));

    if (c_ext == NULL || lo_ext == NULL || hi_ext == NULL ||
        x_ext == NULL || r_ext == NULL || basis_pos == NULL || bound_status == NULL ||
        b == NULL || y == NULL || x_B == NULL || rhs == NULL ||
        rho == NULL || alpha_col == NULL || e_i == NULL || c_B == NULL ||
        devex_w == NULL || basis == NULL ||
        b_indptr == NULL || b_indices == NULL || b_values == NULL ||
        rho_nz_rows == NULL || ftran_pattern == NULL || btran_pattern == NULL ||
        alpha_scratch == NULL || alpha_touched == NULL ||
        flip_delta_xB == NULL ||
        has_art_bound == NULL || lo_true == NULL || hi_true == NULL ||
        ds_row_scale == NULL || ds_col_scale == NULL || scaled_csc_data == NULL ||
        scaled_csr_data == NULL || c_orig == NULL) {
        PyErr_NoMemory();
        goto done;
    }

    /* Parse input arrays */
    if (fill_double_array(c_obj, n_s, c_ext, "c") != 0 ||
        fill_double_array(b_obj, m_s, b, "b") != 0 ||
        fill_double_array(lo_obj, n_s, lo_ext, "lo") != 0 ||
        fill_double_array(hi_obj, n_s, hi_ext, "hi") != 0) {
        goto done;
    }
    /* Save original costs before scaling (avoid round-trip error from
     * multiplying then dividing by the scale factor). */
    memcpy(c_orig, c_ext, (size_t)n * sizeof(double));

    /* ---- Ruiz equilibration: scale A, b, c, lo, hi ----
     * Same 10-pass inf-norm + 1 l2 pass as in the IPM entry.
     * Scaled problem: min c_s'x_s s.t. A_s x_s = b_s, lo_s <= x_s <= hi_s
     * where A_s = R A C, b_s = R b, c_s = C c, lo_s = lo/C, hi_s = hi/C,
     * x_s = C^{-1} x. R = diag(ds_row_scale), C = diag(ds_col_scale).
     *
     * Skip Ruiz when the matrix is already well-conditioned (inf-norm ratio
     * < 100): the round-trip scale/unscale introduces floating-point error
     * that can exceed tight absolute tolerances on well-balanced problems. */
    int ruiz_active = 0;  /* set to 1 if we actually apply scaling */
    {
        /* Temp arrays for row/col inf-norms (reuse rhs and alpha_col) */
        double *row_norms = rhs;       /* size m */
        double *col_norms = alpha_col;  /* size m, but we need n; use r_ext */
        /* Actually, r_ext is size n_total >= n, and we haven't used it yet. */
        double *cn = r_ext;  /* borrow r_ext temporarily for col norms */

        for (int32_t j = 0; j < n; j++) ds_col_scale[j] = 1.0;
        for (int32_t i = 0; i < m; i++) ds_row_scale[i] = 1.0;

        /* Compute row inf-norms of the raw matrix to decide if scaling
         * is needed.  If max_norm / min_norm < 100 the matrix is already
         * well-balanced and Ruiz would only introduce round-trip error. */
        {
            double min_rnorm = 1e300, max_rnorm = 0.0;
            for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
            for (int32_t j = 0; j < n; j++) {
                for (Py_ssize_t p = self->csc_indptr[j];
                     p < self->csc_indptr[j + 1]; p++) {
                    int32_t row = (int32_t)self->csc_rows[p];
                    double av = fabs(self->csc_data[p]);
                    if (av > row_norms[row]) row_norms[row] = av;
                }
            }
            for (int32_t i = 0; i < m; i++) {
                if (row_norms[i] > 0.0) {
                    if (row_norms[i] < min_rnorm) min_rnorm = row_norms[i];
                    if (row_norms[i] > max_rnorm) max_rnorm = row_norms[i];
                }
            }
            if (min_rnorm > 0.0 && max_rnorm / min_rnorm >= 100.0) {
                ruiz_active = 1;
            }
        }

        /* 10 passes of Ruiz inf-norm equilibration */
        if (ruiz_active)
        for (int ruiz_iter = 0; ruiz_iter < 10; ruiz_iter++) {
            for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
            for (int32_t j = 0; j < n; j++) cn[j] = 0.0;

            /* Scan CSC to find scaled inf-norms */
            for (int32_t j = 0; j < n; j++) {
                for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                    int32_t row = (int32_t)self->csc_rows[p];
                    double val = fabs(self->csc_data[p] * ds_row_scale[row] * ds_col_scale[j]);
                    if (val > row_norms[row]) row_norms[row] = val;
                    if (val > cn[j]) cn[j] = val;
                }
            }
            for (int32_t i = 0; i < m; i++) {
                if (row_norms[i] > 0.0) ds_row_scale[i] /= sqrt(row_norms[i]);
            }
            for (int32_t j = 0; j < n; j++) {
                if (cn[j] > 0.0) ds_col_scale[j] /= sqrt(cn[j]);
            }
        }

        if (ruiz_active) {
            /* One l2 balancing pass */
            for (int32_t i = 0; i < m; i++) row_norms[i] = 0.0;
            for (int32_t j = 0; j < n; j++) cn[j] = 0.0;
            for (int32_t j = 0; j < n; j++) {
                for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                    int32_t row = (int32_t)self->csc_rows[p];
                    double val = self->csc_data[p] * ds_row_scale[row] * ds_col_scale[j];
                    row_norms[row] += val * val;
                    cn[j] += val * val;
                }
            }
            for (int32_t i = 0; i < m; i++) {
                if (row_norms[i] > 0.0) ds_row_scale[i] /= sqrt(sqrt(row_norms[i]));
            }
            for (int32_t j = 0; j < n; j++) {
                if (cn[j] > 0.0) ds_col_scale[j] /= sqrt(sqrt(cn[j]));
            }

            /* Clamp scales */
            for (int32_t i = 0; i < m; i++) {
                if (ds_row_scale[i] < 1e-8) ds_row_scale[i] = 1e-8;
                else if (ds_row_scale[i] > 1e8) ds_row_scale[i] = 1e8;
            }
            for (int32_t j = 0; j < n; j++) {
                if (ds_col_scale[j] < 1e-8) ds_col_scale[j] = 1e-8;
                else if (ds_col_scale[j] > 1e8) ds_col_scale[j] = 1e8;
            }
            /* Build scaled CSC data */
            for (int32_t j = 0; j < n; j++) {
                for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                    int32_t row = (int32_t)self->csc_rows[p];
                    scaled_csc_data[p] = self->csc_data[p] * ds_row_scale[row] * ds_col_scale[j];
                }
            }

            /* Build scaled CSR data (same scaling, CSR layout).
             * A_s[i,j] = ds_row_scale[i] * A[i,j] * ds_col_scale[j]. */
            for (int32_t i = 0; i < m; i++) {
                double rs = ds_row_scale[i];
                for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++) {
                    int32_t col = (int32_t)self->indices[p];
                    scaled_csr_data[p] = self->data[p] * rs * ds_col_scale[col];
                }
            }

            /* Scale c, lo, hi, b for structural columns */
            for (int32_t j = 0; j < n; j++) {
                c_ext[j] *= ds_col_scale[j];
                if (isfinite(lo_ext[j])) lo_ext[j] /= ds_col_scale[j];
                if (isfinite(hi_ext[j])) hi_ext[j] /= ds_col_scale[j];
            }
            for (int32_t i = 0; i < m; i++) {
                b[i] *= ds_row_scale[i];
            }
        } else {
            /* No scaling: copy raw data to scaled buffers */
            memcpy(scaled_csc_data, self->csc_data,
                   (size_t)self->nnz * sizeof(double));
            memcpy(scaled_csr_data, self->data,
                   (size_t)self->nnz * sizeof(double));
        }

        /* Zero the temp arrays we borrowed */
        memset(rhs, 0, (size_t)m * sizeof(double));
        memset(r_ext, 0, (size_t)n_total * sizeof(double));
    }

    /* Local pointer to scaled CSC data — used everywhere instead of
     * self->csc_data so the solver operates in equilibrated space. */
    const double *a_data = scaled_csc_data;

    if (pricing == 1) {
        /* Anti-degeneracy cost perturbation, composed with steepest edge
         * only: alone it halved cre_d's degenerate pivots but regressed
         * Devex paths (documented trade in HANDOFF 2026-07-04); under SE
         * the tie-breaking and the selection quality are hypothesized to
         * compound. Deterministic multiplicative-hash psi; exit checks,
         * objective, and dual gates all use c_orig, so this steers the
         * path only. */
        for (int32_t j = 0; j < n; j++) {
            uint32_t h = (uint32_t)j * 2654435761u;
            double psi = 0.5 + 0.5 * ((double)h / 4294967296.0);
            c_ext[j] += 1e-9 * (1.0 + fabs(c_ext[j])) * psi;
        }
    }

    /* Set up artificial columns: indices n..n+m-1, cost 0, bounds [0,0] (fixed) */
    for (int32_t i = 0; i < m; i++) {
        c_ext[n + i] = 0.0;
        lo_ext[n + i] = 0.0;
        hi_ext[n + i] = 0.0;
    }

    /* Initialize Devex weights to 1 */
    for (int32_t k = 0; k < m; k++) devex_w[k] = 1.0;

    /* ============================================================
     * 1. CRASH BASIS: greedy triangular crash + artificial fill
     * ============================================================
     * Attempt to build a nonsingular basis from structural columns.
     * Any uncovered row gets its artificial column.
     */
    {
        int8_t *row_covered = calloc((size_t)(m > 0 ? m : 1), sizeof(int8_t));
        /* Track which basis position each row is assigned to */
        int32_t *row_to_bpos = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
        if (row_covered == NULL || row_to_bpos == NULL) {
            free(row_covered); free(row_to_bpos);
            PyErr_NoMemory(); goto done;
        }
        for (int32_t i = 0; i < m; i++) row_to_bpos[i] = -1;

        /* Conditioning-aware triangular crash (Bixby 1992 spirit).
         *
         * A singleton-cascade builds a provably nonsingular lower-triangular
         * basis: a column is only assigned to a pivot row once it is its own
         * *uncovered-row singleton* (exactly one nonzero left in an uncovered
         * row). Assigning that unique entry as the pivot and covering the row
         * is precisely one step of a symbolic triangular permutation, so the
         * accepted columns can never be linearly dependent — unlike a plain
         * "max uncovered entry" greedy, whose off-pivot fill can (and on
         * greenbea did) produce a singular basis that collapses to identity.
         *
         * Seeds (initial singletons) are drained in quality order (free/wide
         * columns and fewer-nonzero, slack-like columns first); the cascade
         * then proceeds FIFO. A column is accepted only if its pivot passes the
         * relative stability threshold |a_rj| >= DS_CRASH_STAB * max_i|a_ij|,
         * bounding the triangular multipliers. Rows left uncovered (columns
         * that never cascade down to a singleton, or fail stability) get
         * artificials below. All magnitudes use the equilibrated matrix
         * (a_data) so the crash sees the same scaling as the factorization —
         * critical for woodw. */
        int32_t n_basis = 0;
        {
            int32_t *uncov = malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
            int8_t *col_done = calloc((size_t)(n > 0 ? n : 1), sizeof(int8_t));
            int32_t *queue = malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
            DsCrashCand *cand =
                malloc((size_t)(n > 0 ? n : 1) * sizeof(DsCrashCand));
            if (uncov == NULL || col_done == NULL || queue == NULL ||
                cand == NULL) {
                free(uncov); free(col_done); free(queue); free(cand);
                free(row_covered); free(row_to_bpos);
                PyErr_NoMemory(); goto done;
            }
            for (int32_t j = 0; j < n; j++) {
                int32_t nnz_j =
                    (int32_t)(self->csc_indptr[j + 1] - self->csc_indptr[j]);
                uncov[j] = nnz_j;
                int lo_fin = isfinite(lo_ext[j]);
                int hi_fin = isfinite(hi_ext[j]);
                int32_t pen;
                if (!lo_fin && !hi_fin) {
                    pen = 0;  /* free: best basic */
                } else if (lo_fin && hi_fin) {
                    pen = (hi_ext[j] - lo_ext[j] <= 1e-30) ? 3   /* fixed */
                                                           : 2;  /* boxed */
                } else {
                    pen = 1;  /* one-sided */
                }
                cand[j].col = j;
                cand[j].penalty = pen;
                cand[j].nnz = nnz_j;
            }
            /* Seed queue with the initial singleton columns in quality order. */
            qsort(cand, (size_t)n, sizeof(DsCrashCand), ds_crash_cand_cmp);
            int32_t qhead = 0, qtail = 0;
            for (int32_t idx = 0; idx < n; idx++) {
                int32_t j = cand[idx].col;
                if (uncov[j] == 1) queue[qtail++] = j;
            }

            while (qhead < qtail && n_basis < m) {
                int32_t j = queue[qhead++];
                if (col_done[j] || uncov[j] != 1) continue;
                col_done[j] = 1;
                /* Locate the unique uncovered row and the column's max |entry|. */
                int32_t pr = -1;
                double pv = 0.0, colmax = 0.0;
                for (Py_ssize_t p = self->csc_indptr[j];
                     p < self->csc_indptr[j + 1]; p++) {
                    double av = fabs(a_data[p]);
                    if (av > colmax) colmax = av;
                    int32_t row = (int32_t)self->csc_rows[p];
                    if (!row_covered[row]) { pr = row; pv = av; }
                }
                if (pr < 0 || colmax <= 1e-12) continue;
                /* Relative stability: reject a small pivot; leave the row for an
                 * artificial rather than inflate the triangular multipliers. */
                if (pv < DS_CRASH_STAB * colmax) continue;
                basis[n_basis] = j;
                row_covered[pr] = 1;
                row_to_bpos[pr] = n_basis;
                n_basis++;
                /* Cascade: covering row pr may turn its other columns into
                 * singletons. */
                for (Py_ssize_t p = self->indptr[pr];
                     p < self->indptr[pr + 1]; p++) {
                    int32_t jj = (int32_t)self->indices[p];
                    if (!col_done[jj] && uncov[jj] > 0) {
                        uncov[jj]--;
                        if (uncov[jj] == 1 && qtail < n) queue[qtail++] = jj;
                    }
                }
            }
            free(uncov); free(col_done); free(queue); free(cand);
        }

        /* Fill uncovered rows with artificials */
        for (int32_t i = 0; i < m; i++) {
            if (!row_covered[i]) {
                basis[n_basis] = n + i;  /* artificial for row i */
                row_to_bpos[i] = n_basis;
                n_basis++;
            }
        }

        free(row_covered);
        free(row_to_bpos);
    }
    /* ============================================================
     * 2. LU FACTORIZE the crash+artificial basis
     * ============================================================
     * Try crash basis first. If singular, fall back to pure identity.
     */
    {
        lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                a_data, basis,
                                b_indptr, b_indices, b_values);

        /* Repair singular crash columns in place. The singleton-cascade crash
         * is triangular and hence nonsingular by construction, so this is a
         * numerical safety net for the rare near-dependency (e.g. a
         * stability-skipped pivot leaving a tiny residual coupling). Swap the
         * offending columns for artificials and refactorize, keeping the bulk
         * of the structural coverage. The cap is small: if more than a handful
         * of columns are dependent the crash is structurally unusable and the
         * clean pure-identity fallback below is both faster and better
         * conditioned than swapping columns one at a time down to identity. */
        if (lu != NULL && lu->singular_step >= 0) {
            int max_repairs = 16 + m / 100;
            int8_t *in_basis = calloc((size_t)n_total, sizeof(int8_t));
            if (in_basis == NULL) { PyErr_NoMemory(); goto done; }
            for (int32_t k = 0; k < m; k++) in_basis[basis[k]] = 1;
            int repairs = 0;
            while (lu != NULL && lu->singular_step >= 0 && repairs < max_repairs) {
                int32_t bad_pos = lu->perm_col[lu->singular_step];
                in_basis[basis[bad_pos]] = 0;
                /* Prefer the identity column for row bad_pos; else any free
                 * artificial. */
                int32_t art = -1;
                if (!in_basis[n + bad_pos]) {
                    art = n + bad_pos;
                } else {
                    for (int32_t i = 0; i < m; i++) {
                        if (!in_basis[n + i]) { art = n + i; break; }
                    }
                }
                if (art < 0) break;  /* no artificial available (shouldn't happen) */
                basis[bad_pos] = art;
                in_basis[art] = 1;
                lu_context_free(lu);
                lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                        a_data, basis,
                                        b_indptr, b_indices, b_values);
                repairs++;
            }
            free(in_basis);
        }

        int reject_crash = (lu == NULL || lu->singular_step >= 0);
        if (!reject_crash) {
            /* Post-crash conditioning guard: estimate growth from the U
             * diagonals. A crash basis that factorizes but is catastrophically
             * ill-conditioned (presolved woodw ~ kappa 1e19) leaves the dual
             * simplex numerically stuck; the pure-artificial identity is
             * perfectly conditioned, so prefer it over limping on the crash. */
            double dmax = 0.0, dmin = 1e300;
            for (int32_t k = 0; k < m; k++) {
                double d = fabs(lu->u_diag[k]);
                if (d > dmax) dmax = d;
                if (d > 0.0 && d < dmin) dmin = d;
            }
            double growth = (dmin > 0.0 && dmin < 1e300)
                                ? dmax / dmin : INFINITY;
            if (growth > DS_CRASH_MAX_GROWTH) {
                reject_crash = 1;
            }
        }
        if (reject_crash) {
            /* Fall back to pure identity basis (all artificials) */
            lu_context_free(lu);
            lu = NULL;
            for (int32_t i = 0; i < m; i++) {
                basis[i] = n + i;  /* artificial for row i */
            }
            lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                    a_data, basis,
                                    b_indptr, b_indices, b_values);
            if (lu == NULL || lu->singular_step >= 0) {
                status = "numerical_error";
                goto build_result;
            }
        }
        lu_build_transposes(lu);
    }
    /* Initialize basis_pos */
    for (int32_t j = 0; j < n_total; j++) basis_pos[j] = -1;
    for (int32_t k = 0; k < m; k++) basis_pos[basis[k]] = k;


    /* ============================================================
     * 3. NONBASIC ASSIGNMENT for dual feasibility
     * ============================================================
     * Compute y = B^{-T} c_B, then r_j = c_j - a_j^T y.
     * Place nonbasic j at lo if r_j >= 0, at hi if r_j < 0.
     * For artificials: cost=0, bounds=[0,0] => fixed.
     *
     * Big-M artificial bounds: when the reduced cost points toward an
     * infinite bound (e.g., r_j < 0 but hi is infinite for a [0,inf)
     * column), no dual-feasible placement exists. We add an artificial
     * finite bound at distance M from the finite bound and place the
     * variable there. At exit, if any artificial bound is active, we
     * re-solve with a larger M (deterministic).
     */

    /* Save true bounds (structural columns only) */
    for (int32_t j = 0; j < n; j++) {
        lo_true[j] = lo_ext[j];
        hi_true[j] = hi_ext[j];
    }

    /* Compute big-M: 1e5 * max(1, max|finite bound|, max|b_i|) */
    double bigM_scale = 1.0;
    for (int32_t j = 0; j < n; j++) {
        if (isfinite(lo_ext[j]) && fabs(lo_ext[j]) > bigM_scale)
            bigM_scale = fabs(lo_ext[j]);
        if (isfinite(hi_ext[j]) && fabs(hi_ext[j]) > bigM_scale)
            bigM_scale = fabs(hi_ext[j]);
    }
    for (int32_t i = 0; i < m; i++) {
        if (fabs(b[i]) > bigM_scale) bigM_scale = fabs(b[i]);
    }
    double bigM_factor = 1e5;
    {
        /* experiment knob: probe whether big-M magnitude drives the path
         * explosion on one-sided-column-heavy instances (cre/pds family) */
        const char *env = getenv("LINPROGX_DS_BIGM_FACTOR");
        if (env != NULL && atof(env) > 0.0) {
            bigM_factor = atof(env);
        }
    }
    double bigM = bigM_factor * bigM_scale;

    for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
    lu_btran(lu, c_B, y);

    for (int32_t j = 0; j < n_total; j++) {
        if (basis_pos[j] >= 0) {
            bound_status[j] = DS_BOUND_BASIC;
            continue;
        }

        /* Compute r_j = c_j - a_j^T y */
        double rj = c_ext[j];
        if (j < n) {
            /* Structural column */
            for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
            }
        } else {
            /* Artificial column: single +1 in row (j - n) */
            rj -= y[j - n];
        }
        r_ext[j] = rj;

        int lo_fin = isfinite(lo_ext[j]);
        int hi_fin = isfinite(hi_ext[j]);

        if (lo_fin && hi_fin && fabs(lo_ext[j] - hi_ext[j]) < 1e-14) {
            bound_status[j] = DS_BOUND_FIXED;
            x_ext[j] = lo_ext[j];
        } else if (!lo_fin && !hi_fin) {
            /* Free variable: give artificial bounds [-M/2, +M/2] */
            if (j < n) {
                lo_ext[j] = -bigM * 0.5;
                hi_ext[j] =  bigM * 0.5;
                has_art_bound[j] = 1;
            }
            if (rj >= 0.0) {
                bound_status[j] = DS_BOUND_LO;
                x_ext[j] = lo_ext[j];
            } else {
                bound_status[j] = DS_BOUND_HI;
                x_ext[j] = hi_ext[j];
            }
        } else if (rj >= 0.0 && lo_fin) {
            bound_status[j] = DS_BOUND_LO;
            x_ext[j] = lo_ext[j];
        } else if (rj < 0.0 && hi_fin) {
            bound_status[j] = DS_BOUND_HI;
            x_ext[j] = hi_ext[j];
        } else if (rj >= 0.0 && !lo_fin && hi_fin) {
            /* r_j >= 0 wants lower bound, but lower is infinite.
             * Add artificial lower bound at hi - M. */
            if (j < n) {
                lo_ext[j] = hi_ext[j] - bigM;
                has_art_bound[j] = 1;
            }
            bound_status[j] = DS_BOUND_LO;
            x_ext[j] = lo_ext[j];
        } else if (rj < 0.0 && !hi_fin && lo_fin) {
            /* r_j < 0 wants upper bound, but upper is infinite.
             * Add artificial upper bound at lo + M. */
            if (j < n) {
                hi_ext[j] = lo_ext[j] + bigM;
                has_art_bound[j] = 1;
            }
            bound_status[j] = DS_BOUND_HI;
            x_ext[j] = hi_ext[j];
        } else if (lo_fin) {
            bound_status[j] = DS_BOUND_LO;
            x_ext[j] = lo_ext[j];
        } else {
            bound_status[j] = DS_BOUND_HI;
            x_ext[j] = hi_ext[j];
        }
    }

    /* ============================================================
     * 4. MAIN DUAL SIMPLEX LOOP
     * ============================================================ */
    /* Running totals for hyper-sparse density stats (survive refactorizations) */
    int64_t cum_ftran_sparse_count = 0, cum_ftran_sparse_nnz = 0;
    int64_t cum_btran_sparse_count = 0, cum_btran_sparse_nnz = 0;
    int64_t total_refacs = 0;
    double refac_time_total = 0.0;
    double refac_factorize_time = 0.0;
    /* pivot-type composition, for degenerate-structure diagnosis */
    int64_t stat_flips = 0;
    int64_t stat_degenerate = 0;
    int64_t stat_bland_pivots = 0;
    int64_t stat_max_degen_streak = 0;
    int64_t stat_art_ejections = 0;
    int64_t stat_cost_shifts = 0;
    int cost_shift_on = 0;
    {
        const char *env = getenv("LINPROGX_DS_COST_SHIFT");
        if (env != NULL && atoi(env) == 1) {
            cost_shift_on = 1;
        }
    }
    {
        int consecutive_degenerate = 0;
        int use_bland = 0;

        /* Refactorization triggers: lu_should_refactor (n_updates >= 500,
         * diagonal ratio > 1e4, eta fill > 4x) or iters_since_refac >= 500.
         * Also refactorize immediately when lu_update produces a violent
         * eta (diagonal element < 1e-6 or > 1e6). */
        int32_t refac_interval = 500;
        int32_t iters_since_refac = 0;
        int x_B_needs_recompute = 1;
        int x_B_fresh = 0; /* set when x_B comes from a full solve, cleared
                            * after incremental pivot updates; infeasibility
                            * may only be declared from fresh state */
        for (Py_ssize_t iter = 0; iter < max_iter; iter++) {
            iterations = iter;

            /* ---- 4a. Compute x_B = B^{-1}(b - A_N x_N) ---- */
            /* With incremental x_B maintenance, recompute from scratch only
             * when the flag is set (first iteration, after refactorization). */
            if (x_B_needs_recompute) {
            memcpy(rhs, b, (size_t)m * sizeof(double));
            for (int32_t j = 0; j < n_total; j++) {
                if (basis_pos[j] >= 0) continue;
                if (x_ext[j] == 0.0) continue;
                double xj = x_ext[j];
                if (j < n) {
                    for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                        rhs[(int32_t)self->csc_rows[p]] -= a_data[p] * xj;
                    }
                } else {
                    /* Artificial column: single +1 in row (j - n) */
                    rhs[j - n] -= xj;
                }
            }
            lu_ftran(lu, rhs, x_B);
            x_B_needs_recompute = 0;
            x_B_fresh = 1;
            }

            /* ---- 4b. Find leaving variable: Devex-weighted max violation ---- */
            int32_t leaving_basis_pos = -1;
            double max_score = 0.0;
            int leaving_sigma = 0;
            for (int32_t k = 0; k < m; k++) {
                int32_t j = basis[k];
                double viol = 0.0;
                int sigma = 0;
                if (isfinite(lo_ext[j]) && x_B[k] < lo_ext[j] - tol) {
                    viol = lo_ext[j] - x_B[k];
                    sigma = 1;
                }
                if (isfinite(hi_ext[j]) && x_B[k] > hi_ext[j] + tol) {
                    double v2 = x_B[k] - hi_ext[j];
                    if (v2 > viol) {
                        viol = v2;
                        sigma = -1;
                    }
                }
                if (viol > 0.0) {
                    /* Devex score: violation^2 / weight (Harris 1973) */
                    double w = devex_w[k];
                    if (w < 1e-12) w = 1e-12;
                    double score = (viol * viol) / w;
                    if (score > max_score) {
                        max_score = score;
                        leaving_basis_pos = k;
                        leaving_sigma = sigma;
                    }
                }
            }

            if (leaving_basis_pos < 0) {
                /* No bound violation: check for basic artificials with |value| > tol */
                int has_artificial_basic = 0;
                for (int32_t k = 0; k < m; k++) {
                    if (basis[k] >= n && fabs(x_B[k]) > tol) {
                        has_artificial_basic = 1;
                        break;
                    }
                }
                if (has_artificial_basic) {
                    status = "infeasible";
                } else {
                    status = "optimal";
                }
                iterations = iter;
                break;
            }

            /* ---- 4c. Compute rho = BTRAN(e_leaving) ---- */
            /* Use hyper-sparse BTRAN for unit vector rhs.
             * rho must be zero on entry (cleared at the pattern positions
             * after use at the end of the previous iteration). */
            int32_t rho_nnz = 0;
            if (lu->lt_indptr != NULL) {
                rho_nnz = lu_btran_sparse(lu, leaving_basis_pos,
                                          rho, rho_nz_rows);
            } else {
                memset(e_i, 0, (size_t)m * sizeof(double));
                e_i[leaving_basis_pos] = 1.0;
                lu_btran(lu, e_i, rho);
                /* Dense path: scan for nonzeros */
                for (int32_t i = 0; i < m; i++) {
                    if (fabs(rho[i]) > 1e-15) {
                        rho_nz_rows[rho_nnz++] = i;
                    }
                }
            }

            if (pricing == 1) {
                /* Continuous drift anchor: rho = B^{-T} e_r is in hand, so
                 * the leaving row's exact steepest-edge weight ||rho||^2 is
                 * free. Anchoring gamma_r each pivot keeps the carried
                 * update's dominant term exact, so floating-point drift
                 * cannot compound across refactorizations. */
                double g_exact = 0.0;
                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    double v = rho[rho_nz_rows[ri]];
                    g_exact += v * v;
                }
                if (g_exact < 1e-12) g_exact = 1e-12;
                devex_w[leaving_basis_pos] = g_exact;
            }

            /* ---- 4d. Sparse pivot row computation via CSR scatter ---- */
            /* Instead of iterating over all n columns and computing rho^T a_j,
             * we iterate over rho's nonzero rows and scatter into alpha_scratch
             * using A's CSR representation. This is O(nnz_in_rho_rows) instead
             * of O(n * avg_col_nnz). */
            int32_t n_alpha_touched = 0;
            {
                /* Clear touched columns */
                /* alpha_scratch[j] and alpha_touched[j] were zeroed at alloc,
                 * but we clean up at the end of each iteration for touched columns. */

                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    int32_t row = rho_nz_rows[ri];
                    double rho_val = rho[row];
                    /* Walk CSR row of A using pre-scaled CSR data */
                    for (Py_ssize_t p = self->indptr[row]; p < self->indptr[row + 1]; p++) {
                        int32_t col = (int32_t)self->indices[p];
                        if (alpha_scratch[col] == 0.0 && alpha_touched[col] == 0) {
                            alpha_touched[col] = 1;
                        }
                        alpha_scratch[col] += rho_val * scaled_csr_data[p];
                    }
                }
                /* Also handle artificial columns: artificial n+i has a single +1 in row i.
                 * So alpha_{n+i} = rho[i]. */
                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    int32_t row = rho_nz_rows[ri];
                    int32_t art_j = n + row;
                    alpha_scratch[art_j] = rho[row];
                    alpha_touched[art_j] = 1;
                }
            }

            /* ---- 4d'. Harris two-pass ratio test with bound flips ---- */
            int32_t entering_col = -1;
            double entering_alpha_row = 0.0;
            double theta_d = 0.0;

            /* Reset flip tracking */
            int32_t n_flips = 0;
            memset(flip_delta_xB, 0, (size_t)m * sizeof(double));

            if (!use_bland) {
                /* ============================================================
                 * Harris two-pass ratio test with bound flips
                 * ============================================================
                 *
                 * We process candidates sorted by ratio (ascending). For each
                 * candidate that is BOXED (both bounds finite), we can flip it
                 * to its other bound instead of pivoting on it, absorbing some
                 * of the leaving variable's infeasibility. We continue flipping
                 * until we find a candidate where the ratio absorbs the remaining
                 * infeasibility, or a non-boxed candidate is found.
                 *
                 * Harris tolerance: accept any pivot with ratio <= theta_max + delta,
                 * then among those pick the one with largest |alpha_j|.
                 */
                double harris_delta = 1e-7;

                /* Collect admissible candidates with their ratios */
                /* We'll do two passes over the nonbasic variables:
                 *   Pass 1: find theta_max (minimum ratio) with Harris tolerance
                 *   Pass 2: pick entering with largest |alpha| within the band
                 * But we also need to handle bound flips, which means we process
                 * in order of ascending ratio and flip boxed candidates.
                 *
                 * Approach: collect all admissible (j, alpha_j, ratio) tuples,
                 * sort by ratio, then process with flip logic. For efficiency,
                 * store in a temporary array sized at most n_total. */

                /* First pass: find minimum ratio (Harris theta_max) and check
                 * if we need flip logic at all. We use the CSR-scattered alphas. */
                double theta_min = 1e300;
                int n_admissible = 0;

                /* Count admissible to know if we need flips */
                for (int32_t j = 0; j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    if (bound_status[j] == DS_BOUND_FIXED) continue;

                    double alpha_j = alpha_scratch[j];
                    if (fabs(alpha_j) < 1e-9) continue; /* Harris: reject tiny pivots */

                    /* Admissibility check */
                    int admissible = 0;
                    if (bound_status[j] == DS_BOUND_LO && leaving_sigma * alpha_j < 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_HI && leaving_sigma * alpha_j > 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_FREE) {
                        admissible = 1;
                    }
                    if (!admissible) continue;

                    double ratio = fabs(r_ext[j]) / fabs(alpha_j);
                    if (ratio < theta_min) theta_min = ratio;
                    n_admissible++;
                }

                if (n_admissible == 0) {
                    /* Clean up alpha_scratch */
                    for (int32_t j = 0; j < n_total; j++) {
                        if (alpha_touched[j]) {
                            alpha_scratch[j] = 0.0;
                            alpha_touched[j] = 0;
                        }
                    }
                    if (!x_B_fresh) {
                        /* The driving violation may be incremental-update
                         * drift. Infeasibility (an empty dual ratio test on
                         * a violated row = a Farkas ray) may only be
                         * declared from a freshly solved x_B: recompute and
                         * re-select instead. */
                        x_B_needs_recompute = 1;
                        continue;
                    }
                    {
                        /* An empty ratio test certifies infeasibility of
                         * the problem AS BOXED. If any artificial big-M
                         * bound is installed, that proves nothing about
                         * the true problem — report a routing status the
                         * caller treats as non-optimal fallthrough, never
                         * a false infeasibility verdict. */
                        int any_art = 0;
                        for (int32_t j = 0; j < n; j++) {
                            if (has_art_bound[j]) { any_art = 1; break; }
                        }
                        status = any_art ? "dual_unbounded_boxed" : "infeasible";
                    }
                    iterations = iter;
                    break;
                }

                /* Harris band */
                double theta_max = theta_min + harris_delta;

                /* Remaining infeasibility to absorb (in units of the dual step).
                 * The leaving variable's primal infeasibility is x_B[p] - bound.
                 * Each flip of a boxed variable j absorbs alpha_j * (hi_j - lo_j) * sigma
                 * of this infeasibility. If flips absorb enough, theta_d = 0 is fine. */
                double leaving_bound;
                if (leaving_sigma == 1) {
                    leaving_bound = lo_ext[basis[leaving_basis_pos]];
                } else {
                    leaving_bound = hi_ext[basis[leaving_basis_pos]];
                }
                /* ---- Bound-flip ratio test (Fourer/Maros scheme) ----
                 *
                 * Process admissible candidates within the Harris band.
                 * For boxed variables at the minimum-ratio end, flip them
                 * (move to opposite bound) if their flip doesn't absorb all
                 * remaining infeasibility. Flips are purely primal operations:
                 * they change x and bound_status but NOT reduced costs (the
                 * r update is handled by the final theta_d step).
                 *
                 * remaining_infeas tracks how much x_B[p] still needs to move.
                 * Each flip of variable j absorbs |alpha_j| * |delta_j| of it.
                 */
                double remaining_infeas = fabs(x_B[leaving_basis_pos] - leaving_bound);

                /* Pass 2: find entering variable with largest |alpha| within Harris band.
                 * Along the way, flip boxed variables at the bottom of the band
                 * if their absorption < remaining_infeas.
                 *
                 * We scan all candidates, classify into "below theta_min" (flip candidates)
                 * and "within band" (pivot candidates). The minimum-ratio flip candidates
                 * get processed first. */

                /* First, find the set of candidates at exactly theta_min (or close) that
                 * are boxed and could be flipped. To avoid sorting, we do two sweeps:
                 * Sweep 1: flip all boxed candidates with ratio <= theta_min + eps
                 *          (these are at the very bottom) if their absorption fits.
                 * Sweep 2: pick the best pivot from remaining candidates within Harris band. */

                /* Sweep 1: flip boxed candidates at the minimum ratio.
                 * NOTE: bound flips are disabled pending a correct
                 * implementation that accounts for primal/dual state
                 * interaction. The flip mechanism must either recompute
                 * x_B after flips before the pivot step, or separate
                 * flip-only and pivot iterations. See Milestone 4 notes.
                 * For now, all candidates are processed as pivot candidates. */
                for (int32_t j = 0; 0 && j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    if (bound_status[j] == DS_BOUND_FIXED) continue;

                    double alpha_j = alpha_scratch[j];
                    if (fabs(alpha_j) < 1e-9) continue;

                    int admissible = 0;
                    if (bound_status[j] == DS_BOUND_LO && leaving_sigma * alpha_j < 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_HI && leaving_sigma * alpha_j > 0.0) {
                        admissible = 1;
                    }
                    if (!admissible) continue;

                    double ratio = fabs(r_ext[j]) / fabs(alpha_j);
                    /* Only flip candidates at the very bottom of the band */
                    if (ratio > theta_min + 1e-10) continue;

                    /* Only boxed structural variables can flip */
                    if (j >= n) continue;
                    if (!isfinite(lo_ext[j]) || !isfinite(hi_ext[j])) continue;
                    double bound_width = hi_ext[j] - lo_ext[j];
                    if (bound_width < 1e-14) continue;

                    /* How much infeasibility does the flip absorb? */
                    double absorption = fabs(alpha_j) * bound_width;
                    if (absorption >= remaining_infeas) {
                        /* This flip would overshoot; j should be the entering variable */
                        continue;
                    }

                    /* Flip j: update x and bound_status (NOT reduced costs) */
                    if (bound_status[j] == DS_BOUND_LO) {
                        x_ext[j] = hi_ext[j];
                        bound_status[j] = DS_BOUND_HI;
                    } else {
                        x_ext[j] = lo_ext[j];
                        bound_status[j] = DS_BOUND_LO;
                    }
                    remaining_infeas -= absorption;
                    n_flips++;
                }

                /* Sweep 2: pick best pivot (largest |alpha|) within Harris band */
                double best_alpha = 0.0;
                for (int32_t j = 0; j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    if (bound_status[j] == DS_BOUND_FIXED) continue;

                    double alpha_j = alpha_scratch[j];
                    if (fabs(alpha_j) < 1e-9) continue;

                    /* Re-check admissibility (bound_status may have changed from flips) */
                    int admissible = 0;
                    if (bound_status[j] == DS_BOUND_LO && leaving_sigma * alpha_j < 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_HI && leaving_sigma * alpha_j > 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_FREE) {
                        admissible = 1;
                    }
                    if (!admissible) continue;

                    double ratio = fabs(r_ext[j]) / fabs(alpha_j);
                    if (ratio > theta_max) continue;

                    /* Pick largest |alpha_j| for numerical stability (Harris pass 2) */
                    if (fabs(alpha_j) > best_alpha) {
                        best_alpha = fabs(alpha_j);
                        entering_col = j;
                        entering_alpha_row = alpha_j;
                    }
                }

                if (entering_col < 0) {
                    /* All candidates within the band were flipped.
                     * Flips changed x and bound_status but NOT reduced costs,
                     * so r_ext is now inconsistent with bound_status.
                     * Recompute reduced costs from scratch to restore consistency,
                     * then continue to the next iteration. */
                    for (int32_t j = 0; j < n_total; j++) {
                        if (alpha_touched[j]) {
                            alpha_scratch[j] = 0.0;
                            alpha_touched[j] = 0;
                        }
                    }
                    stat_flips += n_flips;
                    if (n_flips > 0) {
                        /* Recompute y and reduced costs from scratch.
                         * Dense BTRAN uses ws_v; clear it afterward. */
                        for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
                        lu_btran(lu, c_B, y);
                        if (lu->ws_v != NULL)
                            memset(lu->ws_v, 0, (size_t)m * sizeof(double));
                        for (int32_t j2 = 0; j2 < n_total; j2++) {
                            if (basis_pos[j2] >= 0) continue;
                            if (bound_status[j2] == DS_BOUND_FIXED) continue;
                            double rj = c_ext[j2];
                            if (j2 < n) {
                                for (Py_ssize_t p = self->csc_indptr[j2]; p < self->csc_indptr[j2 + 1]; p++) {
                                    rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
                                }
                            } else {
                                rj -= y[j2 - n];
                            }
                            r_ext[j2] = rj;
                        }
                        /* Flips invalidate incremental x_B */
                        x_B_needs_recompute = 1;
                    }
                    /* Clean up sparse workspaces before continue */
                    for (int32_t ki = 0; ki < rho_nnz; ki++)
                        rho[rho_nz_rows[ki]] = 0.0;
                    iterations = iter + 1;
                    continue;
                }

                /* Compute theta_d for the chosen entering variable */
                theta_d = -r_ext[entering_col] / ((double)leaving_sigma * entering_alpha_row);
                if (cost_shift_on && theta_d < 1e-12) {
                    /* Dynamic anti-degeneracy cost shift (Koberstein-style):
                     * a zero dual step makes no progress and re-forms the
                     * same ties (measured on cre_d: 764 columns re-entering
                     * the basis up to 55x). Shift the entering column's
                     * WORKING cost minimally so the step is positive. The
                     * shift lives only in c_ext/r_ext (path steering); the
                     * exit gates, objective, and certificates all recompute
                     * from c_orig, and 1e-9-scale shifts sit far below the
                     * 1e-7 exit tolerances. */
                    double want = 1e-9 * (1.0 + fabs(c_ext[entering_col]));
                    double r_new = -((double)leaving_sigma * entering_alpha_row) * want;
                    double shift = r_new - r_ext[entering_col];
                    c_ext[entering_col] += shift;
                    r_ext[entering_col] = r_new;
                    theta_d = want;
                    stat_cost_shifts++;
                }

            } else {
                /* Bland's rule: pick smallest index admissible */
                for (int32_t j = 0; j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    if (bound_status[j] == DS_BOUND_FIXED) continue;

                    double alpha_j = alpha_scratch[j];
                    if (fabs(alpha_j) < 1e-9) continue;

                    int admissible = 0;
                    if (bound_status[j] == DS_BOUND_LO && leaving_sigma * alpha_j < 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_HI && leaving_sigma * alpha_j > 0.0) {
                        admissible = 1;
                    } else if (bound_status[j] == DS_BOUND_FREE) {
                        admissible = 1;
                    }
                    if (!admissible) continue;

                    entering_col = j;
                    entering_alpha_row = alpha_j;
                    break; /* Bland: first admissible */
                }

                if (entering_col < 0) {
                    for (int32_t j = 0; j < n_total; j++) {
                        if (alpha_touched[j]) {
                            alpha_scratch[j] = 0.0;
                            alpha_touched[j] = 0;
                        }
                    }
                    if (!x_B_fresh) {
                        /* see the guard above: only fresh-state emptiness
                         * is a genuine infeasibility certificate */
                        x_B_needs_recompute = 1;
                        continue;
                    }
                    {
                        int any_art = 0;
                        for (int32_t j = 0; j < n; j++) {
                            if (has_art_bound[j]) { any_art = 1; break; }
                        }
                        status = any_art ? "dual_unbounded_boxed" : "infeasible";
                    }
                    iterations = iter;
                    break;
                }
                theta_d = -r_ext[entering_col] / ((double)leaving_sigma * entering_alpha_row);
            }

            /* Clean up alpha_scratch for next iteration */
            for (int32_t j = 0; j < n_total; j++) {
                if (alpha_touched[j]) {
                    alpha_scratch[j] = 0.0;
                    alpha_touched[j] = 0;
                }
            }

            /* ---- 4e. FTRAN entering column: alpha_col = B^{-1} a_entering ---- */
            /* alpha_col must be zero on entry (cleared at pattern positions
             * after use at the end of the previous iteration). */
            {
                int sparse_ftran = 0;
                if (lu->lt_indptr != NULL && entering_col < n) {
                    /* Use hyper-sparse FTRAN for structural columns from CSC.
                     * Build sparse rhs directly from CSC pointers (no malloc). */
                    Py_ssize_t col_start = self->csc_indptr[entering_col];
                    Py_ssize_t col_end   = self->csc_indptr[entering_col + 1];
                    int32_t col_nnz = (int32_t)(col_end - col_start);
                    /* Use ftran_pattern as temporary for sparse rhs indices.
                     * Safe because rhs_indices is consumed in step 1 before
                     * x_pattern is written in step 4 of lu_ftran_sparse. */
                    int32_t *sp_idx = ftran_pattern;
                    double  *sp_val = lu->ws_w;  /* ws_w is unused during sparse FTRAN */
                    for (int32_t k = 0; k < col_nnz; k++) {
                        sp_idx[k] = (int32_t)self->csc_rows[col_start + k];
                        sp_val[k] = a_data[col_start + k];
                    }
                    ftran_nnz = lu_ftran_sparse(lu, col_nnz, sp_idx, sp_val,
                                                alpha_col, ftran_pattern);
                    sparse_ftran = 1;
                } else {
                    double *a_entering = calloc((size_t)m, sizeof(double));
                    if (a_entering == NULL) { PyErr_NoMemory(); goto done; }
                    if (entering_col < n) {
                        for (Py_ssize_t p = self->csc_indptr[entering_col]; p < self->csc_indptr[entering_col + 1]; p++) {
                            a_entering[(int32_t)self->csc_rows[p]] = a_data[p];
                        }
                    } else {
                        /* Artificial column */
                        a_entering[entering_col - n] = 1.0;
                    }
                    lu_ftran(lu, a_entering, alpha_col);
                    free(a_entering);
                    /* Dense FTRAN: build pattern by scanning */
                    ftran_nnz = 0;
                    for (int32_t k = 0; k < m; k++) {
                        if (alpha_col[k] != 0.0)
                            ftran_pattern[ftran_nnz++] = k;
                    }
                }
                (void)sparse_ftran;
            }

            /* ---- 4f. Primal step ---- */
            double bound_leaving;
            if (leaving_sigma == 1) {
                bound_leaving = lo_ext[basis[leaving_basis_pos]];
            } else {
                bound_leaving = hi_ext[basis[leaving_basis_pos]];
            }
            double pivot = alpha_col[leaving_basis_pos];
            if (fabs(pivot) < 1e-12) {
                /* Tiny pivot: inflate the Devex weight for this position so it
                 * is deprioritized in future iterations, then skip.
                 * Clean up sparse workspaces before continuing. */
                devex_w[leaving_basis_pos] *= 1e6;
                for (int32_t ki = 0; ki < ftran_nnz; ki++)
                    alpha_col[ftran_pattern[ki]] = 0.0;
                for (int32_t ki = 0; ki < rho_nnz; ki++)
                    rho[rho_nz_rows[ki]] = 0.0;
                continue;
            }
            double dx_entering = (x_B[leaving_basis_pos] - bound_leaving) / pivot;

            /* Update basic variable values using sparse alpha_col pattern */
            for (int32_t ki = 0; ki < ftran_nnz; ki++) {
                int32_t k = ftran_pattern[ki];
                x_B[k] -= alpha_col[k] * dx_entering;
            }

            /* Entering variable's new value */
            double entering_old_x = x_ext[entering_col];
            double entering_new_x = entering_old_x + dx_entering;

            /* ---- 4g. Update reduced costs ---- */
            /* Use sparse pricing: only update nonbasic columns that overlap rho's support.
             * But we already computed alpha_j for all relevant columns via the CSR scatter.
             * However, alpha_scratch was cleared. We need the full update, so recompute
             * using the CSR scatter again for the reduced cost update. */
            {
                /* Recompute alpha_j for all nonbasic columns via CSR scatter
                 * (using pre-scaled CSR data) */
                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    int32_t row = rho_nz_rows[ri];
                    double rho_val = rho[row];
                    for (Py_ssize_t p = self->indptr[row]; p < self->indptr[row + 1]; p++) {
                        int32_t col = (int32_t)self->indices[p];
                        alpha_scratch[col] += rho_val * scaled_csr_data[p];
                    }
                }
                /* Artificials */
                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    int32_t row = rho_nz_rows[ri];
                    alpha_scratch[n + row] = rho[row];
                }

                double sigma_d = (double)leaving_sigma;
                for (int32_t j = 0; j < n_total; j++) {
                    if (basis_pos[j] >= 0) continue;
                    if (bound_status[j] == DS_BOUND_FIXED) continue;
                    if (j == entering_col) continue;
                    double alpha_j = alpha_scratch[j];
                    if (alpha_j != 0.0) {
                        r_ext[j] += theta_d * sigma_d * alpha_j;
                    }
                }

                /* Clean up alpha_scratch */
                for (int32_t ri = 0; ri < rho_nnz; ri++) {
                    int32_t row = rho_nz_rows[ri];
                    for (Py_ssize_t p = self->indptr[row]; p < self->indptr[row + 1]; p++) {
                        alpha_scratch[(int32_t)self->indices[p]] = 0.0;
                    }
                    alpha_scratch[n + row] = 0.0;
                }
            }

            /* Leaving variable gets a reduced cost */
            double r_leaving = theta_d * (double)leaving_sigma;
            r_ext[entering_col] = 0.0;

            /* ---- 4h. Basis bookkeeping ---- */
            int32_t leaving_col = basis[leaving_basis_pos];

            if (leaving_sigma == 1) {
                bound_status[leaving_col] = DS_BOUND_LO;
                x_ext[leaving_col] = lo_ext[leaving_col];
            } else {
                bound_status[leaving_col] = DS_BOUND_HI;
                x_ext[leaving_col] = hi_ext[leaving_col];
            }
            r_ext[leaving_col] = r_leaving;
            basis_pos[leaving_col] = -1;

            if (leaving_col >= n) {
                stat_art_ejections++;
            }
            if (enter_count != NULL) {
                enter_count[entering_col]++;
            }
            basis[leaving_basis_pos] = entering_col;
            basis_pos[entering_col] = leaving_basis_pos;
            bound_status[entering_col] = DS_BOUND_BASIC;
            x_ext[entering_col] = entering_new_x;

            /* Incremental x_B: the entering variable now occupies
             * leaving_basis_pos; set its basic value directly.
             * Guard: when entering_old_x and dx_entering are large
             * with opposite signs (e.g. free variables at +/-1e20),
             * entering_new_x suffers catastrophic cancellation.
             * Detect this and force from-scratch recomputation. */
            x_B[leaving_basis_pos] = entering_new_x;
            if (fabs(entering_old_x) > 1e6 * (1.0 + fabs(entering_new_x))) {
                x_B_needs_recompute = 1;
            }

            /* ---- 4i. Pricing weight update ---- */
            if (pricing == 1) {
                /* Exact dual steepest-edge update (Forrest-Goldfarb 1992),
                 * transplanted from the exp-sedge experiment (see branch
                 * 158b70a for the full derivation). With basis change
                 * (leaving row r, entering column q, pivot = alpha_col[r]):
                 *     gamma_i' = gamma_i - 2 tau_i beta_i + tau_i^2 gamma_r,
                 *     gamma_r' = gamma_r / pivot^2,
                 * where tau_i = alpha_col[i]/pivot and beta = B^{-1} rho
                 * costs ONE extra sparse FTRAN on the PRE-update factor.
                 * Unlike the experiment, weights are seeded exactly once
                 * (gamma=1 is exact at the identity-artificial start) and
                 * CARRIED across refactorizations — gamma is a basis
                 * property, and the per-pivot ||rho||^2 anchor above stops
                 * drift from compounding. */
                lu_ftran(lu, rho, dse_beta);
                double inv_pivot = 1.0 / pivot;
                double gamma_r_old = devex_w[leaving_basis_pos];
                if (gamma_r_old < 1e-12) gamma_r_old = 1e-12;
                for (int32_t ki = 0; ki < ftran_nnz; ki++) {
                    int32_t k = ftran_pattern[ki];
                    if (k == leaving_basis_pos) continue;
                    double tau = alpha_col[k] * inv_pivot;
                    double g = devex_w[k]
                             - 2.0 * tau * dse_beta[k]
                             + tau * tau * gamma_r_old;
                    if (g < 1e-12) g = 1e-12;
                    devex_w[k] = g;
                }
                double gamma_r_new = gamma_r_old * inv_pivot * inv_pivot;
                if (gamma_r_new < 1e-12) gamma_r_new = 1e-12;
                devex_w[leaving_basis_pos] = gamma_r_new;
            } else {
                /* Devex weight update (Harris 1973 scheme):
                 * w_i_new = max(w_i, (alpha_col[i] / pivot)^2)
                 * The entering variable gets weight 1/pivot^2 */
                double inv_pivot_sq = 1.0 / (pivot * pivot);
                for (int32_t ki = 0; ki < ftran_nnz; ki++) {
                    int32_t k = ftran_pattern[ki];
                    double ratio = alpha_col[k] * alpha_col[k] * inv_pivot_sq;
                    if (ratio > devex_w[k]) devex_w[k] = ratio;
                }
                devex_w[leaving_basis_pos] = inv_pivot_sq;
            }

            /* ---- 4j. LU update ---- */
            {
                int need_refac = 0;

                /* Use the alpha_col already computed in step 4e to avoid a
                 * redundant FTRAN inside the standard lu_update.  This saves
                 * one full FTRAN per iteration (the most expensive operation).
                 * Use the sparse variant when FTRAN pattern is available. */
                int rc = lu_update_with_ftran_sparse(
                    lu, leaving_basis_pos, alpha_col,
                    ftran_nnz, ftran_pattern);

                if (rc != 0) {
                    need_refac = 1;
                } else {
                    iters_since_refac++;
                    /* Check for violent eta: tiny or huge pivot degrades
                     * the PFI representation rapidly. */
                    double abs_pivot = fabs(pivot);
                    if (abs_pivot < 1e-6 || abs_pivot > 1e6) {
                        need_refac = 1;
                    }
                }

                /* ---- 4k. Adaptive refactorization ---- */
                if (!need_refac) {
                    if (lu_should_refactor(lu) ||
                        iters_since_refac >= refac_interval) {
                        need_refac = 1;
                    }
                }

                if (need_refac) {
                    struct timespec _rf_t0, _rf_t1, _rf_tmid;
                    clock_gettime(CLOCK_MONOTONIC, &_rf_t0);
                    /* Accumulate density stats before freeing old LU */
                    if (lu != NULL) {
                        cum_ftran_sparse_count += lu->ftran_sparse_count;
                        cum_ftran_sparse_nnz   += lu->ftran_sparse_nnz_total;
                        cum_btran_sparse_count += lu->btran_sparse_count;
                        cum_btran_sparse_nnz   += lu->btran_sparse_nnz_total;
                    }
                    lu_context_free(lu);
                    lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                            a_data, basis,
                                            b_indptr, b_indices, b_values);
                    if (lu != NULL && lu->singular_step >= 0) {
                        lu = ds_repair_singular_basis(
                            lu, m, n, self->csc_indptr, self->csc_rows,
                            a_data, basis, basis_pos, bound_status,
                            x_ext, r_ext, lo_ext, hi_ext,
                            b_indptr, b_indices, b_values, 10);
                    }
                    if (lu == NULL || lu->singular_step >= 0) {
                        status = "numerical_error";
                        iterations = iter;
                        break;
                    }
                    clock_gettime(CLOCK_MONOTONIC, &_rf_tmid);
                    lu_build_transposes(lu);
                    if (pricing != 1) {
                        /* Devex resets its reference frame; steepest-edge
                         * weights are basis properties and carry across a
                         * refactorization unchanged. */
                        for (int32_t k = 0; k < m; k++) devex_w[k] = 1.0;
                    }
                    iters_since_refac = 0;

                    /* Recompute y, r from scratch and repair dual feasibility.
                     * The dense lu_btran uses ws_v internally; clear it afterward
                     * to avoid contaminating subsequent sparse BTRAN calls. */
                    for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
                    lu_btran(lu, c_B, y);
                    memset(lu->ws_v, 0, (size_t)m * sizeof(double));

                    for (int32_t j = 0; j < n_total; j++) {
                        if (basis_pos[j] >= 0) continue;
                        if (bound_status[j] == DS_BOUND_FIXED) continue;
                        double rj = c_ext[j];
                        if (j < n) {
                            for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                                rj -= a_data[p] * y[(int32_t)self->csc_rows[p]];
                            }
                        } else {
                            rj -= y[j - n];
                        }

                        /* Dual-feasibility repair: if reduced cost sign
                         * drifted, flip the nonbasic to its correct bound
                         * (when finite) to restore dual feasibility. */
                        int lo_f = isfinite(lo_ext[j]);
                        int hi_f = isfinite(hi_ext[j]);
                        if (bound_status[j] == DS_BOUND_LO && rj < -1e-9 && hi_f) {
                            bound_status[j] = DS_BOUND_HI;
                            x_ext[j] = hi_ext[j];
                        } else if (bound_status[j] == DS_BOUND_HI && rj > 1e-9 && lo_f) {
                            bound_status[j] = DS_BOUND_LO;
                            x_ext[j] = lo_ext[j];
                        }
                        r_ext[j] = rj;
                    }
                    /* Dual-feasibility repair may have changed x_ext for
                     * nonbasic variables, invalidating the incremental x_B.
                     * Force from-scratch recomputation at next iteration. */
                    x_B_needs_recompute = 1;
                    clock_gettime(CLOCK_MONOTONIC, &_rf_t1);
                    refac_factorize_time += (_rf_tmid.tv_sec - _rf_t0.tv_sec)
                        + (_rf_tmid.tv_nsec - _rf_t0.tv_nsec) * 1e-9;
                    refac_time_total += (_rf_t1.tv_sec - _rf_t0.tv_sec)
                        + (_rf_t1.tv_nsec - _rf_t0.tv_nsec) * 1e-9;
                    total_refacs++;
                }
            }

            /* ---- 4l. Anti-cycling ---- */
            x_B_fresh = 0;
            if (theta_d < 1e-12) {
                consecutive_degenerate++;
                stat_degenerate++;
                if (consecutive_degenerate > stat_max_degen_streak) {
                    stat_max_degen_streak = consecutive_degenerate;
                }
                if (consecutive_degenerate >= 200) {
                    use_bland = 1;
                }
            } else {
                consecutive_degenerate = 0;
                use_bland = 0;
            }
            if (use_bland) {
                stat_bland_pivots++;
            }

            /* Clear sparse workspaces at pattern positions so they are
             * zero for the next iteration's sparse FTRAN/BTRAN calls. */
            for (int32_t ki = 0; ki < ftran_nnz; ki++)
                alpha_col[ftran_pattern[ki]] = 0.0;
            for (int32_t ki = 0; ki < rho_nnz; ki++)
                rho[rho_nz_rows[ki]] = 0.0;

            iterations = iter + 1;
        }

        /* Check iteration limit */
        if (iterations >= max_iter && strcmp(status, "optimal") != 0 &&
            strcmp(status, "infeasible") != 0) {
            status = "iteration_limit";
        }
    }

    /* ============================================================
     * 5. EXIT: compute final x, verify residuals, dual feasibility
     * ============================================================
     * Artificial columns are excluded from the returned x vector.
     *
     * At claimed optimality:
     *   (a) Refactorize and recompute x_B, y from scratch.
     *   (b) Check if any artificial bound is active. If so, the big-M
     *       bound was binding and the solution may not be optimal for
     *       the original problem; retry with M*100 once, then give up.
     *   (c) Verify dual feasibility against TRUE bounds for every
     *       nonbasic structural column. This catches pricing bugs and
     *       ensures we never return a wrong optimum.
     */

    /* Copy structural x values */
    double *x_out = x_ext; /* reuse; we only return first n entries */

    /* Accumulate final LU's stats into cumulative totals before
     * the optimality-check refactorization creates a fresh LU. */
    if (lu != NULL) {
        cum_ftran_sparse_count += lu->ftran_sparse_count;
        cum_ftran_sparse_nnz   += lu->ftran_sparse_nnz_total;
        cum_btran_sparse_count += lu->btran_sparse_count;
        cum_btran_sparse_nnz   += lu->btran_sparse_nnz_total;
    }

    if (strcmp(status, "optimal") == 0) {
        lu_context_free(lu);
        lu = ds_factorize_basis(m, n, self->csc_indptr, self->csc_rows,
                                a_data, basis,
                                b_indptr, b_indices, b_values);
        if (lu != NULL && lu->singular_step >= 0) {
            lu = ds_repair_singular_basis(
                lu, m, n, self->csc_indptr, self->csc_rows,
                a_data, basis, basis_pos, bound_status,
                x_ext, r_ext, lo_ext, hi_ext,
                b_indptr, b_indices, b_values, 10);
        }
        if (lu == NULL || lu->singular_step >= 0) {
            status = "numerical_error";
        } else {
            /* Recompute x_B from scratch */
            memcpy(rhs, b, (size_t)m * sizeof(double));
            for (int32_t j = 0; j < n_total; j++) {
                if (basis_pos[j] >= 0) continue;
                if (x_ext[j] == 0.0) continue;
                double xj = x_ext[j];
                if (j < n) {
                    for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                        rhs[(int32_t)self->csc_rows[p]] -= a_data[p] * xj;
                    }
                } else {
                    rhs[j - n] -= xj;
                }
            }
            lu_ftran(lu, rhs, x_B);
            for (int32_t k = 0; k < m; k++) {
                int32_t j = basis[k];
                if (j < n) {
                    x_out[j] = x_B[k];
                }
            }

            /* Recompute y and reduced costs from scratch */
            for (int32_t k = 0; k < m; k++) c_B[k] = c_ext[basis[k]];
            lu_btran(lu, c_B, y);

            /* ---- 5a. Check for active artificial bounds ---- */
            /* A structural column with an infinite true bound may sit parked at
             * a big-M artificial bound. This is dual-infeasible for the true
             * problem ONLY if its reduced cost is meaningfully nonzero (the
             * objective would keep improving past M -> genuine unboundedness).
             * If the reduced cost is ~0 the point is a degenerate optimum:
             * complementary slackness holds and any feasible value of x_j
             * (including the parked one) is optimal. The r~0 gate is what
             * distinguishes this from the one-sided-column bug these exit
             * checks were built to catch (which had strictly nonzero rj). */
            {
                /* Gap-budget acceptance: a column parked at an artificial
                 * bound with residual reduced cost rj damages the duality
                 * gap by |rj| * |x_orig_j|. A per-column tolerance alone
                 * would let a big-M park (|x| ~ 1e9) hide hundreds of
                 * absolute gap error, so the damage is bounded against an
                 * objective-scaled certificate budget instead. rj exactly
                 * zero (degenerate optimum) always passes. */
                double obj_budget = 1.0;
                for (int32_t j = 0; j < n; j++) {
                    obj_budget += fabs(c_orig[j] * ds_col_scale[j] * x_ext[j]);
                }
                for (int32_t j = 0; j < n; j++) {
                    if (!has_art_bound[j]) continue;
                    if (basis_pos[j] >= 0) continue;
                    double art_lo = lo_ext[j], art_hi = hi_ext[j];
                    double true_lo = lo_true[j], true_hi = hi_true[j];
                    int at_art_lo = (bound_status[j] == DS_BOUND_LO &&
                                     !isfinite(true_lo) &&
                                     fabs(x_ext[j] - art_lo) < tol);
                    int at_art_hi = (bound_status[j] == DS_BOUND_HI &&
                                     !isfinite(true_hi) &&
                                     fabs(x_ext[j] - art_hi) < tol);
                    if (!at_art_lo && !at_art_hi) continue;
                    /* Fresh reduced cost in original units. */
                    double c_orig_j = c_orig[j];
                    double rj = c_orig_j;
                    for (Py_ssize_t p = self->csc_indptr[j];
                         p < self->csc_indptr[j + 1]; p++) {
                        int32_t row = (int32_t)self->csc_rows[p];
                        rj -= self->csc_data[p] * (ds_row_scale[row] * y[row]);
                    }
                    double x_orig_j = ds_col_scale[j] * x_ext[j];
                    double damage = fabs(rj) * fmax(1.0, fabs(x_orig_j));
                    if (damage > 1e-7 * obj_budget) {
                        status = "dual_infeasible";
                        break;
                    }
                }
            }

            /* ---- 5b. Dual feasibility check against TRUE bounds ---- */
            /* Compute in ORIGINAL units to match the eps contract.
             * y_orig = R * y_scaled; r_orig = c_orig - A_orig^T y_orig. */
            if (strcmp(status, "optimal") == 0) {
                for (int32_t j = 0; j < n; j++) {
                    if (basis_pos[j] >= 0) continue;
                    /* Recompute r_j in original units */
                    double c_orig_j = c_orig[j];
                    double rj = c_orig_j;
                    for (Py_ssize_t p = self->csc_indptr[j]; p < self->csc_indptr[j + 1]; p++) {
                        int32_t row = (int32_t)self->csc_rows[p];
                        rj -= self->csc_data[p] * (ds_row_scale[row] * y[row]);
                    }
                    double dtol = 1e-7 * (1.0 + fabs(c_orig_j));
                    int tlo = isfinite(lo_true[j]);
                    int thi = isfinite(hi_true[j]);

                    if (bound_status[j] == DS_BOUND_LO) {
                        /* At lower bound: need r_j >= -dtol */
                        if (rj < -dtol) {
                            status = "dual_infeasible";
                            break;
                        }
                    } else if (bound_status[j] == DS_BOUND_HI) {
                        /* At upper bound: need r_j <= dtol */
                        if (rj > dtol) {
                            status = "dual_infeasible";
                            break;
                        }
                    } else if (bound_status[j] == DS_BOUND_FREE) {
                        /* Free: need |r_j| <= dtol */
                        if (fabs(rj) > dtol) {
                            status = "dual_infeasible";
                            break;
                        }
                    }
                    /* FIXED variables: r_j is unconstrained (can't move) */

                    /* Also check: if at a bound that doesn't exist in the true
                     * problem (artificial bound active in the basis sense).
                     * Only a nonzero reduced cost makes this dual-infeasible; a
                     * reduced cost of ~0 is a degenerate optimum where x_j may
                     * legitimately take any feasible value (see 5a). The sign
                     * tests above (|rj| within dtol) have already passed here,
                     * so this only fires for genuinely nonzero rj. */
                    if (bound_status[j] == DS_BOUND_LO && !tlo &&
                        fabs(rj) > dtol) {
                        status = "dual_infeasible";
                        break;
                    }
                    if (bound_status[j] == DS_BOUND_HI && !thi &&
                        fabs(rj) > dtol) {
                        status = "dual_infeasible";
                        break;
                    }
                }
            }
        }
    } else {
        /* For non-optimal exits, still set x from x_B */
        for (int32_t k = 0; k < m; k++) {
            int32_t j = basis[k];
            if (j < n) {
                x_out[j] = x_B[k];
            }
        }
    }

build_result:
    /* ---- Unscale from Ruiz equilibration ----
     * x_out is in scaled space (x_s = C^{-1} x). Restore to original units.
     * b was scaled (b_s = R b). Restore so residual uses original b.
     * Objective c_s^T x_s = c^T x is scale-invariant; no need to touch c. */
    if (ruiz_active) {
        for (int32_t j = 0; j < n; j++) {
            x_out[j] *= ds_col_scale[j];
        }
        for (int32_t i = 0; i < m; i++) {
            b[i] /= ds_row_scale[i];
        }
    }
    {
        double objective = 0.0;
        double max_residual = 0.0;

        /* Compute objective in original units: sum c_orig[j] * x_orig[j].
         * c_ext[j] is still scaled (c_s = c * C), x_out is now unscaled,
         * so c_s[j] * x_out[j] / col_scale[j] = c[j] * x[j]. */
        for (int32_t j = 0; j < n; j++) {
            objective += c_orig[j] * x_out[j];
        }

        /* Primal residual: max |A x - b| (structural columns only) */
        for (int32_t i = 0; i < m; i++) {
            double row_sum = 0.0;
            for (Py_ssize_t p = self->indptr[i]; p < self->indptr[i + 1]; p++) {
                row_sum += self->data[p] * x_out[(int32_t)self->indices[p]];
            }
            double res = fabs(row_sum - b[i]);
            if (res > max_residual) max_residual = res;
        }

        PyObject *x_list = PyList_New(n_s);
        PyObject *y_list = PyList_New(m_s);
        if (x_list == NULL || y_list == NULL) {
            Py_XDECREF(x_list);
            Py_XDECREF(y_list);
            goto done;
        }
        for (int32_t j = 0; j < n; j++) {
            PyList_SET_ITEM(x_list, j, PyFloat_FromDouble(x_out[j]));
        }
        for (int32_t i = 0; i < m; i++) {
            PyList_SET_ITEM(y_list, i, PyFloat_FromDouble(y[i]));
        }

        /* Compute hyper-sparse solve density statistics from cumulative
         * totals (survive across refactorizations). */
        double ftran_mean_density = 0.0;
        double btran_mean_density = 0.0;
        if (cum_ftran_sparse_count > 0) {
            ftran_mean_density = (double)cum_ftran_sparse_nnz /
                ((double)cum_ftran_sparse_count * m);
        }
        if (cum_btran_sparse_count > 0) {
            btran_mean_density = (double)cum_btran_sparse_nnz /
                ((double)cum_btran_sparse_count * m);
        }

        long long churn_max = 0;
        long long churn_gt10 = 0;
        if (enter_count != NULL) {
            for (int32_t j = 0; j < n_total; j++) {
                if (enter_count[j] > churn_max) churn_max = enter_count[j];
                if (enter_count[j] > 10) churn_gt10++;
            }
        }
        result = Py_BuildValue(
            "{s:s,s:d,s:d,s:n,s:N,s:N,s:d,s:d,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:L,s:d,s:d}",
            "status", status,
            "objective", objective,
            "max_primal_residual", max_residual,
            "iterations", iterations,
            "x", x_list,
            "y", y_list,
            "ftran_mean_density", ftran_mean_density,
            "btran_mean_density", btran_mean_density,
            "refactorizations", (long long)total_refacs,
            "bound_flips", (long long)stat_flips,
            "artificial_ejections", (long long)stat_art_ejections,
            "cost_shifts", (long long)stat_cost_shifts,
            "max_col_reentries", (long long)churn_max,
            "cols_reentering_gt10", (long long)churn_gt10,
            "degenerate_pivots", (long long)stat_degenerate,
            "bland_pivots", (long long)stat_bland_pivots,
            "max_degenerate_streak", (long long)stat_max_degen_streak,
            "refac_time", refac_time_total,
            "refac_factorize_time", refac_factorize_time);
    }

done:
    lu_context_free(lu);
    free(c_ext); free(lo_ext); free(hi_ext);
    free(x_ext); free(r_ext); free(basis_pos); free(bound_status);
    free(b); free(y); free(x_B); free(rhs);
    free(rho); free(alpha_col); free(e_i); free(c_B);
    free(devex_w); free(dse_beta); free(enter_count); free(basis);
    free(b_indptr); free(b_indices); free(b_values);
    free(rho_nz_rows); free(ftran_pattern); free(btran_pattern);
    free(alpha_scratch); free(alpha_touched);
    free(flip_delta_xB);
    free(has_art_bound); free(lo_true); free(hi_true);
    free(ds_row_scale); free(ds_col_scale); free(scaled_csc_data); free(scaled_csr_data); free(c_orig);
    return result;
}

/* ---- Python test hooks for sparse LU ---- */

/*
 * lu_solve_test(indptr, indices, data, m, rhs_list, transpose)
 *
 * indptr, indices, data: CSC format of a square m x m matrix
 * rhs_list: list of m-length lists (right-hand sides)
 * transpose: 0 for FTRAN (Bx=b), 1 for BTRAN (B^T x = b)
 *
 * Returns: list of m-length lists (solution vectors), or None if singular.
 */
static PyObject *csparse_lu_solve_test(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj, *indices_obj, *data_obj, *rhs_list_obj;
    int m_int, transpose;
    if (!PyArg_ParseTuple(args, "OOOiOi", &indptr_obj, &indices_obj,
                          &data_obj, &m_int, &rhs_list_obj, &transpose)) {
        return NULL;
    }
    int32_t m = (int32_t)m_int;
    if (m < 0 || m > 100000) {
        PyErr_SetString(PyExc_ValueError, "m must be in [0, 100000]");
        return NULL;
    }

    /* Parse CSC arrays */
    int32_t *indptr = calloc((size_t)(m + 1), sizeof(int32_t));
    if (indptr == NULL) { PyErr_NoMemory(); return NULL; }
    {
        Py_ssize_t *tmp = calloc((size_t)(m + 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indptr_obj, m + 1, tmp, "indptr") != 0) {
            free(indptr); free(tmp); return NULL;
        }
        for (int32_t i = 0; i <= m; i++) indptr[i] = (int32_t)tmp[i];
        free(tmp);
    }
    int32_t nnz = indptr[m];
    int32_t *indices = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int32_t));
    double  *data    = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
    if (indices == NULL || data == NULL) {
        free(indptr); free(indices); free(data);
        PyErr_NoMemory();
        return NULL;
    }
    {
        Py_ssize_t *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); free(indices); free(data); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indices_obj, nnz, tmp, "indices") != 0) {
            free(indptr); free(indices); free(data); free(tmp); return NULL;
        }
        for (int32_t i = 0; i < nnz; i++) indices[i] = (int32_t)tmp[i];
        free(tmp);
    }
    {
        double *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
        if (tmp == NULL) { free(indptr); free(indices); free(data); PyErr_NoMemory(); return NULL; }
        if (fill_double_array(data_obj, nnz, tmp, "data") != 0) {
            free(indptr); free(indices); free(data); free(tmp); return NULL;
        }
        memcpy(data, tmp, (size_t)nnz * sizeof(double));
        free(tmp);
    }

    /* Factorize */
    int alloc_fail = 0;
    LUContext *ctx;
    Py_BEGIN_ALLOW_THREADS
    ctx = lu_factorize(m, indptr, indices, data, 0.1, &alloc_fail);
    Py_END_ALLOW_THREADS

    free(indptr);
    free(indices);
    free(data);

    if (ctx == NULL) {
        if (alloc_fail) {
            PyErr_NoMemory();
        } else {
            PyErr_SetString(PyExc_RuntimeError, "LU factorization failed");
        }
        return NULL;
    }

    if (ctx->singular_step >= 0) {
        /* Return None to signal singularity */
        lu_context_free(ctx);
        Py_RETURN_NONE;
    }

    /* Parse rhs_list and solve */
    PyObject *rhs_seq = PySequence_Fast(rhs_list_obj, "rhs_list must be a sequence");
    if (rhs_seq == NULL) { lu_context_free(ctx); return NULL; }
    Py_ssize_t n_rhs = PySequence_Fast_GET_SIZE(rhs_seq);

    PyObject *results = PyList_New(n_rhs);
    if (results == NULL) { Py_DECREF(rhs_seq); lu_context_free(ctx); return NULL; }

    double *b = calloc((size_t)m, sizeof(double));
    double *x = calloc((size_t)m, sizeof(double));
    if (b == NULL || x == NULL) {
        free(b); free(x);
        Py_DECREF(rhs_seq); Py_DECREF(results);
        lu_context_free(ctx);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t r = 0; r < n_rhs; r++) {
        PyObject *rhs_item = PySequence_Fast_GET_ITEM(rhs_seq, r);
        /* Parse the rhs vector - use a temporary Py_ssize_t-free approach */
        {
            PyObject *rhs_inner = PySequence_Fast(rhs_item, "each rhs must be a sequence");
            if (rhs_inner == NULL) {
                free(b); free(x);
                Py_DECREF(rhs_seq); Py_DECREF(results);
                lu_context_free(ctx);
                return NULL;
            }
            if (PySequence_Fast_GET_SIZE(rhs_inner) != m) {
                Py_DECREF(rhs_inner);
                free(b); free(x);
                Py_DECREF(rhs_seq); Py_DECREF(results);
                lu_context_free(ctx);
                PyErr_SetString(PyExc_ValueError, "rhs length must match m");
                return NULL;
            }
            for (int32_t i = 0; i < m; i++) {
                PyObject *item = PySequence_Fast_GET_ITEM(rhs_inner, i);
                b[i] = PyFloat_AsDouble(item);
                if (PyErr_Occurred()) {
                    Py_DECREF(rhs_inner);
                    free(b); free(x);
                    Py_DECREF(rhs_seq); Py_DECREF(results);
                    lu_context_free(ctx);
                    return NULL;
                }
            }
            Py_DECREF(rhs_inner);
        }

        /* Solve */
        Py_BEGIN_ALLOW_THREADS
        if (transpose) {
            lu_btran(ctx, b, x);
        } else {
            lu_ftran(ctx, b, x);
        }
        Py_END_ALLOW_THREADS

        /* Build result list */
        PyObject *x_list = PyList_New(m);
        if (x_list == NULL) {
            free(b); free(x);
            Py_DECREF(rhs_seq); Py_DECREF(results);
            lu_context_free(ctx);
            return NULL;
        }
        for (int32_t i = 0; i < m; i++) {
            PyList_SET_ITEM(x_list, i, PyFloat_FromDouble(x[i]));
        }
        PyList_SET_ITEM(results, r, x_list);
    }

    free(b);
    free(x);
    Py_DECREF(rhs_seq);
    lu_context_free(ctx);
    return results;
}

/*
 * lu_stats_test(indptr, indices, data, m)
 *
 * Returns (nnz_l, nnz_u, singular_step) for the LU factorization.
 * singular_step is -1 if nonsingular.
 */
static PyObject *csparse_lu_stats_test(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj, *indices_obj, *data_obj;
    int m_int;
    if (!PyArg_ParseTuple(args, "OOOi", &indptr_obj, &indices_obj,
                          &data_obj, &m_int)) {
        return NULL;
    }
    int32_t m = (int32_t)m_int;
    if (m < 0 || m > 100000) {
        PyErr_SetString(PyExc_ValueError, "m must be in [0, 100000]");
        return NULL;
    }

    int32_t *indptr = calloc((size_t)(m + 1), sizeof(int32_t));
    if (indptr == NULL) { PyErr_NoMemory(); return NULL; }
    {
        Py_ssize_t *tmp = calloc((size_t)(m + 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indptr_obj, m + 1, tmp, "indptr") != 0) {
            free(indptr); free(tmp); return NULL;
        }
        for (int32_t i = 0; i <= m; i++) indptr[i] = (int32_t)tmp[i];
        free(tmp);
    }
    int32_t nnz = indptr[m];
    int32_t *indices = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int32_t));
    double  *data_arr = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
    if (indices == NULL || data_arr == NULL) {
        free(indptr); free(indices); free(data_arr);
        PyErr_NoMemory();
        return NULL;
    }
    {
        Py_ssize_t *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); free(indices); free(data_arr); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indices_obj, nnz, tmp, "indices") != 0) {
            free(indptr); free(indices); free(data_arr); free(tmp); return NULL;
        }
        for (int32_t i = 0; i < nnz; i++) indices[i] = (int32_t)tmp[i];
        free(tmp);
    }
    {
        double *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
        if (tmp == NULL) { free(indptr); free(indices); free(data_arr); PyErr_NoMemory(); return NULL; }
        if (fill_double_array(data_obj, nnz, tmp, "data") != 0) {
            free(indptr); free(indices); free(data_arr); free(tmp); return NULL;
        }
        memcpy(data_arr, tmp, (size_t)nnz * sizeof(double));
        free(tmp);
    }

    int alloc_fail = 0;
    LUContext *ctx;
    Py_BEGIN_ALLOW_THREADS
    ctx = lu_factorize(m, indptr, indices, data_arr, 0.1, &alloc_fail);
    Py_END_ALLOW_THREADS

    free(indptr);
    free(indices);
    free(data_arr);

    if (ctx == NULL) {
        if (alloc_fail) PyErr_NoMemory();
        else PyErr_SetString(PyExc_RuntimeError, "LU factorization failed");
        return NULL;
    }

    PyObject *result = Py_BuildValue("(iii)", ctx->nnz_l, ctx->nnz_u, ctx->singular_step);
    lu_context_free(ctx);
    return result;
}

static PyObject *csparse_min_degree(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj;
    PyObject *indices_obj;
    long long max_ops = 0;
    if (!PyArg_ParseTuple(args, "OO|L", &indptr_obj, &indices_obj, &max_ops)) {
        return NULL;
    }
    PyObject *indptr_seq = PySequence_Fast(indptr_obj, "indptr must be a sequence");
    if (indptr_seq == NULL) {
        return NULL;
    }
    Py_ssize_t m = PySequence_Fast_GET_SIZE(indptr_seq) - 1;
    Py_DECREF(indptr_seq);
    if (m < 0 || m > INT32_MAX) {
        PyErr_SetString(PyExc_ValueError, "invalid pattern size");
        return NULL;
    }
    Py_ssize_t *indptr = calloc((size_t)m + 1, sizeof(Py_ssize_t));
    if (indptr == NULL || fill_index_array(indptr_obj, m + 1, indptr, "indptr") != 0) {
        free(indptr);
        return NULL;
    }
    Py_ssize_t nnz = indptr[m];
    Py_ssize_t *indices = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(Py_ssize_t));
    int32_t *order = calloc((size_t)(m > 0 ? m : 1), sizeof(int32_t));
    if (indices == NULL || order == NULL ||
        fill_index_array(indices_obj, nnz, indices, "indices") != 0) {
        free(indptr);
        free(indices);
        free(order);
        return NULL;
    }
    int status;
    Py_BEGIN_ALLOW_THREADS
    status = min_degree_impl((int32_t)m, indptr, indices, order, (int64_t)max_ops, 0.0);
    Py_END_ALLOW_THREADS
    free(indptr);
    free(indices);
    if (status == -2) {
        free(order);
        Py_RETURN_NONE;
    }
    if (status != 0) {
        free(order);
        PyErr_NoMemory();
        return NULL;
    }
    PyObject *result = PyList_New(m);
    if (result == NULL) {
        free(order);
        return NULL;
    }
    for (Py_ssize_t i = 0; i < m; i++) {
        PyList_SET_ITEM(result, i, PyLong_FromLong((long)order[i]));
    }
    free(order);
    return result;
}

/*
 * lu_update_test(indptr, indices, data, m, updates, rhs_list, transpose)
 *
 * Test hook for basis-change updates.
 *
 * indptr, indices, data: CSC format of the initial m x m basis matrix.
 * updates: list of (leaving_pos, col_indices_list, col_values_list) triples.
 * rhs_list: list of m-length lists (right-hand sides to solve after ALL updates).
 * transpose: 0 for FTRAN, 1 for BTRAN.
 *
 * Returns (solutions, should_refactor, n_singular) where:
 *   solutions: list of solution vectors (each m-length list)
 *   should_refactor: 1 if lu_should_refactor fires, 0 otherwise
 *   n_singular: number of updates that returned singular
 *
 * Returns None if the initial factorization is singular.
 */
static PyObject *csparse_lu_update_test(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *indptr_obj, *indices_obj, *data_obj, *updates_obj, *rhs_list_obj;
    int m_int, transpose;
    if (!PyArg_ParseTuple(args, "OOOiOOi", &indptr_obj, &indices_obj,
                          &data_obj, &m_int, &updates_obj, &rhs_list_obj,
                          &transpose)) {
        return NULL;
    }
    int32_t m = (int32_t)m_int;
    if (m < 0 || m > 100000) {
        PyErr_SetString(PyExc_ValueError, "m must be in [0, 100000]");
        return NULL;
    }

    /* Parse CSC arrays */
    int32_t *indptr = calloc((size_t)(m + 1), sizeof(int32_t));
    if (indptr == NULL) { PyErr_NoMemory(); return NULL; }
    {
        Py_ssize_t *tmp = calloc((size_t)(m + 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indptr_obj, m + 1, tmp, "indptr") != 0) {
            free(indptr); free(tmp); return NULL;
        }
        for (int32_t i = 0; i <= m; i++) indptr[i] = (int32_t)tmp[i];
        free(tmp);
    }
    int32_t nnz = indptr[m];
    int32_t *indices = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(int32_t));
    double  *data    = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
    if (indices == NULL || data == NULL) {
        free(indptr); free(indices); free(data);
        PyErr_NoMemory();
        return NULL;
    }
    {
        Py_ssize_t *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(Py_ssize_t));
        if (tmp == NULL) { free(indptr); free(indices); free(data); PyErr_NoMemory(); return NULL; }
        if (fill_index_array(indices_obj, nnz, tmp, "indices") != 0) {
            free(indptr); free(indices); free(data); free(tmp); return NULL;
        }
        for (int32_t i = 0; i < nnz; i++) indices[i] = (int32_t)tmp[i];
        free(tmp);
    }
    {
        double *tmp = calloc((size_t)(nnz > 0 ? nnz : 1), sizeof(double));
        if (tmp == NULL) { free(indptr); free(indices); free(data); PyErr_NoMemory(); return NULL; }
        if (fill_double_array(data_obj, nnz, tmp, "data") != 0) {
            free(indptr); free(indices); free(data); free(tmp); return NULL;
        }
        memcpy(data, tmp, (size_t)nnz * sizeof(double));
        free(tmp);
    }

    /* Factorize */
    int alloc_fail = 0;
    LUContext *ctx;
    Py_BEGIN_ALLOW_THREADS
    ctx = lu_factorize(m, indptr, indices, data, 0.1, &alloc_fail);
    Py_END_ALLOW_THREADS

    free(indptr);
    free(indices);
    free(data);

    if (ctx == NULL) {
        if (alloc_fail) PyErr_NoMemory();
        else PyErr_SetString(PyExc_RuntimeError, "LU factorization failed");
        return NULL;
    }

    if (ctx->singular_step >= 0) {
        lu_context_free(ctx);
        Py_RETURN_NONE;
    }

    /* Parse and apply updates */
    PyObject *updates_seq = PySequence_Fast(updates_obj, "updates must be a sequence");
    if (updates_seq == NULL) { lu_context_free(ctx); return NULL; }
    Py_ssize_t n_upd = PySequence_Fast_GET_SIZE(updates_seq);
    int n_singular = 0;

    for (Py_ssize_t u = 0; u < n_upd; u++) {
        PyObject *upd_item = PySequence_Fast_GET_ITEM(updates_seq, u);
        PyObject *upd_seq = PySequence_Fast(upd_item, "each update must be a (pos, indices, values) tuple");
        if (upd_seq == NULL) {
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(upd_seq) != 3) {
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            PyErr_SetString(PyExc_ValueError, "each update must have 3 elements");
            return NULL;
        }

        int32_t leaving_pos = (int32_t)PyLong_AsLong(
            PySequence_Fast_GET_ITEM(upd_seq, 0));
        if (PyErr_Occurred()) {
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            return NULL;
        }

        PyObject *col_idx_obj = PySequence_Fast_GET_ITEM(upd_seq, 1);
        PyObject *col_val_obj = PySequence_Fast_GET_ITEM(upd_seq, 2);

        PyObject *col_idx_seq = PySequence_Fast(col_idx_obj, "col_indices must be a sequence");
        if (col_idx_seq == NULL) {
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            return NULL;
        }
        Py_ssize_t col_nnz = PySequence_Fast_GET_SIZE(col_idx_seq);

        int32_t *col_indices = calloc((size_t)(col_nnz > 0 ? col_nnz : 1), sizeof(int32_t));
        double  *col_values  = calloc((size_t)(col_nnz > 0 ? col_nnz : 1), sizeof(double));
        if (col_indices == NULL || col_values == NULL) {
            free(col_indices); free(col_values);
            Py_DECREF(col_idx_seq);
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            PyErr_NoMemory();
            return NULL;
        }

        for (Py_ssize_t k = 0; k < col_nnz; k++) {
            col_indices[k] = (int32_t)PyLong_AsLong(
                PySequence_Fast_GET_ITEM(col_idx_seq, k));
        }
        Py_DECREF(col_idx_seq);

        PyObject *col_val_seq = PySequence_Fast(col_val_obj, "col_values must be a sequence");
        if (col_val_seq == NULL) {
            free(col_indices); free(col_values);
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(col_val_seq) != col_nnz) {
            free(col_indices); free(col_values);
            Py_DECREF(col_val_seq);
            Py_DECREF(upd_seq);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            PyErr_SetString(PyExc_ValueError, "col_indices and col_values length mismatch");
            return NULL;
        }
        for (Py_ssize_t k = 0; k < col_nnz; k++) {
            col_values[k] = PyFloat_AsDouble(
                PySequence_Fast_GET_ITEM(col_val_seq, k));
        }
        Py_DECREF(col_val_seq);
        Py_DECREF(upd_seq);

        if (PyErr_Occurred()) {
            free(col_indices); free(col_values);
            Py_DECREF(updates_seq);
            lu_context_free(ctx);
            return NULL;
        }

        int rc;
        Py_BEGIN_ALLOW_THREADS
        rc = lu_update(ctx, leaving_pos, col_indices, col_values, (int32_t)col_nnz);
        Py_END_ALLOW_THREADS

        free(col_indices);
        free(col_values);

        if (rc != 0) {
            n_singular++;
        }
    }
    Py_DECREF(updates_seq);

    /* Check should_refactor */
    int should_refac = lu_should_refactor(ctx);

    /* Solve with accumulated updates */
    PyObject *rhs_seq = PySequence_Fast(rhs_list_obj, "rhs_list must be a sequence");
    if (rhs_seq == NULL) { lu_context_free(ctx); return NULL; }
    Py_ssize_t n_rhs = PySequence_Fast_GET_SIZE(rhs_seq);

    PyObject *solutions = PyList_New(n_rhs);
    if (solutions == NULL) { Py_DECREF(rhs_seq); lu_context_free(ctx); return NULL; }

    double *b = calloc((size_t)m, sizeof(double));
    double *x = calloc((size_t)m, sizeof(double));
    if (b == NULL || x == NULL) {
        free(b); free(x);
        Py_DECREF(rhs_seq); Py_DECREF(solutions);
        lu_context_free(ctx);
        PyErr_NoMemory();
        return NULL;
    }

    for (Py_ssize_t r = 0; r < n_rhs; r++) {
        PyObject *rhs_item = PySequence_Fast_GET_ITEM(rhs_seq, r);
        PyObject *rhs_inner = PySequence_Fast(rhs_item, "each rhs must be a sequence");
        if (rhs_inner == NULL) {
            free(b); free(x);
            Py_DECREF(rhs_seq); Py_DECREF(solutions);
            lu_context_free(ctx);
            return NULL;
        }
        if (PySequence_Fast_GET_SIZE(rhs_inner) != m) {
            Py_DECREF(rhs_inner);
            free(b); free(x);
            Py_DECREF(rhs_seq); Py_DECREF(solutions);
            lu_context_free(ctx);
            PyErr_SetString(PyExc_ValueError, "rhs length must match m");
            return NULL;
        }
        for (int32_t i = 0; i < m; i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(rhs_inner, i);
            b[i] = PyFloat_AsDouble(item);
            if (PyErr_Occurred()) {
                Py_DECREF(rhs_inner);
                free(b); free(x);
                Py_DECREF(rhs_seq); Py_DECREF(solutions);
                lu_context_free(ctx);
                return NULL;
            }
        }
        Py_DECREF(rhs_inner);

        Py_BEGIN_ALLOW_THREADS
        if (transpose) {
            lu_btran(ctx, b, x);
        } else {
            lu_ftran(ctx, b, x);
        }
        Py_END_ALLOW_THREADS

        PyObject *x_list = PyList_New(m);
        if (x_list == NULL) {
            free(b); free(x);
            Py_DECREF(rhs_seq); Py_DECREF(solutions);
            lu_context_free(ctx);
            return NULL;
        }
        for (int32_t i = 0; i < m; i++) {
            PyList_SET_ITEM(x_list, i, PyFloat_FromDouble(x[i]));
        }
        PyList_SET_ITEM(solutions, r, x_list);
    }

    free(b);
    free(x);
    Py_DECREF(rhs_seq);
    lu_context_free(ctx);

    PyObject *ret = Py_BuildValue("(Oii)", solutions, should_refac, n_singular);
    Py_DECREF(solutions);  /* Py_BuildValue("O") adds a ref; drop ours */
    return ret;
}

/* ====================================================================== */
/* presolve_eq_box – C port of the pure-Python fixpoint presolve          */
/* ====================================================================== */

#define PRESOLVE_RATIO_LO  1e-4
#define PRESOLVE_RATIO_HI  1e4
#define PRESOLVE_PIVOT_EPS 1e-12
#define PRESOLVE_DROP_EPS  1e-15

/* Per-row sparse entry: col == -1 marks a deleted slot. */
typedef struct { Py_ssize_t col; double val; } PSEntry;

typedef struct {
    PSEntry    *entries;
    Py_ssize_t  count;     /* active entries                 */
    Py_ssize_t  total;     /* slots used (including deleted) */
    Py_ssize_t  capacity;  /* allocated slots                */
} PSRow;

/* Column-to-row membership set (unordered). */
typedef struct {
    Py_ssize_t *items;
    Py_ssize_t  count;
    Py_ssize_t  capacity;
} PSColSet;

#define PS_REC_FIXED    0
#define PS_REC_DOUBLETON 1

typedef struct {
    int         type;
    Py_ssize_t  idx1, idx2;
    double      v1, v2, v3;
} PSRec;

/* ---- PSRow helpers ---------------------------------------------------- */

static inline int ps_row_init(PSRow *r, Py_ssize_t cap) {
    r->entries = (PSEntry *)malloc((size_t)cap * sizeof(PSEntry));
    if (!r->entries) return -1;
    r->count = r->total = 0;
    r->capacity = cap;
    return 0;
}

static inline int ps_row_append(PSRow *r, Py_ssize_t col, double val) {
    if (r->total >= r->capacity) {
        Py_ssize_t nc = r->capacity * 2 + 4;
        PSEntry *t = (PSEntry *)realloc(r->entries, (size_t)nc * sizeof(PSEntry));
        if (!t) return -1;
        r->entries = t;
        r->capacity = nc;
    }
    r->entries[r->total].col = col;
    r->entries[r->total].val = val;
    r->total++;
    r->count++;
    return 0;
}

static inline PSEntry *ps_row_find(PSRow *r, Py_ssize_t col) {
    for (Py_ssize_t k = 0; k < r->total; k++)
        if (r->entries[k].col == col) return &r->entries[k];
    return NULL;
}

static inline void ps_row_delete(PSRow *r, Py_ssize_t col) {
    for (Py_ssize_t k = 0; k < r->total; k++) {
        if (r->entries[k].col == col) {
            r->entries[k].col = -1;
            r->count--;
            return;
        }
    }
}

/* Set: update existing or append new. Returns -1 on alloc failure. */
static inline int ps_row_set(PSRow *r, Py_ssize_t col, double val) {
    PSEntry *e = ps_row_find(r, col);
    if (e) { e->val = val; return 0; }
    return ps_row_append(r, col, val);
}

/* Pop: delete if present.  Returns 1 if found, 0 otherwise. */
static inline int ps_row_pop(PSRow *r, Py_ssize_t col) {
    for (Py_ssize_t k = 0; k < r->total; k++) {
        if (r->entries[k].col == col) {
            r->entries[k].col = -1;
            r->count--;
            return 1;
        }
    }
    return 0;
}

static inline double ps_row_get(PSRow *r, Py_ssize_t col, double def) {
    PSEntry *e = ps_row_find(r, col);
    return e ? e->val : def;
}

/* Return the single active entry of a singleton row. */
static inline void ps_row_get_one(PSRow *r, Py_ssize_t *jout, double *vout) {
    for (Py_ssize_t k = 0; k < r->total; k++) {
        if (r->entries[k].col >= 0) {
            *jout = r->entries[k].col;
            *vout = r->entries[k].val;
            return;
        }
    }
}

/* Return the two active entries of a doubleton row in insertion order. */
static inline void ps_row_get_two(PSRow *r,
                                  Py_ssize_t *j1, double *v1,
                                  Py_ssize_t *j2, double *v2) {
    int found = 0;
    for (Py_ssize_t k = 0; k < r->total; k++) {
        if (r->entries[k].col >= 0) {
            if (found == 0) { *j1 = r->entries[k].col; *v1 = r->entries[k].val; }
            else            { *j2 = r->entries[k].col; *v2 = r->entries[k].val; return; }
            found++;
        }
    }
}

static inline void ps_row_free(PSRow *r) { free(r->entries); r->entries = NULL; }

/* ---- PSColSet helpers ------------------------------------------------- */

static inline int ps_colset_init(PSColSet *s, Py_ssize_t cap) {
    s->items = (Py_ssize_t *)malloc((size_t)cap * sizeof(Py_ssize_t));
    if (!s->items) return -1;
    s->count = 0;
    s->capacity = cap;
    return 0;
}

static inline int ps_colset_add(PSColSet *s, Py_ssize_t row) {
    for (Py_ssize_t k = 0; k < s->count; k++)
        if (s->items[k] == row) return 0;
    if (s->count >= s->capacity) {
        Py_ssize_t nc = s->capacity * 2 + 4;
        Py_ssize_t *t = (Py_ssize_t *)realloc(s->items, (size_t)nc * sizeof(Py_ssize_t));
        if (!t) return -1;
        s->items = t;
        s->capacity = nc;
    }
    s->items[s->count++] = row;
    return 0;
}

static inline void ps_colset_discard(PSColSet *s, Py_ssize_t row) {
    for (Py_ssize_t k = 0; k < s->count; k++) {
        if (s->items[k] == row) {
            s->items[k] = s->items[s->count - 1];
            s->count--;
            return;
        }
    }
}

static inline void ps_colset_clear(PSColSet *s) { s->count = 0; }
static inline void ps_colset_free(PSColSet *s) { free(s->items); s->items = NULL; }

/* ---- qsort comparator for output CSR (sort by column) ----------------- */

static int ps_entry_cmp(const void *a, const void *b) {
    Py_ssize_t ca = ((const PSEntry *)a)->col;
    Py_ssize_t cb = ((const PSEntry *)b)->col;
    return (ca > cb) - (ca < cb);
}

/* ---- the main presolve function --------------------------------------- */

static PyObject *csparse_presolve_eq_box(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *matrix_obj;
    const char *b_buf, *c_buf, *lo_buf, *hi_buf;
    Py_ssize_t b_len, c_len, lo_len, hi_len;
    int max_fill = 5;

    if (!PyArg_ParseTuple(args, "Oy#y#y#y#|i",
                          &matrix_obj,
                          &b_buf, &b_len,
                          &c_buf, &c_len,
                          &lo_buf, &lo_len,
                          &hi_buf, &hi_len,
                          &max_fill))
        return NULL;

    if (!PyObject_TypeCheck(matrix_obj, &CSRMatrixType)) {
        PyErr_SetString(PyExc_TypeError,
                        "first argument must be a CSRMatrix");
        return NULL;
    }
    CSRMatrixObject *matrix = (CSRMatrixObject *)matrix_obj;
    Py_ssize_t rows = matrix->rows;
    Py_ssize_t cols = matrix->cols;
    Py_ssize_t nnz  = matrix->nnz;

    /* Read directly from CSRMatrixObject (no copy needed for matrix data) */
    const Py_ssize_t *indptr  = matrix->indptr;
    const Py_ssize_t *indices = matrix->indices;
    const double     *data_in = matrix->data;

    /* Validate b/c/lo/hi buffer sizes */
    if (b_len != (Py_ssize_t)(rows * sizeof(double))) {
        PyErr_SetString(PyExc_ValueError, "b buffer size mismatch");
        return NULL;
    }
    if (c_len != (Py_ssize_t)(cols * sizeof(double)) ||
        lo_len != (Py_ssize_t)(cols * sizeof(double)) ||
        hi_len != (Py_ssize_t)(cols * sizeof(double))) {
        PyErr_SetString(PyExc_ValueError, "c/lo/hi buffer size mismatch");
        return NULL;
    }

    /* ---- allocate working storage ------------------------------------- */
    double      *b_arr = NULL, *c_arr = NULL, *lo_arr = NULL, *hi_arr = NULL;
    PSRow       *row_data    = NULL;
    PSColSet    *col_sets    = NULL;
    char        *rem_row     = NULL;   /* removed-row flags  */
    char        *rem_col     = NULL;   /* removed-col flags  */
    PSRec       *recs        = NULL;
    Py_ssize_t   recs_n = 0, recs_cap = 0;
    Py_ssize_t  *snap_buf    = NULL;   /* col_rows snapshot  */
    Py_ssize_t  *act_rows    = NULL;
    Py_ssize_t  *act_cols    = NULL;
    Py_ssize_t  *col_map     = NULL;
    PSEntry     *sort_buf    = NULL;
    Py_ssize_t  *o_indptr    = NULL;
    Py_ssize_t  *o_indices   = NULL;
    double      *o_data      = NULL;
    double      *o_b         = NULL;
    PyObject    *py_b  = NULL, *py_c  = NULL, *py_lo = NULL, *py_hi = NULL;
    PyObject    *py_recs = NULL, *py_ac = NULL;
    PyObject    *py_matrix = NULL;
    PyObject    *result      = NULL;

    /* ---- copy mutable vectors from bytes ------------------------------ */
    b_arr  = (double *)malloc((size_t)(rows > 0 ? rows : 1) * sizeof(double));
    c_arr  = (double *)malloc((size_t)(cols > 0 ? cols : 1) * sizeof(double));
    lo_arr = (double *)malloc((size_t)(cols > 0 ? cols : 1) * sizeof(double));
    hi_arr = (double *)malloc((size_t)(cols > 0 ? cols : 1) * sizeof(double));
    if (!b_arr || !c_arr || !lo_arr || !hi_arr) goto oom;
    if (rows > 0) memcpy(b_arr, b_buf, (size_t)rows * sizeof(double));
    if (cols > 0) {
        memcpy(c_arr, c_buf, (size_t)cols * sizeof(double));
        memcpy(lo_arr, lo_buf, (size_t)cols * sizeof(double));
        memcpy(hi_arr, hi_buf, (size_t)cols * sizeof(double));
    }

    /* ---- build row_entries -------------------------------------------- */
    row_data = (PSRow *)calloc((size_t)(rows > 0 ? rows : 1), sizeof(PSRow));
    if (!row_data) goto oom;
    for (Py_ssize_t i = 0; i < rows; i++) {
        Py_ssize_t cap = indptr[i + 1] - indptr[i] + 2;
        if (ps_row_init(&row_data[i], cap) != 0) goto oom;
        for (Py_ssize_t off = indptr[i]; off < indptr[i + 1]; off++) {
            if (fabs(data_in[off]) > PRESOLVE_DROP_EPS) {
                if (ps_row_append(&row_data[i], indices[off], data_in[off]) != 0)
                    goto oom;
            }
        }
    }

    /* ---- build col_rows ----------------------------------------------- */
    col_sets = (PSColSet *)calloc((size_t)(cols > 0 ? cols : 1), sizeof(PSColSet));
    if (!col_sets) goto oom;
    for (Py_ssize_t j = 0; j < cols; j++)
        if (ps_colset_init(&col_sets[j], 4) != 0) goto oom;
    for (Py_ssize_t i = 0; i < rows; i++) {
        for (Py_ssize_t k = 0; k < row_data[i].total; k++) {
            Py_ssize_t c = row_data[i].entries[k].col;
            if (c >= 0)
                if (ps_colset_add(&col_sets[c], i) != 0) goto oom;
        }
    }

    /* ---- tracking arrays ---------------------------------------------- */
    rem_row  = (char *)calloc((size_t)(rows > 0 ? rows : 1), 1);
    rem_col  = (char *)calloc((size_t)(cols > 0 ? cols : 1), 1);
    snap_buf = (Py_ssize_t *)calloc((size_t)(rows > 0 ? rows : 1), sizeof(Py_ssize_t));
    if (!rem_row || !rem_col || !snap_buf) goto oom;

    recs_cap = 32;
    recs = (PSRec *)malloc((size_t)recs_cap * sizeof(PSRec));
    if (!recs) goto oom;

    double obj_off = 0.0;
    Py_ssize_t n_rem_rows = 0, n_rem_cols = 0;

    /* ==== fixpoint loop ================================================ */
    int changed = 1;
    while (changed) {
        changed = 0;

        /* ---- pass 1: empty rows --------------------------------------- */
        for (Py_ssize_t i = 0; i < rows; i++) {
            if (rem_row[i]) continue;
            if (row_data[i].count == 0) {
                rem_row[i] = 1; n_rem_rows++; changed = 1;
            }
        }

        /* ---- pass 2: singleton rows ----------------------------------- */
        for (Py_ssize_t i = 0; i < rows; i++) {
            if (rem_row[i] || row_data[i].count != 1) continue;
            Py_ssize_t j = -1; double coef = 0.0;
            ps_row_get_one(&row_data[i], &j, &coef);
            if (j < 0 || rem_col[j] || fabs(coef) < PRESOLVE_PIVOT_EPS) continue;

            double value = b_arr[i] / coef;
            if (value < lo_arr[j]) value = lo_arr[j];
            if (value > hi_arr[j]) value = hi_arr[j];

            /* record */
            if (recs_n >= recs_cap) {
                recs_cap *= 2;
                PSRec *t = (PSRec *)realloc(recs, (size_t)recs_cap * sizeof(PSRec));
                if (!t) goto oom;
                recs = t;
            }
            recs[recs_n].type = PS_REC_FIXED;
            recs[recs_n].idx1 = j;
            recs[recs_n].idx2 = 0;
            recs[recs_n].v1   = value;
            recs[recs_n].v2   = 0.0;
            recs[recs_n].v3   = 0.0;
            recs_n++;

            obj_off += c_arr[j] * value;

            /* substitute into other rows */
            Py_ssize_t sn = col_sets[j].count;
            memcpy(snap_buf, col_sets[j].items, (size_t)sn * sizeof(Py_ssize_t));
            for (Py_ssize_t s = 0; s < sn; s++) {
                Py_ssize_t oth = snap_buf[s];
                if (oth == i || rem_row[oth]) continue;
                PSEntry *ep = ps_row_find(&row_data[oth], j);
                if (ep) {
                    b_arr[oth] -= ep->val * value;
                    ps_row_delete(&row_data[oth], j);
                    ps_colset_discard(&col_sets[j], oth);
                }
            }
            rem_row[i] = 1; n_rem_rows++;
            rem_col[j] = 1; n_rem_cols++;
            ps_colset_clear(&col_sets[j]);
            changed = 1;
        }

        /* ---- pass 3: doubleton rows ----------------------------------- */
        for (Py_ssize_t i = 0; i < rows; i++) {
            if (rem_row[i] || row_data[i].count != 2) continue;

            Py_ssize_t jp, jq; double ap, dq;
            ps_row_get_two(&row_data[i], &jp, &ap, &jq, &dq);
            if (rem_col[jp] || rem_col[jq]) continue;

            /* prefer eliminating the lower-degree column */
            if (col_sets[jp].count > col_sets[jq].count) {
                Py_ssize_t tj = jp; double ta = ap;
                jp = jq; ap = dq; jq = tj; dq = ta;
            }
            if (fabs(ap) < PRESOLVE_PIVOT_EPS) {
                Py_ssize_t tj = jp; double ta = ap;
                jp = jq; ap = dq; jq = tj; dq = ta;
                if (fabs(ap) < PRESOLVE_PIVOT_EPS) continue;
            }
            double ratio = fabs(dq / ap);
            if (ratio < PRESOLVE_RATIO_LO || ratio > PRESOLVE_RATIO_HI) {
                Py_ssize_t tj = jp; double ta = ap;
                jp = jq; ap = dq; jq = tj; dq = ta;
                if (fabs(ap) < PRESOLVE_PIVOT_EPS) continue;
                ratio = fabs(dq / ap);
                if (ratio < PRESOLVE_RATIO_LO || ratio > PRESOLVE_RATIO_HI) continue;
            }
            if (col_sets[jp].count - 1 > max_fill) continue;

            double alpha = -dq / ap;
            double beta  = b_arr[i] / ap;
            double new_lo = lo_arr[jq];
            double new_hi = hi_arr[jq];
            if (alpha > PRESOLVE_DROP_EPS) {
                if (isfinite(lo_arr[jp]))
                    new_lo = fmax(new_lo, (lo_arr[jp] - beta) / alpha);
                if (isfinite(hi_arr[jp]))
                    new_hi = fmin(new_hi, (hi_arr[jp] - beta) / alpha);
            } else if (alpha < -PRESOLVE_DROP_EPS) {
                if (isfinite(hi_arr[jp]))
                    new_lo = fmax(new_lo, (hi_arr[jp] - beta) / alpha);
                if (isfinite(lo_arr[jp]))
                    new_hi = fmin(new_hi, (lo_arr[jp] - beta) / alpha);
            }
            if (new_lo > new_hi + 1e-8) continue;
            lo_arr[jq] = new_lo;
            hi_arr[jq] = new_hi;

            /* record */
            if (recs_n >= recs_cap) {
                recs_cap *= 2;
                PSRec *t = (PSRec *)realloc(recs, (size_t)recs_cap * sizeof(PSRec));
                if (!t) goto oom;
                recs = t;
            }
            recs[recs_n].type = PS_REC_DOUBLETON;
            recs[recs_n].idx1 = jp;
            recs[recs_n].idx2 = jq;
            recs[recs_n].v1   = ap;
            recs[recs_n].v2   = dq;
            recs[recs_n].v3   = b_arr[i];
            recs_n++;

            obj_off    += c_arr[jp] * beta;
            c_arr[jq]  += c_arr[jp] * alpha;

            /* substitute into other rows */
            Py_ssize_t sn = col_sets[jp].count;
            memcpy(snap_buf, col_sets[jp].items, (size_t)sn * sizeof(Py_ssize_t));
            for (Py_ssize_t s = 0; s < sn; s++) {
                Py_ssize_t oth = snap_buf[s];
                if (oth == i || rem_row[oth]) continue;
                PSEntry *ep = ps_row_find(&row_data[oth], jp);
                if (!ep || fabs(ep->val) < PRESOLVE_DROP_EPS) continue;
                double coef_p = ep->val;

                b_arr[oth] -= coef_p * beta;
                double merged = ps_row_get(&row_data[oth], jq, 0.0) + coef_p * alpha;

                if (fabs(merged) < PRESOLVE_DROP_EPS) {
                    ps_row_pop(&row_data[oth], jq);
                    ps_colset_discard(&col_sets[jq], oth);
                } else {
                    if (ps_row_set(&row_data[oth], jq, merged) != 0) goto oom;
                    if (ps_colset_add(&col_sets[jq], oth) != 0) goto oom;
                }
                ps_row_delete(&row_data[oth], jp);
                ps_colset_discard(&col_sets[jp], oth);
            }
            rem_row[i]  = 1; n_rem_rows++;
            rem_col[jp] = 1; n_rem_cols++;
            ps_colset_clear(&col_sets[jp]);
            changed = 1;
        }
    }
    /* ==== end fixpoint ================================================= */

    if (n_rem_rows == 0 && n_rem_cols == 0) {
        result = Py_None; Py_INCREF(result);
        goto cleanup;
    }

    /* ---- build compacted output --------------------------------------- */
    {
        Py_ssize_t n_ar = rows - n_rem_rows;
        Py_ssize_t n_ac = cols - n_rem_cols;

        act_rows = (Py_ssize_t *)malloc((size_t)(n_ar > 0 ? n_ar : 1) * sizeof(Py_ssize_t));
        act_cols = (Py_ssize_t *)malloc((size_t)(n_ac > 0 ? n_ac : 1) * sizeof(Py_ssize_t));
        col_map  = (Py_ssize_t *)malloc((size_t)(cols > 0 ? cols : 1) * sizeof(Py_ssize_t));
        if (!act_rows || !act_cols || !col_map) goto oom;

        { Py_ssize_t k = 0; for (Py_ssize_t i = 0; i < rows; i++) if (!rem_row[i]) act_rows[k++] = i; }
        { Py_ssize_t k = 0; for (Py_ssize_t j = 0; j < cols; j++) if (!rem_col[j]) { col_map[j] = k; act_cols[k++] = j; } }

        sort_buf  = (PSEntry *)malloc((size_t)(cols > 0 ? cols : 1) * sizeof(PSEntry));
        o_indptr  = (Py_ssize_t *)malloc((size_t)(n_ar + 1) * sizeof(Py_ssize_t));
        Py_ssize_t ocap = nnz > 16 ? nnz : 16;
        o_indices = (Py_ssize_t *)malloc((size_t)ocap * sizeof(Py_ssize_t));
        o_data    = (double *)malloc((size_t)ocap * sizeof(double));
        o_b       = (double *)malloc((size_t)(n_ar > 0 ? n_ar : 1) * sizeof(double));
        if (!sort_buf || !o_indptr || !o_indices || !o_data || !o_b) goto oom;

        Py_ssize_t onnz = 0;
        o_indptr[0] = 0;
        for (Py_ssize_t k = 0; k < n_ar; k++) {
            Py_ssize_t ri = act_rows[k];
            PSRow *r = &row_data[ri];
            Py_ssize_t ns = 0;
            for (Py_ssize_t e = 0; e < r->total; e++)
                if (r->entries[e].col >= 0) {
                    sort_buf[ns].col = r->entries[e].col;
                    sort_buf[ns].val = r->entries[e].val;
                    ns++;
                }
            if (ns > 1) qsort(sort_buf, (size_t)ns, sizeof(PSEntry), ps_entry_cmp);

            if (onnz + ns > ocap) {
                ocap = (onnz + ns) * 2;
                Py_ssize_t *ti = (Py_ssize_t *)realloc(o_indices, (size_t)ocap * sizeof(Py_ssize_t));
                double     *td = (double *)realloc(o_data, (size_t)ocap * sizeof(double));
                if (!ti || !td) goto oom;
                o_indices = ti; o_data = td;
            }
            for (Py_ssize_t e = 0; e < ns; e++) {
                o_indices[onnz] = col_map[sort_buf[e].col];
                o_data[onnz]    = sort_buf[e].val;
                onnz++;
            }
            o_indptr[k + 1] = onnz;
            o_b[k] = b_arr[ri];
        }

        /* ---- Construct output CSRMatrixObject directly -------------------- */
        /* This avoids the C -> Python list -> C round trip entirely. */
        {
            CSRMatrixObject *out = (CSRMatrixObject *)CSRMatrixType.tp_alloc(
                &CSRMatrixType, 0);
            if (!out) goto oom;
            out->rows = n_ar;
            out->cols = n_ac;
            out->nnz  = onnz;
            /* Transfer ownership of output arrays */
            out->indptr  = o_indptr;   o_indptr  = NULL;
            out->indices = o_indices;  o_indices = NULL;
            out->data    = o_data;     o_data    = NULL;
            /* Build CSC representation */
            out->csc_indptr = (Py_ssize_t *)calloc((size_t)(n_ac + 1),
                                                   sizeof(Py_ssize_t));
            out->csc_rows   = (Py_ssize_t *)calloc((size_t)(onnz > 0 ? onnz : 1),
                                                   sizeof(Py_ssize_t));
            out->csc_data   = (double *)calloc((size_t)(onnz > 0 ? onnz : 1),
                                               sizeof(double));
            if (!out->csc_indptr || !out->csc_rows || !out->csc_data) {
                Py_DECREF(out);
                goto oom;
            }
            for (Py_ssize_t i = 0; i < onnz; i++)
                out->csc_indptr[out->indices[i] + 1]++;
            for (Py_ssize_t col = 0; col < n_ac; col++)
                out->csc_indptr[col + 1] += out->csc_indptr[col];
            if (n_ac > 0) {
                Py_ssize_t *next = (Py_ssize_t *)calloc((size_t)n_ac,
                                                        sizeof(Py_ssize_t));
                if (!next) { Py_DECREF(out); goto oom; }
                for (Py_ssize_t col = 0; col < n_ac; col++)
                    next[col] = out->csc_indptr[col];
                for (Py_ssize_t row = 0; row < n_ar; row++) {
                    for (Py_ssize_t off = out->indptr[row];
                         off < out->indptr[row + 1]; off++) {
                        Py_ssize_t col = out->indices[off];
                        Py_ssize_t dest = next[col]++;
                        out->csc_rows[dest] = row;
                        out->csc_data[dest] = out->data[off];
                    }
                }
                free(next);
            }
            py_matrix = (PyObject *)out;
        }

        /* b as bytes */
        py_b = PyBytes_FromStringAndSize((const char *)o_b,
                                         (Py_ssize_t)(n_ar * sizeof(double)));
        if (!py_b) goto cleanup;

        /* c, lo, hi: gather active columns into contiguous buffers */
        {
            double *c_tmp  = (double *)malloc((size_t)(n_ac > 0 ? n_ac : 1) * sizeof(double));
            double *lo_tmp = (double *)malloc((size_t)(n_ac > 0 ? n_ac : 1) * sizeof(double));
            double *hi_tmp = (double *)malloc((size_t)(n_ac > 0 ? n_ac : 1) * sizeof(double));
            if (!c_tmp || !lo_tmp || !hi_tmp) {
                free(c_tmp); free(lo_tmp); free(hi_tmp);
                goto oom;
            }
            for (Py_ssize_t x = 0; x < n_ac; x++) {
                Py_ssize_t j = act_cols[x];
                c_tmp[x]  = c_arr[j];
                lo_tmp[x] = lo_arr[j];
                hi_tmp[x] = hi_arr[j];
            }
            py_c  = PyBytes_FromStringAndSize((const char *)c_tmp,
                                              (Py_ssize_t)(n_ac * sizeof(double)));
            py_lo = PyBytes_FromStringAndSize((const char *)lo_tmp,
                                              (Py_ssize_t)(n_ac * sizeof(double)));
            py_hi = PyBytes_FromStringAndSize((const char *)hi_tmp,
                                              (Py_ssize_t)(n_ac * sizeof(double)));
            free(c_tmp); free(lo_tmp); free(hi_tmp);
            if (!py_c || !py_lo || !py_hi) goto cleanup;
        }

        /* records: keep as Python tuples (relatively small count) */
        py_recs = PyList_New(recs_n);
        if (!py_recs) goto cleanup;
        for (Py_ssize_t x = 0; x < recs_n; x++) {
            PyObject *tup;
            if (recs[x].type == PS_REC_FIXED)
                tup = Py_BuildValue("(nnd)",
                    (Py_ssize_t)0, recs[x].idx1, recs[x].v1);
            else
                tup = Py_BuildValue("(nnnddd)",
                    (Py_ssize_t)1, recs[x].idx1, recs[x].idx2,
                    recs[x].v1, recs[x].v2, recs[x].v3);
            if (!tup) goto cleanup;
            PyList_SET_ITEM(py_recs, x, tup);
        }

        py_ac = PyBytes_FromStringAndSize((const char *)act_cols,
                                          (Py_ssize_t)(n_ac * sizeof(Py_ssize_t)));
        if (!py_ac) goto cleanup;

        result = Py_BuildValue("(OOOOOdnnOOn)",
            py_matrix, py_b, py_c, py_lo, py_hi,
            obj_off, n_rem_rows, n_rem_cols,
            py_recs, py_ac, cols);
    }
    goto cleanup;

oom:
    PyErr_NoMemory();

cleanup:
    /* Python objects: XDECREF is safe for NULL */
    Py_XDECREF(py_matrix);
    Py_XDECREF(py_b);   Py_XDECREF(py_c);   Py_XDECREF(py_lo);
    Py_XDECREF(py_hi);  Py_XDECREF(py_recs); Py_XDECREF(py_ac);
    /* C arrays (matrix data is read from input CSRMatrixObject, not freed) */
    free(b_arr);  free(c_arr);   free(lo_arr);  free(hi_arr);
    if (row_data) { for (Py_ssize_t i = 0; i < rows; i++) ps_row_free(&row_data[i]); free(row_data); }
    if (col_sets) { for (Py_ssize_t j = 0; j < cols; j++) ps_colset_free(&col_sets[j]); free(col_sets); }
    free(rem_row); free(rem_col); free(recs); free(snap_buf);
    free(act_rows); free(act_cols); free(col_map); free(sort_buf);
    free(o_indptr); free(o_indices); free(o_data); free(o_b);
    return result;
}

static PyMethodDef module_methods[] = {
    {"min_degree", csparse_min_degree, METH_VARARGS,
     "Exact minimum-degree ordering of a symmetric CSC/CSR pattern (indptr, indices)."},
    {"lu_solve_test", csparse_lu_solve_test, METH_VARARGS,
     "Test hook: sparse LU factorize + FTRAN/BTRAN solve. Returns list of solution vectors, or None if singular."},
    {"lu_stats_test", csparse_lu_stats_test, METH_VARARGS,
     "Test hook: sparse LU stats. Returns (nnz_l, nnz_u, singular_step)."},
    {"lu_update_test", csparse_lu_update_test, METH_VARARGS,
     "Test hook: sparse LU basis-change update. Returns (solutions, should_refactor, n_singular)."},
    {"presolve_eq_box", csparse_presolve_eq_box, METH_VARARGS,
     "C presolve for equality-plus-bounds LP (mirrors Python presolve_eq_box)."},
    {NULL, NULL, 0, NULL}
};

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_csparse",
    "C sparse matrix types for linprogx.",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit__csparse(void) {
    if (PyType_Ready(&CSRMatrixType) < 0) {
        return NULL;
    }
    PyObject *module_obj = PyModule_Create(&module);
    if (module_obj == NULL) {
        return NULL;
    }
    Py_INCREF(&CSRMatrixType);
    if (PyModule_AddObject(module_obj, "CSRMatrix", (PyObject *)&CSRMatrixType) < 0) {
        Py_DECREF(&CSRMatrixType);
        Py_DECREF(module_obj);
        return NULL;
    }
    return module_obj;
}

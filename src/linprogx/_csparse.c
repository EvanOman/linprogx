#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static void scaled_op_matvec(const ScaledOp *op, const double *restrict x, double *restrict out) {
    const Py_ssize_t *restrict row_start = op->row_start;
    const int32_t *restrict col_index = op->col_index;
    const double *restrict data = op->data;
    for (Py_ssize_t row = 0; row < op->rows; row++) {
        double total = 0.0;
        Py_ssize_t end = row_start[row + 1];
        for (Py_ssize_t offset = row_start[row]; offset < end; offset++) {
            total += data[offset] * x[col_index[offset]];
        }
        out[row] = total;
    }
}

static void scaled_op_transpose_matvec(const ScaledOp *op, const double *restrict y, double *restrict out) {
    const Py_ssize_t *restrict col_start = op->col_start;
    const int32_t *restrict row_index = op->row_index;
    const double *restrict csc_data = op->csc_data;
    for (Py_ssize_t col = 0; col < op->cols; col++) {
        double total = 0.0;
        Py_ssize_t end = col_start[col + 1];
        for (Py_ssize_t offset = col_start[col]; offset < end; offset++) {
            total += csc_data[offset] * y[row_index[offset]];
        }
        out[col] = total;
    }
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
    static char *kwlist[] = {
        "c", "b", "lo", "hi", "max_iter", "tol", "check_interval", "objective_scale",
        "adaptive_weight", "debug", "restart_sufficient", "restart_necessary",
        "restart_artificial", "eval_interval_override", NULL
    };

    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwds,
            "OOOO|ndndiidddn",
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
            &eval_interval_override)) {
        return NULL;
    }
    if (max_iter < 0 || check_interval <= 0) {
        PyErr_SetString(PyExc_ValueError, "max_iter must be nonnegative and check_interval positive");
        return NULL;
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
    double *ax_trial = NULL;
    double *x_sum = NULL;
    double *y_sum = NULL;
    double *avg_x = NULL;
    double *avg_y = NULL;
    double *x_restart = NULL;
    double *y_restart = NULL;
    double *operator_data = NULL;
    double *operator_csc_data = NULL;
    int32_t *op_col_index = NULL;
    int32_t *op_row_index = NULL;
    unsigned char *bound_kind = NULL;

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
    operator_data = calloc((size_t)self->nnz, sizeof(double));
    operator_csc_data = calloc((size_t)self->nnz, sizeof(double));
    op_col_index = calloc((size_t)self->nnz, sizeof(int32_t));
    op_row_index = calloc((size_t)self->nnz, sizeof(int32_t));
    bound_kind = calloc((size_t)self->cols, sizeof(unsigned char));
    if (c == NULL || b == NULL || lo == NULL || hi == NULL ||
        col_scale == NULL || scaled_lo == NULL || scaled_hi == NULL || scaled_c == NULL ||
        row_scale == NULL || scaled_b == NULL ||
        x == NULL || xbar == NULL || y == NULL || ax == NULL || aty == NULL ||
        y_trial == NULL || ax_trial == NULL ||
        x_sum == NULL || y_sum == NULL || avg_x == NULL || avg_y == NULL ||
        x_restart == NULL || y_restart == NULL ||
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

    double norm;
    norm = estimate_scaled_operator_norm(&op);
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
    const char *status = "iteration_limit";
    KKTEval final_ev;
    evaluate_kkt(
        &op, x, y, c, b, lo, hi, bound_kind,
        col_scale, row_scale, scaled_b, b_l2, c_l2, ax, aty, &final_ev);

    if (kkt_terminated(&final_ev, tol, c_inf)) {
        status = "optimal";
    } else if (max_iter > 0) {
        Py_ssize_t eval_interval = check_interval < 40 ? check_interval : 40;
        if (eval_interval_override > 0) {
            eval_interval = eval_interval_override;
        }
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
            for (int trial = 0; trial < 60; trial++) {
                step_trials++;
                double trial_tau = eta / omega;
                double trial_sigma = eta * omega;
                double dx_sq = 0.0;
                for (Py_ssize_t col = 0; col < self->cols; col++) {
                    double updated = x[col] - trial_tau * (aty[col] + scaled_c[col]);
                    switch (bound_kind[col]) {
                        case 1:
                            if (updated < scaled_lo[col]) {
                                updated = scaled_lo[col];
                            }
                            break;
                        case 2:
                            if (updated > scaled_hi[col]) {
                                updated = scaled_hi[col];
                            }
                            break;
                        case 3:
                            if (updated < scaled_lo[col]) {
                                updated = scaled_lo[col];
                            } else if (updated > scaled_hi[col]) {
                                updated = scaled_hi[col];
                            }
                            break;
                    }
                    xbar[col] = updated;
                    double dx = updated - x[col];
                    dx_sq += dx * dx;
                }
                scaled_op_matvec(&op, xbar, ax_trial);
                double dy_sq = 0.0;
                double interaction = 0.0;
                for (Py_ssize_t row = 0; row < self->rows; row++) {
                    double gradient = 2.0 * ax_trial[row] - ax[row] - scaled_b[row];
                    double updated = y[row] + trial_sigma * gradient;
                    double dy = updated - y[row];
                    y_trial[row] = updated;
                    dy_sq += dy * dy;
                    interaction += dy * (ax_trial[row] - ax[row]);
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
            scaled_op_transpose_matvec(&op, y, aty);
            for (Py_ssize_t col = 0; col < self->cols; col++) {
                x_sum[col] += x[col];
            }
            for (Py_ssize_t row = 0; row < self->rows; row++) {
                y_sum[row] += y[row];
            }
            navg++;
            iterations = iter;
            if (iter % eval_interval != 0 && iter != max_iter) {
                continue;
            }

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
            int best_is_average = ev_average.kkt < ev_current.kkt;
            KKTEval ev_best = best_is_average ? ev_average : ev_current;

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
                     * versa. */
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
            }
        }
        Py_END_ALLOW_THREADS
    }

    double max_residual = final_ev.primal_res_max;
    double l2_residual = final_ev.primal_res_l2;

    for (Py_ssize_t col = 0; col < self->cols; col++) {
        x[col] *= col_scale[col];
    }

    if (max_residual > tol) {
        Py_BEGIN_ALLOW_THREADS
        active_set_cgls_cleanup(self, x, b, lo, hi, bound_kind, tol, &max_residual, &l2_residual);
        Py_END_ALLOW_THREADS
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
        result = Py_BuildValue(
            "{s:s,s:d,s:d,s:d,s:n,s:d,s:d,s:d,s:d,s:d,s:d,s:n,s:n,s:N}",
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
            "x",
            x_list);
    }

done:
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
    free(ax_trial);
    free(x_sum);
    free(y_sum);
    free(avg_x);
    free(avg_y);
    free(x_restart);
    free(y_restart);
    free(operator_data);
    free(operator_csc_data);
    free(op_col_index);
    free(op_row_index);
    free(bound_kind);
    return result;
}

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

static PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_csparse",
    "C sparse matrix types for linprogx.",
    -1,
    NULL,
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
